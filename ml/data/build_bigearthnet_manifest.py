"""Build a small (100-500 sample) BigEarthNet v2.0 manifest without downloading the
full ~59 GiB (S2) + ~51 GiB (S1) archives.

Strategy (documented, "do not download the entire dataset blindly"):
  1. Download the small `metadata.parquet` (patch_id <-> s1_name mapping, country,
     season, split hints) fully — this is a few hundred MB, not image data.
  2. Pick the first N *tiles* (Sentinel-2 scene IDs) worth of patches from the
     metadata so every kept sample has a real tile_id for the geographic split.
  3. Stream-decompress BigEarthNet-S2.tar.zst / BigEarthNet-S1.tar.zst /
     Reference_Maps.tar.zst from the network (requests + zstandard + tarfile in
     streaming "r|" mode) and extract ONLY the members belonging to the selected
     patches, stopping as soon as every selected patch has been found or a byte
     budget is exhausted. The rest of each multi-gigabyte archive is never fetched.

This script requires `requests`, `zstandard`, and `pyarrow` (installed on demand,
see docs/ml/sarfuseseg_dataset.md) and network access to zenodo.org.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import requests

from ml.sarfuseseg.config import DATA_ROOT, MANIFEST_DIR, MAX_SAMPLES_FIRST_PASS
from ml.sarfuseseg.manifest import ManifestEntry, assign_geographic_splits, write_manifest

ZENODO_RECORD = "https://zenodo.org/records/10891137/files"
METADATA_URL = f"{ZENODO_RECORD}/metadata.parquet?download=1"
S2_ARCHIVE_URL = f"{ZENODO_RECORD}/BigEarthNet-S2.tar.zst?download=1"
S1_ARCHIVE_URL = f"{ZENODO_RECORD}/BigEarthNet-S1.tar.zst?download=1"
REFMAP_ARCHIVE_URL = f"{ZENODO_RECORD}/Reference_Maps.tar.zst?download=1"

#: Stop scanning an archive after this many decompressed bytes even if some selected
#: patches were never found (keeps the "small subset" promise bandwidth-bounded).
DEFAULT_SCAN_BUDGET_BYTES = 8 * 1024 ** 3  # 8 GiB per archive


@dataclass(frozen=True)
class SelectedPatch:
    s2_patch_id: str
    s1_patch_name: str
    tile_id: str
    country: str | None
    season: str | None


def download_metadata(dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(METADATA_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def select_patches(metadata_path: Path, n_tiles: int) -> list[SelectedPatch]:
    import pyarrow.parquet as pq

    table = pq.read_table(metadata_path)
    df = table.to_pandas()
    # Sentinel-2 tile id = everything before the patch-index suffix in patch_id,
    # e.g. "S2A_MSIL2A_20170613T101031_N0205_R022_T33UUP" from
    # "S2A_MSIL2A_..._T33UUP_12_34".
    df["tile_id"] = df["patch_id"].str.rsplit("_", n=2).str[0]
    chosen_tiles = df["tile_id"].drop_duplicates().head(n_tiles).tolist()
    subset = df[df["tile_id"].isin(chosen_tiles)]
    return [
        SelectedPatch(
            s2_patch_id=row.patch_id,
            s1_patch_name=row.s1_name,
            tile_id=row.tile_id,
            country=getattr(row, "country", None),
            season=getattr(row, "season", None),
        )
        for row in subset.itertuples()
    ]


def download_url_to_file_with_resume(url: str, dest_path: Path, timeout: int = 120, max_retries: int = 3) -> Path:
    """Download a URL to a file, resuming from a partial temp file on transient drops."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    start = dest_path.stat().st_size if dest_path.exists() else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Range": f"bytes={start}-"} if start else {}
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                resp.raise_for_status()
                if start and resp.status_code == 200:
                    # Server ignored the Range header; rewrite the file from scratch.
                    start = 0
                    dest_path.unlink(missing_ok=True)
                mode = "ab" if start else "wb"
                with open(dest_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        start += len(chunk)
            return dest_path
        except (requests.exceptions.RequestException, OSError, ValueError) as exc:  # pragma: no cover - network only
            if attempt == max_retries:
                raise
            # Retry from the last fully written byte to survive mid-stream disconnections.
            start = dest_path.stat().st_size if dest_path.exists() else 0

    raise RuntimeError(f"Failed to download {url} to {dest_path}")


def stream_extract_selected(
    archive_url: str,
    wanted_prefixes: set[str],
    dest_dir: Path,
    budget_bytes: int = DEFAULT_SCAN_BUDGET_BYTES,
    max_retries: int = 3,
) -> set[str]:
    """Stream-decompress a .tar.zst URL and extract only members under a wanted prefix.

    Reads directly off the HTTP response body (never buffers the multi-GiB archive to
    disk first) and stops as soon as ``budget_bytes`` decompressed bytes have been
    scanned or every wanted prefix has been found. On a dropped connection, retries
    from the start of the stream rather than resuming a partial file, so a flaky
    network still cannot balloon total transfer past ``max_retries * budget_bytes``.

    Returns the set of prefixes actually found within the byte budget.
    """
    import time

    import zstandard as zstd

    dest_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    last_exc: Exception | None = None
    archive_name = archive_url.rsplit("/", 1)[-1].split("?")[0]

    for attempt in range(1, max_retries + 1):
        print(f"[{archive_name}] attempt {attempt}/{max_retries}: connecting...", flush=True)
        try:
            with requests.get(archive_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                resp.raw.decode_content = True
                print(f"[{archive_name}] connected, streaming response body", flush=True)
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(resp.raw) as reader:
                    wrapped = io.BufferedReader(reader, buffer_size=1 << 20)
                    with tarfile.open(fileobj=wrapped, mode="r|") as tar:
                        bytes_scanned = 0
                        members_seen = 0
                        start_time = time.monotonic()
                        last_log_time = start_time
                        last_log_bytes = 0
                        next_log_at = 8 * 1024 * 1024
                        for member in tar:
                            bytes_scanned += member.size
                            members_seen += 1
                            top = member.name.split("/")[0]
                            if any(top.startswith(prefix) for prefix in wanted_prefixes):
                                try:
                                    tar.extract(member, path=dest_dir, filter="data")
                                except TypeError:
                                    tar.extract(member, path=dest_dir)
                                found.add(top)
                                print(
                                    f"[{archive_name}] matched {top} "
                                    f"({len(found)}/{len(wanted_prefixes)} found)",
                                    flush=True,
                                )
                            if bytes_scanned >= next_log_at:
                                now = time.monotonic()
                                elapsed = now - start_time
                                interval = max(now - last_log_time, 1e-6)
                                speed_mibs = (bytes_scanned - last_log_bytes) / (1 << 20) / interval
                                avg_mibs = bytes_scanned / (1 << 20) / max(elapsed, 1e-6)
                                print(
                                    f"[{archive_name}] {bytes_scanned / (1 << 20):.1f} MiB scanned "
                                    f"| {speed_mibs:.2f} MiB/s now, {avg_mibs:.2f} MiB/s avg "
                                    f"| {members_seen} members | {len(found)}/{len(wanted_prefixes)} found "
                                    f"| {elapsed:.0f}s elapsed",
                                    flush=True,
                                )
                                last_log_time = now
                                last_log_bytes = bytes_scanned
                                next_log_at += 8 * 1024 * 1024
                            if bytes_scanned >= budget_bytes or found >= wanted_prefixes:
                                print(
                                    f"[{archive_name}] done: {len(found)}/{len(wanted_prefixes)} found, "
                                    f"{bytes_scanned / (1 << 20):.0f} MiB scanned",
                                    flush=True,
                                )
                                return found
            return found
        except (requests.exceptions.RequestException, OSError, ValueError) as exc:  # pragma: no cover - network only
            last_exc = exc
            print(f"[{archive_name}] attempt {attempt} failed: {exc}", flush=True)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to fetch or stream {archive_url}: {exc}") from exc

    raise RuntimeError(f"Failed to fetch or stream {archive_url}: {last_exc}")


class BuildAlreadyRunning(RuntimeError):
    """Raised when another build_bigearthnet_manifest process holds the lock."""


@contextmanager
def _single_instance_lock(lock_path: Path):
    """Exclusive lock so a second concurrent launch fails fast instead of racing on
    the same partially-extracted files (root cause of repeated corrupt-subset crashes).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise BuildAlreadyRunning(
            f"Another build_bigearthnet_manifest run is already using {lock_path}. "
            "Wait for it to finish (or delete the lock file if you are sure no other "
            "instance is running) before starting a new one."
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def build_manifest(n_tiles: int = 20, max_samples: int = MAX_SAMPLES_FIRST_PASS) -> Path:
    """End-to-end: metadata -> patch selection -> bounded streaming extraction -> manifest."""
    with _single_instance_lock(DATA_ROOT / ".build.lock"):
        metadata_path = download_metadata(DATA_ROOT / "metadata.parquet")
        patches = select_patches(metadata_path, n_tiles=n_tiles)[:max_samples]
        print(f"selected {len(patches)} candidate patches from {n_tiles} tiles", flush=True)

        s2_prefixes = {p.s2_patch_id for p in patches}
        s1_prefixes = {p.s1_patch_name for p in patches}

        print("scanning S2 archive...", flush=True)
        found_s2 = stream_extract_selected(S2_ARCHIVE_URL, s2_prefixes, DATA_ROOT / "S2")
        print("scanning S1 archive...", flush=True)
        found_s1 = stream_extract_selected(S1_ARCHIVE_URL, s1_prefixes, DATA_ROOT / "S1")
        print("scanning reference-map archive...", flush=True)
        found_ref = stream_extract_selected(REFMAP_ARCHIVE_URL, s2_prefixes, DATA_ROOT / "reference_maps")

        entries = []
        for p in patches:
            if p.s2_patch_id not in found_s2 or p.s1_patch_name not in found_s1:
                continue  # not found within the scan budget -> excluded, not guessed
            entries.append(
                ManifestEntry(
                    sample_id=p.s2_patch_id,
                    s2_patch_id=p.s2_patch_id,
                    s1_patch_name=p.s1_patch_name,
                    tile_id=p.tile_id,
                    optical_path=str(DATA_ROOT / "S2" / p.s2_patch_id),
                    sar_path=str(DATA_ROOT / "S1" / p.s1_patch_name),
                    label_path=str(DATA_ROOT / "reference_maps" / p.s2_patch_id) if p.s2_patch_id in found_ref else None,
                    country=p.country,
                    season=p.season,
                )
            )

        entries = assign_geographic_splits(entries)
        manifest_path = MANIFEST_DIR / "bigearthnet_subset_manifest.json"
        write_manifest(entries, manifest_path)
        return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_FIRST_PASS)
    args = parser.parse_args()
    path = build_manifest(n_tiles=args.n_tiles, max_samples=args.max_samples)
    print(f"manifest written to {path}")
