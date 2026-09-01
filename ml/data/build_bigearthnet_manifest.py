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
import tarfile
from dataclasses import dataclass
from pathlib import Path

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


def stream_extract_selected(
    archive_url: str,
    wanted_prefixes: set[str],
    dest_dir: Path,
    budget_bytes: int = DEFAULT_SCAN_BUDGET_BYTES,
) -> set[str]:
    """Stream-decompress a .tar.zst URL and extract only members under a wanted prefix.

    Returns the set of prefixes actually found within the byte budget.
    """
    import requests
    import zstandard as zstd

    dest_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()

    with requests.get(archive_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(resp.raw, read_size=1 << 20) as reader:
            wrapped = io.BufferedReader(reader, buffer_size=1 << 20)  # tarfile needs .read()
            with tarfile.open(fileobj=wrapped, mode="r|") as tar:
                bytes_scanned = 0
                for member in tar:
                    bytes_scanned += member.size
                    top = member.name.split("/")[0]
                    if any(top.startswith(prefix) for prefix in wanted_prefixes):
                        # `filter=` (PEP 706) needs Python >= 3.10.12/3.9.17/3.8.17;
                        # fall back silently on older patch releases (still Python 3.10).
                        try:
                            tar.extract(member, path=dest_dir, filter="data")
                        except TypeError:
                            tar.extract(member, path=dest_dir)
                        found.add(top)
                    if bytes_scanned >= budget_bytes or found >= wanted_prefixes:
                        break
    return found


def build_manifest(n_tiles: int = 20, max_samples: int = MAX_SAMPLES_FIRST_PASS) -> Path:
    """End-to-end: metadata -> patch selection -> bounded streaming extraction -> manifest."""
    metadata_path = download_metadata(DATA_ROOT / "metadata.parquet")
    patches = select_patches(metadata_path, n_tiles=n_tiles)[:max_samples]

    s2_prefixes = {p.s2_patch_id for p in patches}
    s1_prefixes = {p.s1_patch_name for p in patches}

    found_s2 = stream_extract_selected(S2_ARCHIVE_URL, s2_prefixes, DATA_ROOT / "S2")
    found_s1 = stream_extract_selected(S1_ARCHIVE_URL, s1_prefixes, DATA_ROOT / "S1")
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
