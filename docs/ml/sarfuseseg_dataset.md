# BigEarthNet v2.0 — source, licence, structure notes (for SAR-FuseSeg)

## Source and version

- Official site: https://bigearth.net/ (RSiM Group + DIMA Group, TU Berlin; BIFOLD).
- Version used: **v2.0**, also called **reBEN** ("Refined BigEarthNet"), paper:
  K. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, "reBEN: Refined
  BigEarthNet Dataset for Remote Sensing Image Analysis", IGARSS 2025
  (https://arxiv.org/abs/2407.03653).
- Scale: 549,488 co-registered Sentinel-1/Sentinel-2 patch pairs, 115 Sentinel-2
  tiles + 312 Sentinel-1 scenes, June 2017-May 2018, 10 European countries.
- Reference (pixel-level) maps derived from CORINE Land Cover 2018 (`CLC2018
  v2020_u1`) — see "coarse label" notes below.

## Licence

**Community Data License Agreement - Permissive, Version 1.0 (CDLA-Permissive-1.0)**
(https://cdla.dev/permissive-1-0/), stated on https://bigearth.net/. This is a
permissive data licence (attribution, no share-alike requirement) — compatible with
this project's use.

## IMPORTANT: `BigEarthNet.txt` is a *different* dataset, reserved for SatVLM

While researching downloads we found a Hugging Face dataset literally named
`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`
(https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt). This is
**not** raw imagery — it is a large (9.6M rows) image-*text* dataset: captions, VQA
pairs (binary/MCQ) and referring-expression instructions built on top of BigEarthNet
v2.0 image pairs, for training/evaluating vision-language models (paper:
arXiv:2603.29630). This exactly matches the task brief's instruction to reserve
`BigEarthNet.txt` for the future SatVLM/Qwen2.5-VL phase and NOT use it for
SAR-FuseSeg. Confirmed and recorded here so nobody accidentally downloads it for
segmentation training.

For SAR-FuseSeg we instead use the **raw BigEarthNet-S1/S2 patch archives** and the
**Reference_Maps** archive, all linked from https://bigearth.net/:

- `BigEarthNet-S2.tar.zst` (~59 GiB) — Sentinel-2 patches, one GeoTIFF per band.
- `BigEarthNet-S1.tar.zst` (~51 GiB) — Sentinel-1 patches (VV, VH GeoTIFFs).
- `Reference_Maps.tar.zst` — pixel-level CLC2018 label rasters, one per S2 patch.
- `metadata.parquet` (few hundred MB) — patch_id <-> s1_name mapping, country,
  season, split hints, snow/cloud/shadow flags.
- Hosted on Zenodo record 10891137 (https://zenodo.org/records/10891137).

## Data structure (per the dataset description document)

Each Sentinel-2 patch is a folder named after its `patch_id`
(`S2A_MSIL2A_<date>_N<...>_R<...>_T<tile>_<row>_<col>`), containing one GeoTIFF per
band (`..._B02.tif`, `..._B03.tif`, ...). Each Sentinel-1 patch (`s1_name`) is a
folder with `..._VV.tif` / `..._VH.tif`. Patches are natively **120x120 px at 10 m
GSD** (1.2 km x 1.2 km) — smaller than this project's 256x256 tile-size standard;
see `ml/sarfuseseg/dataset.py` for the documented reflect-pad-to-256 decision for
this first pass.

## Why we do not download the full archives

~110 GiB combined is not compatible with "start small: 100-500 valid paired
samples" or this machine's disk/bandwidth budget for a first pass. Instead
`ml/data/build_bigearthnet_manifest.py`:

1. Downloads only `metadata.parquet` in full (small).
2. Selects the first N Sentinel-2 *tiles* (not just patches) from the metadata, so
   every kept sample has a real `tile_id` for the geographic split.
3. Stream-decompresses each `.tar.zst` archive (requests + `zstandard` +
   `tarfile` in sequential `"r|"` mode) and extracts **only** the members matching
   the selected patch/S1 names, stopping once every selected patch is found or a
   per-archive byte budget (default 8 GiB) is exhausted — the remainder of each
   50-60 GiB archive is never fetched.

If a selected patch's S1/S2/reference-map member isn't found within the byte
budget, it is dropped from the manifest rather than guessed — the manifest records
only patches that were actually located and verified.

## Weak/coarse label handling

CORINE Land Cover has a minimum mapping unit of 25 ha (polygons), rasterized onto
BigEarthNet's 10 m grid. This means reference-map "pixel" labels near class
boundaries are inherently coarse relative to true 10 m ground resolution. This
project's response (implemented across `ml/sarfuseseg/class_mapping.py` and the
manifest's `label_provenance` / `label_confidence` fields):

- Every manifest entry records `label_provenance = "corine_clc2018_reference_map"`
  so coarse labels are never presented as if they were manually annotated.
- Patches BigEarthNet itself flags as covered by seasonal snow, cloud, or cloud
  shadow (`metadata_for_patches_with_snow_cloud_or_shadow.parquet`) are excluded by
  only ever reading from the clean `metadata.parquet` file.
- CLC codes with no entry in `manifests/class_mapping_v1.json` map to
  `IGNORE_INDEX` (255), not to a guessed class.
