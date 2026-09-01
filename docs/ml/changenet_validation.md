# ChangeNet validation notes (Phase 1 acceptance)

## Residual alignment threshold — revised after real-data testing

Initial design: `MAX_RESIDUAL_OFFSET_PIXELS = 1.5`, using normalized phase
correlation (`cv2.phaseCorrelate`) between T1/T2 luminance, reject above threshold.

**Finding while validating on real LEVIR-CD pairs**: the raw *offset* alone is not
trustworthy on genuine change pairs. Two known-co-registered LEVIR-CD samples
(`test_7_0256_0512.png`, `test_55_0256_0000.png` — vacant lots becoming built-up)
were rejected with 7.9 px and 4.1 px "offsets", far above the 1.5 px threshold. But
their correlation *response* (peak sharpness, 0-1) was only 0.05-0.09 — near noise
level — versus 0.999 for a synthetic, deliberately-shifted control (see debug
session below). A large real change region (new construction covering much of the
256x256 frame) biases/blurs the phase-correlation peak; the "offset" it reports in
that regime is not a reliable misalignment estimate.

Fix (`ml/changenet/config.py::ALIGNMENT_MIN_CONFIDENCE = 0.3`,
`ml/changenet/preprocessing.py::check_residual_alignment`): the offset is only used
to reject a pair when the correlation response is >= 0.3. Below that, the pair is
passed through with a warning ("likely genuine scene change dominating the frame,
not misalignment") instead of a hard rejection. Above 0.3, the original 1.5 px
threshold still applies.

Debug evidence (`.venv-ml` REPL, see session log):

| case | dx, dy | offset (px) | response |
|---|---|---|---|
| image vs itself | 0, 0 | 0.0 | 0.9999 |
| synthetic 3px,2px shift | 3.00, 2.00 | 3.6 | 0.9995 |
| test_7_0256_0512 (real change pair) | -4.84, -6.21 | 7.87 | 0.057 |
| test_55_0256_0000 (real change pair) | -4.07, -0.57 | 4.11 | 0.091 |

This is a first-pass, documented threshold pair (offset limit + confidence floor),
not derived from a large controlled misalignment sweep — revisit once more real
(non-benchmark) T1/T2 pairs are available. It should also be revisited if a future
scene shows *both* high correlation confidence and a genuine geometric shift (which
this rule correctly still rejects).

Separately, `test_2_0000_0000.png` (listed in `samples_LEVIR/list/demo.txt`) turned
out, on visual inspection, to be a **mismatched pair** — T1 is forest, T2 is a
residential neighbourhood, i.e. not the same location at all. The alignment check
correctly rejected it outright (53 px offset) even before the confidence fix, which
is the intended behaviour for a genuinely bad pair; it was swapped out for
`test_7_0256_0512.png` for the acceptance run below.

## Mask cleanup rule

`ml/changenet/config.py` / `ml/changenet/postprocessing.py::CleanupRule`:

1. Binarize the softmax change-probability map at `p >= 0.5`.
2. Binary opening, 3x3 structuring element (removes isolated/diagonal single-pixel
   noise).
3. Binary closing, 3x3 structuring element (fills 1-pixel gaps in otherwise
   contiguous thin regions).
4. Drop connected components smaller than 8 px (documented minimum mapping unit for
   this phase; revisit once a georeferenced accuracy benchmark exists).

## Acceptance run

Run: `python -m ml.changenet.run_demo` (see ml/README.md).

Input: LEVIR-CD sample pair `samples_LEVIR/A/test_7_0256_0512.png` /
`samples_LEVIR/B/test_7_0256_0512.png` (real pixels shipped with the vendored
ChangeFormer repo for its own quick-start demo — vacant lots in T1, new houses built
on the same lots in T2).

Note: `test_2_0000_0000.png` (also listed in `list/demo.txt`) turned out to be a
mismatched pair when inspected visually (T1 = forest, T2 = a residential
neighbourhood) — the phase-correlation alignment check correctly rejected it
(53 px residual, threshold 1.5 px) rather than silently running inference on two
unrelated scenes. That rejection is itself a useful validation of the alignment gate.

Two paths were exercised:

- **Path A (no georeferencing)**: the PNGs as-is. Expected output: pixel area,
  area percent, relative location string, `geographic_coordinates_available: false`,
  no CRS/m^2 anywhere in the result (plan section 4.4 requirement).
- **Path B (synthetic georeferencing)**: the same pixels wrapped in a GeoTIFF with a
  documented, clearly-labeled **synthetic** UTM 33N transform (LEVIR-CD's published
  ~0.5 m/px resolution, but an arbitrary anchor coordinate — not a real-world claim).
  Expected output: polygons in EPSG:32633, `area_m2`, `measurement_crs_epsg: 32633`.

Actual GPU/runtime numbers and the raw JSON result are appended below once the run
completes (see `artifacts/reports/changenet_demo/*/statistics.json`).

<!-- RESULTS_PLACEHOLDER: filled in after ml.changenet.run_demo runs successfully -->
