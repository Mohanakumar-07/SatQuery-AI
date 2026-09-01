# ChangeFormer V6 — provenance and licence record

## Source

- Repository: https://github.com/wgcban/ChangeFormer (author: Wele Gedara Chaminda Bandara, wgcban)
- Paper: "A Transformer-Based Siamese Network for Change Detection", IGARSS 2022.
  ArXiv: https://arxiv.org/abs/2201.01293, IEEE: https://ieeexplore.ieee.org/document/9883686
- Vendored commit: shallow clone (`--depth 1`) of `main` taken 2026-09-01, stored at
  `ml/vendor/ChangeFormer/` (not modified in place; adapter code lives in `ml/changenet/`).
- Release used: `v0.1.0` (only tagged release).

## Pretrained checkpoint

- Variant: `ChangeFormerV6`, `embed_dim=256`, `img_size=256`, trained on LEVIR-CD.
- Download URL:
  `https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip`
- A DSIFN-trained variant also exists at the same release (not used in this phase; LEVIR-CD
  is the closest published benchmark to generic bi-temporal optical scenes).
- "ChangeFormer V6" is the name of the finalized network architecture variant in the paper/repo
  (`net_G=ChangeFormerV6`), not a separate numbered release of the checkpoint file.

## Licence — IMPORTANT CONTRADICTION, flagged for the team

The repository carries **two conflicting licence statements**:

1. `LICENSE` file at the repo root (raw content fetched 2026-09-01):
   ```
   MIT License
   Copyright (c) 2023 Chaminda Bandara
   ```
   — a permissive licence with no field-of-use restriction.

2. The `README.md` "License" section states, verbatim:
   > "Code is released for non-commercial and research purposes only. For commercial
   > purposes, please contact the authors."

These two statements are inconsistent. GitHub's licence detector reports "MIT license" from
the LICENSE file, but the author's prose in the README imposes a non-commercial restriction
that MIT does not carry.

**Decision for SatQuery AI (SIH hackathon / research prototype):** treat the stricter,
non-commercial/research-only statement as controlling, since it is the author's explicit
intent. This is compatible with the current MVP/hackathon use (research and demonstration,
not a commercial product). If SatQuery AI is ever productized commercially, contact the
author (per the README) before continuing to use this code or checkpoint.

The pretrained checkpoint itself is not accompanied by a separate licence file; it is
distributed from the same repository/release and is treated under the same terms above.

## What we vendor vs. what we write

- `ml/vendor/ChangeFormer/`: unmodified upstream code (network definitions, `basic_model.py`,
  `utils.py`, samples). Treated as third-party and excluded from our own code-quality/lint
  scope.
- `ml/changenet/`: original SatQuery code — preprocessing, the `ChangeNetAdapter`, mask
  post-processing, polygonisation and area calculation. This is what plan section 4.4
  describes and is original to this project.
