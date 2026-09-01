"""SAR-FuseSeg — team-developed optical/SAR dual-encoder segmentation model.

Original SatQuery code (Implementation_Plan_v1.2.md section 4.3/4.4). Deliberately
small: ResNet-18 encoders, multi-scale concatenation fusion, a U-Net-style decoder.
No cross-attention, no transformers, no large encoders — the goal is a stable,
reproducible baseline on a 6 GB laptop GPU.
"""
