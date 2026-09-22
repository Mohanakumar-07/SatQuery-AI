"""serve_qwen_local.py -- Local GPU inference server for SatVLM (Qwen2.5-VL-3B-Instruct 4-bit).

Exposes a fast HTTP interface on your laptop so Render (or any cloud deployment) can
dispatch vision queries over a secure Cloudflare Tunnel (`cloudflared`).

Usage:
  python serve_qwen_local.py [--port 8001] [--host 0.0.0.0]

Then run in a separate terminal:
  cloudflared tunnel --url http://localhost:8001
"""

import argparse
import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Add backend and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.models.satvlm_adapter import resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("serve_qwen_local")

app = FastAPI(title="SatQuery Local Qwen2.5-VL Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL = None
_PROCESSOR = None


def get_model_and_processor():
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine. GPU is required for 4-bit NF4 inference.")

    model_path = resolve_model_path()
    if model_path is None:
        raise FileNotFoundError("Local Qwen2.5-VL-3B-Instruct model checkpoint was not found.")

    logger.info("Loading Qwen2.5-VL-3B-Instruct from %s in 4-bit NF4...", model_path)
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map={"": 0},
        local_files_only=True,
    )

    _MODEL = model
    _PROCESSOR = processor
    logger.info("Qwen2.5-VL-3B-Instruct loaded on GPU: %s", torch.cuda.get_device_name(0))
    return _MODEL, _PROCESSOR


class ImagePayload(BaseModel):
    label: str = "primary"
    base64: str
    mime_type: Optional[str] = "image/png"


class InferRequest(BaseModel):
    analysis_id: Optional[str] = None
    question: str
    system_prompt: Optional[str] = None
    evidence_hint: Optional[str] = ""
    chat_history: Optional[list[dict[str, Any]]] = []
    images: Optional[list[ImagePayload]] = []
    task: Optional[str] = "single_scene_vqa"


@app.get("/")
@app.get("/health")
def health_check():
    cuda_ready = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ready else "None"
    return {
        "status": "healthy" if cuda_ready else "degraded",
        "service": "SatQuery Local Qwen2.5-VL Gateway",
        "cuda_available": cuda_ready,
        "device": device_name,
        "model": "Qwen2.5-VL-3B-Instruct",
        "endpoints": ["GET /health", "POST /infer"],
    }



@app.post("/infer")
def infer(req: InferRequest):
    try:
        model, processor = get_model_and_processor()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    from qwen_vl_utils import process_vision_info

    # Parse images
    pil_images = []
    for item in (req.images or []):
        try:
            raw_bytes = base64.b64decode(item.base64)
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_images.append((item.label, img))
        except Exception as exc:
            logger.warning("Failed to decode image %s: %s", item.label, exc)

    sys_prompt = req.system_prompt or "You are SatVLM, an expert geospatial vision-language specialist."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt}
    ]

    for turn in (req.chat_history or []):
        t_role = turn.get("role", "user")
        t_content = turn.get("content", "")
        if t_content:
            messages.append({"role": t_role, "content": str(t_content)})

    user_content: list[dict[str, Any]] = []

    # Check before/after or single image
    before_img = next((img for label, img in pil_images if label == "before"), None)
    after_img = next((img for label, img in pil_images if label == "after"), None)

    if before_img and after_img:
        user_content.append({"type": "text", "text": f"Question: {req.question}\n\nBefore image:"})
        user_content.append({"type": "image", "image": before_img})
        user_content.append({"type": "text", "text": "After image:"})
        user_content.append({"type": "image", "image": after_img})
        if req.evidence_hint:
            user_content.append({"type": "text", "text": req.evidence_hint})
    elif pil_images:
        primary_img = pil_images[0][1]
        user_content.append({"type": "image", "image": primary_img})
        prompt_text = f"Question: {req.question}"
        if req.evidence_hint:
            prompt_text += f"\n{req.evidence_hint}"
        user_content.append({"type": "text", "text": prompt_text})
    else:
        prompt_text = f"Question: {req.question}"
        if req.evidence_hint:
            prompt_text += f"\n{req.evidence_hint}"
        user_content.append({"type": "text", "text": prompt_text})

    messages.append({"role": "user", "content": user_content})

    trace = ["local_gateway_received"]
    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        trace.append("local_gateway_generated")
        return {
            "status": "success",
            "output_text": output_text,
            "trace": trace,
            "model": "Qwen2.5-VL-3B-Instruct",
        }
    except Exception as exc:
        logger.error("Local inference error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SatQuery Local Qwen2.5-VL Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default 8001)")
    args = parser.parse_args()

    # Pre-warm model on startup
    try:
        get_model_and_processor()
    except Exception as e:
        logger.warning("Could not pre-load model on startup (will retry on first request): %s", e)

    logger.info("Starting local Qwen gateway on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)

