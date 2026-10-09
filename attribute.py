import json
import re
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


# -------------------------------------------------------------------------
# 1. Hardware & 4-bit Quantization Config (Fits inside ~1.6 GB VRAM)
# -------------------------------------------------------------------------
def loadQwen2bQuantized() ->  Qwen2VLForConditionalGeneration:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # Force load entirely onto GPU 0; prevents accelerate from spilling to disk
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    return model

def getAutoProcessor() -> AutoProcessor:
    # Cap visual tokens so high-res crops do not cause transient VRAM spikes
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_id,
        size={
            "longest_edge": 512,
            "shortest_edge": 256,
        },
    )

    return processor


# -------------------------------------------------------------------------
# 2. Atribute Extraction
# -------------------------------------------------------------------------
def analyze_crop_features(model: Qwen2VLForConditionalGeneration, processor: AutoProcessor, crop_img: Image.Image) -> dict:
    prompt = (
        "Analyze this clothing crop for an e-commerce catalog. Return ONLY a valid JSON object with:\n"
        "- category (e.g., hoodie, sneakers, cargo pants)\n"
        "- brand_or_logo (name visible brand/logo like Nike, Adidas, Carhartt, or null if none)\n"
        "- dominant_color (precise color, e.g. washed charcoal, olive green)\n"
        "- graphic_or_print (describe visible graphic, typography, pattern, or 'solid')\n"
        "- distinguishing_features (list of 2-4 visual attributes like zippers, ribbed cuffs, distressed hem)"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[crop_img],
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,  
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0].strip()

    # Robust JSON extraction: handles code fences or stray text
    match = re.search(r"(\{.*\})", output_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse JSON", "raw_output": output_text}