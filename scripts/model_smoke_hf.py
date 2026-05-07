#!/usr/bin/env python3
"""Single-model VLM smoke tests for the cluster.

Each invocation loads one Hugging Face VLM on the single GPU exposed through
CUDA_VISIBLE_DEVICES, runs one neutral OCR-style prompt, and writes a small
JSON result next to the log. The launcher runs these invocations in GPU queues.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from types import MethodType

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)


MODEL_IDS = {
    "phi": "microsoft/Phi-3.5-vision-instruct",
    "molmo": "allenai/Molmo-7B-O-0924",
    "internvl": "OpenGVLab/InternVL2-8B",
    "glm": "zai-org/GLM-4.1V-9B-Thinking",
    "cogvlm2": "zai-org/cogvlm2-llama3-chat-19B",
    "deepseek": "deepseek-ai/deepseek-vl2-small",
}

PROMPT = (
    "Read the text in this image and summarize the requested task in one "
    "sentence."
)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def result_payload(
    model_key: str,
    model_id: str,
    status: str,
    elapsed_s: float,
    response: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "model": model_key,
        "model_id": model_id,
        "status": status,
        "elapsed_s": round(elapsed_s, 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "response": response[:1000],
        "error": error[:2000],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("RESULT_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def install_transformers_compat_shims() -> None:
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def run_phi(model_id: str, image_path: Path, cache_dir: str) -> str:
    log("PHASE phi loading_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        cache_dir=cache_dir,
    ).eval()
    log("PHASE phi loading_processor")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=4,
        cache_dir=cache_dir,
    )
    image = load_image(image_path)
    messages = [{"role": "user", "content": "<|image_1|>\n" + PROMPT}]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    log("PHASE phi generating")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=96,
            do_sample=False,
        )
    generated = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def run_molmo(model_id: str, image_path: Path, cache_dir: str) -> str:
    install_transformers_compat_shims()
    log("PHASE molmo loading_processor")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    log("PHASE molmo loading_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).to("cuda:0").eval()
    image = load_image(image_path)
    inputs = processor.process(images=[image], text=PROMPT)
    inputs = {k: v.to("cuda:0").unsqueeze(0) for k, v in inputs.items()}
    log("PHASE molmo generating")
    with torch.inference_mode():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=96, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _internvl_transform(input_size: int = 448):
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def run_internvl(model_id: str, image_path: Path, cache_dir: str) -> str:
    log("PHASE internvl loading_model")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).eval().cuda()
    log("PHASE internvl loading_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=cache_dir,
    )
    image = load_image(image_path)
    pixel_values = _internvl_transform()(image).unsqueeze(0).to(torch.bfloat16).cuda()
    question = "<image>\n" + PROMPT
    generation_config = {"max_new_tokens": 96, "do_sample": False}
    log("PHASE internvl generating")
    with torch.inference_mode():
        return model.chat(tokenizer, pixel_values, question, generation_config)


def run_glm(model_id: str, image_path: Path, cache_dir: str) -> str:
    from transformers import Glm4vForConditionalGeneration

    log("PHASE glm loading_processor")
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=cache_dir,
    )
    log("PHASE glm loading_model")
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    ).eval()
    image = load_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    log("PHASE glm generating")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return processor.decode(
        generated[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def run_cogvlm2(model_id: str, image_path: Path, cache_dir: str) -> str:
    log("PHASE cogvlm2 loading_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    log("PHASE cogvlm2 loading_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to("cuda").eval()
    if not hasattr(model, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, standardize_cache_format=False):
            if hasattr(outputs, "past_key_values"):
                return "past_key_values", outputs.past_key_values
            if isinstance(outputs, dict):
                for cache_name in ("past_key_values", "mems", "past_buckets_states"):
                    if cache_name in outputs:
                        return cache_name, outputs[cache_name]
            return "past_key_values", None

        model._extract_past_from_model_output = MethodType(
            _extract_past_from_model_output,
            model,
        )
    image = load_image(image_path)
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=PROMPT,
        history=[],
        images=[image],
        template_version="chat",
    )
    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to("cuda"),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to("cuda"),
        "images": [[input_by_model["images"][0].to("cuda").to(torch.bfloat16)]],
    }
    log("PHASE cogvlm2 generating")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            pad_token_id=128002,
            do_sample=False,
        )
    output = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(output[0]).split("<|end_of_text|>")[0]


def ensure_deepseek_package() -> None:
    try:
        import deepseek_vl2  # noqa: F401
        return
    except Exception:
        pass

    log("PHASE deepseek installing_package")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "git+https://github.com/deepseek-ai/DeepSeek-VL2.git",
        ]
    )


def run_deepseek(model_id: str, image_path: Path, cache_dir: str) -> str:
    ensure_deepseek_package()
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images

    log("PHASE deepseek loading_processor")
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    tokenizer = processor.tokenizer
    log("PHASE deepseek loading_model")
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = model.to(torch.bfloat16).cuda().eval()
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n" + PROMPT,
            "images": [str(image_path)],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    ).to(model.device)
    log("PHASE deepseek generating")
    with torch.inference_mode():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


RUNNERS = {
    "phi": run_phi,
    "molmo": run_molmo,
    "internvl": run_internvl,
    "glm": run_glm,
    "cogvlm2": run_cogvlm2,
    "deepseek": run_deepseek,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--image", default="stimuli/neutral_01.png")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/Data3/it_FA0571/hf_cache"))
    parser.add_argument("--result-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_id = MODEL_IDS[args.model]
    image_path = Path(args.image).resolve()
    result_file = Path(args.result_file)
    started = time.time()

    log(
        "START "
        f"model={args.model} model_id={model_id} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"image={image_path}"
    )
    log(f"ENV torch={torch.__version__} cuda={torch.cuda.is_available()} visible_count={torch.cuda.device_count()}")

    try:
        response = RUNNERS[args.model](model_id, image_path, args.cache_dir)
        payload = result_payload(args.model, model_id, "ok", time.time() - started, response=response.strip())
        write_result(result_file, payload)
        return 0
    except Exception as exc:
        traceback.print_exc()
        payload = result_payload(
            args.model,
            model_id,
            "error",
            time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_result(result_file, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
