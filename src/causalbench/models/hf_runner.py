from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ALLOWED_QUANTIZATION = {"none", "4bit", "8bit"}
ALLOWED_TORCH_DTYPES = {"auto", "float16", "bfloat16", "float32"}
SYSTEM_INSTRUCTION = (
    'Return ONLY valid JSON with exactly one key: "label". '
    'Allowed values: "obs_gt_do", "do_gt_obs", "approx_equal".'
)


@dataclass
class HFRunner:
    model_name: str
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))  # resolved device
    device_preference: str = "cpu"  # "cpu" | "mps" | "cuda"
    max_new_tokens: int = 64
    temperature: float = 0.0
    torch_dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    quantization: str = "none"  # "none" | "4bit" | "8bit"

    def _resolve_device(self) -> torch.device:
        preferred = self.device_preference.lower()
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_torch_dtype(self, requested_dtype: str) -> torch.dtype | None:
        normalized = requested_dtype.lower().strip()
        if normalized not in ALLOWED_TORCH_DTYPES:
            allowed = ", ".join(sorted(ALLOWED_TORCH_DTYPES))
            raise ValueError(f"torch_dtype must be one of: {allowed}")
        if normalized == "auto":
            return torch.float16 if self.device.type in {"cuda", "mps"} else None
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[normalized]

    def _normalize_encoding(self, value: Any) -> dict[str, torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return {"input_ids": value}
        if hasattr(value, "items"):
            enc = {k: v for k, v in value.items() if isinstance(v, torch.Tensor)}
            if "input_ids" in enc:
                return enc
        raise ValueError("Tokenizer produced an invalid encoding; expected input_ids tensor(s).")

    def _move_inputs(self, enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.input_device) for k, v in enc.items()}

    def __post_init__(self) -> None:
        self.device = self._resolve_device()
        quantization = self.quantization.lower().strip()
        if quantization not in ALLOWED_QUANTIZATION:
            allowed = ", ".join(sorted(ALLOWED_QUANTIZATION))
            raise ValueError(f"quantization must be one of: {allowed}")
        dtype = self._resolve_torch_dtype(self.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self._manual_device_move = True
        if quantization != "none":
            if self.device.type != "cuda":
                raise ValueError("quantization=4bit/8bit requires --device cuda")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ValueError(
                    "quantization requires bitsandbytes. Install it with `pip install bitsandbytes`."
                ) from exc

            if quantization == "4bit":
                compute_dtype = dtype or torch.float16
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            self._manual_device_move = False

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self._manual_device_move:
            self.model = self.model.to(device=self.device)  # type: ignore[assignment]
            self.input_device = self.device
        else:
            self.input_device = torch.device("cuda:0")

        self.model.eval()

        # pad token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        # Chat-template path (Qwen Instruct)
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {"role": "user", "content": prompt},
            ]
            try:
                enc_obj = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                enc_obj = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            enc = self._move_inputs(self._normalize_encoding(enc_obj))

            input_len = enc["input_ids"].shape[-1]

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature

            out = self.model.generate(
                **enc,  # <- includes input_ids + attention_mask if present
                **gen_kwargs,
            )

            completion_ids = out[0][input_len:]
            return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Plain-text fallback
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self._move_inputs(self._normalize_encoding(inputs))
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        out = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        completion_ids = out[0][input_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
