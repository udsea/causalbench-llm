from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class HFRunner:
    model_name: str
    device: torch.device = torch.device("cpu")  # resolved device
    device_preference: str = "cpu"              # "cpu" | "mps" | "cuda"
    max_new_tokens: int = 64
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        preferred = self.device_preference.lower()
        if preferred == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif preferred == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

         # move the model to the chosen device – use the keyword form so
         # the type checker doesn’t think we’re calling some other object
        self.model = self.model.to(device=self.device) # type: ignore
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
                    "content": (
                        'Return ONLY valid JSON with exactly one key: "label". '
                        'Allowed values: "obs_gt_do", "do_gt_obs", "approx_equal".'
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            enc = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # move tensors to device explicitly (works for BatchEncoding)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            input_len = enc["input_ids"].shape[-1]

            out = self.model.generate(
                **enc,  # <- includes input_ids + attention_mask if present
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            completion_ids = out[0][input_len:]
            return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Plain-text fallback
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature if self.temperature > 0 else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        completion_ids = out[0][input_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
