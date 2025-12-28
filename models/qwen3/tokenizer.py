from pathlib import Path

import os
import requests
from tokenizers import Tokenizer


class Qwen3Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = Tokenizer.from_file(model_path)

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        prompt = (
            f"{im_start}system\n{system}{im_end}\n"
            f"{im_start}user\n{user}{im_end}\n"
            f"{im_start}assistant\n"
        )
        return self.tokenizer.encode(prompt).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    @property
    def eos_token_id_instruct(self):
        return self.tokenizer.token_to_id("<|im_end|>")

    @classmethod
    def from_pretrained(cls):
        tokenizer_url = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
        save_path = Path(".cache") / "qwen3-0.6b-instruct"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "tokenizer.json"

        hf_token = os.getenv("HF_TOKEN")
        if not model_file.exists():
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            r = requests.get(tokenizer_url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        return cls(str(model_file))
