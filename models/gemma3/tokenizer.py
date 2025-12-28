from tokenizers import Tokenizer
from pathlib import Path
import requests
import os


class Gemma3Tokenizer:
    def __init__(self, model_path):
        self.tokenizer = Tokenizer.from_file(model_path)
    
    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        return self.tokenizer.encode(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n").ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<eos>")

    @property
    def eos_token_id_instruct(self):
        return self.tokenizer.token_to_id("<end_of_turn>")

    @classmethod
    def from_pretrained(cls):
        TOKENIZER_URL = "https://huggingface.co/google/gemma-3-270m/resolve/main/tokenizer.json"
        save_path = Path(".cache") / "gemma-3-270m-instruct"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "tokenizer.json"

        HF_TOKEN = os.getenv('HF_TOKEN')
        if not model_file.exists():
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(TOKENIZER_URL, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        return cls(str(model_file))
