import argparse
import torch

from models.gpt2 import GPT2, GPT2Tokenizer
from models.llama3 import Llama3, Llama3Tokenizer
from models.gemma3 import Gemma3, Gemma3Tokenizer
from models.qwen3 import Qwen3, Qwen3Tokenizer


MODEL_REGISTRY = {
    "gpt2": {
        "model_cls": GPT2,
        "tokenizer_cls": GPT2Tokenizer,
        "variant": lambda _: "BASE",
        "dtype": lambda _: torch.float32,
    },
    "llama3": {
        "model_cls": Llama3,
        "tokenizer_cls": Llama3Tokenizer,
        "variant": lambda mt: mt,
        "dtype": lambda is_cuda: torch.bfloat16 if is_cuda else torch.float32,
    },
    "gemma3": {
        "model_cls": Gemma3,
        "tokenizer_cls": Gemma3Tokenizer,
        "variant": lambda mt: mt,
        "dtype": lambda is_cuda: torch.bfloat16 if is_cuda else torch.float32,
    },
    "qwen3": {
        "model_cls": Qwen3,
        "tokenizer_cls": Qwen3Tokenizer,
        "variant": lambda mt: mt,
        "dtype": lambda is_cuda: torch.bfloat16 if is_cuda else torch.float32,
    },
}

DEFAULT_PROMPTS = {
    "base": "Quantum mechanics",
    "instruct": "Explain quantum mechanics in depth along with its history.",
}


def load_model_and_tokenizer(model_name: str, model_type: str, device: str):
    name = model_name.lower()
    variant = "INSTRUCT" if model_type.lower() == "instruct" else "BASE"
    is_cuda = device.startswith("cuda")

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from gpt2/llama3/gemma3/qwen3.")

    cfg = MODEL_REGISTRY[name]
    tokenizer = cfg["tokenizer_cls"].from_pretrained()
    torch_dtype = cfg["dtype"](is_cuda)
    model_variant = cfg["variant"](variant)

    model = cfg["model_cls"].from_pretrained(
        model_variant,
        torch_dtype=torch_dtype,
        device=device,
    )

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a chosen model.")
    parser.add_argument("--model", choices=["gpt2", "llama3", "gemma3", "qwen3"], default="llama3")
    parser.add_argument("--type", choices=["base", "instruct"], default="base", help="Model variant to load.")
    parser.add_argument("--prompt", help="Override the default prompt for the chosen variant.")
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.type, args.device)
    model.eval()
    model.setup_hookpoints()

    use_instruct = args.type.lower() == "instruct"
    prompt_text = args.prompt or DEFAULT_PROMPTS["instruct" if use_instruct else "base"]
    encoder = tokenizer.encode_instruct if use_instruct and hasattr(tokenizer, "encode_instruct") else tokenizer.encode
    prompt_ids = encoder(prompt_text)

    eos_token_id = None
    if use_instruct:
        eos_token_id = getattr(tokenizer, "eos_token_id_instruct", None) or getattr(tokenizer, "eos_token_id", None)
    else:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    print(tokenizer.decode(prompt_ids), end="")
    for step_data in model.stream_generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
    ):
        next_id = step_data.token_id
        pred_tok = tokenizer.decode([next_id])
        print(pred_tok, end="")


if __name__ == "__main__":
    main()
