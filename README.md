# minimal pytorch text gen playground

decoder-only llm toybox in pytorch (gpt2, llama3, gemma3, qwen3)

## features
- kv-cached streaming generation for interactive decoding
- safetensors weight streaming that materializes models on meta and fills layers on cpu/gpu
- optional weight-only int8 paths (naive and torchao) you can enable at load time
- device-aware dtype defaults plus simple hooks for inspecting residuals and attention

## usage
- `python generate.py --model llama3 --type base`
- `python generate.py --model llama3 --type instruct`
- add `--prompt "your own text"` to swap in your own
