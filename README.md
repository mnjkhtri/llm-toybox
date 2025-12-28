minimal text gen sandbox: choose a model (gpt2 | llama3 | gemma3 | qwen3), choose base or instruct, and stream tokens on the cli.

usage:
- `python generate.py --model llama3 --type base`   # default prompt: quantum mechanics
- `python generate.py --model llama3 --type instruct`   # default prompt: explain quantum mechanics
- add `--prompt "your own text"` to swap in your own
