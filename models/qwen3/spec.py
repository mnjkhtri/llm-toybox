from models.pretrained import PretrainedSpec


def _qwen3_spec(url, cache_dir):
    config = dict(
        vocab_size=151936,
        max_length=32768,
        d_model=1024,
        d_mlp=3072,
        n_heads=16,
        n_kv_heads=8,
        d_head=128,
        n_layers=28,
    )

    embedding_map = {
        "model.embed_tokens.weight": "embed.W_E.weight",
    }

    lm_head_map = {
        "model.norm.weight": "lm_head.ln_f.weight",
    }

    def blocks_map(i):
        return {
            f"model.layers.{i}.input_layernorm.weight": f"blocks.{i}.ln1.weight",
            f"model.layers.{i}.mlp.up_proj.weight": f"blocks.{i}.mlpf.up_proj.weight",
            f"model.layers.{i}.mlp.gate_proj.weight": f"blocks.{i}.mlpf.gate_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight": f"blocks.{i}.mlpf.down_proj.weight",
            f"model.layers.{i}.self_attn.q_proj.weight": f"blocks.{i}.mhattn.q_proj.weight",
            f"model.layers.{i}.self_attn.q_norm.weight": f"blocks.{i}.mhattn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_proj.weight": f"blocks.{i}.mhattn.k_proj.weight",
            f"model.layers.{i}.self_attn.k_norm.weight": f"blocks.{i}.mhattn.k_norm.weight",
            f"model.layers.{i}.self_attn.v_proj.weight": f"blocks.{i}.mhattn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight": f"blocks.{i}.mhattn.o_proj.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"blocks.{i}.ln2.weight",
        }

    mappings = {**embedding_map, **lm_head_map}
    for i in range(config["n_layers"]):
        mappings.update(blocks_map(i))

    return PretrainedSpec(url=url, cache_dir=cache_dir, config=config, mappings=mappings)


QWEN3_SPECS = {
    "BASE": lambda: _qwen3_spec("https://huggingface.co/qwen/qwen3-0.6b-base/resolve/main/model.safetensors", "qwen3-0.6b"),
    "INSTRUCT": lambda: _qwen3_spec("https://huggingface.co/qwen/qwen3-0.6b/resolve/main/model.safetensors", "qwen3-0.6b-instruct"),
}
