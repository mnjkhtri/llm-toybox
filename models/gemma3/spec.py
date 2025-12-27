from models.pretrained import PretrainedSpec


def _gemma3_spec(url, cache_dir):

    config = dict(vocab_size=262144, max_length=32768, d_model=640, d_mlp=2048, n_heads=4, n_kv_heads=1, d_head=256,
        attentions=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    )

    embedding_map = {
        'model.embed_tokens.weight': 'embed.W_E.weight',
    }

    lm_head_map = {
        'model.norm.weight': 'lm_head.ln_f.weight'
    }

    def blocks_map(i):
        return {
            f'model.layers.{i}.input_layernorm.weight': f'blocks.{i}.ln1.weight',
            f'model.layers.{i}.self_attn.q_proj.weight': f'blocks.{i}.mhattn.q_proj.weight',
            f'model.layers.{i}.self_attn.q_norm.weight': f'blocks.{i}.mhattn.q_norm.weight',
            f'model.layers.{i}.self_attn.k_proj.weight': f'blocks.{i}.mhattn.k_proj.weight',
            f'model.layers.{i}.self_attn.k_norm.weight': f'blocks.{i}.mhattn.k_norm.weight',
            f'model.layers.{i}.self_attn.v_proj.weight': f'blocks.{i}.mhattn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight': f'blocks.{i}.mhattn.o_proj.weight',
            f'model.layers.{i}.post_attention_layernorm.weight': f'blocks.{i}.ln1_post.weight',
            f'model.layers.{i}.pre_feedforward_layernorm.weight': f'blocks.{i}.ln2.weight',
            f'model.layers.{i}.mlp.up_proj.weight': f'blocks.{i}.mlpf.up_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight': f'blocks.{i}.mlpf.gate_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight': f'blocks.{i}.mlpf.down_proj.weight',
            f'model.layers.{i}.post_feedforward_layernorm.weight':  f'blocks.{i}.ln2_post.weight'
        }

    mappings = {**embedding_map, **lm_head_map}

    for i in range(config["attentions"].__len__()):
        mappings.update(blocks_map(i))

    return PretrainedSpec(url=url, cache_dir=cache_dir, config=config, mappings=mappings)


GEMMA3_SPECS={
  "BASE":lambda:_gemma3_spec("https://huggingface.co/google/gemma-3-270m/resolve/main/model.safetensors", "gemma-3-270m"),
  "INSTRUCT":lambda:_gemma3_spec("https://huggingface.co/google/gemma-3-270m-it/resolve/main/model.safetensors", "gemma-3-270m-instruct"),
}
