import json
import torch
import numpy as np
from fastapi import FastAPI, WebSocket
from sklearn.decomposition import PCA

from models.gpt2 import GPT2, GPT2Tokenizer
from models.llama3 import Llama3, Llama3Tokenizer
from models.gemma3 import Gemma3, Gemma3Tokenizer
from models.qwen3 import Qwen3, Qwen3Tokenizer


MODEL_REGISTRY = {
    "gpt2": {
        "model_cls": GPT2,
        "tokenizer_cls": GPT2Tokenizer,
        "variant": lambda _: "BASE",
        "dtype": lambda is_cuda: torch.float32,
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

_MODEL_CACHE = {}
_ACTIVE_KEY = None


def unload_models(except_key=None):
    """
    Drop cached models except the specified key to free device memory.
    """
    removed = []
    for k in list(_MODEL_CACHE.keys()):
        if except_key is not None and k == except_key:
            continue
        bundle = _MODEL_CACHE.pop(k, None)
        if bundle is None:
            continue
        model = bundle.get("model")
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        removed.append(k)
    if removed and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed


def load_bundle(model_name: str, model_type: str, device: str, *, exclusive=True):
    """
    Load (or fetch cached) model/tokenizer/PCA bundle for the requested model/type/device.
    If exclusive=True, evict other cached models before loading to keep peak memory low.
    """
    key = (model_name.lower(), model_type.upper(), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if key[0] not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_REGISTRY)}.")

    if exclusive:
        unload_models()

    cfg = MODEL_REGISTRY[key[0]]
    is_cuda = str(device).startswith("cuda")
    torch_dtype = cfg["dtype"](is_cuda)
    variant = cfg["variant"](key[1])

    tokenizer = cfg["tokenizer_cls"].from_pretrained()
    model = cfg["model_cls"].from_pretrained(
        variant,
        device=device,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.setup_hookpoints()

    # Precompute projection basis for residual streams.
    with torch.no_grad():
        W = model.embed.W_E_weight.detach().float().cpu().numpy()
        pca = PCA(n_components=3, svd_solver="randomized").fit(W)
        P = torch.tensor(pca.components_.T, dtype=torch.float32)   # [d_model, 3]
        mu = torch.tensor(pca.mean_, dtype=torch.float32)          # [d_model]

    bundle = {"model": model, "tokenizer": tokenizer, "P": P, "mu": mu}
    _MODEL_CACHE[key] = bundle
    global _ACTIVE_KEY
    _ACTIVE_KEY = key
    return bundle


app = FastAPI()

def proj_batch(vt, P, mu):
    """
    Projects a batch of vectors to 3D.
    Input: [B, T, d_model]
    Output: List of T lists, each containing [x, y, z]
    """
    vt = vt.detach().float().cpu()          # [B, T, d_model]
    z = (vt - mu) @ P                       # [B, T, 3]
    z = z[0].tolist()                       # [T, 3] (Assuming B=1)
    return [[round(float(a), 3), round(float(b), 3), round(float(c), 3)] for a, b, c in z]

def layer_idx(k):
    if not k.startswith("L"): return None
    try:
        part = k.split(".")[0]
        return int(part[1:])
    except:
        return None

def resid_kind(k):
    if "._resid_pre" in k: return "pre"
    if "._resid_mid" in k: return "mid"
    if "._resid_post" in k: return "post"
    return None

def attn_kind(k):
    if "._attn_probs" in k: return "probs"
    if "._attn_scores" in k: return "scores"
    return None

def extract_attention(v, decimals=4):
    """
    Extracts attention matrices.
    Input v shape: [Batch, Heads, Queries, Keys]
    Output: [Heads, Queries, Keys] as a nested list of floats.
    """
    data = v[0].detach().float().cpu()
    return [[[round(float(x), decimals) for x in row] for row in head] for head in data.tolist()]

def extract_attn_avg(v, decimals=4):
    """
    Returns average attention across heads.
    Output: [Queries, Keys]
    """
    data = v[0].detach().float().cpu() # [H, Q, K]
    avg = data.mean(dim=0)             # [Q, K]
    return [[round(float(x), decimals) for x in row] for row in avg.tolist()]


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data_text = await ws.receive_text()
        except Exception:
            break
        req = json.loads(data_text)
        action = req.get("action", "generate")

        if action == "unload":
            removed = unload_models()
            await ws.send_text(json.dumps({"type": "unloaded", "removed": [list(r) for r in removed]}))
            continue

        model_name = req.get("model", "gemma3")
        model_type = "INSTRUCT" if str(req.get("type", "BASE")).lower() == "instruct" else "BASE"
        device = req.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        bundle = load_bundle(model_name, model_type, device)
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        P = bundle["P"]
        mu = bundle["mu"]

        if action == "load":
            await ws.send_text(json.dumps({"type": "loaded", "model": model_name, "variant": model_type, "device": device}))
            continue

        # generation path
        prompt = req.get("prompt", "Life is beautiful because")
        max_new = int(req.get("max_new", 10))

        include_pre  = bool(req.get("include_pre", False))
        include_mid  = bool(req.get("include_mid", True))
        include_post = bool(req.get("include_post", True))
        
        include_attn_probs  = bool(req.get("include_attn", True)) 
        include_attn_scores = bool(req.get("include_attn_scores", False))

        if model_type == "INSTRUCT" and hasattr(tokenizer, "encode_instruct"):
            prompt_ids = tokenizer.encode_instruct(prompt)
        else:
            prompt_ids = tokenizer.encode(prompt)
        ctx_ids = list(prompt_ids)

        pos_offset = 0
        step = 0
        prev_pred_id = None

        order_map = {"pre": 0, "mid": 1, "post": 2}
        allow_map = {"pre": include_pre, "mid": include_mid, "post": include_post}

        for step_data, cache in model.stream_generate_with_cache(prompt_ids, max_new_tokens=max_new):
            
            next_id = step_data.token_id
            
            # 1. Determine what tokens are being sent in this payload
            if step == 0:
                current_ids = prompt_ids
                current_toks = [tokenizer.decode([i]) for i in current_ids]
                payload_pos = 0
            else:
                current_ids = [prev_pred_id]
                current_toks = [tokenizer.decode([prev_pred_id])]
                payload_pos = pos_offset

            pred_tok = tokenizer.decode([next_id])
            
            # 2. Process Residual Stream (Projections)
            projection_items = []
            if "embed._resid_emb" in cache.data:
                projection_items.append({
                    "layer": -1, 
                    "type": "emb",
                    "name": "embed._resid_emb",
                    "vectors": proj_batch(cache.data["embed._resid_emb"], P, mu)
                })

            temp_resid = []
            for k, v in cache.data.items():
                li = layer_idx(k)
                kind = resid_kind(k)
                if li is None or kind is None: continue
                if not allow_map[kind]: continue
                temp_resid.append((li, order_map[kind], k, v))

            temp_resid.sort(key=lambda t: (t[0], t[1]))

            for li, _, k, v in temp_resid:
                projection_items.append({
                    "layer": li,
                    "type": resid_kind(k),
                    "name": k,
                    "vectors": proj_batch(v, P, mu)
                })

            # 3. Process Attention
            attn_payload = []
            if include_attn_probs or include_attn_scores:
                temp_attn = []
                for k, v in cache.data.items():
                    li = layer_idx(k)
                    ak = attn_kind(k)
                    if li is None or ak is None: continue
                    if ak == "probs" and not include_attn_probs: continue
                    if ak == "scores" and not include_attn_scores: continue
                    temp_attn.append((li, ak, k, v))

                temp_attn.sort(key=lambda t: t[0])

                for li, ak, k, v in temp_attn:
                    attn_payload.append({
                        "layer": li,
                        "type": ak,
                        "name": k,
                        "heads": extract_attention(v), 
                        "avg": extract_attn_avg(v)
                    })

            response = {
                "type": "step",
                "step_idx": step,
                "new_toks": {
                    "ids": current_ids,
                    "text": current_toks,
                    "start_pos": payload_pos
                },
                "pred": {
                    "id": next_id,
                    "tok": pred_tok,
                    "abs_pos": payload_pos + len(current_ids),
                    "distribution": {
                        "top_k_ids": step_data.top_k_ids,
                        "top_k_toks": [tokenizer.decode([i]) for i in step_data.top_k_ids],
                        "top_k_probs": step_data.top_k_probs
                    }
                },
                "ctx": {
                    "ids": ctx_ids,
                    "text": [tokenizer.decode([i]) for i in ctx_ids]
                },
                "projections": projection_items,
                "attentions": attn_payload
            }

            await ws.send_text(json.dumps(response))

            # Update state for next iteration
            pos_offset = payload_pos + len(current_ids)
            prev_pred_id = next_id
            step += 1
            ctx_ids.append(next_id)

        await ws.send_text(json.dumps({"type": "end"}))
