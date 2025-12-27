import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.infer import HookPoint, InferenceMixin
from models.pretrained import FromPretrainedMixin
from models.gemma3.spec import GEMMA3_SPECS


@dataclass
class PastKV:
    K: torch.Tensor  # [B, Hkv, T_cache, Dh]
    V: torch.Tensor  # [B, Hkv, T_cache, Dh]
    # Absolute token positions for the cache:
    # - start_pos: absolute position of the first cached KV 
    # - end_pos:   absolute position *after* the last cached KV (start_pos + T_cache)
    start_pos: int
    end_pos: int


class Gemma3RMSNorm(torch.nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        dtype = x.dtype
        y = F.rms_norm(
            x.float(),
            (self.weight.numel(),),
            weight=(1.0 + self.weight.float()),
            eps=self.eps,
        )
        return y.to(dtype)


class Gemma3Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, scale=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = scale
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.hook_resid_emb = HookPoint()

    @property
    def W_E_weight(self):
        return self.W_E.weight

    def forward(self, toks):
        resid = self.W_E(toks)
        if self.scale:
            resid = resid * (self.d_model ** 0.5)
        resid = self.hook_resid_emb(resid)
        return resid


class Gemma3LMHead(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.ln_f = Gemma3RMSNorm(d_model, eps=eps)

    def forward(self, resid, W_U):
        resid = self.ln_f(resid)
        return F.linear(resid, W_U)  # output: [B, T, vocab]


class Gemma3MLPF(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.up_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.gate_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.down_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=False)

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        x = self.down_proj(F.gelu(gate, approximate="tanh") * up)
        return x


class Gemma3MHAttention(nn.Module):
    def __init__(self, max_length, d_model, n_heads, n_kv_heads, d_head, is_sliding):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        assert d_head % 2 == 0

        self.max_length = max_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_heads // n_kv_heads
        self.is_sliding = is_sliding

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.q_norm = Gemma3RMSNorm(d_head, eps=1e-6)
        self.k_norm = Gemma3RMSNorm(d_head, eps=1e-6)

        self.hook_attn_q = HookPoint()
        self.hook_attn_k = HookPoint()
        self.hook_attn_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_attn_probs = HookPoint()

        if self.is_sliding:
            self.window_size, self.rope_theta = 512, 10000.0
        else:
            self.rope_theta = 1_000_000.0

    @staticmethod
    def _rope_cos_sin(head_dim, rope_theta, x_pos, device, dtype):
        # x_pos: [B, T] absolute token positions for the *new* tokens.
        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim
        # This ends up equivalent to the usual theta^{-i/d} once the 2π factors cancel below.
        freq = (1.0 / (2 * math.pi)) * (rope_theta ** (-idx))

        freq = freq[None, :, None].expand(x_pos.shape[0], -1, 1).to(device)
        phase = 2 * math.pi * (freq @ x_pos[:, None, :].float()).transpose(1, 2)  # [B, T, Dh/2]
        emb = torch.cat((phase, phase), dim=-1)  # [B, T, Dh]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        # cos/sin: [B, T, Dh] -> unsqueeze to broadcast over heads.
        cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, T, Dh]
        sin = sin.unsqueeze(unsqueeze_dim)  # [B, 1, T, Dh]
        q_embed = (q * cos) + (Gemma3MHAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Gemma3MHAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, resid, past_kv=None, use_cache=False):
        B, T_new, _ = resid.size()
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.d_head
        G = self.group_size

        # Project to Q/K/V.
        q = self.q_proj(resid).view(B, T_new, H, Dh).permute(0, 2, 1, 3)   # [B, H,   T_new, Dh]
        k = self.k_proj(resid).view(B, T_new, Hkv, Dh).permute(0, 2, 1, 3) # [B, Hkv, T_new, Dh]
        v = self.v_proj(resid).view(B, T_new, Hkv, Dh).permute(0, 2, 1, 3) # [B, Hkv, T_new, Dh]

        # Normalize per head vector.
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.hook_attn_q(q)
        k = self.hook_attn_k(k)
        v = self.hook_attn_v(v)

        # Cache position semantics:
        # - If past_kv exists, it covers absolute positions [past_kv.start_pos, past_kv.end_pos)
        # - New token positions start at past_kv.end_pos
        start_pos = 0 if past_kv is None else past_kv.start_pos
        end_pos = 0 if past_kv is None else past_kv.end_pos

        # Absolute positions for the new tokens (used for RoPE).
        x_pos = torch.arange(end_pos, end_pos + T_new, device=resid.device).unsqueeze(0).expand(B, -1)
        cos, sin = self._rope_cos_sin(Dh, self.rope_theta, x_pos=x_pos, device=resid.device, dtype=resid.dtype)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # Append cache if provided.
        if past_kv is not None:
            K = torch.cat([past_kv.K, k], dim=2)  # [B, Hkv, T_total, Dh]
            V = torch.cat([past_kv.V, v], dim=2)
        else:
            K, V = k, v

        T_total = K.size(2)

        # Attention scores with grouped-query attention:
        # reshape heads: H = Hkv * G
        q = q * (Dh ** -0.5)
        q = q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        qg = q.reshape(B, Hkv, G, T_new, Dh)
        att_g = torch.einsum("bhgqd,bhkd->bhgqk", qg.float(), K.float())  # [B, Hkv, G, T_new, T_total]

        # Build causal (+ optional sliding window) mask using absolute positions.
        # Keys are assumed to correspond to consecutive absolute positions [start_pos, start_pos + T_total)
        # This is guaranteed if PastKV always preserves: end_pos == start_pos + cache_len.
        qpos = end_pos + torch.arange(T_new, device=resid.device).view(T_new, 1)  # [T_new, 1]
        kpos = torch.arange(start_pos, start_pos + T_total, device=resid.device).view(1, T_total)  # [1, T_total]
        dist = qpos - kpos  # [T_new, T_total]
        m = (dist >= 0)
        if self.is_sliding:
            m = m & (dist <= self.window_size)
        m = m.view(1, 1, 1, T_new, T_total)  # broadcast over [B, Hkv, G, ...]

        att_g = att_g.masked_fill(~m, float("-inf"))

        att_scores = att_g.reshape(B, H, T_new, T_total)
        att_scores = self.hook_attn_scores(att_scores)
        att_g = att_scores.reshape(B, Hkv, G, T_new, T_total)

        att_g = F.softmax(att_g, dim=-1).to(q.dtype)

        att_probs = att_g.reshape(B, H, T_new, T_total)
        att_probs = self.hook_attn_probs(att_probs)
        att_g = att_probs.reshape(B, Hkv, G, T_new, T_total)

        ctx = torch.einsum("bhgqk,bhkd->bhgqd", att_g, V)  # [B, Hkv, G, T_new, Dh]
        ctx = ctx.contiguous()

        out = ctx.reshape(B, H, T_new, Dh).permute(0, 2, 1, 3).contiguous()  # [B, T_new, H, Dh]
        y = self.o_proj(out.view(B, T_new, H * Dh))

        if not use_cache:
            return y

        # Update cache positions.
        new_end_pos = end_pos + T_new

        # If using a sliding window, keep only the last window_size tokens in the cache
        # and advance start_pos accordingly so [start_pos, end_pos) remains consistent.
        if self.is_sliding and T_total > self.window_size:
            K_new = K[:, :, -self.window_size :, :]
            V_new = V[:, :, -self.window_size :, :]
            new_start_pos = new_end_pos - self.window_size
            return y, PastKV(K_new, V_new, start_pos=new_start_pos, end_pos=new_end_pos)

        # Non-sliding (or short) cache: start_pos unchanged, end_pos advances.
        return y, PastKV(K, V, start_pos=start_pos, end_pos=new_end_pos)


class Gemma3Block(nn.Module):
    def __init__(self, max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head, is_sliding):
        super().__init__()
        self.mhattn = Gemma3MHAttention(max_length, d_model, n_heads, n_kv_heads, d_head, is_sliding)
        self.mlpf = Gemma3MLPF(d_model, d_mlp)

        self.ln1 = Gemma3RMSNorm(d_model, eps=1e-6)
        self.ln2 = Gemma3RMSNorm(d_model, eps=1e-6)
        self.ln1_post = Gemma3RMSNorm(d_model, eps=1e-6)
        self.ln2_post = Gemma3RMSNorm(d_model, eps=1e-6)

        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, resid, past_kv=None, use_cache=False):
        resid = self.hook_resid_pre(resid)

        if use_cache:
            attn_out, present_kv = self.mhattn(self.ln1(resid), past_kv=past_kv, use_cache=True)
        else:
            attn_out = self.mhattn(self.ln1(resid))
            present_kv = None

        attn_out = self.ln1_post(attn_out)
        attn_out = self.hook_attn_out(attn_out)
        resid = resid + attn_out
        resid = self.hook_resid_mid(resid)

        mlp_out = self.mlpf(self.ln2(resid))
        mlp_out = self.ln2_post(mlp_out)
        mlp_out = self.hook_mlp_out(mlp_out)

        resid = resid + mlp_out
        resid = self.hook_resid_post(resid)

        return (resid, present_kv) if use_cache else resid


class Gemma3(nn.Module, FromPretrainedMixin, InferenceMixin):
    PRETRAINED_SPECS = GEMMA3_SPECS

    def __init__(self, vocab_size, max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head, attentions):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.attentions = attentions

        self.embed = Gemma3Embedding(vocab_size, d_model)
        self.lm_head = Gemma3LMHead(d_model, eps=1e-6)

        self.blocks = nn.ModuleList(
            Gemma3Block(max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head, is_sliding=(t == 1))
            for t in attentions
        )

    def forward(self, toks, past_key_values=None, use_cache=False):
        resid = self.embed(toks)

        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if use_cache:
                resid, present_kv = block(resid, past_kv=past_key_values[i], use_cache=True)
                new_past_key_values.append(present_kv)
            else:
                resid = block(resid)

        logits = self.lm_head(resid, self.embed.W_E_weight)
        out = {"logits": logits}
        if use_cache:
            out["past_key_values"] = new_past_key_values
        return out