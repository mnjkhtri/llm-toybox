import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from models.infer import HookPoint, InferenceMixin
from models.pretrained import FromPretrainedMixin
from models.qwen3.spec import QWEN3_SPECS


@dataclass
class PastKV:
    K: torch.Tensor
    V: torch.Tensor
    start_pos: int
    end_pos: int


class Qwen3RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(emb_dim))

    def forward(self, x):
        dtype = x.dtype
        y = F.rms_norm(
            x.float(),
            (self.weight.numel(),),
            weight=self.weight.float(),
            eps=self.eps,
        )
        return y.to(dtype)


class Qwen3Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W_E = nn.Embedding(self.vocab_size, self.d_model)
        self.hook_resid_emb = HookPoint()

    @property
    def W_E_weight(self):
        return self.W_E.weight

    def forward(self, toks):
        resid = self.W_E(toks)
        resid = self.hook_resid_emb(resid)
        return resid


class Qwen3LMHead(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.ln_f = Qwen3RMSNorm(d_model, eps=eps)

    def forward(self, resid, W_U):
        resid = self.ln_f(resid)
        return F.linear(resid, W_U)

class Qwen3MLPF(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.up_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.down_proj = nn.Linear(self.d_mlp, self.d_model, bias=False)

    def forward(self, resid):
        up = self.up_proj(resid)
        gate = self.gate_proj(resid)
        return self.down_proj(up * F.silu(gate))


class Qwen3MHAttention(nn.Module):
    def __init__(self, max_length, d_model, n_heads, n_kv_heads, d_head):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.max_length = max_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)

        self.q_norm = Qwen3RMSNorm(d_head, eps=1e-6)
        self.k_norm = Qwen3RMSNorm(d_head, eps=1e-6)

        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        self.rope = {"rope_theta": 1_000_000.0}

        self.hook_attn_q = HookPoint()
        self.hook_attn_k = HookPoint()
        self.hook_attn_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_attn_probs = HookPoint()

    @staticmethod
    def _rope_cos_sin(head_dim, rope, position_ids, device, dtype):
        freq = (1.0 / (2 * math.pi)) * (
            rope["rope_theta"] ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim)
        )
        freq = freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(device)

        phase = (freq @ position_ids[:, None, :].float()).transpose(1, 2) * (2 * math.pi)

        emb = torch.cat((phase, phase), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        return cos, sin

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (Qwen3MHAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3MHAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, resid, past_kv=None, use_cache=False):
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.d_head
        G = self.group_size
        B, T_new, _ = resid.size()

        q = self.q_proj(resid)
        k = self.k_proj(resid)
        v = self.v_proj(resid)

        q = q.view(B, T_new, H, Dh).permute(0, 2, 1, 3)   # [B, H,   T_new, Dh]
        k = k.view(B, T_new, Hkv, Dh).permute(0, 2, 1, 3) # [B, Hkv, T_new, Dh]
        v = v.view(B, T_new, Hkv, Dh).permute(0, 2, 1, 3) # [B, Hkv, T_new, Dh]

        start_pos = 0 if past_kv is None else past_kv.start_pos
        end_pos = 0 if past_kv is None else past_kv.end_pos

        position_ids = torch.arange(end_pos, end_pos + T_new, device=q.device).unsqueeze(0).expand(B, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.hook_attn_q(q)
        k = self.hook_attn_k(k)
        v = self.hook_attn_v(v)

        cos, sin = Qwen3MHAttention._rope_cos_sin(Dh, self.rope, position_ids=position_ids, device=q.device, dtype=q.dtype)
        q, k = Qwen3MHAttention.apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            K = torch.cat([past_kv.K, k], dim=2)
            V = torch.cat([past_kv.V, v], dim=2)
        else:
            K, V = k, v

        T_total = K.size(2)
        qg = q.reshape(B, Hkv, G, T_new, Dh)

        att_g = torch.einsum("bhgtd,bhkd->bhgtk", [qg.float(), K.float()]) / (Dh ** 0.5)

        query_positions = end_pos + torch.arange(T_new, device=q.device).view(T_new, 1)
        key_positions = torch.arange(start_pos, start_pos + T_total, device=q.device).view(1, T_total)
        dist = query_positions - key_positions
        attn_mask = (dist >= 0)
        attn_mask = attn_mask.view(1, 1, 1, T_new, T_total)
        att_g = att_g.masked_fill(~attn_mask, torch.finfo(att_g.dtype).min)

        att_scores = att_g.reshape(B, H, T_new, T_total)
        att_scores = self.hook_attn_scores(att_scores)
        att_g = att_scores.reshape(B, Hkv, G, T_new, T_total)

        att_g = F.softmax(att_g, dim=-1).to(q.dtype)

        att_probs = att_g.reshape(B, H, T_new, T_total)
        att_probs = self.hook_attn_probs(att_probs)
        att_g = att_probs.reshape(B, Hkv, G, T_new, T_total)

        ctx = torch.einsum("bhgtk,bhkd->bhgtd", [att_g, V])
        out = ctx.reshape(B, H, T_new, Dh)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.o_proj(out.reshape(B, T_new, H * Dh))

        if use_cache:
            new_end_pos = end_pos + T_new
            return out, PastKV(K, V, start_pos=start_pos, end_pos=new_end_pos)

        return out


class Qwen3Block(nn.Module):
    def __init__(self, max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head

        self.mhattn = Qwen3MHAttention(self.max_length, self.d_model, self.n_heads, self.n_kv_heads, self.d_head)
        self.mlpf = Qwen3MLPF(self.d_model, self.d_mlp)

        self.ln1 = Qwen3RMSNorm(d_model, eps=1e-6)
        self.ln2 = Qwen3RMSNorm(d_model, eps=1e-6)

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

        attn_out = self.hook_attn_out(attn_out)
        resid = resid + attn_out
        resid = self.hook_resid_mid(resid)

        mlp_out = self.mlpf(self.ln2(resid))
        mlp_out = self.hook_mlp_out(mlp_out)
        resid = resid + mlp_out
        resid = self.hook_resid_post(resid)

        return (resid, present_kv) if use_cache else resid


class Qwen3(nn.Module, FromPretrainedMixin, InferenceMixin):
    PRETRAINED_SPECS = QWEN3_SPECS

    def __init__(self, vocab_size, max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_layers = n_layers

        self.embed = Qwen3Embedding(self.vocab_size, self.d_model)
        self.lm_head = Qwen3LMHead(self.d_model, eps=1e-6)

        self.blocks = nn.ModuleList(
            Qwen3Block(max_length, d_model, d_mlp, n_heads, n_kv_heads, d_head) for _ in range(self.n_layers)
        )

    def forward(self, toks, past_key_values=None, use_cache=False):
        resid = self.embed(toks)

        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * self.n_layers
            new_past_key_values = []
            for i, block in enumerate(self.blocks):
                resid, present_kv = block(resid, past_kv=past_key_values[i], use_cache=True)
                new_past_key_values.append(present_kv)

            logits = self.lm_head(resid, self.embed.W_E_weight)

            return {"logits": logits, "past_key_values": new_past_key_values}

        for block in self.blocks:
            resid = block(resid)

        logits = self.lm_head(resid, self.embed.W_E_weight)

        return {"logits": logits}
