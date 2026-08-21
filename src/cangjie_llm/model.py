import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    block_size,
    cangjie_auxiliary_loss_weight,
    dropout,
    embed_size,
    ffn_hidden_size,
    n_head,
    n_kv_head,
    n_layer,
    return_training_logits,
    rope_theta,
    sampled_softmax_negatives,
    vocab_size,
)
from .tokenization import tokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x_float * rms
        return (normalized.to(x.dtype)) * self.weight


def precompute_freqs_cis(dim, end, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    seq_len = x.size(-2)
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x_float[..., 0]
    x_imag = x_float[..., 1]

    cos = freqs_cos[:seq_len].view(1, 1, seq_len, -1)
    sin = freqs_sin[:seq_len].view(1, 1, seq_len, -1)

    rotated_real = x_real * cos - x_imag * sin
    rotated_imag = x_real * sin + x_imag * cos
    rotated = torch.stack((rotated_real, rotated_imag), dim=-1).flatten(-2)
    return rotated.to(x.dtype)


class embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        tok = tokenizer()
        self.CJ_START = tok.vocab["cj_a"]
        self.CJ_END = tok.vocab["cj_z"]
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.position = nn.Linear(embed_size * 5, embed_size, bias=False)

    def forward(self, x):
        all_emb = self.token_emb(x)
        first_id = x[:, :, 0]
        is_cj = (first_id >= self.CJ_START) & (first_id <= self.CJ_END)

        non_cj_emb = all_emb[:, :, 0, :]
        cj_emb = self.position(all_emb.flatten(2))
        if cj_emb.dtype != non_cj_emb.dtype:
            cj_emb = cj_emb.to(non_cj_emb.dtype)

        return torch.where(is_cj.unsqueeze(-1), cj_emb, non_cj_emb)


class Mutihead(nn.Module):
    def __init__(self, n_head, n_kv_head, head_size, embed_size):
        super().__init__()
        if n_head % n_kv_head != 0:
            raise ValueError("n_head 必須能被 n_kv_head 整除")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_size = head_size
        self.q_per_kv = n_head // n_kv_head
        self.query = nn.Linear(embed_size, head_size * n_head, bias=False)
        self.key = nn.Linear(embed_size, head_size * n_kv_head, bias=False)
        self.value = nn.Linear(embed_size, head_size * n_kv_head, bias=False)
        self.proj = nn.Linear(head_size * n_head, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        freqs_cos, freqs_sin = precompute_freqs_cis(head_size, block_size, rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.head_size).transpose(1, 2)
        q = apply_rotary_emb(q, self.freqs_cos, self.freqs_sin)
        k = apply_rotary_emb(k, self.freqs_cos, self.freqs_sin)
        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_size)
        out = self.proj(out)
        return self.dropout(out)


class FF(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.gate_proj = nn.Linear(embed_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(embed_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class layer(nn.Module):
    def __init__(self, n_head, embed_size):
        super().__init__()
        assert embed_size % n_head == 0, "embed_size 必須能被 n_head 整除"
        head_size = embed_size // n_head
        self.mh = Mutihead(n_head, n_kv_head, head_size, embed_size)
        self.ff = FF(embed_size)
        self.ln1 = RMSNorm(embed_size)
        self.ln2 = RMSNorm(embed_size)

    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class cj_head(nn.Module):
    def __init__(self, emb_layer):
        super().__init__()
        self.emb_layer = emb_layer
        tok = tokenizer()
        codes = tok.all_vocab()
        self.register_buffer("output_codes", codes)
        self.register_buffer("slot_pad_id", torch.tensor(tok.vocab["[PAD]"], dtype=torch.long), persistent=False)
        cj_start = tok.vocab["cj_a"]
        cj_end = tok.vocab["cj_z"]
        is_cj_output = (codes[:, 0] >= cj_start) & (codes[:, 0] <= cj_end)
        self.register_buffer(
            "non_cj_output_indices",
            torch.nonzero(~is_cj_output, as_tuple=False).squeeze(1),
            persistent=False,
        )
        self.register_buffer(
            "cj_output_indices",
            torch.nonzero(is_cj_output, as_tuple=False).squeeze(1),
            persistent=False,
        )
        self.register_buffer("non_cj_output_ids", codes[~is_cj_output, 0].to(torch.long), persistent=False)
        self.register_buffer("cj_output_codes", codes[is_cj_output].to(torch.long), persistent=False)
        codes_long = codes.to(torch.long)
        hashes = (
            (codes_long[:, 0] << 36)
            | (codes_long[:, 1] << 27)
            | (codes_long[:, 2] << 18)
            | (codes_long[:, 3] << 9)
            | codes_long[:, 4]
        )  # needs to be fixed if vocab_size add
        sorted_hashes, sorted_indices = torch.sort(hashes)
        self.register_buffer("sorted_hashes", sorted_hashes)
        self.register_buffer("sorted_indices", sorted_indices)
        self.register_buffer("_cached_output_emb", torch.empty(0), persistent=False)
        self._cached_output_versions = None

    def input_to_output_idx(self, target):
        target_long = target.to(torch.long)
        target_hash = (
            (target_long[..., 0] << 36)
            | (target_long[..., 1] << 27)
            | (target_long[..., 2] << 18)
            | (target_long[..., 3] << 9)
            | target_long[..., 4]
        )
        pos = torch.searchsorted(self.sorted_hashes, target_hash)
        return self.sorted_indices[pos]

    def slot_targets(self, target):
        """Map complete-code targets to five Cangjie-key targets, masking specials."""
        target_idx = target.to(torch.long) if target.dim() == 2 else self.input_to_output_idx(target)
        code_rows = self.output_codes.index_select(0, target_idx.reshape(-1)).view(*target_idx.shape, 5)
        cj_start = self.emb_layer.CJ_START
        cj_end = self.emb_layer.CJ_END
        is_cj = (code_rows[..., 0] >= cj_start) & (code_rows[..., 0] <= cj_end)
        slot_targets = code_rows - cj_start
        slot_targets = torch.where(
            code_rows == self.slot_pad_id,
            torch.full_like(slot_targets, 26),
            slot_targets,
        )
        return torch.where(is_cj.unsqueeze(-1), slot_targets, torch.full_like(slot_targets, -100))

    def _build_output_emb(self):
        token_weight = self.emb_layer.token_emb.weight
        output_emb = token_weight.new_empty((self.output_codes.size(0), token_weight.size(1)))

        if self.non_cj_output_ids.numel() > 0:
            non_cj_emb = F.embedding(self.non_cj_output_ids, token_weight)
            output_emb.index_copy_(0, self.non_cj_output_indices, non_cj_emb)

        if self.cj_output_codes.numel() > 0:
            cj_token_emb = F.embedding(self.cj_output_codes, token_weight).flatten(1)
            cj_emb = self.emb_layer.position(cj_token_emb)
            output_emb.index_copy_(0, self.cj_output_indices, cj_emb)

        return output_emb

    def _build_output_emb_for_indices(self, output_indices):
        token_weight = self.emb_layer.token_emb.weight
        codes = self.output_codes.index_select(0, output_indices).to(torch.long)
        first_ids = codes[:, 0]
        cj_start = self.emb_layer.CJ_START
        cj_end = self.emb_layer.CJ_END
        is_cj = (first_ids >= cj_start) & (first_ids <= cj_end)

        non_cj_emb = F.embedding(first_ids, token_weight)
        cj_token_emb = F.embedding(codes, token_weight).flatten(1)
        cj_emb = self.emb_layer.position(cj_token_emb)
        if cj_emb.dtype != non_cj_emb.dtype:
            cj_emb = cj_emb.to(non_cj_emb.dtype)
        return torch.where(is_cj.unsqueeze(-1), cj_emb, non_cj_emb)

    def output_emb(self):
        if self.training and torch.is_grad_enabled():
            return self._build_output_emb()

        token_weight = self.emb_layer.token_emb.weight
        position_weight = self.emb_layer.position.weight
        current_versions = (token_weight._version, position_weight._version)
        needs_refresh = (
            self._cached_output_emb.numel() == 0
            or self._cached_output_versions != current_versions
            or self._cached_output_emb.device != token_weight.device
            or self._cached_output_emb.dtype != token_weight.dtype
        )
        if needs_refresh:
            self._cached_output_emb = self._build_output_emb().detach()
            self._cached_output_versions = current_versions
        return self._cached_output_emb

    def logits(self, hidden):
        return hidden @ self.output_emb().T

    def loss(self, hidden, target):
        target_idx = target.to(torch.long) if target.dim() == 2 else self.input_to_output_idx(target)
        hidden = hidden.flatten(0, 1)
        target_idx = target_idx.reshape(-1)

        if sampled_softmax_negatives <= 0 or sampled_softmax_negatives >= self.output_codes.size(0):
            logits = self.logits(hidden)
            return F.cross_entropy(logits, target_idx)

        negative_idx = torch.randint(
            self.output_codes.size(0),
            (sampled_softmax_negatives,),
            device=target_idx.device,
            dtype=target_idx.dtype,
        )
        sampled_idx, inverse = torch.unique(
            torch.cat((target_idx, negative_idx)),
            sorted=True,
            return_inverse=True,
        )
        target_pos = inverse[:target_idx.numel()]
        sampled_emb = self._build_output_emb_for_indices(sampled_idx)
        logits = hidden @ sampled_emb.T
        return F.cross_entropy(logits, target_pos)

    def forward(self, hidden):
        return self.logits(hidden)


class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding(vocab_size, embed_size)
        self.layers = nn.Sequential(*[layer(n_head, embed_size) for _ in range(n_layer)])
        self.ln_f = RMSNorm(embed_size)
        self.head = cj_head(self.embedding)
        self.slot_heads = nn.ModuleList([nn.Linear(embed_size, 27) for _ in range(5)])
        self.apply(self._init_weights)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Allow older pure-Cangjie checkpoints to omit newly added slot heads."""
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        allowed_missing = [
            key for key in incompatible.missing_keys
            if key.startswith("slot_heads.")
        ]
        unexpected_missing = [
            key for key in incompatible.missing_keys
            if key not in allowed_missing
        ]
        if strict and (unexpected_missing or incompatible.unexpected_keys):
            raise RuntimeError(
                "checkpoint 與目前模型架構不相容："
                f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
            )
        return incompatible

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        _, _, _ = x.shape
        emb = self.embedding(x)
        h = self.ln_f(self.layers(emb))

        loss = None
        if target is not None:
            loss = self.head.loss(h, target)
            if cangjie_auxiliary_loss_weight > 0:
                slot_targets = self.head.slot_targets(target)
                if torch.any(slot_targets != -100):
                    slot_loss = sum(
                        F.cross_entropy(
                            slot_head(h).flatten(0, 1),
                            slot_targets[..., slot_index].reshape(-1),
                            ignore_index=-100,
                        )
                        for slot_index, slot_head in enumerate(self.slot_heads)
                    ) / len(self.slot_heads)
                    loss = loss + cangjie_auxiliary_loss_weight * slot_loss
            logits = self.head(h) if return_training_logits else None
        else:
            logits = self.head(h)

        return logits, loss


__all__ = ["LLM", "RMSNorm", "cj_head", "embedding", "layer"]
