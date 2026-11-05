"""SDPA-based attention forward replacements + Q-chunked SDPA helper.

Provides the per-architecture forward factories used by every SDPA-based
circuit-discovery algorithm (activation patching, attribution, subnetwork
probing) to install a pre-softmax additive mask. Centralised here so memory
fixes (mem-eff backend selection, Q-chunking) live in one place.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils.circuit_discovery.common import expand_sentence_mask_to_tokens


# Prefer mem-efficient SDPA over math (which materialises the full
# (1, H, seq, seq) attention pattern). Flash rejects attn_mask, so it's
# omitted from this priority list.
_PREFERRED_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

# Above this q_len, the differentiable broadcasted mask (1, 1, q, k) makes
# autograd materialise a (1, H, q, k) gradient (~35 GiB at q=24k for H=32) —
# Q-chunking caps that intermediate at (1, H, chunk, k).
_SDPA_Q_CHUNK_THRESHOLD = 8192
_SDPA_Q_CHUNK_SIZE = 4096


def _chunked_sdpa(q, k, v, attn_mask, dropout_p):
    """SDPA with Q chunked along the seq dim when q_len exceeds the threshold.

    Identical numerics to the monolithic call (each Q chunk still attends to
    the full K/V), but per-chunk memory is bounded by ``(1, H, chunk, k_len)``
    instead of ``(1, H, q_len, k_len)``. Critical when ``attn_mask`` requires
    grad and broadcasts in the head dim (autograd materialises the full
    (1, H, q, k) gradient before summing back to the broadcast source shape).
    """
    q_len = q.shape[-2]
    if q_len <= _SDPA_Q_CHUNK_THRESHOLD:
        with sdpa_kernel(_PREFERRED_SDPA):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            )
    out_chunks = []
    for i in range(0, q_len, _SDPA_Q_CHUNK_SIZE):
        end = min(i + _SDPA_Q_CHUNK_SIZE, q_len)
        q_chunk = q[:, :, i:end]
        mask_chunk = (
            attn_mask[:, :, i:end] if attn_mask is not None else None
        )
        with sdpa_kernel(_PREFERRED_SDPA):
            out_chunks.append(F.scaled_dot_product_attention(
                q_chunk, k, v, attn_mask=mask_chunk, dropout_p=dropout_p,
            ))
    return torch.cat(out_chunks, dim=2)


def _expand_mask_to_additive(
    module,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Convert binary sentence mask on *module* to an additive SDPA mask.

    Returns ``(1, H, q_len, k_len)`` with 0.0 for attend and ``min_dtype``
    for mask-out, or *None* when no circuit mask is set.

    **Fast path**: If all query tokens are continuation tokens (i.e.
    ``token_to_sent == -1``), the mask is all-ones and we return *None*,
    avoiding a huge allocation on long sequences.
    """
    mask = getattr(module, "_circuit_mask", None)
    token_to_sent = getattr(module, "_token_to_sent", None)
    gap_filter = getattr(module, "_gap_filter", None)

    if mask is None or token_to_sent is None:
        return None

    # Fast path: if every query is outside all sentences → mask is all-ones
    if cache_position is not None:
        q_sent = token_to_sent[cache_position.to(token_to_sent.device).long()]
        if (q_sent == -1).all():
            return None

    # (num_heads, q_len, k_len) binary mask
    binary = expand_sentence_mask_to_tokens(
        mask, token_to_sent, gap_filter, q_len, k_len, cache_position,
        out_dtype=dtype,
    )
    min_val = torch.finfo(dtype).min
    additive = torch.where(
        binary > 0.5,
        torch.tensor(0.0, device=binary.device, dtype=dtype),
        torch.tensor(min_val, device=binary.device, dtype=dtype),
    )
    return additive.unsqueeze(0)  # (1, H, q_len, k_len)


def _expand_mask_to_additive_memory_efficient(
    module,
    q_len: int,
    k_len: int,
    cache_position: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Opt-in, non-differentiable conversion that reuses expanded mask storage.

    Returns ``(1, H, q_len, k_len)`` with 0.0 for attend and ``min_dtype``
    for mask-out, or *None* when no circuit mask is set.

    **Fast path**: If all query tokens are continuation tokens (i.e.
    ``token_to_sent == -1``), the mask is all-ones and we return *None*,
    avoiding a huge allocation on long sequences.
    """
    mask = getattr(module, "_circuit_mask", None)
    token_to_sent = getattr(module, "_token_to_sent", None)
    gap_filter = getattr(module, "_gap_filter", None)

    if mask is None or token_to_sent is None:
        return None

    # Fast path: if every query is outside all sentences → mask is all-ones
    if cache_position is not None:
        q_sent = token_to_sent[cache_position.to(token_to_sent.device).long()]
        if (q_sent == -1).all():
            return None

    # (num_heads, q_len, k_len) binary mask
    binary = expand_sentence_mask_to_tokens(
        mask, token_to_sent, gap_filter, q_len, k_len, cache_position,
        out_dtype=dtype,
    )
    min_val = torch.finfo(dtype).min
    # Advanced indexing produced fresh storage, separate from the sentence mask.
    # This threshold converter is non-differentiable, just like the original.
    binary = binary.detach()
    with torch.no_grad():
        binary.gt_(0.5).neg_().add_(1).mul_(min_val)
        binary.add_(0.0)  # Preserve the original converter's positive zero.
    return binary.unsqueeze(0)  # (1, H, q_len, k_len)


def _make_llama_forward_sdpa(mask_converter: Callable = None):
    """Build SDPA-based LlamaAttention / Qwen2Attention forward.

    Args:
        mask_converter: Callable ``(module, q_len, k_len, cache_position, dtype)
            -> Optional[Tensor]`` producing an additive mask of shape
            ``(1, H, q_len, k_len)`` in ``dtype``. Defaults to
            ``_expand_mask_to_additive`` (binary → 0 / -inf, non-differentiable).
            Pass a differentiable variant (e.g. ``log(mask)``) for IG.
    """
    if mask_converter is None:
        mask_converter = _expand_mask_to_additive

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # --- Q / K / V projections ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim,
        ).transpose(1, 2)

        # --- RoPE ---
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        # --- KV cache ---
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin, "cos": cos, "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        # --- GQA repeat ---
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        k_len = key_states.shape[-2]

        # --- Build combined additive mask ---
        combined_mask = None
        if attention_mask is not None:
            combined_mask = attention_mask
            if combined_mask.shape[-1] != k_len:
                combined_mask = combined_mask[:, :, :, :k_len]

        sent_additive = mask_converter(
            self, q_len, k_len, cache_position, query_states.dtype,
        )
        if sent_additive is not None:
            sent_additive = sent_additive.to(device=query_states.device)
            if combined_mask is not None:
                combined_mask = combined_mask + sent_additive
            else:
                combined_mask = sent_additive

        # --- SDPA (mem-efficient backend; Q-chunked for long sequences) ---
        attn_output = _chunked_sdpa(
            query_states, key_states, value_states,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        # --- Output projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.config.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_values

    return _forward


def _make_qwen3_forward_sdpa(mask_converter: Callable = None):
    """Build SDPA-based Qwen3Attention forward (Q/K RMSNorm).

    See ``_make_llama_forward_sdpa`` for the ``mask_converter`` contract.
    """
    if mask_converter is None:
        mask_converter = _expand_mask_to_additive

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim,
        )
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim,
        )

        # Qwen3-specific Q/K normalisation
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim,
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin, "cos": cos, "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        k_len = key_states.shape[-2]

        combined_mask = None
        if attention_mask is not None:
            combined_mask = attention_mask
            if combined_mask.shape[-1] != k_len:
                combined_mask = combined_mask[:, :, :, :k_len]

        sent_additive = mask_converter(
            self, q_len, k_len, cache_position, query_states.dtype,
        )
        if sent_additive is not None:
            sent_additive = sent_additive.to(device=query_states.device)
            if combined_mask is not None:
                combined_mask = combined_mask + sent_additive
            else:
                combined_mask = sent_additive

        attn_output = _chunked_sdpa(
            query_states, key_states, value_states,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, -1)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_values

    return _forward


def _make_gpt_oss_forward_sdpa(mask_converter: Callable = None):
    """Build a patched GptOssAttention forward (attention sinks).

    GPT-OSS attention appends a learnable per-head *sink* logit as an extra
    column before the softmax and drops it afterwards, so part of the
    probability mass leaks to the sink and the retained scores no longer
    sum to 1. SDPA cannot express this, so the score matrix is materialised
    eagerly — Q-chunked to bound memory at ``(1, H, chunk, k_len + 1)``.
    The circuit mask enters as a pre-softmax additive bias exactly as in
    the SDPA forwards; the sink logit is unaffected by the mask, so masked
    attention mass is partly absorbed by the sink rather than fully
    redistributed (matching how the architecture treats suppressed keys).

    Sliding-window layers need no special handling: with eager attention
    the model passes each layer an additive mask that already encodes its
    window.
    """
    if mask_converter is None:
        mask_converter = _expand_mask_to_additive

    from transformers.models.gpt_oss.modeling_gpt_oss import (
        apply_rotary_pos_emb as gpt_oss_apply_rotary_pos_emb,
    )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = gpt_oss_apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
        )

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        k_len = key_states.shape[-2]

        combined_mask = None
        if attention_mask is not None:
            combined_mask = attention_mask
            if combined_mask.shape[-1] != k_len:
                combined_mask = combined_mask[:, :, :, :k_len]

        sent_additive = mask_converter(
            self, q_len, k_len, cache_position, query_states.dtype,
        )
        if sent_additive is not None:
            sent_additive = sent_additive.to(device=query_states.device)
            if combined_mask is not None:
                combined_mask = combined_mask + sent_additive
            else:
                combined_mask = sent_additive

        out_chunks = []
        sink = self.sinks.reshape(1, -1, 1, 1)
        for i in range(0, q_len, _SDPA_Q_CHUNK_SIZE):
            end = min(i + _SDPA_Q_CHUNK_SIZE, q_len)
            scores = torch.matmul(
                query_states[:, :, i:end], key_states.transpose(2, 3),
            ) * self.scaling
            if combined_mask is not None:
                scores = scores + combined_mask[:, :, i:end]
            sink_col = sink.expand(bsz, -1, end - i, -1).to(scores.dtype)
            combined = torch.cat([scores, sink_col], dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
            out_chunks.append(torch.matmul(probs[..., :-1], value_states))
        attn_output = torch.cat(out_chunks, dim=2)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, -1)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_values

    return _forward


_SDPA_BUILDERS = {
    "llama": _make_llama_forward_sdpa,
    "qwen2": _make_llama_forward_sdpa,
    "qwen":  _make_llama_forward_sdpa,
    "qwen3": _make_qwen3_forward_sdpa,
    "gpt_oss": _make_gpt_oss_forward_sdpa,
}


def make_sdpa_attention_forward(
    model_type: str,
    mask_converter: Optional[Callable] = None,
):
    """Build an SDPA-based patched attention forward for *model_type*.

    ``mask_converter`` is forwarded to the per-architecture builder and
    controls how ``_circuit_mask`` becomes an additive pre-softmax bias.
    Defaults to the binary ``_expand_mask_to_additive`` used by activation
    patching. IG callers pass a differentiable log-mask variant instead.
    """
    key = model_type.lower()
    if key not in _SDPA_BUILDERS:
        raise ValueError(
            f"Unsupported model type for SDPA patching: {model_type!r}. "
            f"Supported: {sorted(_SDPA_BUILDERS)}"
        )
    return _SDPA_BUILDERS[key](mask_converter=mask_converter)
