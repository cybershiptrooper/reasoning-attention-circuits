"""Evaluate a learned circuit mask at different sparsity thresholds.

Provides helpers for:
- Building binary masks from a NodeMask at a given threshold
- Installing / removing attention ablation hooks
- Running the model with masked attention and computing all metrics
- Orchestrating threshold sweeps with random-mask baselines
"""

import types
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from utils.utils import Sentence, get_attention_module
from utils.masks import NodeMask, build_gap_filter, apply_gap_filter, build_mode_filter, build_combined_filter, build_causal_filter
from utils.cot_analysis import split_tokens_into_sentences
from utils.importance_sampling import chain_log_prob, importance_weights, effective_sample_size, snis_answer_probs
from utils.circuit_discovery.common import (
    make_attention_forward,
    apply_sentence_mask,
)
from utils.circuit_discovery.sdpa_forward import (
    make_sdpa_attention_forward,
    _expand_mask_to_additive_memory_efficient,
)
from utils.circuit_discovery.base import AblationHandle


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def build_token_to_sent_map(
    sentences: list[Sentence],
    total_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each token position to its sentence index (-1 if none)."""
    token_to_sent = torch.full((total_seq_len,), -1, dtype=torch.long)
    for idx, sent in enumerate(sentences):
        token_to_sent[sent.start : sent.end + 1] = idx
    return token_to_sent.to(device)


def _threshold_2d(
    scores_2d: list[list[float]], threshold: float, num_sents: int, device: torch.device,
) -> torch.Tensor:
    """Threshold a 2D score matrix into a binary (S, S) tensor."""
    m = torch.ones(num_sents, num_sents, device=device)
    for i in range(num_sents):
        for j in range(num_sents):
            if scores_2d[i][j] < threshold:
                m[i, j] = 0.0
    return m


def build_binary_masks(
    scores: NodeMask | dict | list,
    threshold: float,
    layers: list[int],
    num_heads: int,
    num_sents: int,
    gap_filter: torch.Tensor,
    device: torch.device,
    granularity: str = "head",
) -> dict[int, torch.Tensor]:
    """Threshold scores into binary masks of shape ``(H, S, S)`` per layer.

    Handles all three granularities:

    - ``"head"``: ``scores[layer][head][i][j]``
    - ``"layer"``: ``scores[layer][i][j]`` — broadcast to all heads
    - ``"pair"``: ``scores[i][j]`` — broadcast to all heads and layers

    Returns ``{layer: (num_heads, num_sents, num_sents)}`` tensors.
    """
    if isinstance(scores, NodeMask):
        granularity = scores.granularity
        scores_data = scores.scores
    else:
        scores_data = scores

    binary_masks: dict[int, torch.Tensor] = {}

    if granularity == "pair":
        # scores_data is a 2D list
        base = _threshold_2d(scores_data, threshold, num_sents, device)
        base = apply_gap_filter(base, gap_filter, fill_value=1.0)
        expanded = base.unsqueeze(0).expand(num_heads, -1, -1)
        for layer in layers:
            binary_masks[layer] = expanded
    elif granularity == "layer":
        for layer in layers:
            base = _threshold_2d(scores_data[layer], threshold, num_sents, device)
            base = apply_gap_filter(base, gap_filter, fill_value=1.0)
            binary_masks[layer] = base.unsqueeze(0).expand(num_heads, -1, -1)
    else:  # "head"
        for layer in layers:
            m = torch.ones(num_heads, num_sents, num_sents, device=device)
            for h in range(num_heads):
                layer_scores = scores_data[layer][h]
                for i in range(num_sents):
                    for j in range(num_sents):
                        if layer_scores[i][j] < threshold:
                            m[h, i, j] = 0.0
            binary_masks[layer] = apply_gap_filter(m, gap_filter, fill_value=1.0)

    return binary_masks


def build_random_masks(
    keep_prob: float,
    layers: list[int],
    num_heads: int,
    num_sents: int,
    gap_filter: torch.Tensor,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Build random binary masks that keep *keep_prob* fraction of edges."""
    random_masks: dict[int, torch.Tensor] = {}
    for layer in layers:
        rand = torch.rand(num_heads, num_sents, num_sents, device=device)
        random_masks[layer] = apply_gap_filter(
            (rand < keep_prob).float(), gap_filter, fill_value=1.0
        )
    return random_masks


def build_random_score_masks(
    node_mask: NodeMask,
    num_samples: int,
    layers: list[int],
    combined_filter: torch.Tensor,
):
    """Create *num_samples* random score masks by permuting learned scores.

    Permutes at the native granularity of the mask so the random baseline
    has the same structural constraints as the learned mask:

    - ``"head"``: permute ``(layer, head, i, j)`` positions
    - ``"layer"``: permute ``(layer, i, j)`` positions
    - ``"pair"``: permute ``(i, j)`` positions
    """
    num_sents = combined_filter.shape[0]
    filter_bool = combined_filter.bool()
    g = node_mask.granularity

    if g == "head":
        positions: list[tuple] = []
        score_values: list[float] = []
        for layer in layers:
            for h in node_mask.scores[layer]:
                scores_2d = node_mask.scores[layer][h]
                for i in range(num_sents):
                    for j in range(num_sents):
                        if not filter_bool[i, j]:
                            positions.append((layer, h, i, j))
                            score_values.append(scores_2d[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_dict: dict = {}
            for layer in layers:
                scores_dict[layer] = {}
                for h in node_mask.scores[layer]:
                    scores_dict[layer][h] = [
                        [0.0] * num_sents for _ in range(num_sents)
                    ]
            for idx, (layer, h, i, j) in enumerate(positions):
                scores_dict[layer][h][i][j] = permuted[idx]
            random_masks.append(scores_dict)
        return random_masks

    elif g == "layer":
        positions = []
        score_values = []
        for layer in layers:
            scores_2d = node_mask.scores[layer]
            for i in range(num_sents):
                for j in range(num_sents):
                    if not filter_bool[i, j]:
                        positions.append((layer, i, j))
                        score_values.append(scores_2d[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_dict = {}
            for layer in layers:
                scores_dict[layer] = [
                    [0.0] * num_sents for _ in range(num_sents)
                ]
            for idx, (layer, i, j) in enumerate(positions):
                scores_dict[layer][i][j] = permuted[idx]
            random_masks.append(scores_dict)
        return random_masks

    else:  # "pair"
        positions = []
        score_values = []
        for i in range(num_sents):
            for j in range(num_sents):
                if not filter_bool[i, j]:
                    positions.append((i, j))
                    score_values.append(node_mask.scores[i][j])

        random_masks = []
        for _ in range(num_samples):
            perm = torch.randperm(len(score_values))
            permuted = [score_values[p] for p in perm.tolist()]
            scores_2d = [[0.0] * num_sents for _ in range(num_sents)]
            for idx, (i, j) in enumerate(positions):
                scores_2d[i][j] = permuted[idx]
            random_masks.append(scores_2d)
        return random_masks


def compute_clean_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
) -> list[torch.Tensor]:
    """Run the model on each continuation and return logits on CPU.

    Called after non-target-layer ablation hooks are installed, so the
    resulting logits reflect the baseline with non-target layers ablated
    but target layers unmasked.  Uses chunked forward when sequences are
    long to avoid OOM from eager attention.
    """
    prefix_len = input_ids.shape[-1]
    max_cont_len = max(c.shape[-1] for c in continuations)
    _do_chunk = use_chunked_forward and (prefix_len + max_cont_len) > chunk_size

    clean_logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cont in continuations:
            full_input = torch.cat([input_ids, cont], dim=-1)
            if _do_chunk:
                logits = _chunked_forward(
                    model, full_input, prefix_len, chunk_size
                )
            else:
                logits = model(full_input).logits
            clean_logits.append(logits.cpu())
            del logits
            torch.cuda.empty_cache()
    return clean_logits


# ------------------------------------------------------------------
# Attention-hook management
# ------------------------------------------------------------------


def install_mask_hooks(
    model: torch.nn.Module,
    layers: list[int],
    binary_masks: dict[int, torch.Tensor],
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
) -> list[AblationHandle]:
    """Monkey-patch attention modules with *binary_masks* and return handles."""
    forward_fn = make_attention_forward(model.config.model_type, apply_sentence_mask)
    handles: list[AblationHandle] = []
    for layer_idx in layers:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = binary_masks[layer_idx]
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def install_non_target_ablation(
    model: torch.nn.Module,
    target_layers: list[int],
    num_heads: int,
    num_sents: int,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    device: torch.device,
) -> list[AblationHandle]:
    """Zero-out attention in every layer *not* in *target_layers*."""
    num_total_layers = model.config.num_hidden_layers
    target_set = set(target_layers)
    non_target = [l for l in range(num_total_layers) if l not in target_set]
    print(f"Ablating {len(non_target)} non-target layers for evaluation...")

    zero_mask = torch.zeros(num_heads, num_sents, num_sents, device=device)
    filled_mask = apply_gap_filter(zero_mask, gap_filter, fill_value=1.0)

    forward_fn = make_attention_forward(model.config.model_type, apply_sentence_mask)
    handles: list[AblationHandle] = []
    for layer_idx in non_target:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = filled_mask
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def install_sdpa_mask_hooks(
    model: torch.nn.Module,
    layers: list[int],
    binary_masks: dict[int, torch.Tensor],
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    *,
    memory_efficient_binary_mask: bool = False,
) -> list[AblationHandle]:
    """SDPA equivalent of ``install_mask_hooks`` for *binary* masks.

    Installs the SDPA-based attention forward (pre-softmax additive mask
    via ``_expand_mask_to_additive`` from
    :mod:`utils.circuit_discovery.sdpa_forward`). For binary 0/1 masks
    with renormalization on, this is mathematically equivalent to the
    eager post-softmax × m + ratio renorm path used by
    ``install_mask_hooks``: 0 → -inf gets softmaxed to 0; 1 → 0 leaves
    the logit unchanged; softmax over surviving entries renormalizes
    automatically. ``renormalize`` is accepted for interface
    compatibility but has no separate effect under SDPA (softmax *is*
    the renormalization).
    """
    if memory_efficient_binary_mask:
        forward_fn = make_sdpa_attention_forward(
            model.config.model_type,
            mask_converter=_expand_mask_to_additive_memory_efficient,
        )
    else:
        forward_fn = make_sdpa_attention_forward(model.config.model_type)
    handles: list[AblationHandle] = []
    for layer_idx in layers:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = binary_masks[layer_idx]
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def install_non_target_ablation_sdpa(
    model: torch.nn.Module,
    target_layers: list[int],
    num_heads: int,
    num_sents: int,
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    device: torch.device,
) -> list[AblationHandle]:
    """SDPA equivalent of ``install_non_target_ablation``."""
    num_total_layers = model.config.num_hidden_layers
    target_set = set(target_layers)
    non_target = [l for l in range(num_total_layers) if l not in target_set]
    print(f"Ablating {len(non_target)} non-target layers (SDPA backend)...")

    zero_mask = torch.zeros(num_heads, num_sents, num_sents, device=device)
    filled_mask = apply_gap_filter(zero_mask, gap_filter, fill_value=1.0)

    forward_fn = make_sdpa_attention_forward(model.config.model_type)
    handles: list[AblationHandle] = []
    for layer_idx in non_target:
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        attn_module._circuit_mask = filled_mask
        attn_module._token_to_sent = token_to_sent
        attn_module._gap_filter = gap_filter
        attn_module._renormalize_masked_attn = renormalize
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def install_clean_sdpa_forward(
    model: torch.nn.Module,
) -> list[AblationHandle]:
    """Patch every attention layer to use the SDPA forward with no mask installed.

    Used by ``--backend sdpa`` callers that want the *clean* reference
    forward to also run through the SDPA-patched path (so the masked
    eval and the clean baseline differ only in the mask, not the
    backend). Without ``_circuit_mask`` set on the module, the SDPA
    forward's mask converter returns ``None`` and SDPA runs unmodified.
    """
    forward_fn = make_sdpa_attention_forward(model.config.model_type)
    handles: list[AblationHandle] = []
    num_total_layers = model.config.num_hidden_layers
    for layer_idx in range(num_total_layers):
        attn_module = get_attention_module(model, layer_idx)
        original_forward = attn_module.forward
        # Explicitly clear any leftover state from prior runs.
        for attr in ("_circuit_mask", "_token_to_sent", "_gap_filter"):
            if hasattr(attn_module, attr):
                delattr(attn_module, attr)
        attn_module.forward = types.MethodType(forward_fn, attn_module)
        handles.append(AblationHandle(attn_module, original_forward))
    return handles


def remove_handles(handles: list[AblationHandle]):
    """Remove all ablation handles (restore original forwards)."""
    for h in handles:
        h.remove()


# ------------------------------------------------------------------
# Direct-answer probe metric (deterministic suffix evaluation)
# ------------------------------------------------------------------


def compute_probe_metric(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    suffix_token_ids: list[int] | torch.Tensor,
    answer_token_ids: list[int] | torch.Tensor,
    clean_answer_logprobs: torch.Tensor,
    target_answer_id: Optional[int] = None,
) -> dict:
    """One-forward-pass answer-distribution metric for the deterministic probe.

    Assumes the caller has already installed the ablation hooks for the
    threshold under test.  Runs the model on ``prefix + suffix +
    placeholder``, reads the logits at the position whose next token is
    the answer, restricts to ``answer_token_ids``, softmax-renormalises,
    and computes:

    - ``answer_probs_masked``: list of P(letter) under the masked model.
    - ``probe_kl``: KL(P_clean || P_masked) over the answer simplex.
    - ``probe_reward_gap``: ``P_masked(target) - max_{other} P_masked(other)``
      (only when *target_answer_id* is provided).

    Args:
        clean_answer_logprobs: ``(num_answers,)`` log-probs over the answer
            simplex under the clean (unablated) model.  Computed once
            outside this helper and reused across thresholds.
    """
    device = next(model.parameters()).device
    if isinstance(suffix_token_ids, list):
        suffix_token_ids = torch.tensor(suffix_token_ids, dtype=torch.long)
    if isinstance(answer_token_ids, list):
        answer_token_ids = torch.tensor(answer_token_ids, dtype=torch.long)
    suffix_token_ids = suffix_token_ids.to(device)
    answer_token_ids = answer_token_ids.to(device)
    clean_lp = clean_answer_logprobs.to(device)

    placeholder = answer_token_ids[0:1]
    cont = torch.cat([suffix_token_ids, placeholder]).unsqueeze(0)
    full_input = torch.cat([input_ids.to(device), cont], dim=-1)
    prefix_len = input_ids.shape[-1]
    answer_pos = prefix_len + suffix_token_ids.shape[-1] - 1

    model.eval()
    with torch.no_grad():
        logits = model(full_input).logits
    row = logits[0, answer_pos].float()
    masked_lp = torch.log_softmax(row[answer_token_ids], dim=-1)
    masked_p = masked_lp.exp()

    p_clean = clean_lp.exp()
    kl = (p_clean * (clean_lp - masked_lp)).sum().item()

    out = {
        "probe_kl": kl,
        "answer_probs_masked": masked_p.cpu().tolist(),
    }
    if target_answer_id is not None:
        target = int(target_answer_id)
        other = torch.ones(
            masked_p.shape[0], dtype=torch.bool, device=masked_p.device,
        )
        other[target] = False
        out["probe_reward_gap"] = float(
            (masked_p[target] - masked_p[other].max()).item()
        )
        out["probe_p_target"] = float(masked_p[target].item())
        out["probe_p_best_other"] = float(masked_p[other].max().item())
    return out


# ------------------------------------------------------------------
# Unified evaluation pass — all metrics from a single forward pass
# ------------------------------------------------------------------


def _chunked_forward(
    model: torch.nn.Module,
    full_input: torch.Tensor,
    prefix_len: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Run a forward pass in chunks using KV cache to reduce peak memory.

    Splits the input into prefix + continuation chunks.  Each chunk's
    attention matrix is only ``(heads, chunk_len, accumulated_len)`` instead
    of ``(heads, full_len, full_len)``.  The model's attention hooks (masks)
    are active for all chunks because hooks are installed by monkey-patching
    each attention module's ``.forward`` — ``model(chunk, past_key_values=...)``
    still calls the hooked forward with ``cache_position`` so the mask
    correctly indexes sentence positions for each chunk.

    This is mathematically equivalent to a single full forward pass because
    causal attention means token *i*'s output only depends on tokens 0..i,
    all of which are in the KV cache by the time token *i* is processed.

    Args:
        model: The (hooked) model.
        full_input: ``(1, full_len)`` input token IDs.
        prefix_len: Number of prefix tokens (first chunk boundary).
        chunk_size: Max tokens per continuation chunk.

    Returns:
        ``(1, full_len, vocab)`` logits — identical to ``model(full_input).logits``.
    """
    from transformers import DynamicCache

    device = full_input.device
    full_len = full_input.shape[-1]
    past_key_values = DynamicCache()
    all_logits: list[torch.Tensor] = []

    # Build chunk boundaries: [0, prefix_len, prefix_len+chunk, ...]
    boundaries = [0, prefix_len]
    pos = prefix_len
    while pos < full_len:
        boundaries.append(min(pos + chunk_size, full_len))
        pos += chunk_size

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        chunk_ids = full_input[:, start:end]
        cache_position = torch.arange(start, end, device=device)

        outputs = model(
            chunk_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        all_logits.append(outputs.logits)
        past_key_values = outputs.past_key_values

    # Free KV cache
    del past_key_values

    return torch.cat(all_logits, dim=1)


def eval_all_metrics(
    model: torch.nn.Module,
    binary_masks: dict[int, torch.Tensor],
    layers: list[int],
    input_ids: torch.Tensor,
    continuations: list[torch.Tensor],
    clean_logits_list: list[torch.Tensor],
    token_to_sent: torch.Tensor,
    gap_filter: torch.Tensor,
    renormalize: bool,
    tokenizer=None,
    min_sentence_length: int = 10,
    branch_rewards: list[float] | None = None,
    position_mask_overrides: list[torch.Tensor | None] | None = None,
    chain_logprobs_clean: Optional[torch.Tensor] = None,
    answer_ids_fine: Optional[torch.Tensor] = None,
    num_answers_fine: Optional[int] = None,
    answer_ids_binary: Optional[torch.Tensor] = None,
    num_answers_binary: Optional[int] = None,
    collect_per_sentence: bool = True,
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
    temperature: float = 1.0,
    is_method: str = "snis",
    is_temperature: Optional[float] = None,
    chain_lengths: Optional[torch.Tensor] = None,
    probe_metric_fn: Optional[Callable] = None,
) -> dict:
    """Run model with *binary_masks* and compute all metrics in a single pass.

    For each continuation, one forward pass yields logits from which we extract:

    **Local metrics (always computed):**
    - ``kl_divergence``: plain unweighted mean per-token KL over the full
      continuation (all tokens after the prefix).
    - ``reward_weighted_kl``: same KL multiplied by per-branch reward weights
      (only when ``branch_rewards`` is provided).
    - ``per_sentence_kl``: per-sentence mean KL for each branch.

    **IS-based metrics (when ``answer_ids_fine`` is provided):**
    - ``answer_kl``: KL(P_clean || P_m) over the fine-grained answer
      distribution (each distinct answer is its own bucket, ``__no_answer``
      variants merged).

    **IS-based metrics (when ``answer_ids_binary`` is provided):**
    - ``reward_gap``, ``p_target``, ``p_best_other``: reward gap using binary
      correct/incorrect bucketing.
    - ``answer_probs_masked``: per-answer probabilities (binary).
    - ``n_eff``, ``n_eff_ratio``: effective sample size diagnostics.
    - ``log_weights``: raw log importance weights.

    **Contrastive metrics (when ``answer_ids_binary`` is provided):**
    - ``kl_a``: mean per-token KL over correct-answer chains.
    - ``kl_b``: mean per-token KL over incorrect-answer chains.
    - ``contrastive_loss``: ``kl_a - kl_b``.

    Args:
        collect_per_sentence: Whether to compute per-sentence KL breakdown.
        use_chunked_forward: If True, use KV-cache chunked forward to reduce
            peak attention memory from O(seq^2) to O(chunk * seq).  Set to
            False for raw full forward passes (faster for short sequences).
        chunk_size: Max tokens per chunk when using chunked forward.
    """
    from utils.objectives import (
        answer_distribution_kl_loss,
        answer_distribution_kl_loss_weighted,
        reward_gap_loss,
    )

    device = next(model.parameters()).device
    prefix_len = input_ids.shape[-1]
    compute_is_fine = (
        answer_ids_fine is not None
        and num_answers_fine is not None
        and chain_logprobs_clean is not None
    )
    compute_is_binary = (
        answer_ids_binary is not None
        and num_answers_binary is not None
        and chain_logprobs_clean is not None
    )

    # Auto-select: only chunk when sequences are long enough to benefit
    max_cont_len = max(c.shape[-1] for c in continuations)
    _do_chunk = use_chunked_forward and (prefix_len + max_cont_len) > chunk_size

    handles = install_mask_hooks(
        model, layers, binary_masks, token_to_sent, gap_filter, renormalize
    )

    # Accumulators
    total_kl = 0.0
    total_weighted_kl = 0.0
    per_branch_kl: list[float] = []  # mean KL per branch (for contrastive)
    per_sent_kl_branches: list[list[dict]] = []
    chain_lps: list[torch.Tensor] = []
    total_branches = 0

    with torch.no_grad():
        # Optional direct-answer probe metric (one extra forward on
        # `prefix + suffix + placeholder` while hooks are alive).
        if probe_metric_fn is not None:
            probe_out = probe_metric_fn(model)

        for cont_idx, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            full_len = full_input.shape[-1]

            if _do_chunk:
                logits = _chunked_forward(
                    model, full_input, prefix_len, chunk_size
                )
            else:
                logits = model(full_input).logits
            out_device = logits.device
            clean = clean_logits_list[cont_idx][:, :full_len].to(out_device)

            # --- Per-token KL (computed for all metrics, not saved raw) ---
            # Chunk over seq dim to bound peak fp32 memory: vocab x seq fp32
            # tensors can be tens of GB for long sequences with large vocabs.
            _kl_chunk = 256
            _kl_parts = []
            for _i in range(0, logits.shape[1], _kl_chunk):
                _lc = F.log_softmax(
                    clean[:, _i : _i + _kl_chunk].detach().float(), dim=-1
                )
                _lm = F.log_softmax(
                    logits[:, _i : _i + _kl_chunk].float(), dim=-1
                )
                _kl_parts.append(
                    F.kl_div(_lm, _lc, log_target=True, reduction="none").sum(dim=-1)
                )
                del _lc, _lm
            kl_tokens = torch.cat(_kl_parts, dim=1)  # (1, seq_len)
            del _kl_parts
            # Keep dummy refs so existing `del log_clean, log_masked` below works
            log_clean = log_masked = kl_tokens

            # Local KL: always average over the full continuation
            cont_len = full_len - prefix_len
            analyse_end = full_len

            # Build position mask for local KL
            local_pos_mask = torch.zeros(1, full_len, device=out_device)
            local_pos_mask[0, prefix_len - 1 : analyse_end - 1] = 1.0

            # Apply position_mask_overrides if provided (e.g. answer-only)
            if position_mask_overrides is not None and position_mask_overrides[cont_idx] is not None:
                pos_mask_override = position_mask_overrides[cont_idx].to(out_device)
                # Intersect: override mask AND local window
                effective_mask = torch.zeros(1, full_len, device=out_device)
                effective_mask[0, :pos_mask_override.shape[-1]] = pos_mask_override[0, :full_len]
                effective_mask[0, analyse_end - 1 :] = 0.0  # clip to analysis window
                local_kl = (kl_tokens * effective_mask).sum() / effective_mask.sum().clamp(min=1)
            else:
                local_kl = (kl_tokens * local_pos_mask).sum() / local_pos_mask.sum().clamp(min=1)

            branch_kl_val = local_kl.item()
            total_kl += branch_kl_val
            per_branch_kl.append(branch_kl_val)

            # Reward-weighted KL
            if branch_rewards is not None:
                total_weighted_kl += branch_kl_val * branch_rewards[cont_idx]

            # --- Per-sentence KL (within analysis window) ---
            if collect_per_sentence and tokenizer is not None:
                # Per-token KL for the continuation (full, for sentence splitting)
                branch_kl_tokens = kl_tokens[0, prefix_len - 1 : full_len - 1].cpu().tolist()
                cont_token_ids = cont[0]
                cont_sents = split_tokens_into_sentences(
                    cont_token_ids,
                    tokenizer,
                    min_sentence_length=min_sentence_length,
                )
                sent_kl_list = []
                for s in cont_sents:
                    s_kl = branch_kl_tokens[s.start : s.end + 1]
                    avg = sum(s_kl) / max(len(s_kl), 1)
                    text = tokenizer.decode(
                        cont_token_ids[s.start : s.end + 1].tolist()
                    )
                    sent_kl_list.append({"text": text, "mean_kl": avg})
                per_sent_kl_branches.append(sent_kl_list)

            # --- Chain log-prob for IS metrics (over FULL branch) ---
            if compute_is_fine or compute_is_binary:
                # Chunked equivalent of chain_log_prob to avoid materializing
                # a full (1, seq, vocab) fp32 tensor.
                _seq = full_input.shape[-1]
                _scale = (1.0 / temperature) if temperature != 1.0 else 1.0
                _targets = full_input[:, prefix_len:_seq]  # (1, cont_len)
                _cont_lp_parts = []
                _start = prefix_len - 1
                _end = _seq - 1
                _chunk = 256
                for _i in range(_start, _end, _chunk):
                    _j = min(_i + _chunk, _end)
                    _seg = logits[:, _i:_j].float() * _scale
                    _lp_seg = F.log_softmax(_seg, dim=-1)
                    _tgt_seg = _targets[:, _i - _start : _j - _start]
                    _tok_lp = _lp_seg.gather(-1, _tgt_seg.unsqueeze(-1)).squeeze(-1)
                    _cont_lp_parts.append(_tok_lp)
                    del _seg, _lp_seg
                lp = torch.cat(_cont_lp_parts, dim=-1).sum(dim=-1).squeeze(0)
                chain_lps.append(lp)
                del _cont_lp_parts

            total_branches += 1

            # Free GPU memory between branches
            del logits, clean, log_clean, log_masked, kl_tokens
            del full_input, local_pos_mask
            torch.cuda.empty_cache()

    remove_handles(handles)

    # Assemble results
    result: dict = {
        "kl_divergence": total_kl / max(total_branches, 1),
    }

    if branch_rewards is not None:
        result["reward_weighted_kl"] = total_weighted_kl / max(total_branches, 1)

    if per_sent_kl_branches:
        result["per_sentence_kl"] = per_sent_kl_branches

    # --- IS-based metrics (shared importance weights) ---
    if compute_is_fine or compute_is_binary:
        chain_lps_t = torch.stack(chain_lps).to(device)
        clean_lps = chain_logprobs_clean.to(device)

        chain_lengths_dev = (
            chain_lengths.to(device) if chain_lengths is not None else None
        )
        w = importance_weights(
            chain_lps_t, clean_lps,
            method=is_method, chain_lengths=chain_lengths_dev,
            temperature=is_temperature,
        )
        n_eff = effective_sample_size(w)
        result["n_eff"] = n_eff
        result["n_eff_ratio"] = n_eff / len(continuations)
        result["log_weights"] = (
            (chain_lps_t - clean_lps).detach().cpu().tolist()
        )
        result["chain_weights_normalized"] = w.detach().cpu().tolist()
        result["importance_sampling_method"] = is_method
        if is_temperature is not None:
            result["importance_sampling_temperature"] = float(is_temperature)

    # Answer KL uses fine-grained buckets (each distinct answer is its own bucket)
    if compute_is_fine:
        answer_ids_fine_dev = answer_ids_fine.to(device)
        p_m_fine = snis_answer_probs(w, answer_ids_fine_dev, num_answers_fine)

        # Paper-style (Eq. 1) outcome distribution under the masked model:
        # within-sample histogram weighted by softmax(chain_logprobs_masked).
        sample_weights_m = torch.softmax(chain_lps_t, dim=0)
        p_m_weighted = torch.zeros(num_answers_fine, device=device)
        for a in range(num_answers_fine):
            a_mask = (answer_ids_fine_dev == a).float()
            p_m_weighted[a] = (sample_weights_m * a_mask).sum()

        answer_kl = answer_distribution_kl_loss(
            chain_lps_t, clean_lps, answer_ids_fine_dev, num_answers_fine,
            chain_lengths=chain_lengths_dev, is_method=is_method,
            is_temperature=is_temperature,
        ).item()
        answer_kl_w = answer_distribution_kl_loss_weighted(
            chain_lps_t, clean_lps, answer_ids_fine_dev, num_answers_fine,
            chain_lengths=chain_lengths_dev, is_method=is_method,
            is_temperature=is_temperature,
        ).item()

        result["answer_kl"] = answer_kl
        result["answer_kl_weighted"] = answer_kl_w
        result["answer_probs_masked_fine"] = p_m_fine.detach().cpu().tolist()
        result["answer_probs_masked_weighted"] = p_m_weighted.detach().cpu().tolist()

    # Reward gap uses binary buckets (correct vs incorrect)
    if compute_is_binary:
        answer_ids_binary_dev = answer_ids_binary.to(device)
        p_m_binary = snis_answer_probs(w, answer_ids_binary_dev, num_answers_binary)

        p_target = p_m_binary[0].item()   # correct
        p_other = p_m_binary[1].item() if num_answers_binary > 1 else 0.0
        reward_gap = p_target - p_other

        result["reward_gap"] = reward_gap
        result["p_target"] = p_target
        result["p_best_other"] = p_other
        result["answer_probs_masked"] = p_m_binary.detach().cpu().tolist()

    # --- Contrastive metrics (group per-branch KL by binary answer_ids) ---
    if answer_ids_binary is not None and num_answers_binary is not None:
        target_answer = 0
        kl_a_vals = []
        kl_b_vals = []
        for i, bkl in enumerate(per_branch_kl):
            if answer_ids_binary[i].item() == target_answer:
                kl_a_vals.append(bkl)
            else:
                kl_b_vals.append(bkl)
        kl_a = sum(kl_a_vals) / max(len(kl_a_vals), 1) if kl_a_vals else 0.0
        kl_b = sum(kl_b_vals) / max(len(kl_b_vals), 1) if kl_b_vals else 0.0
        result["kl_a"] = kl_a
        result["kl_b"] = kl_b
        result["contrastive_loss"] = kl_a - kl_b

    if probe_metric_fn is not None:
        result.update({f"probe_{k}" if not k.startswith("probe_") else k: v
                       for k, v in probe_out.items()})

    return result


# ------------------------------------------------------------------
# Main threshold sweep
# ------------------------------------------------------------------


def evaluate_at_thresholds(
    model: torch.nn.Module,
    node_mask: NodeMask,
    input_ids: torch.Tensor,
    sentences: list[Sentence],
    continuations: list[torch.Tensor],
    objective_fn: Callable,
    thresholds: list[float],
    layers: list[int],
    ablate_non_target_layers: bool = False,
    renormalize_masked_attention: bool = True,
    tokenizer=None,
    min_sentence_length: int = 10,
    num_random_samples: int = 5,
    branch_rewards: list[float] | None = None,
    position_mask_overrides: list[torch.Tensor | None] | None = None,
    answer_ids_fine: Optional[torch.Tensor] = None,
    num_answers_fine: Optional[int] = None,
    answer_ids_binary: Optional[torch.Tensor] = None,
    num_answers_binary: Optional[int] = None,
    use_chunked_forward: bool = True,
    chunk_size: int = 2048,
    temperature: float = 1.0,
    importance_sampling_method: str = "snis",
    importance_sampling_temperature: Optional[float] = None,
    probe_suffix_token_ids: Optional[list[int]] = None,
    probe_answer_token_ids: Optional[list[int]] = None,
    probe_target_answer_id: Optional[int] = None,
) -> list[dict]:
    """Evaluate all metrics at different mask thresholds.

    For each threshold, runs :func:`eval_all_metrics` once on the learned mask
    and *K* times on random baseline masks.  All available metrics (local, IS,
    contrastive) are computed in a single forward pass per continuation.

    Args:
        answer_ids_fine: Fine-grained answer IDs (each distinct answer is its
            own bucket, ``__no_answer`` variants merged).  Used for ``answer_kl``.
        num_answers_fine: Number of fine-grained answer buckets.
        answer_ids_binary: Binary answer IDs (0=correct, 1=incorrect).
            Used for ``reward_gap``.
        num_answers_binary: Number of binary answer buckets (always 2).
    """
    device = next(model.parameters()).device
    num_heads = model.config.num_attention_heads
    prefix_len = input_ids.shape[-1]
    num_sents = len(sentences)
    max_cont_len = max(c.shape[-1] for c in continuations)
    total_seq_len = prefix_len + max_cont_len

    token_to_sent = build_token_to_sent_map(sentences, total_seq_len, device)

    sentence_gap = 0
    if hasattr(node_mask, "metadata"):
        sentence_gap = node_mask.metadata.get("sentence_gap", 0)
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=device)

    # Build combined filter (gap + mode + causal) to match discovery-time filtering
    mask_mode = node_mask.metadata.get("mask_mode", "prefix")
    num_prefix_sents = node_mask.metadata.get("num_prefix_sentences", num_sents)
    mode_filter = build_mode_filter(num_prefix_sents, num_sents, mask_mode, device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
    combined_filter_cpu = combined_filter.cpu()

    # Optionally ablate non-target layers
    non_target_handles: list[AblationHandle] = []
    if ablate_non_target_layers:
        non_target_handles = install_non_target_ablation(
            model,
            layers,
            num_heads,
            num_sents,
            token_to_sent,
            combined_filter,
            renormalize_masked_attention,
            device,
        )

    # Compute clean logits (no target-layer masks; ratio-based renormalization
    # in apply_sentence_mask ensures patched forwards are bit-identical to
    # unpatched when mask=1, so no all-ones hooks are needed here).
    print("Computing clean logits for threshold evaluation...")
    clean_logits_list = compute_clean_logits(
        model, input_ids, continuations,
        use_chunked_forward=use_chunked_forward,
        chunk_size=chunk_size,
    )

    # Compute clean chain logprobs when any answer_ids are available (for IS metrics)
    chain_logprobs_clean = None
    has_fine = answer_ids_fine is not None and num_answers_fine is not None
    has_binary = answer_ids_binary is not None and num_answers_binary is not None
    if has_fine or has_binary:
        chain_logprobs_clean = []
        for ci, cont in enumerate(continuations):
            full_input = torch.cat([input_ids, cont], dim=-1)
            clean_logits = clean_logits_list[ci][:, : full_input.shape[-1]]
            lp = chain_log_prob(clean_logits, full_input.cpu(), prefix_len, temperature=temperature)
            chain_logprobs_clean.append(lp.detach())
        chain_logprobs_clean = torch.stack(chain_logprobs_clean).to(device)

    # Paper-style (Eq. 1) outcome distribution under the clean model.
    # Threshold-independent — computed once and stashed on node_mask.metadata
    # so the dashboard can read it at top level.
    if has_fine and chain_logprobs_clean is not None:
        answer_ids_fine_dev_top = answer_ids_fine.to(device)
        sample_weights_clean = torch.softmax(chain_logprobs_clean.detach(), dim=0)
        p_clean_weighted = torch.zeros(num_answers_fine, device=device)
        for a in range(num_answers_fine):
            a_mask = (answer_ids_fine_dev_top == a).float()
            p_clean_weighted[a] = (sample_weights_clean * a_mask).sum()
        node_mask.metadata["answer_probs_clean_weighted"] = (
            p_clean_weighted.cpu().tolist()
        )

    # Per-chain continuation lengths — invariant across thresholds.
    chain_lengths = torch.tensor(
        [c.shape[-1] for c in continuations], dtype=torch.long, device=device,
    )

    # Optional direct-answer probe metric: pre-compute clean answer
    # log-probs once and build a closure that the threshold-loop can
    # invoke while masked-hooks are alive.
    probe_metric_fn = None
    if probe_suffix_token_ids is not None and probe_answer_token_ids is not None:
        suffix_ids_t = torch.tensor(probe_suffix_token_ids, dtype=torch.long, device=device)
        ans_ids_t = torch.tensor(probe_answer_token_ids, dtype=torch.long, device=device)
        placeholder = ans_ids_t[0:1]
        probe_full = torch.cat(
            [input_ids.to(device), suffix_ids_t.unsqueeze(0), placeholder.unsqueeze(0)],
            dim=-1,
        )
        probe_answer_pos = prefix_len + suffix_ids_t.shape[-1] - 1
        with torch.no_grad():
            clean_probe_logits = model(probe_full).logits
        clean_probe_lp = torch.log_softmax(
            clean_probe_logits[0, probe_answer_pos, ans_ids_t].float(), dim=-1,
        ).detach()
        node_mask.metadata["probe_answer_probs_clean"] = (
            clean_probe_lp.exp().cpu().tolist()
        )
        del clean_probe_logits

        def _probe_fn(_model):
            return compute_probe_metric(
                _model,
                input_ids=input_ids,
                suffix_token_ids=suffix_ids_t,
                answer_token_ids=ans_ids_t,
                clean_answer_logprobs=clean_probe_lp,
                target_answer_id=probe_target_answer_id,
            )
        probe_metric_fn = _probe_fn

    # Shared kwargs for eval_all_metrics
    shared_kwargs = dict(
        layers=layers,
        input_ids=input_ids,
        continuations=continuations,
        clean_logits_list=clean_logits_list,
        token_to_sent=token_to_sent,
        gap_filter=combined_filter,
        renormalize=renormalize_masked_attention,
        tokenizer=tokenizer,
        min_sentence_length=min_sentence_length,
        branch_rewards=branch_rewards,
        position_mask_overrides=position_mask_overrides,
        chain_logprobs_clean=chain_logprobs_clean,
        answer_ids_fine=answer_ids_fine,
        num_answers_fine=num_answers_fine,
        answer_ids_binary=answer_ids_binary,
        num_answers_binary=num_answers_binary,
        use_chunked_forward=use_chunked_forward,
        chunk_size=chunk_size,
        temperature=temperature,
        is_method=importance_sampling_method,
        is_temperature=importance_sampling_temperature,
        chain_lengths=chain_lengths,
        probe_metric_fn=probe_metric_fn,
    )

    # Pre-generate K random score masks by permuting learned scores
    print(f"Generating {num_random_samples} random score masks (permuted)...")
    random_score_masks = build_random_score_masks(
        node_mask, num_random_samples, layers, combined_filter_cpu
    )

    results = []
    for threshold in thresholds:
        sparsity = node_mask.sparsity(threshold, gap_filter=combined_filter_cpu)

        binary_masks = build_binary_masks(
            node_mask,
            threshold,
            layers,
            num_heads,
            num_sents,
            combined_filter,
            device,
        )

        # Single call: all metrics from one forward pass per continuation
        learned = eval_all_metrics(
            model=model, binary_masks=binary_masks, **shared_kwargs,
        )

        # Evaluate K random score masks at this threshold
        random_results: list[dict] = []
        granularity = node_mask.granularity
        for k in range(num_random_samples):
            rand_binary = build_binary_masks(
                random_score_masks[k],
                threshold,
                layers,
                num_heads,
                num_sents,
                combined_filter,
                device,
                granularity=granularity,
            )
            rand_result = eval_all_metrics(
                model=model,
                binary_masks=rand_binary,
                # Skip per-sentence KL for random baselines (expensive, not needed)
                collect_per_sentence=False,
                **shared_kwargs,
            )
            random_results.append(rand_result)
            del rand_binary
        del binary_masks
        torch.cuda.empty_cache()

        # --- Build entry dict ---
        entry: dict = {
            "threshold": threshold,
            "sparsity": sparsity,
            "kl_divergence": learned["kl_divergence"],
            "random_kl_divergence": _mean_field(random_results, "kl_divergence"),
            "random_kl_divergences": [r["kl_divergence"] for r in random_results],
        }

        # Reward-weighted KL
        if "reward_weighted_kl" in learned:
            entry["reward_weighted_kl"] = learned["reward_weighted_kl"]
            entry["random_reward_weighted_kl"] = _mean_field(random_results, "reward_weighted_kl")
            entry["random_reward_weighted_kls"] = [
                r.get("reward_weighted_kl", 0.0) for r in random_results
            ]

        # Per-sentence KL
        if "per_sentence_kl" in learned:
            entry["per_sentence_kl"] = learned["per_sentence_kl"]

        # IS-based shared diagnostics
        if "n_eff" in learned:
            entry["n_eff"] = learned["n_eff"]
            entry["n_eff_ratio"] = learned["n_eff_ratio"]
            entry["log_weights"] = learned["log_weights"]
            entry["chain_weights_normalized"] = learned["chain_weights_normalized"]
            entry["importance_sampling_method"] = learned["importance_sampling_method"]
            if "importance_sampling_temperature" in learned:
                entry["importance_sampling_temperature"] = learned["importance_sampling_temperature"]
            entry["random_n_effs"] = [r.get("n_eff", 0.0) for r in random_results]

        # Answer KL (fine-grained buckets)
        if "answer_kl" in learned:
            entry["answer_kl"] = learned["answer_kl"]
            entry["answer_kl_weighted"] = learned["answer_kl_weighted"]
            entry["answer_probs_masked_fine"] = learned["answer_probs_masked_fine"]
            entry["answer_probs_masked_weighted"] = learned["answer_probs_masked_weighted"]
            entry["random_answer_kl"] = _mean_field(random_results, "answer_kl")
            entry["random_answer_kls"] = [r.get("answer_kl", 0.0) for r in random_results]
            entry["random_answer_kl_weighted"] = _mean_field(random_results, "answer_kl_weighted")
            entry["random_answer_kl_weighteds"] = [r.get("answer_kl_weighted", 0.0) for r in random_results]

        # Reward gap (binary buckets)
        if "reward_gap" in learned:
            entry["reward_gap"] = learned["reward_gap"]
            entry["p_target"] = learned["p_target"]
            entry["p_best_other"] = learned["p_best_other"]
            entry["answer_probs_masked"] = learned["answer_probs_masked"]
            entry["random_reward_gap"] = _mean_field(random_results, "reward_gap")
            entry["random_reward_gaps"] = [r.get("reward_gap", 0.0) for r in random_results]
            entry["random_p_target"] = _mean_field(random_results, "p_target")
            entry["random_p_targets"] = [r.get("p_target", 0.0) for r in random_results]
            entry["random_p_best_other"] = _mean_field(random_results, "p_best_other")
            entry["random_p_best_others"] = [r.get("p_best_other", 0.0) for r in random_results]

        # Contrastive metrics
        if "kl_a" in learned:
            entry["kl_a"] = learned["kl_a"]
            entry["kl_b"] = learned["kl_b"]
            entry["contrastive_loss"] = learned["contrastive_loss"]
            entry["random_kl_a"] = _mean_field(random_results, "kl_a")
            entry["random_kl_as"] = [r.get("kl_a", 0.0) for r in random_results]
            entry["random_kl_b"] = _mean_field(random_results, "kl_b")
            entry["random_kl_bs"] = [r.get("kl_b", 0.0) for r in random_results]
            entry["random_contrastive_loss"] = _mean_field(random_results, "contrastive_loss")
            entry["random_contrastive_losses"] = [
                r.get("contrastive_loss", 0.0) for r in random_results
            ]

        # Direct-answer probe metrics (forced-suffix evaluation)
        if "probe_kl" in learned:
            entry["probe_kl"] = learned["probe_kl"]
            entry["probe_answer_probs_masked"] = learned["probe_answer_probs_masked"]
            entry["random_probe_kl"] = _mean_field(random_results, "probe_kl")
            entry["random_probe_kls"] = [r.get("probe_kl", 0.0) for r in random_results]
            if "probe_reward_gap" in learned:
                entry["probe_reward_gap"] = learned["probe_reward_gap"]
                entry["probe_p_target"] = learned["probe_p_target"]
                entry["probe_p_best_other"] = learned["probe_p_best_other"]
                entry["random_probe_reward_gap"] = _mean_field(random_results, "probe_reward_gap")
                entry["random_probe_reward_gaps"] = [r.get("probe_reward_gap", 0.0) for r in random_results]

        results.append(entry)

        # --- Logging ---
        extra_parts = []
        if "answer_kl" in learned:
            extra_parts.append(f"answer_kl={learned['answer_kl']:.6f}")
            if "reward_gap" in learned:
                extra_parts.append(f"reward_gap={learned['reward_gap']:.4f}")
            if "n_eff" in learned:
                extra_parts.append(
                    f"N_eff={learned['n_eff']:.1f} ({learned['n_eff_ratio']:.1%})"
                )
        if "kl_a" in learned:
            extra_parts.append(
                f"kl_a={learned['kl_a']:.6f} kl_b={learned['kl_b']:.6f}"
            )
        if "probe_kl" in learned:
            extra_parts.append(f"probe_kl={learned['probe_kl']:.6f}")
            if "probe_reward_gap" in learned:
                extra_parts.append(f"probe_gap={learned['probe_reward_gap']:+.4f}")
        extra = (" | " + " | ".join(extra_parts)) if extra_parts else ""
        print(
            f"  threshold={threshold:.1e} | sparsity={sparsity:.2%} "
            f"| KL={learned['kl_divergence']:.6f} "
            f"| random KL={entry['random_kl_divergence']:.6f} "
            f"(K={num_random_samples})"
            f"{extra}"
        )

    remove_handles(non_target_handles)
    return results


def _mean_field(results: list[dict], key: str) -> float:
    """Mean of a field across result dicts, skipping missing entries."""
    vals = [r[key] for r in results if key in r]
    return sum(vals) / max(len(vals), 1)
