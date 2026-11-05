"""Thought Anchors leave-one-out scores for the termination study, with the
per-token KL computed on the GPU in chunks of positions.

Same algorithm and output as
``expts.thought_anchor_analysis.compute_suppression_scores`` (kept
unchanged for the other tasks that import it): for each sentence i, zero
all attention to i in every layer and head, run the prefix, and score
sentence j by the mean over its tokens of KL(clean || suppressed) of the
next-token distribution.

The original moves the full (seq_len, vocab) fp32 log-probabilities of the
clean run and of every suppressed run to the CPU and takes the KL there,
holding about three such copies at once. On a 12k-15k token prefix one copy
is 7-9 GB, which exceeds the 32 GB of host memory a one-GPU job gets
(aqua_p006_s506 and gpqa_p023_s651 were OOM-killed on 2026-09-23/24). Here
the clean log-probabilities stay on the GPU and each KL is computed over
``chunk`` positions at a time, so host memory holds only the per-token KL.
"""
from typing import List

import torch
import torch.nn.functional as F

from utils.circuit_eval import (
    install_clean_sdpa_forward,
    install_mask_hooks,
    install_sdpa_mask_hooks,
    remove_handles,
)
from utils.utils import get_attention_module

KL_CHUNK = 1024  # positions per chunk of the per-token KL computation


def _log_probs(logits: torch.Tensor, chunk: int) -> torch.Tensor:
    """fp32 log-softmax of (1, T, V) logits, filled chunk by chunk."""
    out = torch.empty(logits.shape[1:], dtype=torch.float32,
                      device=logits.device)
    for a in range(0, logits.shape[1], chunk):
        out[a:a + chunk] = F.log_softmax(logits[0, a:a + chunk].float(), dim=-1)
    return out


def _kl_per_token(logits: torch.Tensor, log_clean: torch.Tensor,
                  chunk: int) -> torch.Tensor:
    """KL(clean || masked) at every position, returned on the CPU as (T,)."""
    T = logits.shape[1]
    kl = torch.empty(T, dtype=torch.float32)
    for a in range(0, T, chunk):
        log_masked = F.log_softmax(logits[0, a:a + chunk].float(), dim=-1)
        kl[a:a + chunk] = F.kl_div(
            log_masked, log_clean[a:a + chunk], log_target=True,
            reduction="none",
        ).sum(dim=-1).cpu()
    return kl


def compute_suppression_scores(
    model,
    input_ids,
    sentences,
    token_to_sent,
    sentence_gap,
    backend: str = "sdpa",
    chunk: int = KL_CHUNK,
    *,
    memory_efficient_binary_mask: bool = False,
) -> List[List[float]]:
    """scores[src][tgt] = mean KL at sentence src when attention to tgt is
    suppressed (higher = tgt more important for src). See the module
    docstring; arguments as in the original."""
    num_sents = len(sentences)
    num_heads = model.config.num_attention_heads
    all_layers = list(range(model.config.num_hidden_layers))
    device = next(model.parameters()).device
    seq_len = input_ids.shape[-1]

    # No gap filter for suppression: any column can be zeroed.
    gap_filter = torch.zeros(num_sents, num_sents, dtype=torch.bool, device=device)

    sdpa_clean_handles = None
    if backend == "sdpa":
        sdpa_clean_handles = install_clean_sdpa_forward(model)

    model.eval()
    with torch.no_grad():
        clean_logits = model(input_ids).logits
        log_clean = _log_probs(clean_logits, chunk)
    del clean_logits
    torch.cuda.empty_cache()

    ones_mask = torch.ones(num_heads, num_sents, num_sents, device=device)
    binary_masks = {layer: ones_mask for layer in all_layers}
    install = install_sdpa_mask_hooks if backend == "sdpa" else install_mask_hooks
    if memory_efficient_binary_mask:
        if backend != "sdpa":
            raise ValueError("memory_efficient_binary_mask requires backend='sdpa'")
        handles = install(
            model, all_layers, binary_masks, token_to_sent, gap_filter,
            renormalize=True, memory_efficient_binary_mask=True,
        )
    else:
        handles = install(model, all_layers, binary_masks, token_to_sent, gap_filter,
                          renormalize=True)

    # suppression_kl[i][j] = KL at sentence j when attention to i is suppressed
    suppression_kl = [[0.0] * num_sents for _ in range(num_sents)]
    for s_suppress in range(num_sents):
        mask = torch.ones(num_heads, num_sents, num_sents, device=device)
        mask[:, :, s_suppress] = 0.0
        for layer_idx in all_layers:
            get_attention_module(model, layer_idx)._circuit_mask = mask

        with torch.no_grad():
            logits = model(input_ids).logits
            kl_tokens = _kl_per_token(logits, log_clean, chunk)

        for s_affected in range(num_sents):
            if s_affected < s_suppress + sentence_gap:
                continue
            start = sentences[s_affected].start
            end = min(sentences[s_affected].end, seq_len - 1)
            if start >= seq_len:
                continue
            suppression_kl[s_suppress][s_affected] = (
                kl_tokens[start:end + 1].mean().item()
            )
        print(f"  [{s_suppress}/{num_sents - 1}] max KL = "
              f"{max(suppression_kl[s_suppress]):.4f}", flush=True)
        del logits, kl_tokens
        torch.cuda.empty_cache()

    remove_handles(handles)
    if sdpa_clean_handles is not None:
        remove_handles(sdpa_clean_handles)
    del log_clean
    torch.cuda.empty_cache()

    return [[suppression_kl[i][j] for i in range(num_sents)]
            for j in range(num_sents)]
