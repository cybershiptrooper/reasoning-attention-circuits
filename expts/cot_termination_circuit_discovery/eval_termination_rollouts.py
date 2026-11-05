"""Ground-truth evaluation of a termination mask: generate under the
installed mask and measure termination within the horizon.

For the learned mask (top-k at its trained target sparsity), matched-
sparsity random masks, and the clean model, generates --n_rollouts
continuations of at most --horizon tokens from the analysis-point
prefix (mask active during generation), then reports:

- termination rate within the horizon
- tokens to ``</think>`` among terminated rollouts
- answer accuracy among terminated rollouts (same regex grading as the
  bank builder)

Also reports the training objective's own estimate of the masked
target-cluster probability on the held-out bank candidates
(self-normalised importance sampling over teacher-forced log-probs),
with its effective sample size — a check that the training estimator
tracks the generation ground truth.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
    build_prompt_filter,
)
from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.utils import set_seed, clear_cuda, get_attention_module

from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.cot_termination_circuit_discovery.eval_utils import (
    scores_to_log_alpha,
    build_binary_mask,
    random_binary_mask,
    binary_to_per_layer_masks,
)


def _install(model, layers, binary, num_heads, token_to_sent, combined_filter):
    masks = binary_to_per_layer_masks(binary, layers, num_heads)
    for l in layers:
        attn = get_attention_module(model, l)
        attn._circuit_mask = masks[l]
        attn._token_to_sent = token_to_sent
        attn._gap_filter = combined_filter
        attn._renormalize_masked_attn = True


def _clear(model, layers):
    for l in layers:
        attn = get_attention_module(model, l)
        for a in ("_circuit_mask", "_token_to_sent", "_gap_filter",
                  "_renormalize_masked_attn"):
            if hasattr(attn, a):
                delattr(attn, a)


THINK_END_ID = 151668  # dedicated "</think>" token in the Qwen3 vocab
PROBE_SUFFIX_TEXT = " I think the answer is"


def _probe_label(model, tokenizer, prefix_ids, gen_ids, think_pos,
                 suffix_ids, letter_ids, device):
    """Answer letter read from the forced suffix placed right after the
    generation's own </think> token (mask state unchanged — the label is
    measured under whatever model state is currently installed)."""
    full = torch.cat([
        prefix_ids.to(device),
        gen_ids[: think_pos + 1].unsqueeze(0).to(device),
        torch.tensor([suffix_ids], device=device),
    ], dim=-1)
    with torch.no_grad():
        logits = model(full, logits_to_keep=1).logits
    row = logits[0, -1].float()
    lvals = {L: float(row[tid]) for L, tid in letter_ids.items()}
    probs = torch.softmax(torch.tensor(list(lvals.values())), dim=0)
    label = max(lvals, key=lvals.get)
    del logits, full
    return label, {L: float(p) for L, p in zip(lvals, probs)}


def _generate_and_grade(model, tokenizer, prefix_ids, n_rollouts, horizon,
                        temperature, device, suffix_ids, letter_ids,
                        trace_answer, batch_size=8, probe_at_horizon=False,
                        store_token_ids=True):
    """Generate ``n_rollouts`` continuations and read each answer with the
    forced probe after its own ``</think>``.

    With ``probe_at_horizon`` a continuation that never emits ``</think>``
    is also probed, with ``</think>`` forced after its last generated
    token (``probe_forced_at_horizon`` is set on it), so every rollout
    carries a letter distribution.  With ``store_token_ids`` the generated
    ids (up to the first pad/eos token) are kept.
    """
    stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id} - {None}
    prefix_len = prefix_ids.shape[-1]
    results = []
    done = 0
    while done < n_rollouts:
        b = min(batch_size, n_rollouts - done)
        out = model.generate(
            input_ids=prefix_ids.to(device).expand(b, -1),
            max_new_tokens=horizon,
            temperature=temperature,
            top_p=1.0,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for i in range(b):
            new = out[i, prefix_len:]
            ids = new.tolist()
            think_pos = ids.index(THINK_END_ID) if THINK_END_ID in ids else None
            # the batch is padded to its longest member: cut at the first
            # pad/eos so the tail is this rollout's own text, not padding
            n_gen = next((k for k, t in enumerate(ids) if t in stop_ids),
                         len(ids))
            r = {
                "n_tokens": len(ids),
                "think_pos": think_pos,
                "terminated": think_pos is not None,
                "tokens_to_termination": (think_pos + 1
                                          if think_pos is not None else None),
                "probe_label": None,
                "text_tail": tokenizer.decode(ids[:n_gen][-60:]),
            }
            if think_pos is not None:
                label, probs = _probe_label(
                    model, tokenizer, prefix_ids, new.cpu(), think_pos,
                    suffix_ids, letter_ids, device)
                r["probe_label"] = label
                r["probe_letter_probs"] = probs
                r["matches_trace"] = label == trace_answer
            elif probe_at_horizon:
                forced = torch.tensor(ids[:n_gen] + [THINK_END_ID])
                label, probs = _probe_label(
                    model, tokenizer, prefix_ids, forced, n_gen,
                    suffix_ids, letter_ids, device)
                r["probe_label"] = label
                r["probe_letter_probs"] = probs
                r["probe_forced_at_horizon"] = True
            if store_token_ids:
                r["gen_token_ids"] = ids[:n_gen]
            results.append(r)
        done += b
        del out
        clear_cuda()
    return results


def _summarize(rollouts, trace_answer, gold_letter):
    n = len(rollouts)
    term = [r for r in rollouts if r["terminated"]]
    return {
        "n_rollouts": n,
        "termination_rate": len(term) / n if n else None,
        "match_trace_rate_among_terminated": (
            float(np.mean([r["probe_label"] == trace_answer for r in term]))
            if term else None
        ),
        "match_gold_rate_among_terminated": (
            float(np.mean([r["probe_label"] == gold_letter for r in term]))
            if term else None
        ),
        "target_cluster_rate": (
            sum(1 for r in term if r["probe_label"] == trace_answer) / n
            if n else None
        ),
        "median_tokens_to_termination": (
            float(np.median([r["tokens_to_termination"] for r in term]))
            if term else None
        ),
    }


def _teacher_forced_logprobs(model, prefix_ids, candidates, device):
    """Teacher-forced sequence log-prob of each candidate continuation
    under the currently-installed model state (masked or clean).

    Uses ``logits_to_keep`` so only the continuation window's logits are
    materialised.
    """
    lps = []
    prefix_len = prefix_ids.shape[-1]
    with torch.no_grad():
        for c in candidates:
            cont = torch.tensor([c["continuation_token_ids"]], device=device)
            cont_len = cont.shape[-1]
            full = torch.cat([prefix_ids.to(device), cont], dim=-1)
            logits = model(full, logits_to_keep=cont_len + 1).logits
            # kept positions are full[-(cont_len+1):]; position i predicts
            # token i+1, so rows 0..cont_len-1 predict the cont tokens.
            lsm = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            lp = lsm.gather(-1, cont[0].unsqueeze(-1)).sum()
            lps.append(float(lp.item()))
            del full, logits, lsm
    clear_cuda()
    return lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--bank_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda",
                    help='"auto" spreads the model over all visible GPUs (Qwen3-32B)')
    ap.add_argument("--n_rollouts", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_random_masks", type=int, default=3)
    ap.add_argument("--sentence_gap", type=int, default=0)
    ap.add_argument("--mask_mode", default="prefix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_snis_check", action="store_true")
    ap.add_argument("--skip_clean", action="store_true",
                    help="Skip the clean-model rollouts (identical across "
                    "runs of the same prompt at the same seed — run clean "
                    "once per prompt and skip it elsewhere).")
    ap.add_argument("--probe_at_horizon", action="store_true",
                    help="Also probe continuations that hit the horizon, "
                    "with </think> forced after their last token.")
    ap.add_argument("--store_token_ids", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Keep every rollout's generated token ids (default on; "
                    "--no-store_token_ids to drop them).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    nm = NodeMask.from_json(args.mask_path)
    with open(args.bank_path) as f:
        bank = json.load(f)
    meta = nm.metadata
    data_path = meta["data_path"]
    prompt_index = int(meta["prompt_index"])
    step = int(meta["analysis_sentence_step"])
    target_sparsity = float(meta.get("target_sparsity"))
    assert bank["prompt_index"] == prompt_index and \
        bank["analysis_sentence_step"] == step, "bank/mask mismatch"

    with open(data_path) as f:
        record = json.load(f)[prompt_index]
    trace_answer = (record.get("clean_answer") or "").strip()
    gold_letter = (record.get("correct_letter") or "").strip()
    all_letters = record.get("all_letters") or ["A", "B", "C", "D"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]
    gen_kwargs = dict(suffix_ids=suffix_ids, letter_ids=letter_ids,
                      trace_answer=trace_answer, batch_size=args.batch_size,
                      probe_at_horizon=args.probe_at_horizon,
                      store_token_ids=args.store_token_ids)
    prefix_ids, sentences, _, _, _, num_prompt_sentences = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=0, min_sentence_length=10, sentence_chunk=1,
    )
    num_sents = len(sentences)
    prefix_len = prefix_ids.shape[-1]

    model, _ = load_model_eager(args.model_name, device=args.device)
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    num_heads = model.config.num_attention_heads
    handles = install_clean_sdpa_forward(model)

    gap_filter = build_gap_filter(num_sents, args.sentence_gap, device=device)
    mode_filter = build_mode_filter(num_sents, num_sents, args.mask_mode,
                                    device=device)
    causal_filter = build_causal_filter(num_sents, device=device)
    num_frozen = int(meta.get("num_frozen_prompt_sentences", 0) or 0)
    prompt_filter = (build_prompt_filter(num_frozen, num_sents, device=device)
                     if num_frozen else None)
    combined_filter = build_combined_filter(
        gap_filter, mode_filter, causal_filter, prompt_filter)
    valid_filter = ~combined_filter.bool()

    log_alpha = scores_to_log_alpha(
        nm.scores, meta.get("score_readout", "hard_concrete_mean")
    ).to(device)
    learned_binary = build_binary_mask(
        log_alpha, "top_k", target_sparsity, valid_filter)

    max_len = prefix_len + args.horizon + 8
    token_to_sent = torch.full((max_len,), -1, dtype=torch.long, device=device)
    for idx, s in enumerate(sentences):
        token_to_sent[s.start:s.end + 1] = idx

    out = {
        "mask_path": args.mask_path,
        "bank_path": args.bank_path,
        "data_path": data_path,
        "prompt_index": prompt_index,
        "analysis_sentence_step": step,
        "target_sparsity": target_sparsity,
        "prefix_len": prefix_len,
        "horizon": args.horizon,
        "temperature": args.temperature,
        "clean_fraction_terminated_scan": bank.get("clean_fraction_terminated"),
        "variants": {},
    }

    out["trace_answer"] = trace_answer
    out["gold_letter"] = gold_letter

    # ---- clean ----
    if not args.skip_clean:
        set_seed(args.seed)
        rolls = _generate_and_grade(model, tokenizer, prefix_ids,
                                    args.n_rollouts, args.horizon,
                                    args.temperature, device, **gen_kwargs)
        out["variants"]["clean"] = {
            "summary": _summarize(rolls, trace_answer, gold_letter),
            "rollouts": rolls,
        }
        print("clean:", out["variants"]["clean"]["summary"])

    # ---- learned mask ----
    _install(model, layers, learned_binary, num_heads, token_to_sent,
             combined_filter)
    set_seed(args.seed)
    rolls = _generate_and_grade(model, tokenizer, prefix_ids, args.n_rollouts,
                                args.horizon, args.temperature, device,
                                **gen_kwargs)
    learned_summary = {
        "summary": _summarize(rolls, trace_answer, gold_letter),
        "rollouts": rolls,
    }

    if not args.skip_snis_check:
        held = bank.get("heldout_candidates") or []
        if held:
            lp_masked = _teacher_forced_logprobs(model, prefix_ids, held,
                                                 device)
            _clear(model, layers)
            lp_clean = _teacher_forced_logprobs(model, prefix_ids, held,
                                                device)
            lw = torch.tensor(lp_masked) - torch.tensor(lp_clean)
            w = torch.softmax(lw, dim=0)
            cl = torch.tensor([c["cluster_id"] for c in held])
            p0 = float(w[cl == 0].sum().item())
            ess = float(1.0 / (w ** 2).sum().item())
            learned_summary["snis_heldout"] = {
                "p_target_cluster": p0,
                "ess": ess,
                "n_heldout": len(held),
            }
            print(f"SNIS held-out: P(cluster0)={p0:.3f}, ESS={ess:.1f}/{len(held)}")
    _clear(model, layers)
    out["variants"]["learned"] = learned_summary
    print("learned:", learned_summary["summary"])

    # ---- random masks ----
    for ri in range(args.n_random_masks):
        rb = random_binary_mask(
            tuple(log_alpha.shape), target_sparsity,
            valid_filter.cpu(), seed=args.seed + 1000 + ri,
        ).to(device)
        _install(model, layers, rb, num_heads, token_to_sent, combined_filter)
        set_seed(args.seed)
        rolls = _generate_and_grade(
            model, tokenizer, prefix_ids, args.n_rollouts, args.horizon,
            args.temperature, device, **gen_kwargs)
        _clear(model, layers)
        out["variants"][f"random_{ri}"] = {
            "summary": _summarize(rolls, trace_answer, gold_letter),
            "rollouts": rolls,
        }
        print(f"random_{ri}:", out["variants"][f"random_{ri}"]["summary"])

    remove_handles(handles)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
