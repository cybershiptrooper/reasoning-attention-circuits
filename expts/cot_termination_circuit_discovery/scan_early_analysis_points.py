"""Collect (prompt, analysis point) pairs far from the trace's ending, and
build candidate banks at them.

Motivation: every termination bank so far was collected at analysis points
where the model's answer is already decided (the forced answer probe returns
the trace's answer at essentially every paragraph break).  This script
collects the opposite regime: the analysis point is placed a fixed number of
tokens BEFORE the trace's own ``</think>`` (default 2,200), where sampled
continuations still disagree about the final answer.

Selection per collection file:

1. keep prompts whose accuracy over the 16 stored rollouts (fraction of
   ``all_sampled_answers`` equal to ``correct_letter``) lies in
   [--accuracy_min, --accuracy_max], whose trace answer (``clean_answer``)
   is a parseable letter, and whose base rollout contains ``</think>``;
2. analysis point = the sentence ending closest to
   (position of ``</think>``) - --offset_tokens, clamped to at least
   10 reasoning sentences into the trace and to --max_prefix_tokens;
3. sample --n_samples continuations from that point with vLLM
   (temperature 1.0, at most --horizon tokens);
4. label every continuation with an answer letter: continuations that
   terminate are probed with `` I think the answer is`` after their own
   ``</think>``; continuations that do not terminate are probed with
   `` </think> I think the answer is`` forced at their truncation point
   (this second label is used only for the diversity check, never for
   cluster assignment);
5. **diversity check**: if all continuations carry the same answer letter,
   the answer is still effectively decided at this point — re-scan the
   prompt at --fallback_offset_tokens (default 3,000) before the ending;
6. keep up to --per_dataset prompts per dataset (files sharing a name
   prefix, e.g. aqua.json + aqua_train.json, count as one dataset),
   preferring points whose continuations disagree the most, and requiring
   at least --min_target_cluster cluster-0 continuations (so a training/
   held-out split is possible) and at least one continuation outside
   cluster 0 (so pairwise losses have pairs);
7. build banks (same schema as build_termination_bank.py, reusing its
   split/candidate functions) and a manifest.

Cluster rule is unchanged from the 512-token-horizon banks: cluster 0 =
terminated within the horizon AND probed answer equals the trace's own
answer; cluster 1 = terminated with a different answer; cluster 2 = not
terminated.  The horizon here is a generation cap (--horizon, default
4,096), not part of any loss.

Usage (one GPU):
    uv run python -m expts.cot_termination_circuit_discovery.scan_early_analysis_points \
        --data_paths data/collection/qwen3_8b/gpqa.json \
                     data/collection/qwen3_8b/aqua.json \
                     data/collection/qwen3_8b/aqua_train.json \
        --output_dir results/cot_termination/early_2200
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from utils.cot_analysis import (
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)
from expts.cot_termination_circuit_discovery.build_termination_bank import (
    PROBE_SUFFIX,
    NUM_CLUSTERS,
    TARGET_CLUSTER,
    stratified_split,
    to_candidate,
)
import random

MIN_SENTENCE_LENGTH = 10
THINK_END_ID = 151668  # dedicated "</think>" token in the Qwen3 vocab
PROBE_SUFFIX_TEXT = " I think the answer is"
MIN_REASONING_SENTENCES_BEFORE_POINT = 10


def record_accuracy(rec):
    answers = rec.get("all_sampled_answers") or []
    gold = (rec.get("correct_letter") or "").strip()
    if not answers or not gold:
        return None
    return sum(1 for a in answers if (a or "").strip() == gold) / len(answers)


def dataset_key(data_path):
    """aqua.json and aqua_train.json count as one dataset ("aqua")."""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return stem.split("_")[0]


def build_pair(rec, pi, data_path, tokenizer, offset_tokens, max_prefix_tokens):
    """Analysis point = sentence ending closest to (</think> - offset)."""
    trace_answer = (rec.get("clean_answer") or "").strip()
    all_letters = rec.get("all_letters") or ["A", "B", "C", "D"]
    if len(trace_answer) != 1 or trace_answer not in all_letters:
        return None
    out_ids = list(rec["output_token_ids"])
    if THINK_END_ID not in out_ids:
        return None  # trace was cut short; no ending point to offset from
    prompt_len = len(rec["prompt_token_ids"])
    end_pos = prompt_len + out_ids.index(THINK_END_ID)  # position of </think>
    full = torch.tensor(list(rec["prompt_token_ids"]) + out_ids)
    sents = split_tokens_into_sentences(
        full, tokenizer, min_sentence_length=MIN_SENTENCE_LENGTH
    )
    sents = remove_bos_from_sentences(sents)
    sents = chunk_sentences(sents, 1)
    n_sent = len(sents)
    n_prompt = sum(1 for s in sents if s.start < prompt_len)
    lo = n_prompt + MIN_REASONING_SENTENCES_BEFORE_POINT
    hi = n_sent - 2
    if lo > hi:
        return None
    target_cut = end_pos - offset_tokens
    # sentence step whose prefix cut is closest to the target, within limits
    best_step, best_dist = None, None
    for step in range(lo, hi + 1):
        cut = sents[step - 1].end + 1
        if cut > max_prefix_tokens:
            break  # cuts only grow with step
        d = abs(cut - target_cut)
        if best_dist is None or d < best_dist:
            best_step, best_dist = step, d
    if best_step is None:
        return None
    cut = sents[best_step - 1].end + 1
    return {
        "data_path": data_path,
        "prompt_index": pi,
        "analysis_sentence_step": best_step,
        "offset_tokens_requested": int(offset_tokens),
        "tokens_before_think_end": int(end_pos - cut),
        "prefix_len": int(cut),
        "n_sentences": n_sent,
        "n_prompt_sentences": n_prompt,
        "prefix_ids": full[:cut].tolist(),
        "trace_answer": trace_answer,
        "correct_letter": (rec.get("correct_letter") or "").strip(),
        "all_letters": all_letters,
        "accuracy": record_accuracy(rec),
    }


def letter_token_ids(tokenizer, letters):
    out = {}
    for L in letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            out[L] = ids[0]
    return out


def scan_pairs(llm, sp, sp_probe, tokenizer, pairs, suffix_ids, horizon):
    """Sample branches for each pair and label them.  Mutates pairs' dicts by
    attaching 'samples', counts, and 'all_same_answer'."""
    from vllm import SamplingParams  # noqa: F401  (type only)

    outs = llm.generate(
        [{"prompt_token_ids": p["prefix_ids"]} for p in pairs], sp
    )
    probe_prompts, probe_meta = [], []
    per_pair_samples = []
    for pair, out in zip(pairs, outs):
        samples = []
        for comp in out.outputs:
            ids = list(comp.token_ids)
            think_pos = ids.index(THINK_END_ID) if THINK_END_ID in ids else None
            s = {
                "token_ids": ids,
                "n_tokens": len(ids),
                "think_pos": think_pos,
                "terminated": think_pos is not None,
                "probe_label": None,
                "probe_letter_probs": None,
                "forced_probe_label": None,
                "regex_answer": None,
            }
            if think_pos is not None:
                # probe right after the branch's own </think>
                probe_prompts.append({"prompt_token_ids":
                    pair["prefix_ids"] + ids[:think_pos + 1] + suffix_ids})
            else:
                # branch never terminated: force " </think> I think the
                # answer is" at its truncation point.  Used ONLY for the
                # answer-diversity check, never for clustering.
                probe_prompts.append({"prompt_token_ids":
                    pair["prefix_ids"] + ids + [THINK_END_ID] + suffix_ids})
            probe_meta.append((len(per_pair_samples), len(samples)))
            samples.append(s)
        per_pair_samples.append(samples)

    probe_outs = llm.generate(probe_prompts, sp_probe)
    for (pair_i, samp_i), pout in zip(probe_meta, probe_outs):
        pair = pairs[pair_i]
        lids = letter_token_ids(tokenizer, pair["all_letters"])
        lp_dict = pout.outputs[0].logprobs[0]
        raw = {L: lp_dict[tid].logprob
               for L, tid in lids.items() if tid in lp_dict}
        s = per_pair_samples[pair_i][samp_i]
        if raw:
            mx = max(raw.values())
            exp = {L: math.exp(v - mx) for L, v in raw.items()}
            z = sum(exp.values())
            label = max(raw, key=raw.get)
            if s["terminated"]:
                s["probe_letter_probs"] = {L: v / z for L, v in exp.items()}
                s["probe_label"] = label
            else:
                s["forced_probe_label"] = label

    for pair, samples in zip(pairs, per_pair_samples):
        for s in samples:
            if s["terminated"] and s["probe_label"] == pair["trace_answer"]:
                s["cluster_id"] = 0
            elif s["terminated"]:
                s["cluster_id"] = 1
            else:
                s["cluster_id"] = 2
            s["matches_gold"] = (
                s["probe_label"] == pair["correct_letter"]
                if s["terminated"] else None
            )
        labels = [s["probe_label"] if s["terminated"] else s["forced_probe_label"]
                  for s in samples]
        labels = [l for l in labels if l is not None]
        pair["samples"] = samples
        pair["n_samples"] = len(samples)
        pair["n_terminated"] = sum(s["terminated"] for s in samples)
        pair["n_target_cluster"] = sum(s["cluster_id"] == 0 for s in samples)
        pair["f_terminated"] = pair["n_terminated"] / len(samples)
        pair["branch_answer_labels"] = labels
        pair["n_distinct_answers"] = len(set(labels))
        pair["all_same_answer"] = len(set(labels)) <= 1
        pair["majority_answer_fraction"] = (
            max(labels.count(l) for l in set(labels)) / len(labels)
            if labels else 1.0
        )


def eligible(pair, min_target_cluster):
    n0 = pair["n_target_cluster"]
    n_other = pair["n_samples"] - n0
    return (not pair["all_same_answer"]
            and n0 >= min_target_cluster
            and n_other >= 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_paths", nargs="+", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--accuracy_min", type=float, default=0.5)
    ap.add_argument("--accuracy_max", type=float, default=0.75)
    ap.add_argument("--offset_tokens", type=int, default=2200)
    ap.add_argument("--fallback_offset_tokens", type=int, default=3000)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=4096)
    ap.add_argument("--max_prefix_tokens", type=int, default=15000)
    ap.add_argument("--per_dataset", type=int, default=5)
    ap.add_argument("--candidates_per_dataset", type=int, default=10,
                    help="prompts scanned per dataset (extras survive the "
                         "diversity filter)")
    ap.add_argument("--min_target_cluster", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_scan", nargs="*", default=[],
                    help="scan_early.json files of earlier scans; their "
                    "SELECTED (data_path, prompt_index) pairs are not scanned "
                    "again (used to add prompts to an existing pool).")
    ap.add_argument("--max_total", type=int, default=None,
                    help="cap on the total number of selected prompts, filled "
                    "round-robin over datasets in their ranking order (a "
                    "dataset with fewer eligible prompts leaves its share to "
                    "the others).")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(args.output_dir, exist_ok=True)
    bank_dir = os.path.join(args.output_dir, "banks")
    os.makedirs(bank_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)

    # ---- candidate prompts per dataset, accuracy band first ----
    excluded = set()
    for path in args.exclude_scan:
        with open(path) as f:
            for p in json.load(f)["selected"]:
                excluded.add((os.path.normpath(p["data_path"]), p["prompt_index"]))
    if excluded:
        print(f"Excluding {len(excluded)} prompts selected by earlier scans")
    by_dataset: dict[str, list] = {}
    for dp in args.data_paths:
        with open(dp) as f:
            records = json.load(f)
        n_band = 0
        for pi, rec in enumerate(records):
            if (os.path.normpath(dp), pi) in excluded:
                continue
            acc = record_accuracy(rec)
            if acc is None or not (args.accuracy_min <= acc <= args.accuracy_max):
                continue
            n_band += 1
            pair = build_pair(rec, pi, dp, tokenizer, args.offset_tokens,
                              args.max_prefix_tokens)
            if pair is not None:
                by_dataset.setdefault(dataset_key(dp), []).append((rec, pair))
        print(f"{dp}: {len(records)} records, {n_band} in accuracy band, "
              f"usable so far: "
              f"{sum(len(v) for v in by_dataset.values())}")

    # cap candidates per dataset (closest to the accuracy-band midpoint first)
    mid = 0.5 * (args.accuracy_min + args.accuracy_max)
    round1 = []
    kept_records = {}
    for ds, entries in by_dataset.items():
        entries.sort(key=lambda e: abs((e[1]["accuracy"] or 0.0) - mid))
        for rec, pair in entries[: args.candidates_per_dataset]:
            round1.append(pair)
            kept_records[(pair["data_path"], pair["prompt_index"])] = rec
    print(f"Scanning {len(round1)} candidate pairs at "
          f"offset {args.offset_tokens}")

    llm = LLM(model=args.model_name, gpu_memory_utilization=0.9,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_prefix_tokens + args.horizon + 64,
              seed=args.seed)
    sp = SamplingParams(n=args.n_samples, max_tokens=args.horizon,
                        temperature=1.0, top_p=1.0, seed=args.seed)
    sp_probe = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    scan_pairs(llm, sp, sp_probe, tokenizer, round1, suffix_ids, args.horizon)

    # ---- round 2: re-scan answer-uniform prompts at the earlier offset ----
    round2 = []
    for pair in round1:
        if not pair["all_same_answer"]:
            continue
        rec = kept_records[(pair["data_path"], pair["prompt_index"])]
        earlier = build_pair(rec, pair["prompt_index"], pair["data_path"],
                             tokenizer, args.fallback_offset_tokens,
                             args.max_prefix_tokens)
        if earlier is None:
            continue
        if earlier["analysis_sentence_step"] == pair["analysis_sentence_step"]:
            continue  # trace too short for a genuinely earlier point
        round2.append(earlier)
    if round2:
        print(f"Re-scanning {len(round2)} answer-uniform pairs at "
              f"offset {args.fallback_offset_tokens}")
        scan_pairs(llm, sp, sp_probe, tokenizer, round2, suffix_ids,
                   args.horizon)

    # replace uniform round-1 pairs by their round-2 version when that helped
    by_key = {(p["data_path"], p["prompt_index"]): p for p in round1}
    for p in round2:
        key = (p["data_path"], p["prompt_index"])
        if not p["all_same_answer"]:
            by_key[key] = p

    # ---- selection: most answer-diverse first, per dataset ----
    selected, skipped = [], []
    ranked = {}
    for ds in sorted({dataset_key(dp) for dp in args.data_paths}):
        cands = [p for p in by_key.values() if dataset_key(p["data_path"]) == ds]
        good = [p for p in cands if eligible(p, args.min_target_cluster)]
        good.sort(key=lambda p: (-p["n_distinct_answers"],
                                 p["majority_answer_fraction"]))
        ranked[ds] = (cands, good[: args.per_dataset])
        print(f"dataset {ds}: {len(good)} eligible of {len(cands)} scanned, "
              f"at most {min(len(good), args.per_dataset)} kept")
    if args.max_total is None:
        keep = {ds: good for ds, (_, good) in ranked.items()}
    else:
        keep = {ds: [] for ds in ranked}
        rank = 0
        n_kept = 0
        while n_kept < args.max_total and any(
                rank < len(good) for _, good in ranked.values()):
            for ds, (_, good) in ranked.items():
                if rank < len(good) and n_kept < args.max_total:
                    keep[ds].append(good[rank])
                    n_kept += 1
            rank += 1
    for ds, (cands, _) in ranked.items():
        selected.extend(keep[ds])
        skipped.extend([p for p in cands if p not in keep[ds]])
        print(f"dataset {ds}: keeping {len(keep[ds])}")

    # ---- scan record (samples included — banks are built from these) ----
    def strip(pair, with_samples):
        d = {k: v for k, v in pair.items() if k != "prefix_ids"}
        if not with_samples:
            d = {k: v for k, v in d.items() if k != "samples"}
        return d

    scan_path = os.path.join(args.output_dir, "scan_early.json")
    with open(scan_path, "w") as f:
        json.dump({"args": {k: v for k, v in vars(args).items()},
                   "selected": [strip(p, True) for p in selected],
                   "skipped": [strip(p, False) for p in skipped]}, f)
    print(f"Saved {scan_path}")

    # ---- banks (same schema as build_termination_bank.py) ----
    manifest = []
    for pair in selected:
        rng = random.Random(args.seed + pair["prompt_index"])
        train, heldout = stratified_split(pair["samples"], args.n_train, rng)
        if sum(s["cluster_id"] == TARGET_CLUSTER for s in train) == 0:
            print(f"SKIP p{pair['prompt_index']}: no cluster-0 candidate "
                  f"in the training half")
            continue
        stem = os.path.splitext(os.path.basename(pair["data_path"]))[0]
        name = f"{stem}_p{pair['prompt_index']:03d}_s{pair['analysis_sentence_step']}"
        bank = {
            "bank_type": "termination",
            "data_path": pair["data_path"],
            "prompt_index": pair["prompt_index"],
            "analysis_sentence_step": pair["analysis_sentence_step"],
            "sentences_after_prefix": 0,
            "probe_suffix": PROBE_SUFFIX,
            "horizon": max(s["n_tokens"] for s in pair["samples"]),
            "num_clusters": NUM_CLUSTERS,
            "target_cluster": TARGET_CLUSTER,
            "gold_answer_normalized": pair["correct_letter"],
            "trace_answer": pair["trace_answer"],
            "all_letters": pair["all_letters"],
            "clean_fraction_correct": pair["n_target_cluster"] / pair["n_samples"],
            "clean_fraction_terminated": pair["f_terminated"],
            "candidates": [to_candidate(s) for s in train],
            "heldout_candidates": [to_candidate(s) for s in heldout],
            # provenance of the early-analysis-point regime
            "offset_tokens_requested": pair["offset_tokens_requested"],
            "tokens_before_think_end": pair["tokens_before_think_end"],
            "prompt_accuracy": pair["accuracy"],
            "branch_answer_labels": pair["branch_answer_labels"],
        }
        out_path = os.path.join(bank_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(bank, f)
        counts = [sum(s["cluster_id"] == c for s in train) for c in range(3)]
        print(f"{name}: offset {pair['tokens_before_think_end']} tokens before "
              f"</think>, train clusters {counts}, "
              f"distinct answers {pair['n_distinct_answers']}")
        manifest.append({
            "bank_path": out_path,
            "data_path": pair["data_path"],
            "prompt_index": pair["prompt_index"],
            "analysis_sentence_step": pair["analysis_sentence_step"],
            "prefix_len": pair["prefix_len"],
            "f_terminated": pair["f_terminated"],
            "tokens_before_think_end": pair["tokens_before_think_end"],
            "n_distinct_answers": pair["n_distinct_answers"],
        })
    man_path = os.path.join(bank_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} banks + {man_path}")


if __name__ == "__main__":
    main()
