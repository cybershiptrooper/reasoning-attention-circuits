"""Boundary data for the answer-distribution termination loss.

``boundary_answer_dist_kl_length`` (fix 2 of the 2026-09-21 meeting notes)
needs, for every bank continuation, the clean model's full forced-probe
distribution over the option letters

- at every paragraph break (``probe_probs``): where a stop at that break
  would land, and
- at the end of the continuation (``final_probs``): the answer the
  continuation itself ends with.  For a continuation that reached its own
  ``</think>`` the probe suffix is appended right after it; for one that
  hit the horizon, ``</think>`` is forced after its last token.

The mean of ``final_probs`` over the bank is the reference distribution
of the loss (the bank's final-answer distribution, not the distribution at
any break).  Everything else (boundaries, event tokens, clean hazards) is
copied from the reference boundary-data file.

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.build_answer_dist_boundary_data \
        --boundary_data_path results/cot_termination/early_2200/boundary_data/gpqa_p028_s407.json \
        --output_dir results/cot_termination/early_2200/boundary_data_answer_dist
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np

from utils.utils import set_seed, clear_cuda
from utils.circuit_eval import install_clean_sdpa_forward

from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.cot_termination_circuit_discovery.answer_bank_utils import (
    flatten_bank_candidates,
)
from expts.cot_termination_circuit_discovery.build_boundary_data import (
    THINK_END_ID, PROBE_SUFFIX_TEXT, _probe_at_boundary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary_data_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda",
                    help='"auto" spreads the model over all visible GPUs (Qwen3-32B)')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    bd = json.load(open(args.boundary_data_path))
    bank = json.load(open(bd["bank_path"]))
    stem = os.path.splitext(os.path.basename(args.boundary_data_path))[0]
    all_letters = bank.get("all_letters") or ["A", "B", "C", "D"]

    model, tokenizer = load_model_eager(args.model_name, device=args.device)
    install_clean_sdpa_forward(model)
    device = next(model.parameters()).device
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]
    letters = list(letter_ids)

    prefix_ids, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=bd["data_path"],
        prompt_index=int(bd["prompt_index"]), base_answer_type="stored",
        analysis_timestep=None,
        analysis_sentence_step=int(bd["analysis_sentence_step"]),
        sentences_after_prefix=int(bank.get("sentences_after_prefix", 0)),
        min_sentence_length=int(bd.get("min_sentence_length", 10)),
        sentence_chunk=1,
    )
    token_lists, _, _ = flatten_bank_candidates(bank)
    assert len(token_lists) == len(bd["candidates"])

    out = copy.deepcopy(bd)
    out["answer_letters"] = letters
    out["source_boundary_data_path"] = args.boundary_data_path
    n_label_mismatch = 0
    for c, ids in zip(out["candidates"], token_lists):
        terminated = THINK_END_ID in ids
        content = ids[: ids.index(THINK_END_ID)] if terminated else list(ids)
        probs_all, labels = [], []
        for j in c["boundaries"]:
            label, probs = _probe_at_boundary(
                model, prefix_ids, content, j, suffix_ids, letter_ids, device,
            )
            labels.append(label)
            probs_all.append(probs)
        n_label_mismatch += sum(
            1 for a, b in zip(labels, c["probe_labels"]) if a != b
        )
        c["probe_probs"] = probs_all
        final_label, final_probs = _probe_at_boundary(
            model, prefix_ids, content, len(content) - 1, suffix_ids,
            letter_ids, device,
        )
        c["final_probs"] = final_probs
        c["final_label"] = final_label
        c["final_probe_source"] = (
            "own_think_end" if terminated else "forced_at_last_token"
        )
        clear_cuda()

    ref = np.mean([[c["final_probs"][L] for L in letters]
                   for c in out["candidates"]], axis=0)
    out["reference_answer_distribution"] = {
        L: float(p) for L, p in zip(letters, ref)
    }
    n_b = sum(len(c["boundaries"]) for c in out["candidates"])
    print(f"{stem}: {len(out['candidates'])} candidates, {n_b} boundaries, "
          f"{n_label_mismatch} recomputed probe labels differ from stored")
    print(f"  final labels {[c['final_label'] for c in out['candidates']]}")
    print(f"  reference distribution {out['reference_answer_distribution']}")

    os.makedirs(args.output_dir, exist_ok=True)
    p = os.path.join(args.output_dir, f"{stem}.json")
    json.dump(out, open(p, "w"))
    print(f"  saved {p}")


if __name__ == "__main__":
    main()
