"""Precompute boundary-hazard metadata for a termination bank.

**Event definition.**  Qwen3 does not emit ``</think>`` directly after a
sentence: termination is a multi-token sequence — a paragraph break,
then final-answer text (e.g. ``**Final Answer**\\n\\boxed{A}\\n``), then
``</think>``.  Measured single-token ``</think>`` probability at sentence
ends is ~e^-30 to e^-45 everywhere except inside that sequence, so it
carries no usable signal.  The decision point the model actually faces is
*at each paragraph break*: continue reasoning (next token ``But`` /
``Wait`` / ``So`` ...) or start the wrap-up (next token ``**`` of
``**Final Answer**``, or ``</think>`` directly).  We therefore define:

- **boundaries** = positions of paragraph-break tokens (decoded token
  contains a double newline) in the continuation;
- **hazard** h_b = total probability, at boundary b, of the *wrap-up
  head token set*: the tokens observed to start the final-answer text
  across this prompt's terminated candidates, plus the ``</think>``
  token itself.  A single-forward, single-position readout.

For every (flattened) bank candidate and each boundary b this script
records:

- ``clean_log_h``: log h_b under the clean model;
- ``eligible``: whether the clean model's forced answer probe — append
  the ``</think>`` token plus ``" I think the answer is"`` at the
  boundary and read the answer-letter logits — yields the trace's own
  final answer.  This is the per-position version of the bank's
  cluster-0 rule: only boundaries where the model would already conclude
  with the trace's answer are rewarded by the hazard objectives;
- ``gaps``: tokens from boundary b to the next boundary (horizon-capped
  for the last), used by the expected-remaining-length objective.

Output JSON is consumed by the ``nodewise_subnetwork_probing_boundary_hazard``
algorithm via ``--boundary_data_path``.

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.build_boundary_data \
        --bank_path results/cot_termination/banks/aqua_p080_s57.json \
        --output results/cot_termination/boundary_data/aqua_p080_s57.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from utils.utils import set_seed, clear_cuda
from utils.circuit_eval import install_clean_sdpa_forward

from expts.cot_termination_circuit_discovery.learn import (
    _build_prefix, load_model_eager,
)
from expts.cot_termination_circuit_discovery.answer_bank_utils import (
    flatten_bank_candidates,
)

THINK_END_ID = 151668  # dedicated "</think>" token in the Qwen3 vocab
PROBE_SUFFIX_TEXT = " I think the answer is"


def _paragraph_boundaries(content_ids, tokenizer):
    """Indices of paragraph-break tokens (decoded token contains '\\n\\n').

    ``content_ids`` excludes the ``</think>`` token.  For terminated
    candidates the last such position is the paragraph break at which the
    model actually started its wrap-up; positions after it are inside the
    final-answer text (single newlines, so they do not match).
    """
    return [
        j for j, t in enumerate(content_ids)
        if "\n\n" in tokenizer.decode([t])
    ]


def _wrap_up_head_tokens(token_lists, tokenizer,
                         max_nonfinal_frequency=0.02):
    """Per-prompt wrap-up head token set.

    Candidate heads: for each terminated candidate, the token immediately
    after its *last* paragraph break (the first token of the final-answer
    text).  A head is kept only if it is discriminative — it must start at
    most ``max_nonfinal_frequency`` of all *non-final* paragraphs across
    the bank (otherwise it is an ordinary sentence starter like ``But`` /
    ``So`` and the hazard would measure "next paragraph begins with a
    common word", not wrap-up).  The ``</think>`` token is always included
    for direct termination.

    Raises if no discriminative head survives and no terminated candidate
    terminates directly — on such a prompt the wrap-up start is not
    marked by any single token and the boundary-hazard objectives have no
    valid event to read.
    """
    raw_heads = {}
    nonfinal_counts = {}
    n_nonfinal = 0
    for ids in token_lists:
        terminated = THINK_END_ID in ids
        content = ids[: ids.index(THINK_END_ID)] if terminated else list(ids)
        pb = _paragraph_boundaries(content, tokenizer)
        final_pb = pb[-1] if (terminated and pb) else None
        for j in pb:
            if j + 1 >= len(content):
                continue
            t = content[j + 1]
            if j == final_pb:
                raw_heads[t] = tokenizer.decode([t])
            else:
                nonfinal_counts[t] = nonfinal_counts.get(t, 0) + 1
                n_nonfinal += 1
    heads, head_texts, dropped = {THINK_END_ID}, {}, {}
    for t, txt in raw_heads.items():
        freq = nonfinal_counts.get(t, 0) / max(1, n_nonfinal)
        if freq <= max_nonfinal_frequency:
            heads.add(t)
            head_texts[t] = txt
        else:
            dropped[txt] = round(freq, 3)
    if dropped:
        print(f"  dropped non-discriminative head tokens "
              f"(fraction of non-final paragraphs they start): {dropped}")
    if not head_texts:
        raise RuntimeError(
            "No discriminative wrap-up head token on this prompt (all "
            f"observed heads are common paragraph starters: {dropped}). "
            "The boundary-hazard objectives cannot be trained here."
        )
    return sorted(heads), head_texts


@torch.no_grad()
def _clean_hazards(model, prefix_ids, content_ids, boundaries,
                   event_token_ids, device):
    """log sum_{t in event set} p(t) at each boundary under the clean
    model, fp32."""
    cont = torch.tensor([content_ids], device=device)
    full = torch.cat([prefix_ids.to(device), cont], dim=-1)
    logits = model(full, logits_to_keep=len(content_ids) + 1).logits
    ev = torch.tensor(event_token_ids, dtype=torch.long,
                      device=logits.device)
    # Kept rows start at absolute position prefix_len - 1, so the row that
    # predicts the token after content j is row j + 1.
    out = []
    for j in boundaries:
        lsm = torch.log_softmax(logits[0, j + 1].float(), dim=-1)
        out.append(float(torch.logsumexp(lsm[ev], dim=0).item()))
    del logits, full
    return out


@torch.no_grad()
def _probe_at_boundary(model, prefix_ids, content_ids, j, suffix_ids,
                       letter_ids, device):
    """Forced-probe answer letter distribution at boundary j."""
    full = torch.cat([
        prefix_ids.to(device),
        torch.tensor([content_ids[: j + 1] + [THINK_END_ID] + suffix_ids],
                     device=device),
    ], dim=-1)
    logits = model(full, logits_to_keep=1).logits
    row = logits[0, -1].float()
    lvals = {L: float(row[tid]) for L, tid in letter_ids.items()}
    probs = torch.softmax(torch.tensor(list(lvals.values())), dim=0)
    label = max(lvals, key=lvals.get)
    del logits, full
    return label, {L: float(p) for L, p in zip(lvals, probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_path", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda",
                    help='"auto" spreads the model over all visible GPUs (Qwen3-32B)')
    ap.add_argument("--min_sentence_length", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(args.bank_path) as f:
        bank = json.load(f)
    data_path = bank["data_path"]
    prompt_index = int(bank["prompt_index"])
    step = int(bank["analysis_sentence_step"])
    horizon = int(bank["horizon"])
    trace_answer = (bank["trace_answer"] or "").strip()
    all_letters = bank.get("all_letters") or ["A", "B", "C", "D"]

    model, tokenizer = load_model_eager(args.model_name, device=args.device)
    # Route forwards through the SDPA attention path (no mask installed, so
    # numerically a no-op) — eager attention materialises the full
    # attention matrix and runs out of memory on the long early-analysis-
    # point prefixes (~15k tokens).
    install_clean_sdpa_forward(model)
    device = next(model.parameters()).device

    enc = tokenizer.encode("</think>", add_special_tokens=False)
    assert enc == [THINK_END_ID], f"</think> tokenizes to {enc}"
    suffix_ids = tokenizer.encode(PROBE_SUFFIX_TEXT, add_special_tokens=False)
    letter_ids = {}
    for L in all_letters:
        ids = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids[L] = ids[0]

    prefix_ids, _, _, _, _, _ = _build_prefix(
        tokenizer=tokenizer, prompt=None, data_path=data_path,
        prompt_index=prompt_index, base_answer_type="stored",
        analysis_timestep=None, analysis_sentence_step=step,
        sentences_after_prefix=int(bank.get("sentences_after_prefix", 0)),
        min_sentence_length=args.min_sentence_length, sentence_chunk=1,
    )

    token_lists, cluster_ids, counts = flatten_bank_candidates(bank)
    event_token_ids, head_texts = _wrap_up_head_tokens(token_lists, tokenizer)
    print(f"wrap-up event token set: {event_token_ids} "
          f"(heads: {head_texts}, plus </think>={THINK_END_ID})")
    out_candidates = []
    for flat_idx, (ids, cl) in enumerate(zip(token_lists, cluster_ids)):
        terminated = THINK_END_ID in ids
        content = ids[: ids.index(THINK_END_ID)] if terminated else list(ids)
        boundaries = _paragraph_boundaries(content, tokenizer)
        if not boundaries:
            print(f"  WARNING: candidate {flat_idx} has no boundaries "
                  f"({len(content)} content tokens)")
        clean_log_h = _clean_hazards(
            model, prefix_ids, content, boundaries, event_token_ids, device,
        )
        eligible, probe_labels, probe_p_trace = [], [], []
        for j in boundaries:
            label, probs = _probe_at_boundary(
                model, prefix_ids, content, j, suffix_ids, letter_ids,
                device,
            )
            eligible.append(label == trace_answer)
            probe_labels.append(label)
            probe_p_trace.append(probs.get(trace_answer))
        gaps = [
            float(boundaries[b + 1] - boundaries[b])
            for b in range(len(boundaries) - 1)
        ] + ([float(max(1, horizon - boundaries[-1]))] if boundaries else [])
        out_candidates.append({
            "flat_index": flat_idx,
            "cluster_id": cl,
            "terminated": terminated,
            "n_content_tokens": len(content),
            "boundaries": boundaries,
            "eligible": eligible,
            "probe_labels": probe_labels,
            "probe_p_trace": probe_p_trace,
            "clean_log_h": clean_log_h,
            "gaps": gaps,
        })
        n_el = sum(eligible)
        print(f"  candidate {flat_idx} (cluster {cl}, "
              f"{'terminated' if terminated else 'not terminated'}): "
              f"{len(boundaries)} boundaries, {n_el} eligible, "
              f"clean log h range "
              f"[{min(clean_log_h):.1f}, {max(clean_log_h):.1f}]"
              if boundaries else
              f"  candidate {flat_idx}: no boundaries")
        clear_cuda()

    n_any = sum(1 for c in out_candidates if any(c["eligible"]))
    if n_any == 0:
        raise RuntimeError(
            "No candidate has any probe-correct boundary — the hazard "
            "objectives would have no signal on this prompt."
        )
    print(f"{n_any}/{len(out_candidates)} candidates have >=1 eligible boundary")

    out = {
        "bank_path": args.bank_path,
        "data_path": data_path,
        "prompt_index": prompt_index,
        "analysis_sentence_step": step,
        "model_name": args.model_name,
        "think_end_id": THINK_END_ID,
        "event_token_ids": event_token_ids,
        "event_head_texts": {str(k): v for k, v in head_texts.items()},
        "horizon": horizon,
        "trace_answer": trace_answer,
        "probe_suffix_text": PROBE_SUFFIX_TEXT,
        "min_sentence_length": args.min_sentence_length,
        "candidates": out_candidates,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
