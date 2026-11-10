"""Direct-answer circuit discovery using existing algorithms.

Skips vLLM branch generation entirely.  At each analysis boundary we
construct a single deterministic continuation
(``" </think> I think the answer is "`` + placeholder letter) and run
the model's existing circuit-discovery algorithms (IG, activation
patching, ...) with that continuation plus a one-position position mask.

The objective is one of the new local probe objectives
(:func:`utils.objectives.answer_probe_kl_loss`,
:func:`utils.objectives.answer_probe_reward_gap_loss`).  Because they
are local objectives, ``utils.objectives.is_global_objective`` returns
False and the algorithms' importance-sampling code paths are entirely
bypassed.
"""

from __future__ import annotations

import os
import json
from functools import partial
from typing import List, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda
from utils.cot_analysis import (
    Sentence,
    split_tokens_into_sentences,
    remove_bos_from_sentences,
    chunk_sentences,
)
# The only trainer that restricts the learnable pool to a named region
# (utils/circuit_discovery/edits/nodewise_subnetwork_probing_region.py).
REGION_ALGORITHM = "nodewise_subnetwork_probing_region"

from utils.objectives import (
    answer_probe_kl_loss,
    answer_probe_logit_margin_loss,
    answer_probe_prefix_kl_loss,
    answer_probe_reward_gap_loss,
    answer_probe_target_kl_loss,
    candidate_logprob_margin_loss,
    candidate_reward_gap_loss,
    candidate_snis_reward_gap_loss,
)
from utils.masks import NodeMask
from utils.circuit_discovery.factory import create_circuit_discovery
from utils.base_path_selection import select_base_from_record
from utils.rewards import extract_boxed

from expts.direct_answer_circuit_discovery.probe import (
    AnswerProbe,
    DEFAULT_ANSWER_LETTERS,
    DEFAULT_SUFFIX,
    answer_logprobs_from_logits,
    answer_probs_from_logits,
    build_answer_probe,
)


PROBE_OBJECTIVES = {
    "answer_probe_kl": answer_probe_kl_loss,
    "answer_probe_reward_gap": answer_probe_reward_gap_loss,
    "answer_probe_logit_margin": answer_probe_logit_margin_loss,
    "answer_probe_prefix_kl": answer_probe_prefix_kl_loss,
    "answer_probe_target_kl": answer_probe_target_kl_loss,
}

# Candidate-set objectives for open-ended answers.  These are global
# (chain-logprob) objectives: continuations are (probe suffix + one
# candidate answer string) from a pre-built answer bank, not the synthetic
# single-token probe continuation.  Requires --answer_bank_path.
CANDIDATE_OBJECTIVES = {
    "candidate_reward_gap": candidate_reward_gap_loss,
    "candidate_logprob_margin": candidate_logprob_margin_loss,
    "candidate_snis_reward_gap": candidate_snis_reward_gap_loss,
}

# Answer-distribution KL over a candidate bank on a rollout bank of sampled
# clean continuations (open-ended answers). The loss is computed inside the
# trainer, which reads the candidate paths and the clean cluster
# distributions from the candidate rollout-bank file
# (build_candidate_rollout_bank.py) passed as rollout_bank_path.
CANDIDATE_ROLLOUT_OBJECTIVES = {"candidate_rollout_kl"}
CANDIDATE_ROLLOUT_ALGORITHM = "nodewise_subnetwork_probing_candidate_rollouts"


def load_model_eager(
    model_name: str,
    device: str = "cuda",
    gradient_checkpointing: bool = False,
):
    """Eager-attention model loader (matches learn_circuit.py)."""
    print(f"Loading {model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print("  Gradient checkpointing: ENABLED")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _resolve_layers(layers_to_analyse, model) -> List[int]:
    if layers_to_analyse is None:
        return [8, 12, 16, 20, 24]
    if isinstance(layers_to_analyse, str):
        if layers_to_analyse.lower() == "all":
            return list(range(model.config.num_hidden_layers))
        return [int(layers_to_analyse)]
    if any(isinstance(l, str) and l.lower() == "all" for l in layers_to_analyse):
        if len(layers_to_analyse) > 1:
            raise ValueError("Use 'all' by itself.")
        return list(range(model.config.num_hidden_layers))
    return [int(l) for l in layers_to_analyse]


def split_with_prompt_chunks(full_tensor, tokenizer, prompt_len, prompt_chunk_spans, min_sentence_length, skip_reasoning_tokens=0):
    """Sentence list for a sequence whose prompt part is pre-chunked.

    ``prompt_chunk_spans`` are inclusive token spans over the prompt (from
    ``prompt_chunking.chunk_prompt``); they are used verbatim. The
    reasoning part (tokens from ``prompt_len`` on) is split with the usual
    delimiter splitter, so no sentence straddles the prompt/reasoning
    boundary (the legacy splitter merges the first reasoning words into
    the last prompt sentence when the chat-template tail is short).

    Tokens outside every span (a leading chat-template chunk left out of
    ``prompt_chunk_spans``, or the first ``skip_reasoning_tokens`` reasoning
    tokens such as ``<think>``) belong to no sentence: the mask machinery
    treats them as always attended, i.e. they are never ablated."""
    sents = [Sentence(start=int(s), end=int(e)) for s, e in prompt_chunk_spans if e < full_tensor.shape[-1]]
    start = prompt_len + int(skip_reasoning_tokens or 0)
    if full_tensor.shape[-1] > start:
        rest = split_tokens_into_sentences(full_tensor[start:], tokenizer, min_sentence_length=min_sentence_length)
        sents += [Sentence(start=s.start + start, end=s.end + start) for s in rest]
    return sents


def _build_prefix(
    *,
    tokenizer,
    prompt: str,
    data_path: Optional[str],
    prompt_index: Optional[int],
    base_answer_type: str,
    analysis_timestep: Optional[int],
    analysis_sentence_step: Optional[int],
    min_sentence_length: int,
    sentence_chunk: int,
    sentences_after_prefix: int = 0,
):
    """Build the prefix token sequence and split into sentences.

    Two modes:

    - ``data_path + prompt_index``: load record, take its stored base
      path (or alternate via ``base_answer_type``), and cut at either an
      explicit ``analysis_timestep`` or sentence index
      ``analysis_sentence_step``.
    - ``prompt`` alone with explicit ``analysis_timestep``: just
      tokenize the prompt and cut.  Useful for a sanity-check run with
      no base completion.

    If ``sentences_after_prefix > 0``, k extra reasoning sentences
    (drawn from the model's stored base path) are appended after the
    analysis-sentence prefix but *before* the forced-answer suffix.
    These k sentences are included in the returned token sequence
    (``prefix_ids``) so the model sees them, but they are **not**
    included in the returned ``sentences_to_be_masked`` list — their
    attention is never masked.  This is a "middle ground" between the
    fully-local probe (k = 0, suffix directly after the masked prefix)
    and a fully-resampled global outcome: the model gets k more
    unmasked reasoning sentences before being forced to commit.

    **Why ``prefix_ids`` includes the k extra tokens even though they
    are not masked:** ``prefix_len = prefix_ids.shape[-1]`` is used
    downstream to compute the answer-logit position
    (``prefix_len + suffix_len - 1``).  If ``prefix_ids`` excluded the
    k extra tokens, the answer position would point into the middle of
    the unmasked context instead of at the probe suffix's last token.
    """
    record = None
    correct_answer = None
    if data_path is not None and prompt_index is not None:
        with open(data_path) as f:
            records = json.load(f)
        record = records[prompt_index]
        if prompt is None:
            prompt = record.get("question_with_choices") or record["question"]
        # Prefer the explicit correct_letter for multiple-choice datasets;
        # fall back to extract_boxed(correct_answer), then the raw text.
        cl = record.get("correct_letter")
        if cl:
            correct_answer = cl
        else:
            ca_raw = record.get("correct_answer")
            correct_answer = extract_boxed(ca_raw) if ca_raw else None
            if correct_answer is None and ca_raw:
                correct_answer = ca_raw
    if prompt is None:
        raise ValueError("Either prompt or (data_path + prompt_index) is required.")

    if record is not None and "prompt_token_ids" in record:
        prompt_input_ids = torch.tensor(
            record["prompt_token_ids"]
        ).unsqueeze(0)
        formatted = record.get("prompt", "")
    else:
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        prompt_input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"]
    prompt_len = prompt_input_ids.shape[-1]

    if record is not None:
        base_token_ids = select_base_from_record(record, base_answer_type, tokenizer)
        full_token_ids = prompt_input_ids[0].tolist() + list(base_token_ids)
    else:
        full_token_ids = prompt_input_ids[0].tolist()

    full_tensor = torch.tensor(full_token_ids)

    if analysis_sentence_step is not None:
        if record is None:
            raise ValueError(
                "analysis_sentence_step requires data_path + prompt_index."
            )
        if record is not None and record.get("prompt_chunk_spans"):
            raw_sentences = split_with_prompt_chunks(
                full_tensor, tokenizer, prompt_len, record["prompt_chunk_spans"], min_sentence_length,
                skip_reasoning_tokens=record.get("reasoning_skip_tokens", 0))
        else:
            raw_sentences = split_tokens_into_sentences(
                full_tensor, tokenizer, min_sentence_length=min_sentence_length,
            )
        raw_sentences = remove_bos_from_sentences(raw_sentences)
        raw_sentences = chunk_sentences(raw_sentences, sentence_chunk)
        if analysis_sentence_step >= len(raw_sentences):
            raise ValueError(
                f"analysis_sentence_step={analysis_sentence_step} past last "
                f"sentence index ({len(raw_sentences) - 1})."
            )
        # analysis_sentence_step=50 means sentences 0-49 are the masked
        # prefix.  The k extra context sentences occupy indices 50
        # through 49+k.  The token cut includes all of them so the
        # model sees the unmasked context before the probe suffix.
        last_included_sentence = analysis_sentence_step - 1 + sentences_after_prefix
        if last_included_sentence >= len(raw_sentences):
            raise ValueError(
                f"analysis_sentence_step={analysis_sentence_step} + "
                f"sentences_after_prefix={sentences_after_prefix} - 1 = "
                f"{last_included_sentence} past last sentence index "
                f"({len(raw_sentences) - 1})."
            )
        cut = raw_sentences[last_included_sentence].end + 1
    elif analysis_timestep is not None:
        if sentences_after_prefix:
            raise ValueError(
                "sentences_after_prefix requires analysis_sentence_step "
                "(token-step mode does not know sentence boundaries beyond "
                "the cut)."
            )
        cut = prompt_len + analysis_timestep
        cut = min(cut, full_tensor.shape[-1])
    else:
        if sentences_after_prefix:
            raise ValueError(
                "sentences_after_prefix requires analysis_sentence_step."
            )
        cut = full_tensor.shape[-1]

    prefix_ids = full_tensor[:cut].unsqueeze(0)
    if record is not None and record.get("prompt_chunk_spans"):
        all_sentences = split_with_prompt_chunks(
            prefix_ids[0], tokenizer, prompt_len, record["prompt_chunk_spans"], min_sentence_length,
            skip_reasoning_tokens=record.get("reasoning_skip_tokens", 0))
    else:
        all_sentences = split_tokens_into_sentences(
            prefix_ids[0], tokenizer, min_sentence_length=min_sentence_length,
        )
    all_sentences = remove_bos_from_sentences(all_sentences)
    all_sentences = chunk_sentences(all_sentences, sentence_chunk)

    # Only the first analysis_sentence_step sentences form the masked
    # prefix.  Tokens from the remaining k context sentences are in
    # prefix_ids (the model sees them) but excluded from the sentence
    # list so the masking hooks leave them unmasked (token_to_sent = -1
    # → sentinel padding → mask = 1.0 in expand_sentence_mask_to_tokens).
    if analysis_sentence_step is not None and sentences_after_prefix > 0:
        sentences_to_be_masked = all_sentences[:analysis_sentence_step]
        n_context = len(all_sentences) - len(sentences_to_be_masked)
        assert n_context == sentences_after_prefix, (
            f"Expected exactly {sentences_after_prefix} unmasked context "
            f"sentences after the masked prefix, but the re-split produced "
            f"{n_context} (total {len(all_sentences)}, masked "
            f"{len(sentences_to_be_masked)}). This can happen if sentence "
            f"splitting is inconsistent between the full sequence and the "
            f"truncated prefix."
        )
    else:
        sentences_to_be_masked = all_sentences

    # Number of sentences that contain prompt tokens (question, choices,
    # chat-template tokens). A sentence straddling the prompt/reasoning
    # boundary counts as a prompt sentence (conservative). Used by
    # freeze_prompt_sentences to keep all attention to/from the prompt
    # unmasked.
    num_prompt_sentences = sum(
        1 for s in sentences_to_be_masked if s.start < prompt_len
    )

    return (
        prefix_ids,
        sentences_to_be_masked,
        prompt,
        correct_answer,
        formatted,
        num_prompt_sentences,
    )


def main(
    *,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    model_to_analyse: Optional[str] = None,
    prompt: Optional[str] = None,
    data_path: Optional[str] = None,
    prompt_index: Optional[int] = None,
    correct_answer: Optional[str] = None,
    base_answer_type: str = "stored",
    analysis_timestep: Optional[int] = None,
    analysis_sentence_step: Optional[int] = None,
    sentences_after_prefix: int = 0,
    masking_algorithm: str = "nodewise_attribution",
    objective: str = "answer_probe_kl",
    answer_bank_path: Optional[str] = None,
    # Train the probe objective on a bank of sampled clean rollouts instead of
    # the stored suffix: a clean_rollouts_k32 file of eval_onpolicy_kl.py
    rollout_bank_path: Optional[str] = None,
    rollout_bank_set: str = "B",
    continuations_per_step: Optional[int] = None,
    clean_logits_dtype: Optional[str] = None,
    target_letter: Optional[str] = None,
    target_probs: Optional[List[float]] = None,
    logit_margin_reduce: str = "mean",
    probe_suffix: str = DEFAULT_SUFFIX,
    answer_letters: Optional[List[str]] = None,
    layers_to_analyse=None,
    sentence_gap: int = 1,
    sentence_chunk: int = 1,
    mask_mode: str = "prefix",
    freeze_prompt_sentences: bool = False,
    freeze_sentences_before: Optional[int] = None,
    # Named learnable region: None (default pool) or "prompt_to_trace"
    # (only cells where a reasoning sentence reads a prompt sentence are
    # learnable; every reasoning-to-reasoning cell is frozen at 1.0).
    learnable_region: Optional[str] = None,
    frozen_key_sentences: Optional[list] = None,
    mask_granularity: str = "head",
    pair_aggregation: str = "sum",  # attribution only; see run.py
    ablate_non_target_layers: bool = False,
    renormalize_masked_attention: bool = True,
    num_ig_steps: int = 10,
    no_negate_scores: bool = False,
    include_zero_ablation: bool = True,
    zero_ablation_epsilon: float = 1e-10,
    min_sentence_length: int = 10,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/direct_answer_circuit_discovery",
    file_name: Optional[str] = None,
    batch_chunk_size: Optional[int] = None,
    torch_compile: bool = False,
    gradient_checkpointing: bool = False,
    l0_lambda: Optional[float] = None,
    num_training_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    log_alpha_init: Optional[float] = None,
    log_every: Optional[int] = None,
    plot_every: Optional[int] = None,
    sparsity_loss_mode: Optional[str] = None,
    l0_normalize_hinge: Optional[bool] = None,
    l0_warmup_frac: Optional[float] = None,
    l0_ramp_frac: Optional[float] = None,
    target_sparsity: Optional[float] = None,
    optimizer: Optional[str] = None,
    momentum: Optional[float] = None,
    l0_lr_multiplier: Optional[float] = None,
    dropout_p: Optional[float] = None,
    save_log_alpha: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    resume_from_checkpoint: Optional[bool] = None,
    # D2 — HC variance reduction
    num_hc_samples_per_step: Optional[int] = None,
    log_alpha_init_mask_path: Optional[str] = None,
    log_alpha_init_mask_alpha: Optional[float] = None,
    polyak_ema_log_alpha: Optional[float] = None,
    hc_beta_anneal: Optional[bool] = None,
    hc_beta_start: Optional[float] = None,
    hc_beta_end: Optional[float] = None,
    hc_beta_anneal_end_frac: Optional[float] = None,
    # D3 — LR scheduler
    lr_schedule: Optional[str] = None,
    lr_min_ratio: Optional[float] = None,
    lr_plateau_patience: Optional[int] = None,
    lr_plateau_factor: Optional[float] = None,
    # Column subnetwork probing
    training_gap_mode: Optional[str] = None,
    # DCM + PID (nodewise_dcm_pid_sdpa)
    pid_kp: Optional[float] = None,
    pid_ki: Optional[float] = None,
    pid_kd: Optional[float] = None,
    pid_mult_init: Optional[float] = None,
    pid_ramp_end_frac: Optional[float] = None,
    pid_max_target_sparsity: Optional[float] = None,
    snapshot_sparsities: Optional[List[float]] = None,
    dcm_mask_init: Optional[float] = None,
    dcm_l0_optimizer: Optional[str] = None,
    dcm_polarization: Optional[float] = None,
    pid_snapshot_hold_steps: Optional[int] = None,
    dcm_lr_init: Optional[float] = None,
    dcm_lr_warmup_frac: Optional[float] = None,
    algorithm_kwargs: Optional[dict] = None,
):
    if model_to_analyse is None:
        model_to_analyse = model_name
    if answer_letters is None:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)
    if (objective not in PROBE_OBJECTIVES and objective not in CANDIDATE_OBJECTIVES
            and objective not in CANDIDATE_ROLLOUT_OBJECTIVES):
        raise ValueError(
            f"objective must be one of "
            f"{sorted(PROBE_OBJECTIVES) + sorted(CANDIDATE_OBJECTIVES) + sorted(CANDIDATE_ROLLOUT_OBJECTIVES)}, "
            f"got {objective!r}"
        )
    use_candidate_objective = objective in CANDIDATE_OBJECTIVES
    use_candidate_rollouts = objective in CANDIDATE_ROLLOUT_OBJECTIVES
    candidate_rollout_bank = None
    if use_candidate_rollouts:
        if rollout_bank_path is None or masking_algorithm != CANDIDATE_ROLLOUT_ALGORITHM:
            raise ValueError(f"{objective} needs rollout_bank_path (a candidate rollout bank) and "
                             f"masking_algorithm={CANDIDATE_ROLLOUT_ALGORITHM!r}")
        with open(rollout_bank_path) as f:
            candidate_rollout_bank = json.load(f)
        if candidate_rollout_bank.get("kind") != "candidate_rollout_bank":
            raise ValueError(f"{rollout_bank_path} is not a candidate rollout bank")
        for key, val in [("data_path", data_path), ("prompt_index", prompt_index),
                         ("analysis_sentence_step", analysis_sentence_step), ("probe_suffix", probe_suffix)]:
            if candidate_rollout_bank.get(key) != val:
                raise ValueError(f"rollout bank {rollout_bank_path}: {key}={candidate_rollout_bank.get(key)!r} "
                                 f"does not match this run ({val!r})")
    answer_bank = None
    if use_candidate_objective:
        if answer_bank_path is None:
            raise ValueError(f"{objective} requires answer_bank_path")
        with open(answer_bank_path) as f:
            answer_bank = json.load(f)
        if (
            answer_bank.get("data_path") != data_path
            or answer_bank.get("prompt_index") != prompt_index
            or answer_bank.get("analysis_sentence_step") != analysis_sentence_step
            or answer_bank.get("sentences_after_prefix") != sentences_after_prefix
        ):
            raise ValueError(
                f"Answer bank {answer_bank_path} was built for "
                f"(data_path={answer_bank.get('data_path')}, "
                f"prompt_index={answer_bank.get('prompt_index')}, "
                f"analysis_sentence_step={answer_bank.get('analysis_sentence_step')}, "
                f"sentences_after_prefix={answer_bank.get('sentences_after_prefix')}) "
                f"but this run uses (data_path={data_path}, "
                f"prompt_index={prompt_index}, "
                f"analysis_sentence_step={analysis_sentence_step}, "
                f"sentences_after_prefix={sentences_after_prefix})."
            )
        if answer_bank.get("probe_suffix") != probe_suffix:
            raise ValueError(
                f"Answer bank probe_suffix {answer_bank.get('probe_suffix')!r} "
                f"!= run probe_suffix {probe_suffix!r}"
            )

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Step 1: Build prefix and split into sentences")
    print("=" * 80)
    tokenizer_for_split = AutoTokenizer.from_pretrained(model_name)
    (
        prefix_ids,
        sentences_to_be_masked,
        prompt,
        correct_answer_from_record,
        formatted,
        num_prompt_sentences,
    ) = _build_prefix(
        tokenizer=tokenizer_for_split,
        prompt=prompt,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type=base_answer_type,
        analysis_timestep=analysis_timestep,
        analysis_sentence_step=analysis_sentence_step,
        # with a rollout bank the sampled rollouts replace the stored suffix
        sentences_after_prefix=0 if rollout_bank_path else sentences_after_prefix,
        min_sentence_length=min_sentence_length,
        sentence_chunk=sentence_chunk,
    )
    if correct_answer is None:
        correct_answer = correct_answer_from_record
    prefix_len = prefix_ids.shape[-1]
    print(f"  Prefix length: {prefix_len} tokens")
    sentences = sentences_to_be_masked
    print(f"  Sentences (masked): {len(sentences)}")
    _frozen_prompt_algorithms = {
        "nodewise_subnetwork_probing_sdpa",
        "nodewise_subnetwork_probing_hc_batched",
        # subclasses of the batched trainer; the prompt filter is inherited
        "nodewise_straight_through_topk",
        "nodewise_deterministic_continuous",
        "nodewise_subnetwork_probing_candidate_rollouts",
        "nodewise_matryoshka_attribution",
        "column_subnetwork_probing",
        "nodewise_activation_patching_flash",
        "nodewise_attribution_sdpa",
    }
    if freeze_prompt_sentences:
        if masking_algorithm not in _frozen_prompt_algorithms:
            raise ValueError(
                "freeze_prompt_sentences is only implemented for "
                f"{_frozen_prompt_algorithms}, got {masking_algorithm!r}."
            )
        print(
            f"  Prompt sentences frozen (not learnable): "
            f"{num_prompt_sentences} (sentences 0-{num_prompt_sentences - 1})"
        )
    # Optional window restriction: freeze all sentences before index N
    # (superset of the prompt sentences), so only the last
    # (num_sentences - N) prefix sentences are learnable. Used for
    # late-anchor analysis points where the masked window must stay
    # comparable to the standard 50-sentence pool.
    if freeze_sentences_before is not None:
        if not freeze_prompt_sentences:
            raise ValueError(
                "freeze_sentences_before requires freeze_prompt_sentences"
            )
        if freeze_sentences_before < num_prompt_sentences:
            raise ValueError(
                f"freeze_sentences_before={freeze_sentences_before} < "
                f"num_prompt_sentences={num_prompt_sentences}"
            )
        num_prompt_sentences = freeze_sentences_before
        print(f"  Window restriction: sentences 0-{num_prompt_sentences - 1} "
              f"frozen; learnable window = last "
              f"{len(sentences) - num_prompt_sentences} sentences")
    if correct_answer is not None:
        print(f"  Correct answer: {correct_answer!r}")

    print("=" * 80)
    print("Step 2: Build answer probe")
    print("=" * 80)
    probe = build_answer_probe(
        tokenizer_for_split,
        suffix=probe_suffix,
        answer_letters=answer_letters,
    )
    print(f"  Suffix: {probe_suffix!r} → {probe.suffix_len} tokens")
    print(f"  Answer letters: {probe.answer_letters}")
    print(f"  Answer token ids: {probe.answer_token_ids.tolist()}")
    target_answer_id: Optional[int] = None
    if target_letter is not None:
        # Strip-match so a plain "C" matches the " C" answer letter, same
        # as the correct_answer fallback below.
        stripped_letters = [l.strip() for l in probe.answer_letters]
        if target_letter.strip() not in stripped_letters:
            raise ValueError(
                f"target_letter={target_letter!r} not in answer_letters="
                f"{probe.answer_letters}"
            )
        target_answer_id = stripped_letters.index(target_letter.strip())
    elif correct_answer is not None:
        # Match against answer_letters, stripping leading whitespace so a
        # plain ``"C"`` from a dataset matches a ``" C"`` answer letter.
        stripped = [l.strip() for l in probe.answer_letters]
        if correct_answer.strip() in stripped:
            target_answer_id = stripped.index(correct_answer.strip())
    if objective in ("answer_probe_reward_gap", "answer_probe_logit_margin") and target_answer_id is None:
        raise ValueError(
            f"{objective} requires --target_letter or a correct_answer "
            "that matches one of the answer_letters."
        )

    print("=" * 80)
    print(f"Step 3: Loading model with eager attention ({model_to_analyse})")
    print("=" * 80)
    model, tokenizer = load_model_eager(
        model_to_analyse,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)
    layers = _resolve_layers(layers_to_analyse, model)
    print(f"  Layers to analyse: {layers}")

    print("=" * 80)
    print("Step 4: Build synthetic continuation + position mask")
    print("=" * 80)
    clean_p = None
    clean_lp = None
    rollout_bank_info = None
    if use_candidate_objective:
        # Continuations are (probe suffix + candidate answer tokens) from
        # the pre-built answer bank; the global objective sums log-probs
        # over each whole continuation, so no position mask is needed.
        from expts.direct_answer_circuit_discovery.answer_bank_utils import (
            flatten_bank_candidates,
        )
        bank_candidates = answer_bank["candidates"]
        _token_lists, _cluster_ids, _counts = flatten_bank_candidates(answer_bank)
        continuations = [
            torch.tensor([ids], device=target_device) for ids in _token_lists
        ]
        candidate_answer_ids = torch.tensor(_cluster_ids, dtype=torch.long)
        candidate_counts = torch.tensor(_counts, dtype=torch.float)
        num_answer_clusters = int(answer_bank["num_clusters"])
        target_cluster = int(answer_bank["target_cluster"])
        position_mask = None
        print(f"  Candidate bank: {answer_bank_path}")
        for c in bank_candidates:
            print(
                f"    cluster {c['cluster_id']} "
                f"({'correct' if c['is_correct'] else 'wrong'}, "
                f"count {c['count']}, {c['grade_method']}): "
                f"{c['answer_text']!r}"
            )
        print(f"  Clean fraction correct at probe context: "
              f"{answer_bank.get('clean_fraction_correct')}")
    else:
        continuation = probe.make_continuation(target_device)        # (1, L)
        continuations = [continuation]
        position_mask = probe.make_position_mask(prefix_len, target_device)  # (1, P+L)
        position_masks = [position_mask]
        rollout_bank_info = None
        if use_candidate_rollouts:
            crb = candidate_rollout_bank
            if crb["prefix_len"] != prefix_len:
                raise ValueError(f"rollout bank prefix_len {crb['prefix_len']} != {prefix_len}")
            rolls = crb["sets"][rollout_bank_set]["rollouts"]
            # The trainer rebuilds every (rollout, candidate) row from the bank;
            # these tensors only fix the number of continuations and the length
            # of the token-to-sentence map (rollout + suffix + longest path).
            pad = len(crb["probe_suffix_token_ids"]) + max(len(p) for p in crb["candidate_paths"])
            continuations = [torch.zeros((1, len(r["tokens"]) + pad), dtype=torch.long, device=target_device)
                             for r in rolls]
            position_masks = [None] * len(rolls)
            rollout_bank_info = {
                "path": rollout_bank_path, "set": rollout_bank_set, "num_rollouts": len(rolls),
                "continuations_per_step": continuations_per_step,
                "mean_clean_cluster_probs": torch.tensor([r["clean_cluster_probs"] for r in rolls]).mean(0).tolist(),
                "rollout_n_tokens": [len(r["tokens"]) for r in rolls],
                "num_candidate_paths": len(crb["candidate_paths"]),
            }
            print(f"  Candidate rollout bank: {rollout_bank_path} set {rollout_bank_set}, {len(rolls)} "
                  f"continuations x {len(crb['candidate_paths'])} candidate paths "
                  f"({continuations_per_step or 'all'} continuations per step)")
        elif rollout_bank_path is not None:
            if use_candidate_objective:
                raise ValueError("rollout_bank_path is for the probe objectives only")
            with open(rollout_bank_path) as f:
                bank = json.load(f)
            for key, val in [("data_path", data_path), ("prompt_index", prompt_index),
                             ("analysis_sentence_step", analysis_sentence_step),
                             ("prefix_len", prefix_len), ("probe_suffix", probe_suffix),
                             ("answer_letters", list(probe.answer_letters))]:
                if bank.get(key) != val:
                    raise ValueError(f"rollout bank {rollout_bank_path}: {key}={bank.get(key)!r} "
                                     f"does not match this run ({val!r})")
            rolls = bank["sets"][rollout_bank_set]["rollouts"]
            continuations = [
                torch.cat([torch.tensor([r["tokens"]], dtype=torch.long, device=target_device), continuation], dim=-1)
                for r in rolls
            ]
            position_masks = [probe.make_position_mask(prefix_len + len(r["tokens"]), target_device) for r in rolls]
            rollout_bank_info = {
                "path": rollout_bank_path, "set": rollout_bank_set, "num_rollouts": len(rolls),
                "continuations_per_step": continuations_per_step,
                "mean_clean_answer_probs": torch.tensor([r["clean_answer_probs"] for r in rolls]).mean(0).tolist(),
                "rollout_n_tokens": [len(r["tokens"]) for r in rolls],
            }
            print(f"  Rollout bank: {rollout_bank_path} set {rollout_bank_set}, {len(rolls)} continuations "
                  f"({continuations_per_step or 'all'} per step)")
        print(f"  Continuation length: {continuation.shape[-1]} tokens")
        print(f"  Answer logit position: {probe.answer_logit_position(prefix_len)}")

        print("=" * 80)
        print("Step 5: Compute clean answer distribution (sanity)")
        print("=" * 80)
        with torch.no_grad():
            full_input = torch.cat([input_ids, continuation], dim=-1)
            clean_logits = model(full_input).logits
        clean_p = answer_probs_from_logits(clean_logits, probe, prefix_len).cpu()
        clean_lp = answer_logprobs_from_logits(clean_logits, probe, prefix_len).cpu()
        print(f"  Clean P(answer): {dict(zip(probe.answer_letters, clean_p.tolist()))}")
        del clean_logits, full_input
        clear_cuda()

    print("=" * 80)
    print(f"Step 6: Running {masking_algorithm} (objective={objective})")
    print("=" * 80)
    base_objective = (
        CANDIDATE_OBJECTIVES[objective]
        if use_candidate_objective
        else PROBE_OBJECTIVES.get(objective)
    )
    if use_candidate_rollouts:
        from utils.circuit_discovery.edits.nodewise_subnetwork_probing_candidate_rollouts import (
            candidate_rollout_kl,
        )
        objective_fn = candidate_rollout_kl
    elif use_candidate_objective:
        if objective == "candidate_snis_reward_gap":
            objective_fn = partial(
                base_objective,
                target_answer=target_cluster,
                sample_counts=candidate_counts,
            )
        else:
            objective_fn = partial(
                base_objective,
                target_answer=target_cluster,
            )
    elif objective == "answer_probe_reward_gap":
        objective_fn = partial(
            base_objective,
            answer_token_ids=probe.answer_token_ids,
            target_answer=target_answer_id,
        )
    elif objective == "answer_probe_logit_margin":
        objective_fn = partial(
            base_objective,
            answer_token_ids=probe.answer_token_ids,
            target_answer=target_answer_id,
            reduce=logit_margin_reduce,
        )
    elif objective == "answer_probe_target_kl":
        if target_probs is None:
            raise ValueError("answer_probe_target_kl requires target_probs")
        if len(target_probs) != len(probe.answer_letters):
            raise ValueError(
                f"target_probs has {len(target_probs)} entries but the probe "
                f"has {len(probe.answer_letters)} answer letters"
            )
        objective_fn = partial(
            base_objective,
            answer_token_ids=probe.answer_token_ids,
            target_probs=torch.tensor(target_probs, dtype=torch.float32),
        )
    elif objective == "answer_probe_prefix_kl":
        # Bind sentences + prefix_len so the loss can iterate per-sentence.
        # The SNP step still passes a `position_mask` and `token_ids`; the
        # prefix-KL loss ignores both via **kwargs.
        objective_fn = partial(
            base_objective,
            sentences=sentences,
            prefix_len=prefix_len,
        )
    else:
        objective_fn = partial(
            base_objective,
            answer_token_ids=probe.answer_token_ids,
        )
    if not use_candidate_rollouts:
        objective_fn.__name__ = base_objective.__name__

    discovery_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        objective_fn=objective_fn,
        sentence_gap=sentence_gap,
        ablate_non_target_layers=ablate_non_target_layers,
        renormalize_masked_attention=renormalize_masked_attention,
        num_ig_steps=num_ig_steps,
        negate_scores=not no_negate_scores,
        include_zero_ablation=include_zero_ablation,
        zero_ablation_epsilon=zero_ablation_epsilon,
        pair_aggregation=pair_aggregation,
        mask_granularity=mask_granularity,
        torch_compile=torch_compile,
    )
    if batch_chunk_size is not None:
        discovery_kwargs["batch_chunk_size"] = batch_chunk_size
    # Subnetwork-probing kwargs (only consumed by the SNP algorithm; harmless
    # when passed to algorithms that ignore them via **kwargs).
    for _k, _v in {
        "l0_lambda": l0_lambda,
        "num_training_steps": num_training_steps,
        "learning_rate": learning_rate,
        "log_alpha_init": log_alpha_init,
        "log_alpha_init_mask_path": log_alpha_init_mask_path,
        "log_alpha_init_mask_alpha": log_alpha_init_mask_alpha,
        "log_every": log_every,
        "plot_every": plot_every,
        "sparsity_loss_mode": sparsity_loss_mode,
        "l0_normalize_hinge": l0_normalize_hinge,
        "l0_warmup_frac": l0_warmup_frac,
        "l0_ramp_frac": l0_ramp_frac,
        "target_sparsity": target_sparsity,
        "optimizer": optimizer,
        "momentum": momentum,
        "l0_lr_multiplier": l0_lr_multiplier,
        "dropout_p": dropout_p,
        "save_log_alpha": save_log_alpha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_every": checkpoint_every,
        "resume_from_checkpoint": resume_from_checkpoint,
        "num_hc_samples_per_step": num_hc_samples_per_step,
        "polyak_ema_log_alpha": polyak_ema_log_alpha,
        "hc_beta_anneal": hc_beta_anneal,
        "hc_beta_start": hc_beta_start,
        "hc_beta_end": hc_beta_end,
        "hc_beta_anneal_end_frac": hc_beta_anneal_end_frac,
        "lr_schedule": lr_schedule,
        "lr_min_ratio": lr_min_ratio,
        "lr_plateau_patience": lr_plateau_patience,
        "lr_plateau_factor": lr_plateau_factor,
        "training_gap_mode": training_gap_mode,
        "continuations_per_step": continuations_per_step,
        "clean_logits_dtype": clean_logits_dtype,
        "pid_kp": pid_kp,
        "pid_ki": pid_ki,
        "pid_kd": pid_kd,
        "pid_mult_init": pid_mult_init,
        "pid_ramp_end_frac": pid_ramp_end_frac,
        "pid_max_target_sparsity": pid_max_target_sparsity,
        "snapshot_sparsities": snapshot_sparsities,
        "dcm_mask_init": dcm_mask_init,
        "dcm_l0_optimizer": dcm_l0_optimizer,
        "dcm_polarization": dcm_polarization,
        "pid_snapshot_hold_steps": pid_snapshot_hold_steps,
        "dcm_lr_init": dcm_lr_init,
        "dcm_lr_warmup_frac": dcm_lr_warmup_frac,
    }.items():
        if _v is not None:
            discovery_kwargs[_k] = _v
    if use_candidate_rollouts:
        discovery_kwargs["candidate_rollout_bank"] = candidate_rollout_bank
        discovery_kwargs["rollout_bank_set"] = rollout_bank_set
    if algorithm_kwargs:
        # Algorithm-specific keys without a dedicated argument (YAML mapping
        # ``algorithm_kwargs``); they go straight to the constructor.
        discovery_kwargs.update(dict(algorithm_kwargs))
    log_dir = os.path.join(output_dir, file_name.removesuffix(".json")) if file_name else os.path.join(output_dir, masking_algorithm)
    discovery_kwargs["log_dir"] = log_dir

    discoverer = create_circuit_discovery(masking_algorithm, **discovery_kwargs)
    discover_kwargs = dict(
        input_ids=input_ids,
        sentences=sentences,
        continuations=continuations,
        mask_mode=mask_mode,
        num_prefix_sentences=len(sentences),
        branch_rewards=None,
        position_mask_overrides=(
            None if use_candidate_objective else position_masks
        ),
        num_frozen_prompt_sentences=(
            num_prompt_sentences if freeze_prompt_sentences else 0
        ),
    )
    if learnable_region is not None:
        # Only the region-aware trainer variant understands these kwargs;
        # every other algorithm would swallow them in **kwargs and train
        # over the default pool, so refuse loudly instead.
        if masking_algorithm != REGION_ALGORITHM:
            raise ValueError(
                f"learnable_region={learnable_region!r} requires "
                f"masking_algorithm={REGION_ALGORITHM!r}, got "
                f"{masking_algorithm!r}."
            )
        if freeze_prompt_sentences:
            raise ValueError(
                "learnable_region and freeze_prompt_sentences are mutually "
                "exclusive: the frozen-prompt filter removes every "
                "prompt-key cell that learnable_region='prompt_to_trace' needs."
            )
        discover_kwargs["learnable_region"] = learnable_region
        discover_kwargs["num_prompt_sentences"] = num_prompt_sentences
        if frozen_key_sentences:
            discover_kwargs["frozen_key_sentences"] = [int(j) for j in frozen_key_sentences]
            print(f"  Frozen key sentences (always readable): {sorted(int(j) for j in frozen_key_sentences)}")
    elif frozen_key_sentences:
        raise ValueError("frozen_key_sentences requires learnable_region (the region-aware trainer)")
    if use_candidate_objective:
        discover_kwargs["answer_ids"] = candidate_answer_ids
        discover_kwargs["num_answers"] = num_answer_clusters
    node_mask = discoverer.discover(**discover_kwargs)

    for i, s in enumerate(node_mask.sentences):
        s["text"] = tokenizer.decode(input_ids[0, s["start"] : s["end"] + 1])

    print("=" * 80)
    print("Step 7: Save")
    print("=" * 80)
    node_mask.metadata.update({
        "objective": objective,
        "seed": seed,
        "mode": "direct_answer",
        "probe_suffix": probe_suffix,
        "probe_suffix_token_ids": probe.suffix_ids.tolist(),
        "prefix_len": prefix_len,
        "freeze_prompt_sentences": freeze_prompt_sentences,
        "num_frozen_prompt_sentences": (
            num_prompt_sentences if freeze_prompt_sentences else 0
        ),
        "learnable_region": learnable_region,
        "num_prompt_sentences": num_prompt_sentences,
        "frozen_key_sentences": list(frozen_key_sentences or []),
    })
    if rollout_bank_info is not None:
        node_mask.metadata["rollout_bank"] = rollout_bank_info
    if use_candidate_objective:
        node_mask.metadata.update({
            "answer_bank_path": answer_bank_path,
            "num_candidates": len(answer_bank["candidates"]),
            "num_answer_clusters": num_answer_clusters,
            "target_cluster": target_cluster,
            "gold_answer": answer_bank["gold_answer_normalized"],
            "clean_fraction_correct": answer_bank.get("clean_fraction_correct"),
            "candidate_answers": [
                {
                    "answer_text": c["answer_text"],
                    "cluster_id": c["cluster_id"],
                    "count": c["count"],
                    "is_correct": c["is_correct"],
                }
                for c in answer_bank["candidates"]
            ],
        })
    else:
        node_mask.metadata.update({
            "answer_letters": probe.answer_letters,
            "answer_token_ids": probe.answer_token_ids.tolist(),
            "answer_logit_position": probe.answer_logit_position(prefix_len),
            "answer_probs_clean": clean_p.tolist(),
            "answer_logprobs_clean": clean_lp.tolist(),
        })
    if use_candidate_rollouts:
        # No answer letters: the answer distribution is over the clusters of
        # the candidate bank the rollout bank was scored with.
        for key in ("answer_letters", "answer_token_ids", "answer_logit_position",
                    "answer_probs_clean", "answer_logprobs_clean"):
            node_mask.metadata.pop(key, None)
        crb = candidate_rollout_bank
        node_mask.metadata.update({
            "answer_bank_path": crb["answer_bank_path"],
            "num_answer_clusters": crb["num_clusters"],
            "target_cluster": crb["target_cluster"],
            "gold_answer": crb["gold_answer"],
            "candidate_answers": crb["candidate_answers"],
        })
        target_answer_id = None
    if correct_answer is not None:
        node_mask.metadata["correct_answer"] = correct_answer
    if target_answer_id is not None:
        node_mask.metadata["target_answer_id"] = int(target_answer_id)
        node_mask.metadata["target_letter"] = probe.answer_letters[target_answer_id]
    if data_path is not None and prompt_index is not None:
        node_mask.metadata["data_path"] = data_path
        node_mask.metadata["prompt_index"] = prompt_index
        node_mask.metadata["base_answer_type"] = base_answer_type
    if analysis_sentence_step is not None:
        node_mask.metadata["analysis_sentence_step"] = analysis_sentence_step
    if analysis_timestep is not None:
        node_mask.metadata["analysis_timestep"] = analysis_timestep
    if sentences_after_prefix:
        node_mask.metadata["sentences_after_prefix"] = sentences_after_prefix

    if file_name is not None:
        if not file_name.endswith(".json"):
            file_name += ".json"
        output_file = os.path.join(output_dir, file_name)
    else:
        layers_str = "_".join(str(l) for l in layers)
        output_file = os.path.join(
            output_dir,
            f"direct_answer_{masking_algorithm}_layers{layers_str}_{objective}.json",
        )
    node_mask.to_json(output_file)
    print(f"Saved NodeMask to {output_file}")

    # Algorithms that snapshot the mask at multiple sparsities during one
    # training run (nodewise_dcm_pid_sdpa) expose `snapshot_scores`
    # ({target_sparsity: 2D score list}). Write one NodeMask JSON per
    # snapshot, sharing the enriched metadata, so the standard
    # matched-target evaluation applies to each file unchanged.
    snapshot_scores = getattr(discoverer, "snapshot_scores", None)
    if snapshot_scores:
        base_stem = output_file.removesuffix(".json")
        snapshot_steps = getattr(discoverer, "snapshot_steps", {})
        for tsp, scores in sorted(snapshot_scores.items()):
            snap_meta = dict(node_mask.metadata)
            snap_meta["target_sparsity"] = float(tsp)
            snap_meta["snapshot_step"] = snapshot_steps.get(tsp)
            snap_mask = NodeMask(
                model_name=node_mask.model_name,
                algorithm=node_mask.algorithm,
                layers=node_mask.layers,
                sentences=node_mask.sentences,
                objective_name=node_mask.objective_name,
                metadata=snap_meta,
                scores=scores,
            )
            snap_file = f"{base_stem}_tsp{int(round(tsp * 100)):02d}.json"
            snap_mask.to_json(snap_file)
            print(f"Saved snapshot NodeMask (target sparsity {tsp}) to {snap_file}")

    del model
    clear_cuda()
    print("\nDone!")
    return output_file
