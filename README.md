# Learning Attention Circuits over Reasoning Sentences

Code for learning sparse masks over sentence-to-sentence attention in a reasoning model's chain of thought. A mask removes attention from sentence `i` to earlier sentence `j` in every layer and head. It is trained with subnetwork probing (hard-concrete mask values plus a quadratic penalty on the number of kept connections) so that the masked model still satisfies an objective defined on the reasoning outcome.

There are three tasks:

| Task | Objective | Training entry point |
|---|---|---|
| Answer preservation | KL between the clean and masked answer distributions | `expts.direct_answer_circuit_discovery.run` |
| Correct-answer probability | reward gap on the correct answer letter | `expts.direct_answer_circuit_discovery.run` |
| Shortening | expected remaining reasoning length | `expts.cot_termination_circuit_discovery.run` |

There are two baselines:

- A local-connection baseline adapted from Thought Anchors. It scores each pair by next-token KL when the attention is removed.
- Random masks with the same number of kept connections.

## Setup

```bash
uv sync            # Python 3.12, dependencies from pyproject.toml / uv.lock
```

- Run every command from the repository root; `base_config` paths and data paths are resolved relative to it.
- The models are `Qwen/Qwen3-8B` and `Qwen/Qwen3-32B` from the Hugging Face Hub.
- Training loads the model with attention hooks, so one GPU is enough for 8B. For 32B, set `device: auto` to shard the model over the visible GPUs.

## Restoring outputs from the backup

Masks, evaluations, rollouts, data, notes and other files not in git are backed up as zstd tar shards in two private Hugging Face dataset repos. `backup/restore.sh` rebuilds the original layout from them.

You need:
- `git`, `tar`, `zstd` and `uv` on `PATH`.
- An HF token with read access to the private repos.
- Optionally, an Overleaf git token, used to clone the paper.

```bash
export HF_TOKEN=hf_...      # or: uv run --with huggingface_hub hf auth login
git clone -b is_variants https://github.com/cybershiptrooper/forking-paths-clone.git /tmp/fpc
bash /tmp/fpc/backup/restore.sh ~/work                  # core: ~3 GB download
bash /tmp/fpc/backup/restore.sh ~/work --with-results   # also the rest of results/: +17 GB
```

The script writes the following under `~/work`:

| Path | Source | Contents |
|---|---|---|
| `is_expts/` | GitHub clone, then the HF shards extracted on top | Core repo [`cybershiptrooper/is-expts-backup`](https://huggingface.co/datasets/cybershiptrooper/is-expts-backup): `final_results/` (every run behind the main-body figures), `paper_figure_code/`, `notes/`, `new_reports/`, `claude_scripts/`, ignored configs, `data/`, `logs/`, `cache/`. With `--with-results`, the rest of `results/` from [`cybershiptrooper/is-expts-results`](https://huggingface.co/datasets/cybershiptrooper/is-expts-results). |
| `paper_draft/` | Overleaf clone | Skipped with a warning if there is no Overleaf token. |
| `website/` | GitHub clone | |
| `claude_home/` | HF | Claude Code memory. |

Every shard is checked against its sha256 and per-file listing. If the restore is interrupted, rerun the same command; it skips shards that are already restored. Afterwards:

```bash
cd ~/work/is_expts && uv sync      # rebuild .venv
# recreate .env (OPENROUTER_API_KEY, WANDB_API_KEY, RUNPOD_API_KEY); it is not backed up
git branch -a                      # local-only branches (vm_local) and the stash come back as remotes/bundle/*
```

Old `results/...` paths of the main-body runs are symlinks into `final_results/`, so the paper's figure scripts work unchanged. See `backup/README.md` for these options:
- `--groups` / `--exclude-groups` to restore only some groups.
- `SKIP_WEBSITE=1` and `SKIP_OVERLEAF=1` to skip those clones.
- How to make a new backup.

## How configs work

Every training script takes `--config <file.yaml>`. Configs are loaded by `utils/expt_config.py:load_config` with these rules:

1. **Inheritance.** A config may set `base_config: <path>`. The base is loaded first, recursively, so chains of any depth work. The child's keys then override it (nested dicts are merged key by key).
2. **CLI beats config beats default.** `run.py` passes the loaded config to `parser.set_defaults(...)`, so any flag given on the command line overrides the YAML value.
3. **Extra keys pass through.** Keys without a matching CLI flag still reach the trainer. For example, `deterministic_algorithms: true` works this way.
4. **Output location.** Each run writes its mask to `<output_dir>/<file_name>`. Give every config a unique `file_name`.

A typical layout is a shared base, plus one small file per run that sets only what varies. The `configs/` directory has ready-to-run examples of the paper recipe:

| Config | What it runs |
|---|---|
| `configs/base_answer.yaml` | shared settings for the two answer tasks |
| `configs/thought_anchors/gpqa8b_p08_s80.yaml` | local-connection baseline |
| `configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml` | answer preservation, Qwen3-8B, stored suffix |
| `configs/answer_kl/gpqa32b_p00_s80_tsp40.yaml` | answer preservation, Qwen3-32B |
| `configs/answer_kl_bank/gpqa8b_p08_s80_tsp40.yaml` | answer preservation on sampled continuations, initialised from the baseline |
| `configs/matryoshka/gpqa8b_p08_s80.yaml` | answer preservation with Matryoshka attribution: one ordering for every sparsity |
| `configs/correct_answer/gpqa8b_p08_s80_tsp40.yaml` | correct-answer probability |
| `configs/shortening/aqua_p004_s198_tsp40.yaml` | shortening |

`configs/base_answer.yaml`:

```yaml
mode: learn
probe_suffix: " </think> I think the answer is"   # forced suffix; answer read from next-token logits
answer_letters: [" A", " B", " C", " D"]          # AQuA: add " E"
mask_mode: prefix               # mask only inside the prefix
mask_granularity: pair          # sentence-to-sentence connections
sentence_gap: 1                 # never mask a sentence's attention to itself
freeze_prompt_sentences: true   # the prompt's sentences are always attended to
sentences_after_prefix: 5       # stored suffix used for training and fixed-suffix evaluation
...
sparsity_loss_mode: target_size_l2   # quadratic penalty on (#kept - k)
l0_lambda: 1000.0
optimizer: hybrid
learning_rate: 0.1
log_alpha_init: 2.0
num_training_steps: 1000
seed: 42
device: cuda
```

`configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml` inherits it and sets only the run:

```yaml
base_config: configs/base_answer.yaml
masking_algorithm: nodewise_subnetwork_probing_hc_batched
objective: answer_probe_kl
num_hc_samples_per_step: 4      # masks sampled per step
batch_chunk_size: 4

model_name: Qwen/Qwen3-8B
data_path: data/collection/qwen3_8b/gpqa_filtered.json
prompt_index: 8
analysis_sentence_step: 80      # prefix = prompt + first 80 reasoning sentences
target_sparsity: 0.4            # fraction of eligible connections removed

output_dir: results/answer_kl/masks
file_name: gpqa8b_p08_s80_tsp40
```

Inheritance can chain. `configs/answer_kl_bank/gpqa8b_p08_s80_tsp40.yaml` inherits the config above and adds the continuation bank and the initial mask.

Run a config:

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml
```

Any config value can be overridden on the command line. For example, to use a different sparsity or prompt without writing a new file:

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml \
    --target_sparsity 0.8 --file_name gpqa8b_p08_s80_tsp80

uv run python -m expts.direct_answer_circuit_discovery.run --config configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml \
    --prompt_index 0 --analysis_sentence_step 50 --file_name gpqa8b_p00_s50_tsp40
```

### Sweeps

A sweep is a directory with one YAML per run. The simplest way to make one is a short generator script:

```python
# make_sweep.py
import os, yaml

OUT = "configs/kl_sweep"
os.makedirs(OUT, exist_ok=True)
for p, step in [(0, 50), (8, 80), (22, 65)]:
    for tsp in [0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        stem = f"gpqa_p{p:02d}_s{step}_tsp{round(tsp * 100):02d}"
        cfg = {
            "base_config": "configs/base_answer.yaml",
            "masking_algorithm": "nodewise_subnetwork_probing_hc_batched",
            "objective": "answer_probe_kl",
            "num_hc_samples_per_step": 4,
            "model_name": "Qwen/Qwen3-8B",
            "data_path": "data/collection/qwen3_8b/gpqa_filtered.json",
            "prompt_index": p, "analysis_sentence_step": step,
            "target_sparsity": tsp,
            "output_dir": "results/kl_sweep/masks", "file_name": stem,
        }
        with open(f"{OUT}/{stem}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
```

Then run every config in the directory:

```bash
uv run python make_sweep.py
for cfg in configs/kl_sweep/*.yaml; do
    uv run python -m expts.direct_answer_circuit_discovery.run --config "$cfg"
done
```

On a cluster, submit one job per file, for example as a job array indexed into the sorted file list.

### Main config keys

| Key | Meaning |
|---|---|
| `model_name`, `data_path`, `prompt_index` | model and question (see Data below) |
| `analysis_sentence_step` | number of reasoning sentences in the prefix |
| `sentences_after_prefix` | length of the stored suffix used for training and fixed-suffix evaluation |
| `masking_algorithm` | `nodewise_subnetwork_probing_hc_batched` (several masks per step), `nodewise_subnetwork_probing_sdpa` (one mask per step), `nodewise_subnetwork_probing_boundary_hazard_batched` (shortening) |
| `objective` | `answer_probe_kl`, `answer_probe_reward_gap`, `boundary_expected_length_eligible` |
| `target_sparsity` | fraction of eligible connections removed; the final mask keeps the top `k = round((1 - s)·\|E\|)` |
| `sparsity_loss_mode`, `l0_lambda` | `target_size_l2` with λ = 1000 is the quadratic penalty used throughout |
| `num_hc_samples_per_step` | hard-concrete masks sampled per step (4 for answer preservation, 1 otherwise) |
| `rollout_bank_path`, `rollout_bank_set`, `continuations_per_step` | train on sampled continuations instead of the stored suffix |
| `log_alpha_init_mask_path`, `log_alpha_init_mask_alpha` | initialise from another mask's ranking (e.g. the local-connection baseline) |
| `mask_granularity` | `pair` (default), `head` or `layer` |

## 1. Data

Traces are sampled with vLLM, 16 per question, and only questions whose accuracy over the 16 samples is between 25% and 75% are kept. Dataset names are listed in `utils/data_utils.py` (`GPQA`, `AQuA`, `AQuA_train`, `MATH_open`, `MMLU_<subject>`, ...).

```bash
# Sample traces (use --tensor_parallel_size 2 for Qwen3-32B)
uv run python -m expts.forking_paths.data_collection_new \
    --model_name Qwen/Qwen3-8B --dataset_names GPQA --num_examples 198 --shuffle \
    --num_paths 16 --max_new_tokens 50000 --temperature 0.6 --batch_size 8 \
    --return_logprobs --return_alternate_texts --seed 42 --enable_prefix_caching

# Keep questions with 25-75% accuracy
uv run python -m expts.analyse_collected_data \
    --data data/collection/qwen3_8b/gpqa.json \
    --output-dir results/data_analysis/gpqa_8b \
    --filtered-output data/collection/qwen3_8b/gpqa_filtered.json
```

The answer-preservation and correct-answer tasks use the filtered files. The shortening task uses the unfiltered ones.

## 2. Answer preservation

### Local-connection baseline

This baseline has no training loop. Its config (`configs/thought_anchors/gpqa8b_p08_s80.yaml`) is flat: it has no `base_config` and does not accept CLI overrides.

```bash
uv run python -m expts.direct_answer_circuit_discovery.thought_anchors_compat --config configs/thought_anchors/gpqa8b_p08_s80.yaml
```

The baseline produces one score matrix per prompt. It is thresholded to any target sparsity at evaluation time.

### Learned masks

**On the stored suffix:**

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/answer_kl/gpqa8b_p08_s80_tsp40.yaml
```

**On sampled continuations, initialised from the baseline (paper recipe).** First sample 32 continuations of the prefix per clean set (at most 200 tokens, temperature 0.7). Set B is used for training; set A is held out.

```python
# build_bank.py
from argparse import Namespace
from expts.direct_answer_circuit_discovery.eval_onpolicy_kl import build_clean_rollouts

build_clean_rollouts(Namespace(
    model_name="Qwen/Qwen3-8B", data_path="data/collection/qwen3_8b/gpqa_filtered.json",
    prompt_index=8, analysis_sentence_step=80, sentence_gap=1, mask_mode="prefix",
    answer_letters=None, probe_suffix=None,          # AQuA: answer_letters=" A, B, C, D, E"
    n_rollouts=32, max_new_tokens=200, temperature=0.7, gen_batch=16, seed=42, alpha=0.5,
    output="results/banks/gpqa8b_p08_s80.json"), ctx={})
```

For open-ended answers (MATH), use `python -m expts.direct_answer_circuit_discovery.build_candidate_rollout_bank` instead, with a candidate answer bank from `build_answer_bank`.

Then train with the baseline mask and the bank:

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/answer_kl_bank/gpqa8b_p08_s80_tsp40.yaml
```

**Qwen3-32B.** Use `configs/answer_kl/gpqa32b_p00_s80_tsp40.yaml`, which sets `device: auto`. Make 2 GPUs visible.

### One ordering for every sparsity (Matryoshka attribution)

`nodewise_matryoshka_attribution` is a port of the optimiser of [Matryoshka attribution](https://github.com/aryamanarora/matryoshka-attribution) (Arora et al., 2026) to the sentence-pair masks of this repository. It is not part of the paper's results.

- Each connection has one score, initialised at zero.
- Every step draws a number of kept connections `k ~ Uniform(1, n)` and applies the sigmoid top-k mask of the scores at that `k`.
- Adam minimises the task loss only. There is no sparsity penalty and no target sparsity.
- One run gives an ordering of the connections. The evaluators threshold it at any sparsity.

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/matryoshka/gpqa8b_p08_s80.yaml
```

The run writes the ordering to `<output_dir>/<file_name>.json`. Next to it, `<file_name>/bank_topk_eval.jsonl` has the loss of the binary top-k mask on the training bank at 12 sparsities every `bank_eval_every` steps, and `<file_name>/score_snapshots.npz` has the scores at the steps listed in `mattr_snapshot_steps`. The trainer is `utils/circuit_discovery/edits/nodewise_matryoshka_attribution.py`; its tensor-level tests are in `tests/test_matryoshka_attribution.py`.

## 3. Correct-answer probability

This task uses the same base config and the same prompts as answer preservation. It changes the objective (`answer_probe_reward_gap`), uses one mask per step, and trains on the 5 stored sentences after the prefix.

```bash
uv run python -m expts.direct_answer_circuit_discovery.run --config configs/correct_answer/gpqa8b_p08_s80_tsp40.yaml
```

## 4. Evaluation (answer tasks)

All evaluation scripts read a saved mask, keep its top-`k` connections at the requested sparsity, and compare the masked model to the clean model. The flags must match the training setup: `sentence_gap` and `sentences_after_prefix` default to 0 in the evaluation scripts, so always pass them.

**Fixed-suffix evaluation.** This teacher-forces the stored suffix and reports answer KL, answer probabilities, and the reward gap. Evaluate at the training target:

```bash
uv run python -m expts.direct_answer_circuit_discovery.eval_log_alpha \
    --mask_path results/answer_kl/masks/gpqa8b_p08_s80_tsp40.json \
    --model_name Qwen/Qwen3-8B --data_path data/collection/qwen3_8b/gpqa_filtered.json \
    --prompt_index 8 --analysis_sentence_step 80 --sentences_after_prefix 5 \
    --sentence_gap 1 --mask_mode prefix --top_k_sparsities 0.4 \
    --output results/answer_kl/eval/gpqa8b_p08_s80_tsp40.eval.json
```

For the local-connection baseline, pass its score file as `--mask_path`, add `--force_freeze_prompt`, and give all target sparsities at once, for example `--top_k_sparsities 0.01,0.05,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99`.

**Random masks.** These use the same eligible-connection pool and the same `k`:

```bash
uv run python -m expts.direct_answer_circuit_discovery.eval_random_masks \
    --model_name Qwen/Qwen3-8B --data_path data/collection/qwen3_8b/gpqa_filtered.json \
    --prompt_index 8 --analysis_sentence_step 80 --sentences_after_prefix 5 --sentence_gap 1 \
    --mask_mode prefix --sparsities 0.2,0.4,0.6,0.8 --n_samples 3 --seed 42 --force_freeze_prompt \
    --output results/random/gpqa8b_p08_s80.random_eval.json
```

**Generated-suffix evaluation.** This samples continuations from the masked model at temperature 0.7 and reads the answer after the forced suffix:

```bash
uv run python -m expts.direct_answer_circuit_discovery.eval_masked_rollouts \
    --mask_path results/answer_kl/masks/gpqa8b_p08_s80_tsp40.json \
    --model_name Qwen/Qwen3-8B --data_path data/collection/qwen3_8b/gpqa_filtered.json \
    --prompt_index 8 --analysis_sentence_step 80 --sentences_after_prefix 5 --sentence_gap 1 \
    --mask_mode prefix --target_sparsity 0.4 --n_rollouts 5 --max_new_tokens 200 \
    --temperature 0.7 --seed 42 --output results/answer_kl/rollouts/gpqa8b_p08_s80_tsp40.json
```

To compare the masked model's answer-outcome distribution with the clean bank over 32 rollouts, call `run_cell` from `expts/direct_answer_circuit_discovery/eval_onpolicy_kl.py`. It takes the same arguments as `build_clean_rollouts` above, plus `mask_path`, `target_sparsity`, `sentences_after_prefix`, `force_freeze_prompt` and `clean_path` (the bank file).

## 5. Shortening

**1. Pick questions and analysis points, and sample continuation banks.**

- Questions are kept if their accuracy is between 0.5 and 0.75.
- The analysis point is placed about 2200 tokens before `</think>`.

```bash
uv run python -m expts.cot_termination_circuit_discovery.scan_early_analysis_points \
    --data_paths data/collection/qwen3_8b/gpqa.json data/collection/qwen3_8b/aqua.json \
                 data/collection/qwen3_8b/aqua_train.json \
    --offset_tokens 2200 --fallback_offset_tokens 3000 --n_samples 16 --horizon 4096 \
    --max_prefix_tokens 15000 --per_dataset 5 --candidates_per_dataset 10 \
    --output_dir results/termination/banks_root
```

**2. Build the sentence-boundary training data for each selected bank:**

```bash
uv run python -m expts.cot_termination_circuit_discovery.build_boundary_data \
    --bank_path results/termination/banks_root/banks/aqua_p004_s198.json \
    --output results/termination/boundary_data/aqua_p004_s198.json
```

**3. Train.** `configs/shortening/aqua_p004_s198_tsp40.yaml` is self-contained and has no `base_config`. It points at the bank and boundary data from steps 1-2, and saves checkpoints so that a preempted run resumes.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m expts.cot_termination_circuit_discovery.run --config configs/shortening/aqua_p004_s198_tsp40.yaml
```

**4. Evaluate.** This samples 16 continuations with the mask applied, up to 4096 tokens. It reports reasoning length and answer outcome, together with 2 random masks at the same sparsity:

```bash
uv run python -m expts.cot_termination_circuit_discovery.eval_termination_rollouts \
    --mask_path results/termination/masks/aqua_p004_s198_tsp40.json \
    --bank_path results/termination/banks_root/banks/aqua_p004_s198.json \
    --n_rollouts 16 --horizon 4096 --batch_size 16 --n_random_masks 2 --sentence_gap 1 \
    --probe_at_horizon --store_token_ids --skip_clean --skip_snis_check \
    --output results/termination/eval/aqua_p004_s198_tsp40.json
```

**5. Local-connection baseline.** This scores, thresholds and evaluates in one process. Add `--include_clean` once per prompt to also sample unmasked continuations.

```bash
uv run python -m expts.cot_termination_circuit_discovery.eval_thought_anchors_termination \
    --bank_path results/termination/banks_root/banks/aqua_p004_s198.json \
    --target_sparsities 0.2 0.4 0.6 0.8 --n_rollouts 16 --horizon 4096 --batch_size 16 \
    --sentence_gap 1 --probe_at_horizon --store_token_ids \
    --output_dir results/termination/eval_ta
```

After the first run, you can pass `--scores_path <saved score file>` to reuse the baseline scores instead of recomputing them.

## Code layout

| Path | Contents |
|---|---|
| `configs/` | example configs for every task (paper recipe) |
| `expts/direct_answer_circuit_discovery/` | answer-preservation and correct-answer training (`run.py`, `learn.py`), baseline (`thought_anchors_compat.py`), evaluation (`eval_*.py`) |
| `expts/cot_termination_circuit_discovery/` | shortening: bank and boundary-data builders, training (`run.py`), evaluation |
| `expts/forking_paths/data_collection_new.py` | trace sampling with vLLM |
| `utils/circuit_discovery/edits/` | subnetwork-probing trainers (registered in `utils/circuit_discovery/factory.py`) |
| `utils/masks.py` | mask container and eligible-connection filters |
| `utils/expt_config.py` | config loading |

Tests: `uv run pytest tests/`.
