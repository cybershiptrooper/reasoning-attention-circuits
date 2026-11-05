# Backup and restore

The code lives in git (this GitHub repo, which is **public**). Everything else is stored as zstd
tar shards in two **private** HF dataset repos:

- [`cybershiptrooper/is-expts-backup`](https://huggingface.co/datasets/cybershiptrooper/is-expts-backup)
  (core, restored by default): notes, reports, `claude_scripts/`, ignored configs, `final_results/`
  (every run behind the paper's main-body figures), `paper_figure_code/`, `data/`, `logs/`, `cache/`,
  a git bundle, and the Claude Code memory.
- [`cybershiptrooper/is-expts-results`](https://huggingface.co/datasets/cybershiptrooper/is-expts-results):
  the rest of `results/` (sweeps, diagnostics, head/layer masks), restored with `--with-results`.

## Restore

```bash
export HF_TOKEN=...                                   # read access to the private repos
git clone -b is_variants https://github.com/cybershiptrooper/forking-paths-clone.git /tmp/fpc
bash /tmp/fpc/backup/restore.sh ~/work                  # core only
bash /tmp/fpc/backup/restore.sh ~/work --with-results   # core + the rest of results/
```

`restore.sh DEST` clones this repo into `DEST/is_expts`, `website` into `DEST/website` and the
Overleaf project into `DEST/paper_draft` (needs an Overleaf git token; skipped with a warning otherwise),
then `restore.py` downloads each shard, checks its sha256 against `manifest.json`, extracts
it, and checks every extracted file's size against the shard's listing. It is resumable;
rerun the same command after an interruption. Afterwards:

- `cd DEST/is_expts && uv sync` to rebuild `.venv`.
- Recreate `.env` (`OPENROUTER_API_KEY`, `WANDB_API_KEY`, `RUNPOD_API_KEY`); it is deliberately not backed up.
- Local-only branches (`vm_local`) and the stash come from a git bundle: `git branch -a` shows them as `remotes/bundle/*`.
- `DEST/claude_home/` holds the Claude Code memory; copy it into `~/.claude/` by hand.

## What is in the backup

| group | contents | restored to |
|---|---|---|
| `git` | `git bundle --all` of this repo (all branches, stash, codex refs) | refs in `DEST/is_expts` |
| `workspace` | working tree outside the bulk dirs: `notes/`, `new_reports/`, `claude_scripts/`, `configs/`, `expts/` configs, `*.sh`, `CLAUDE.md`, `.claude/`, notebooks, the `paper_draft`/`website` symlinks | `DEST/is_expts` |
| `final_results` | KL and reward-gap runs, plus the runs moved from `results/` on 2026-11-05 for the shortening, perplexity, coherence, sentence-type and seed-variance figures | `DEST/is_expts/final_results` |
| `results_links` | symlinks at the old `results/...` paths of those moved runs (`PAPER_RESULTS` in `pack.py`) | `DEST/is_expts/results` |
| `paper_figure_code` | copy of the paper's `scripts/`, `figures/html/` templates and `PLOTTING_GUIDE.md` | `DEST/is_expts/paper_figure_code` |
| `data`, `logs`, `cache` | the corresponding directories | `DEST/is_expts/...` |
| `results` (results repo) | the rest of `results/`, split into ~8 GB shards | `DEST/is_expts/results` |
| `claude` | `~/.claude/CLAUDE.md`, `shared-memory/`, project memory for `is_expts` and `forking-paths-clone` | `DEST/claude_home` |

Not backed up: `~/paper_draft` (lives on Overleaf), `.venv`, `__pycache__`, `.pytest_cache`, `node_modules`, `.env`, and the deprecated or
regenerable paths in `EXCLUDE_PATHS` in `pack.py` (`wandb/`, streamlit, legacy code, runpod scripts,
`results/activations`, `results/snp_sweep_old`, cached attention arrays in `results/receiver_heads_0925`).
`listings/<group>/<shard>.tsv` in the HF repo lists every file in each shard, so you can find
which shard holds a path without downloading anything, e.g. `--groups results` after checking.

## Making a new backup

```bash
uv run --no-project --with huggingface_hub python backup/pack.py --staging ~/is_expts_backup_staging --upload
uv run --no-project --with huggingface_hub python backup/pack.py --staging ~/is_expts_backup_staging --part results --upload
```

If a new main-body figure reads from `results/`, move that run into `final_results/`, leave a
symlink at the old path, and add the old path to `PAPER_RESULTS`. Re-copy `paper_figure_code/`
from the Overleaf clone when the figure scripts change.

`--dry-run` prints the shard plan; `--groups a,b` packs only some groups (the manifest then
lists only those). Packing is resumable via `*.done` markers in the staging dir; delete the
staging dir to start a fresh snapshot. `backup/compare_trees.py --dest DEST` compares a
restore byte-for-byte against the original trees.
