#!/usr/bin/env bash
# Rebuild ~/is_expts (+ ../paper_draft from Overleaf, ../website, Claude memory) from git and the private HF backups.
#
#   export HF_TOKEN=...            # or `hf auth login`; the backup repos are private
#   bash backup/restore.sh DEST                   # core: code, notes, final_results, figure code, data
#   bash backup/restore.sh DEST --with-results    # also the rest of results/ (separate HF repo, ~17 GB)
#   bash backup/restore.sh DEST --exclude-groups logs,cache   # extra args go to restore.py
#
# Env overrides: GIT_URL, BRANCH, WEBSITE_URL, SKIP_WEBSITE=1, OVERLEAF_URL, SKIP_OVERLEAF=1,
# HF_BACKUP_REPO, HF_RESULTS_REPO. Cloning Overleaf needs an Overleaf git token (username "git").
# Needs git, tar, zstd and uv on PATH.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 DEST [--with-results] [restore.py args]" >&2; exit 2; }
DEST=$(realpath -m "$1"); shift
WITH_RESULTS=
ARGS=()
for a in "$@"; do
  if [ "$a" = "--with-results" ]; then WITH_RESULTS=1; else ARGS+=("$a"); fi
done
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIT_URL=${GIT_URL:-https://github.com/cybershiptrooper/forking-paths-clone.git}
BRANCH=${BRANCH:-is_variants}
WEBSITE_URL=${WEBSITE_URL:-https://github.com/cybershiptrooper/website.git}
OVERLEAF_URL=${OVERLEAF_URL:-https://git@git.overleaf.com/69c2f8d8e2c524f0e3f5189c}
HF_BACKUP_REPO=${HF_BACKUP_REPO:-cybershiptrooper/is-expts-backup}
HF_RESULTS_REPO=${HF_RESULTS_REPO:-cybershiptrooper/is-expts-results}

for tool in git tar zstd uv; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

mkdir -p "$DEST"
if [ ! -d "$DEST/is_expts/.git" ]; then
  git clone -b "$BRANCH" "$GIT_URL" "$DEST/is_expts"
fi
if [ -z "${SKIP_WEBSITE:-}" ] && [ ! -d "$DEST/website/.git" ]; then
  git clone "$WEBSITE_URL" "$DEST/website"
fi
if [ -z "${SKIP_OVERLEAF:-}" ] && [ ! -d "$DEST/paper_draft/.git" ]; then
  # Not fatal: the paper lives on Overleaf, and its figure code is also in is_expts/paper_figure_code.
  GIT_TERMINAL_PROMPT=0 git clone "$OVERLEAF_URL" "$DEST/paper_draft" \
    || echo "WARNING: could not clone the Overleaf project (set up an Overleaf git token, or SKIP_OVERLEAF=1)" >&2
fi

restore() {
  uv run --no-project --with huggingface_hub python "$HERE/restore.py" --dest "$DEST" --repo "$1" "${ARGS[@]+"${ARGS[@]}"}"
}
restore "$HF_BACKUP_REPO"
[ -z "$WITH_RESULTS" ] || restore "$HF_RESULTS_REPO"

cat <<EOF

Next steps:
  cd $DEST/is_expts && uv sync          # rebuild .venv
  recreate $DEST/is_expts/.env          # OPENROUTER_API_KEY, WANDB_API_KEY, RUNPOD_API_KEY (not backed up)
  copy $DEST/claude_home/* into ~/.claude/ if you want the Claude Code memory back
  git -C $DEST/is_expts branch -a       # local-only branches and the stash came from the bundle (remotes/bundle/*)
EOF
