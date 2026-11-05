#!/usr/bin/env python3
"""Download shards from a private HF backup repo, verify, and extract them under --dest.

Layout produced under DEST (matches the original ~/ layout, so the repo's
`paper_draft -> ../paper_draft/` and `website -> ../website/` symlinks resolve once restore.sh
has cloned them):

    DEST/is_expts/       git clone (made by restore.sh) + extracted workspace and bulk shards
    DEST/claude_home/    Claude Code memory (copy into ~/.claude by hand)

    uv run --no-project --with huggingface_hub python backup/restore.py --dest ~/restore
    ... --repo cybershiptrooper/is-expts-results  # the non-paper results/ (separate repo)
    ... --groups workspace,data,final_results     # only some groups (see manifest.json)
    ... --exclude-groups logs,cache               # everything but these

Resumable: units already extracted (marker in DEST/.backup_state) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_REPO = "cybershiptrooper/is-expts-backup"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def check_listing(listing: Path, target: Path) -> list[str]:
    """Every listed file exists with its recorded size; every symlink has its recorded target."""
    problems = []
    for line in listing.read_text().splitlines():
        kind, size, path, link = line.split("\t")
        p = target / path
        if kind == "f":
            if not p.is_file() or p.is_symlink():
                problems.append(f"missing file {path}")
            elif p.stat().st_size != int(size):
                problems.append(f"size mismatch {path}: {p.stat().st_size} != {size}")
        elif kind == "l":
            if not p.is_symlink() or os.readlink(p) != link:
                problems.append(f"bad symlink {path}")
        elif kind == "d" and not p.is_dir():
            problems.append(f"missing dir {path}")
    return problems


def restore_unit(u, dest: Path, dl: Path, state: Path, keep: bool, repo: str):
    from huggingface_hub import hf_hub_download

    marker = state / (u["archive"].replace("/", "__") + ".ok")
    if marker.exists() and marker.read_text() == u["sha256"]:
        return u["name"], "skipped (already restored)", []
    arc = Path(hf_hub_download(repo, u["archive"], repo_type="dataset", local_dir=dl))
    got = sha256(arc)
    if got != u["sha256"]:
        return u["name"], f"SHA256 MISMATCH {got}", [f"sha256 mismatch on {u['archive']}"]

    target = dest / u["extract_to"]
    if u["group"] == "git":
        repo_dir = dest / "is_expts"
        # Pull every ref of the original repo (local-only branches, stash, codex refs) next to the GitHub clone.
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "-q", str(arc),
                        "+refs/heads/*:refs/remotes/bundle/*", "+refs/stash:refs/stash"], check=True)
        local = set(subprocess.run(["git", "-C", str(repo_dir), "for-each-ref", "--format=%(refname:short)",
                                    "refs/heads"], capture_output=True, text=True).stdout.split())
        for ref in u["refs"]:
            name = ref.split()[1]
            if name.startswith("refs/heads/") and name[len("refs/heads/"):] not in local:
                b = name[len("refs/heads/"):]
                subprocess.run(["git", "-C", str(repo_dir), "branch", b, f"bundle/{b}"], check=True)
        problems = []
    else:
        target.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(f"zstd -dc -T4 '{arc}' | tar -xpf - -C '{target}'", shell=True)
        if r.returncode != 0:
            return u["name"], "EXTRACT FAILED", [f"extract failed on {u['archive']}"]
        lst = Path(hf_hub_download(repo, u["listing"], repo_type="dataset", local_dir=dl))
        problems = check_listing(lst, target)
    if not keep:
        arc.unlink()
    if not problems:
        marker.write_text(got)
    return u["name"], ("ok" if not problems else f"{len(problems)} problems"), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--groups", help="comma-separated groups to restore (default: all)")
    ap.add_argument("--exclude-groups", default="", help="comma-separated groups to skip")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    dest = args.dest.expanduser().resolve()
    dl, state = dest / ".backup_download" / args.repo.replace("/", "__"), dest / ".backup_state"
    state.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(hf_hub_download(args.repo, "manifest.json", repo_type="dataset",
                                               local_dir=dl)).read_text())
    groups = set(args.groups.split(",")) if args.groups else {u["group"] for u in manifest["units"]}
    groups -= set(filter(None, args.exclude_groups.split(",")))
    units = [u for u in manifest["units"] if u["group"] in groups]
    print(f"backup from {manifest['source_host']} at {manifest['created']}; restoring {len(units)} units "
          f"({sum(u['compressed_bytes'] for u in units) / 1e9:.1f} GB) of groups {sorted(groups)}", flush=True)

    git_units = [u for u in units if u["group"] == "git"]
    if git_units and not (dest / "is_expts/.git").exists():
        sys.exit(f"{dest}/is_expts is not a git clone; run restore.sh (it clones first) or drop the git group")

    failures = []
    # Git first; then biggest shards first.
    rest = sorted((u for u in units if u["group"] != "git"), key=lambda u: -u["compressed_bytes"])
    for u in git_units:
        name, status, probs = restore_unit(u, dest, dl, state, args.keep_archives, args.repo)
        print(f"[git] {name}: {status}", flush=True)
        failures += probs
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(restore_unit, u, dest, dl, state, args.keep_archives, args.repo) for u in rest]
        for i, f in enumerate(as_completed(futs), 1):
            name, status, probs = f.result()
            print(f"[{i}/{len(rest)}] {name}: {status}", flush=True)
            for p in probs[:5]:
                print(f"    {p}", flush=True)
            failures += probs

    if failures:
        print(f"RESTORE FAILED: {len(failures)} problems", flush=True)
        sys.exit(1)
    print(f"RESTORE OK: {len(units)} units verified against their listings under {dest}", flush=True)


if __name__ == "__main__":
    main()
