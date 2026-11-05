#!/usr/bin/env python3
"""Byte-for-byte comparison of the original trees against a restore (for testing a backup).

    python backup/compare_trees.py --dest ~/backup_test

Compares ~/is_expts vs DEST/is_expts and the backed-up Claude memory vs DEST/claude_home
(paper_draft is a fresh Overleaf clone, so only its HEAD is checked against Overleaf's master),
skipping what pack.py never archives (.venv, __pycache__, .env, EXCLUDE_PATHS). With --core-only,
results/ is limited to PAPER_RESULTS (a restore without the separate results repo). For is_expts/.git it compares refs instead of bytes (the restore is a fresh clone
plus the bundle's refs). Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack import CLAUDE_HOME, CLAUDE_MEMBERS, HOME, PAPER_RESULTS, REPO, excluded, skipped  # noqa: E402

CORE_ONLY = False


def outside_core(r: str) -> bool:
    """Under results/ but neither inside nor above one of the PAPER_RESULTS dirs."""
    r = os.path.normpath(r)
    if not (r == "results" or r.startswith("results/")):
        return False
    return not any(r == p or r.startswith(p + "/") or p.startswith(r + "/") for p in PAPER_RESULTS)


def tree(base: Path, rel: str, skip_top_git: bool, deref: bool = False) -> dict[str, tuple]:
    out = {}
    root = base / rel
    # Apply the repo's EXCLUDE_PATHS to both the source and the restored copy of is_expts.
    is_repo = base.name == "is_expts"

    def skip(r):
        return is_repo and (skipped(REPO, r) or (CORE_ONLY and outside_core(r)))

    if not (root.exists() or root.is_symlink()) or skip(rel):
        return out
    if root.is_symlink() and not deref:  # e.g. is_expts/paper_draft -> ../paper_draft: compare the link, don't descend
        return {os.path.normpath(rel): ("l", os.readlink(root))}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=deref):
        dirnames[:] = [d for d in dirnames
                       if not excluded(d) and not (skip_top_git and d == ".git" and Path(dirpath) == base)
                       and not skip(os.path.relpath(os.path.join(dirpath, d), base))]
        for name in filenames + [d for d in dirnames if (Path(dirpath) / d).is_symlink() and not deref]:
            p = Path(dirpath) / name
            r = os.path.relpath(p, base)
            if excluded(name) or skip(r):
                continue
            if p.is_symlink() and not deref:
                out[r] = ("l", os.readlink(p))
            else:
                out[r] = ("f", p.stat().st_size)
        if not deref:
            dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
    return out


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def compare(name: str, a_base: Path, b_base: Path, rels: list[str], skip_top_git: bool, deref: bool, workers: int):
    a, b = {}, {}
    for rel in rels:
        a.update(tree(a_base, rel, skip_top_git, deref))
        b.update(tree(b_base, rel, skip_top_git, deref))
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    meta_diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    to_hash = [k for k in set(a) & set(b) if a[k] == b[k] and a[k][0] == "f"]

    def same(k):
        return k, digest(a_base / k) == digest(b_base / k)

    with ThreadPoolExecutor(workers) as ex:
        content_diff = sorted(k for k, ok in ex.map(same, to_hash, chunksize=64) if not ok)
    nbytes = sum(a[k][1] for k in to_hash)
    print(f"== {name}: {len(a)} source entries, {len(to_hash)} files / {nbytes / 1e9:.2f} GB hashed; "
          f"missing={len(missing)} extra={len(extra)} size/link-diff={len(meta_diff)} content-diff={len(content_diff)}",
          flush=True)
    for label, items in [("missing", missing), ("extra", extra), ("size/link-diff", meta_diff),
                         ("content-diff", content_diff)]:
        for k in items[:20]:
            print(f"   {label}: {k}", flush=True)
    return len(missing) + len(extra) + len(meta_diff) + len(content_diff)


def git_refs(repo: Path, prefix_map: dict[str, str]) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(repo), "for-each-ref", "--format=%(objectname) %(refname)"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    refs = {}
    for line in filter(None, out):
        sha, ref = line.split()
        for src, dst in prefix_map.items():
            if ref.startswith(src):
                refs[dst + ref[len(src):]] = sha
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--core-only", action="store_true", help="restore did not include the results repo")
    args = ap.parse_args()
    global CORE_ONLY
    CORE_ONLY = args.core_only
    dest = args.dest.expanduser().resolve()
    bad = 0

    top = sorted(n for n in os.listdir(HOME / "is_expts") if n != ".git" and not excluded(n) and not skipped(REPO, n))
    bad += compare("is_expts", HOME / "is_expts", dest / "is_expts", top, True, False, args.workers)
    bad += compare("claude memory", CLAUDE_HOME, dest / "claude_home", CLAUDE_MEMBERS, False, True, args.workers)

    src = git_refs(HOME / "is_expts", {"refs/heads/": "heads/", "refs/stash": "stash"})
    dst = git_refs(dest / "is_expts", {"refs/remotes/bundle/": "heads/", "refs/stash": "stash"})
    ref_diff = {k for k in src if src.get(k) != dst.get(k)}
    print(f"== is_expts git refs: {len(src)} branches+stash in source, {len(ref_diff)} differ "
          f"{sorted(ref_diff) if ref_diff else ''}", flush=True)
    bad += len(ref_diff)
    overleaf = subprocess.run(["git", "-C", str(HOME / "paper_draft"), "ls-remote", "origin", "refs/heads/master"],
                              capture_output=True, text=True).stdout.split()[:1]
    for repo in ["is_expts", "website", "paper_draft"]:
        heads = [subprocess.run(["git", "-C", str(r / repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() for r in (HOME, dest)]
        if repo == "paper_draft":  # restored from Overleaf, so compare with Overleaf's master, not the local checkout
            heads[0] = overleaf[0] if overleaf else "<ls-remote failed>"
        same = heads[0] == heads[1]
        # The restore clones website from GitHub; a source that is behind its remote is fine if the clone contains it.
        if not same and repo == "website":
            same = subprocess.run(["git", "-C", str(dest / repo), "merge-base", "--is-ancestor", heads[0], heads[1]],
                                  capture_output=True).returncode == 0
        print(f"== {repo} HEAD: source {heads[0][:10]} restored {heads[1][:10]} "
              f"{'OK' if heads[0] == heads[1] else 'OK (restored is newer)' if same else 'DIFFERENT'}", flush=True)
        bad += not same

    print("COMPARE OK" if bad == 0 else f"COMPARE FOUND {bad} DIFFERENCES", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
