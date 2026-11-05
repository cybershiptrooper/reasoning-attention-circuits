#!/usr/bin/env python3
"""Pack is_expts (and the Claude memory) into zstd tar shards for two private HF dataset repos.

Part `core` (default, repo cybershiptrooper/is-expts-backup): git bundle, working tree outside the
bulk dirs, final_results/ (every run behind the paper's main-body figures), paper_figure_code/,
data/, logs/, cache/, Claude memory. Part `results` (repo cybershiptrooper/is-expts-results): the
rest of results/.

Each shard ("unit") is one .tar.zst plus a listing of its members. Each part's manifest.json
records every unit's sha256, member count and raw size so restore.py can verify downloads and
extracted trees. Re-running skips units whose archive is already done and unchanged.

    python backup/pack.py --staging ~/is_expts_backup_staging --upload                  # core
    python backup/pack.py --staging ~/is_expts_backup_staging --part results --upload   # results
    python backup/pack.py --staging ... --dry-run                                       # print units only

Needs: tar, zstd, git on PATH; huggingface_hub only for --upload.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]

HF_REPOS = {"core": "cybershiptrooper/is-expts-backup", "results": "cybershiptrooper/is-expts-results"}

# Large output trees in the core part: split into shards of at most --max-gb raw bytes each.
BULK_ROOTS = ["final_results", "data", "logs", "cache"]

# Main-body runs that used to live in results/ were moved to final_results/; these old paths are now
# symlinks into final_results/ (kept so paper scripts and stored relative paths still resolve). The
# core part carries the symlinks; the results part leaves them out.
PAPER_RESULTS = [
    "results/snp_sweep/termination_gap1_0925",  # Fig 3 shortening curves (learned + local baseline, gap 1)
    "results/snp_sweep/answer_dist_0923",       # Fig 3 local-baseline evals, perplexity panel
    "results/snp_sweep/early_2200",             # perplexity panel full-text rollouts
    "results/objective_rollouts_logprobs",      # perplexity panel
    "results/onpolicy_k32_logprobs",
    "results/rollout_clean_logprobs",
    "results/noreason_prefix_baseline",
    "results/swap_prompt_baseline",
    "results/judge_coherence",                  # coherence figure
    "results/judge_gradings",                   # sentence-type tags
    "results/sentence_type_paper",              # sentence-type figure
    "results/seed_variance_0918",               # seed-variance figure
    "results/seed_variance_0924",
    "results/seed_variance_0926",
    "results/seed_variance_random_0926",
]

# Never archived. `.env` holds API keys; `.venv` is rebuilt by `uv sync`.
EXCLUDE_NAMES = {".venv", "__pycache__", ".pytest_cache", ".env", "node_modules"}
EXCLUDE_GLOBS = [".nfs*", "*.pyc"]

# Repo-relative paths (fnmatch) left out of the backup: deprecated or regenerable, and not used by the
# paper or by current (non-outdated) reports. Tracked files matching these still come back via git.
EXCLUDE_PATHS = [
    "wandb",                                    # offline W&B runs, not needed
    ".streamlit", "data/streamlit",             # streamlit app is deprecated
    "legacy", "expts/legacy", "*_legacy.py", "*_legacy.html",
    "claude_scripts/runpod", "claude_scripts/*runpod*",
    "results/activations",                      # cached activations, unused
    "results/snp_sweep_old",                    # runs from before the sentence-splitting and prompt fixes
    "results/receiver_heads_0925/attention*",   # cached attention arrays; analysis/ and figures kept
]

# Claude Code memory (files under projects/*/memory are symlinks into shared-memory; archived dereferenced).
CLAUDE_HOME = HOME / ".claude"
CLAUDE_MEMBERS = [
    "CLAUDE.md",
    "shared-memory",
    "projects/-home-rgupta-is-expts/memory",
    "projects/-home-rgupta-forking-paths-clone/memory",
]


def excluded(name: str) -> bool:
    return name in EXCLUDE_NAMES or any(fnmatch.fnmatch(name, g) for g in EXCLUDE_GLOBS)


def skipped(base: Path, rel: str, extra=()) -> bool:
    """True for repo paths listed in EXCLUDE_PATHS or `extra` (only applies to walks rooted at the repo)."""
    rel = os.path.normpath(rel)
    return Path(base) == REPO and any(fnmatch.fnmatch(rel, p) for p in (*EXCLUDE_PATHS, *extra))


def walk(base: Path, rel: str, skip_top_git: bool = False, extra=()):
    """Yield (relpath, kind, size, link_target) under base/rel; kind in {d, f, l}. Dirs come before their contents."""
    root = base / rel
    if skipped(base, rel, extra):
        return
    if root.is_symlink() or root.is_file():
        st = root.lstat()
        yield (rel, "l" if root.is_symlink() else "f", st.st_size, os.readlink(root) if root.is_symlink() else "")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if not excluded(d) and not (skip_top_git and d == ".git" and Path(dirpath) == base)
            and not skipped(base, os.path.relpath(os.path.join(dirpath, d), base), extra)
        )
        drel = os.path.normpath(os.path.relpath(dirpath, base))
        yield (drel, "d", 0, "")
        # os.walk lists symlinks-to-dirs in dirnames without descending; archive them as links.
        for d in list(dirnames):
            p = Path(dirpath) / d
            if p.is_symlink():
                dirnames.remove(d)
                yield (os.path.join(drel, d), "l", 0, os.readlink(p))
        for f in sorted(filenames):
            if excluded(f) or skipped(base, os.path.join(drel, f), extra):
                continue
            p = Path(dirpath) / f
            st = p.lstat()
            if p.is_symlink():
                yield (os.path.join(drel, f), "l", 0, os.readlink(p))
            else:
                yield (os.path.join(drel, f), "f", st.st_size, "")


def entries_size(entries) -> int:
    return sum(e[2] for e in entries)


def split_bulk(base: Path, rel: str, max_bytes: int, extra=()):
    """Split base/rel into units of <= max_bytes raw.

    A subtree that fits is one unit. Otherwise children bigger than max_bytes are split
    recursively, and the remaining children (small subdirs and loose files, whole) are
    bin-packed in name order into `<rel>/_part_NNN` units.
    """
    entries = list(walk(base, rel, extra=extra))
    if entries_size(entries) <= max_bytes:
        return [(rel, entries)]
    by_child: dict[str, list] = {}
    for e in entries[1:]:
        child = e[0][len(rel) + 1:].split("/", 1)[0]
        by_child.setdefault(child, []).append(e)
    units, chunk, size, idx = [], [(rel, "d", 0, "")], 0, 0
    for child in sorted(by_child):
        ents = by_child[child]
        csize = entries_size(ents)
        if csize > max_bytes and ents[0][1] == "d":
            units.extend(split_bulk(base, ents[0][0], max_bytes, extra))
            continue
        if size + csize > max_bytes and len(chunk) > 1:
            units.append((f"{rel}/_part_{idx:03d}", chunk))
            chunk, size, idx = [(rel, "d", 0, "")], 0, idx + 1
        chunk.extend(ents)
        size += csize
    if len(chunk) > 1:
        units.append((f"{rel}/_part_{idx:03d}", chunk))
    return units


# Copy of the paper's figure-building code (scripts, HTML templates, PLOTTING_GUIDE); its own shard.
FIGURE_CODE = "paper_figure_code"


def plan_units(max_bytes: int, part: str = "core"):
    """Return list of unit dicts: name, group, base (dir tar runs in), extract_to, entries, dereference."""
    units = []
    if part == "results":
        for uname, ents in split_bulk(REPO, "results", max_bytes, extra=PAPER_RESULTS):
            units.append(dict(name=uname, group="results", base=str(REPO), extract_to="is_expts", entries=ents))
        return units

    # 1. Working-tree snapshot outside the bulk roots (includes tracked files, notes/, claude_scripts/, configs, symlinks).
    ws = []
    for name in sorted(os.listdir(REPO)):
        if excluded(name) or name in (".git", "results", FIGURE_CODE) or name in BULK_ROOTS or skipped(REPO, name):
            continue
        ws.extend(walk(REPO, name))
    units.append(dict(name="workspace", group="workspace", base=str(REPO), extract_to="is_expts", entries=ws))

    # 2. Bulk roots.
    for root in BULK_ROOTS:
        if not (REPO / root).exists() or skipped(REPO, root):
            continue
        for uname, ents in split_bulk(REPO, root, max_bytes):
            units.append(dict(name=uname, group=root, base=str(REPO), extract_to="is_expts", entries=ents))

    # 3. results/ -> final_results/ symlinks for the main-body runs, and the figure code.
    links = [e for rel in PAPER_RESULTS for e in walk(REPO, rel)]
    assert all(e[1] == "l" for e in links), "PAPER_RESULTS entries should be symlinks into final_results/"
    units.append(dict(name="results_links", group="results_links", base=str(REPO), extract_to="is_expts", entries=links))
    if (REPO / FIGURE_CODE).exists():
        units.append(dict(name=FIGURE_CODE, group=FIGURE_CODE, base=str(REPO), extract_to="is_expts",
                          entries=list(walk(REPO, FIGURE_CODE))))

    # 4. Claude memory, dereferenced so the shared-memory symlinks become real files.
    cents = []
    for m in CLAUDE_MEMBERS:
        p = CLAUDE_HOME / m
        if p.exists():
            for e in walk(CLAUDE_HOME, m):
                full = CLAUDE_HOME / e[0]
                if e[1] == "l" and full.exists():
                    e = (e[0], "f" if full.is_file() else "d", full.stat().st_size if full.is_file() else 0, "")
                cents.append(e)
    units.append(dict(name="claude_memory", group="claude", base=str(CLAUDE_HOME), extract_to="claude_home",
                      entries=cents, dereference=True))
    return units


def safe_name(name: str) -> str:
    return name.replace("/", "__")


def pack_unit(u, staging: Path, level: int, threads: int):
    arc_rel = f"archives/{u['group']}/{safe_name(u['name'])}.tar.zst"
    lst_rel = f"listings/{u['group']}/{safe_name(u['name'])}.tsv"
    arc, lst, done = staging / arc_rel, staging / lst_rel, staging / (arc_rel + ".done")
    ehash = hashlib.sha256("".join(f"{e[0]}\t{e[1]}\t{e[2]}\n" for e in u["entries"]).encode()).hexdigest()
    if done.exists():
        rec = json.loads(done.read_text())
        listing_now = "".join(f"{e[1]}\t{e[2]}\t{e[0]}\t{e[3]}\n" for e in u["entries"])
        if rec.get("entries_sha256") == ehash or (
                "entries_sha256" not in rec and lst.exists() and lst.read_text() == listing_now):
            rec["entries_sha256"] = ehash
            done.write_text(json.dumps(rec))
            return rec
        done.unlink()
    arc.parent.mkdir(parents=True, exist_ok=True)
    lst.parent.mkdir(parents=True, exist_ok=True)

    with open(lst, "w") as fh:
        for path, kind, size, target in u["entries"]:
            fh.write(f"{kind}\t{size}\t{path}\t{target}\n")

    member_file = staging / "tmp" / (safe_name(u["name"]) + f".{u['group']}.members")
    member_file.parent.mkdir(exist_ok=True)
    member_file.write_bytes(b"".join(e[0].encode() + b"\0" for e in u["entries"]))

    tar_cmd = ["tar", "-cf", "-", "-C", u["base"], "--no-recursion", "--null", "-T", str(member_file)]
    if u.get("dereference"):
        tar_cmd.insert(1, "--dereference")
    t0 = time.time()
    tmp = arc.with_suffix(".partial")
    h = hashlib.sha256()
    tar = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    zst = subprocess.Popen(["zstd", "-q", f"-{level}", f"-T{threads}", "-c"], stdin=tar.stdout, stdout=subprocess.PIPE)
    tar.stdout.close()
    with open(tmp, "wb") as out:
        while chunk := zst.stdout.read(1 << 22):
            h.update(chunk)
            out.write(chunk)
    if zst.wait() != 0 or tar.wait() != 0:
        raise RuntimeError(f"pack failed for {u['name']} (tar={tar.returncode}, zstd={zst.returncode})")
    tmp.rename(arc)
    member_file.unlink()

    rec = dict(
        name=u["name"], group=u["group"], extract_to=u["extract_to"], archive=arc_rel, listing=lst_rel,
        sha256=h.hexdigest(), compressed_bytes=arc.stat().st_size,
        raw_bytes=entries_size(u["entries"]), n_entries=len(u["entries"]),
        n_files=sum(e[1] == "f" for e in u["entries"]), seconds=round(time.time() - t0, 1), entries_sha256=ehash,
    )
    done.write_text(json.dumps(rec))
    return rec


def git_bundle(staging: Path):
    out = staging / "archives/git/is_expts.bundle"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(out), "--all"], check=True,
                   stderr=subprocess.DEVNULL)
    refs = subprocess.run(["git", "-C", str(REPO), "for-each-ref", "--format=%(objectname) %(refname)"],
                          check=True, capture_output=True, text=True).stdout.splitlines()
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    return dict(name="is_expts.bundle", group="git", extract_to="is_expts", archive="archives/git/is_expts.bundle",
                sha256=h, compressed_bytes=out.stat().st_size, refs=refs,
                head_branch=subprocess.run(["git", "-C", str(REPO), "branch", "--show-current"],
                                           capture_output=True, text=True).stdout.strip(),
                origin=subprocess.run(["git", "-C", str(REPO), "remote", "get-url", "origin"],
                                      capture_output=True, text=True).stdout.strip())


def prune_local(staging: Path, keep: set[str]):
    """Delete archives/listings in staging that the current plan no longer produces."""
    for sub in ("archives", "listings"):
        for p in (staging / sub).rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(staging))
                base = rel[:-len(".done")] if rel.endswith(".done") else rel
                if base not in keep:
                    print(f"pruning stale {rel}", flush=True)
                    p.unlink()


def upload(staging: Path, repo_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo_id, repo_type="dataset").private, f"{repo_id} is not private; refusing to upload"
    api.upload_large_folder(repo_id=repo_id, folder_path=str(staging), repo_type="dataset",
                            ignore_patterns=["*.done", "*.partial", "tmp/**"], num_workers=16)
    # Remove remote shards from earlier snapshots that the current manifest no longer lists.
    keep = {"manifest.json", ".gitattributes"}
    for u in json.loads((staging / "manifest.json").read_text())["units"]:
        keep |= {u["archive"], u.get("listing")}
    stale = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f not in keep]
    if stale:
        print(f"deleting {len(stale)} stale remote files: {stale}", flush=True)
        api.delete_files(repo_id, delete_patterns=stale, repo_type="dataset",
                         commit_message="Remove shards no longer in the manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", type=Path, required=True, help="parent dir; each part packs into STAGING/<part>")
    ap.add_argument("--part", choices=["core", "results"], default="core")
    ap.add_argument("--max-gb", type=float, default=8.0, help="max raw bytes per shard")
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=6, help="shards packed concurrently")
    ap.add_argument("--threads", type=int, default=10, help="zstd threads per shard")
    ap.add_argument("--groups", help="comma-separated groups to pack (default: all; manifest then lists only these)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--upload", action="store_true", help="upload the part to its private HF dataset repo")
    ap.add_argument("--repo", help="override the HF repo id for this part")
    ap.add_argument("--upload-only", action="store_true", help="skip packing; just upload")
    args = ap.parse_args()
    staging = args.staging.expanduser().resolve() / args.part
    repo_id = args.repo or HF_REPOS[args.part]

    if not args.upload_only:
        groups = set(args.groups.split(",")) if args.groups else None
        units = [u for u in plan_units(int(args.max_gb * 1e9), args.part) if groups is None or u["group"] in groups]
        total = sum(entries_size(u["entries"]) for u in units)
        print(f"{len(units)} units, {total / 1e9:.1f} GB raw", flush=True)
        if args.dry_run:
            for u in units:
                print(f"  {u['group']:14s} {entries_size(u['entries']) / 1e9:8.3f} GB {len(u['entries']):7d}  {u['name']}")
            return
        staging.mkdir(parents=True, exist_ok=True)
        records = [git_bundle(staging)] if args.part == "core" and (groups is None or "git" in groups) else []
        # Biggest first so the long shards overlap with the many small ones.
        order = sorted(units, key=lambda u: -entries_size(u["entries"]))
        with ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(pack_unit, u, staging, args.level, args.threads): u["name"] for u in order}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                print(f"[{i}/{len(units)}] {r['group']}/{r['name']}: {r['raw_bytes'] / 1e9:.2f} -> "
                      f"{r['compressed_bytes'] / 1e9:.2f} GB ({r.get('seconds', 0)} s)", flush=True)
                records.append(r)
        records.sort(key=lambda r: (r["group"], r["name"]))
        manifest = dict(
            part=args.part, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source_host=os.uname().nodename,
            source_repo=str(REPO), exclude_paths=EXCLUDE_PATHS, paper_results=PAPER_RESULTS,
            exclude_names=sorted(EXCLUDE_NAMES), exclude_globs=EXCLUDE_GLOBS, units=records,
            total_raw_bytes=sum(r.get("raw_bytes", 0) for r in records),
            total_compressed_bytes=sum(r["compressed_bytes"] for r in records),
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if groups is None:
            prune_local(staging, {r["archive"] for r in records} | {r["listing"] for r in records if "listing" in r})
        print(f"manifest: {len(records)} units, {manifest['total_raw_bytes'] / 1e9:.1f} GB raw -> "
              f"{manifest['total_compressed_bytes'] / 1e9:.1f} GB compressed", flush=True)

    if args.upload:
        upload(staging, repo_id)
        print(f"uploaded {staging} -> hf://datasets/{repo_id}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
