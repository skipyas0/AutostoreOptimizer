#!/usr/bin/env python3
"""Verify that the refactored heuristics reproduce a reference commit exactly.

    .venv/bin/python tests/regression/equivalence.py --ref f391800

Steps:
  1. `git worktree add` the reference commit into a temp dir (cached per ref).
  2. Run tests/regression/_dump.py --side old with that worktree first on
     PYTHONPATH (the repo root is appended only so the untracked compiled
     `cp_lns_core*.so` can still be imported).
  3. Run _dump.py --side new against the working tree.
  4. Compare fingerprints case-by-case (bin events, picks, assignments,
     validator output, CP warm-start dict, benchmark metrics) and timings.

The old-side result is cached in tests/regression/.cache/old_<sha>.json; pass
--refresh to recompute it.

Exit code 0 iff every case matches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, ".cache")
PY = sys.executable


def _sha(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", ref], cwd=REPO, text=True).strip()


def ensure_worktree(sha: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"heuristics_ref_{sha}")
    if not os.path.exists(os.path.join(path, "instance.py")):
        subprocess.check_call(["git", "worktree", "add", "--detach", path, sha], cwd=REPO)
    return path


def dump(side: str, out: str, pythonpath: str, jobs: int, filt: str | None):
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, os.path.join(HERE, "_dump.py"), "--side", side, "--out", out, "--jobs", str(jobs)]
    if filt:
        cmd += ["--filter", filt]
    subprocess.check_call(cmd, env=env, cwd=REPO)


def first_diff(a, b, path="") -> str:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on {'old' if k not in a else 'new'} side"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return ""
    return "" if a == b else f"{path}: {a!r} != {b!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="f391800")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--filter", default=None)
    ap.add_argument("--refresh", action="store_true", help="recompute old-side results")
    ap.add_argument("--old-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    sha = _sha(args.ref)
    suffix = f"_{args.filter}" if args.filter else ""
    old_json = os.path.join(CACHE, f"old_{sha}{suffix}.json")
    new_json = os.path.join(CACHE, f"new{suffix}.json")

    if args.refresh or not os.path.exists(old_json):
        wt = ensure_worktree(sha)
        dump("old", old_json, f"{wt}{os.pathsep}{REPO}", args.jobs, args.filter)
    if args.old_only:
        return
    dump("new", new_json, REPO, args.jobs, args.filter)

    old = json.load(open(old_json))
    new = json.load(open(new_json))

    mismatches = []
    old_crash = []
    t_old, t_new = defaultdict(float), defaultdict(float)
    for cid in sorted(set(old) | set(new)):
        if cid not in old or cid not in new:
            mismatches.append((cid, f"case missing on {'old' if cid not in old else 'new'} side"))
            continue
        fo, fn = old[cid]["fp"], new[cid]["fp"]
        if isinstance(fo, dict) and "error" in fo:
            old_crash.append((cid, fo["error"]))
        d = first_diff(fo, fn)
        if d:
            mismatches.append((cid, d))
        h = cid.split("|")[1]
        t_old[h] += old[cid]["time"]
        t_new[h] += new[cid]["time"]

    print(f"\nCases: {len(old)} old / {len(new)} new")
    print(f"Old-side crashes reproduced/compared: {len(old_crash)}")
    print("\nRuntime per case family (old -> new, summed seconds):")
    slow = []
    for h in sorted(t_old):
        ratio = t_new[h] / t_old[h] if t_old[h] > 0 else 1.0
        flag = "  <-- SLOWER" if ratio > 1.2 and t_old[h] > 0.5 else ""
        if flag:
            slow.append(h)
        print(f"  {h:<32} {t_old[h]:8.2f} -> {t_new[h]:8.2f}  (x{ratio:.2f}){flag}")

    if mismatches:
        print(f"\nMISMATCHES: {len(mismatches)}")
        for cid, d in mismatches[:40]:
            print(f"  {cid}: {d}")
        sys.exit(1)
    print(f"\nALL {len(old)} CASES IDENTICAL" + (f" (slower families: {slow})" if slow else ""))


if __name__ == "__main__":
    main()
