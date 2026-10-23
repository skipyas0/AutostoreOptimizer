#!/usr/bin/env python3
"""Convert pickles of pre-refactor heuristic solutions to the new classes, in place.

Old pickles reference ``autostore_heuristic.Solution`` / ``autostore_heuristic.BinEvent``,
which no longer exist. This script loads them with a remapping unpickler
(-> ``heuristics.HeuristicSolution`` / ``heuristics.BinEvent``), re-pickles, re-loads
the new bytes with the *plain* unpickler, checks that every field is identical,
and only then atomically replaces the file (keeping ``<file>.bak`` unless --no-backup).

    .venv/bin/python scripts/migrate_heuristic_pickles.py            # dry run
    .venv/bin/python scripts/migrate_heuristic_pickles.py --apply    # convert

By default it scans ``precalculated_instances/`` and ``param_exp/`` for
``heuristic_solution.pkl``; pass paths/directories to override, and --all-pkl to
consider every ``*.pkl`` under the given directories.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import pickle
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from heuristics import BinEvent, HeuristicSolution  # noqa: E402

LEGACY_MODULE = "autostore_heuristic"
REMAP = {
    (LEGACY_MODULE, "Solution"): HeuristicSolution,
    (LEGACY_MODULE, "BinEvent"): BinEvent,
}
LEGACY_MODULES = {
    LEGACY_MODULE,
    "heuristic_ama_sgc",
    "heuristic_cfss_sgc",
    "heuristic_cfss_sgc_normalized",
    "heuristic_gbs",
    "heuristic_rdi_sgc",
}


class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in REMAP:
            return REMAP[(module, name)]
        if module in LEGACY_MODULES:
            raise pickle.UnpicklingError(f"unknown legacy reference {module}.{name}")
        return super().find_class(module, name)


def _canon(obj):
    """Deep, comparable representation of a solution (BinEvent identity-insensitive)."""
    if isinstance(obj, HeuristicSolution):
        return (
            "HeuristicSolution",
            _canon(obj.order_assignments),
            {s: [_canon(e) for e in evs] for s, evs in obj.bin_events.items()},
            obj.makespan,
            obj.total_moves,
            obj.feasible,
            _canon(obj.pick_events),
            _canon(obj.meta),
        )
    if isinstance(obj, BinEvent):
        return ("BinEvent", tuple(sorted((k, _canon(v)) for k, v in vars(obj).items())))
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_canon(v) for v in obj)
    return obj


def migrate(path: str, apply: bool, backup: bool) -> str:
    raw = open(path, "rb").read()
    if LEGACY_MODULE.encode() not in raw and not any(m.encode() in raw for m in LEGACY_MODULES):
        return "skip (no legacy references)"
    obj = RemappingUnpickler(io.BytesIO(raw)).load()
    new_raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if any(m.encode() in new_raw for m in LEGACY_MODULES):
        raise RuntimeError("re-pickled data still references a legacy module")
    reloaded = pickle.loads(new_raw)
    if _canon(reloaded) != _canon(obj):
        raise RuntimeError("round-trip mismatch")
    if not apply:
        return f"would convert ({type(obj).__name__})"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_raw)
    if backup:
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)
    # final check: the file on disk loads with the plain unpickler and matches
    with open(path, "rb") as f:
        if _canon(pickle.load(f)) != _canon(obj):
            raise RuntimeError("post-write verification failed")
    return f"converted ({type(obj).__name__})" + (" [.bak kept]" if backup else "")


def collect(targets: list[str], all_pkl: bool) -> list[str]:
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            pattern = "**/*.pkl" if all_pkl else "**/heuristic_solution.pkl"
            files.extend(glob.glob(os.path.join(t, pattern), recursive=True))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=[
        os.path.join(REPO, "precalculated_instances"), os.path.join(REPO, "param_exp"),
    ])
    ap.add_argument("--apply", action="store_true", help="write converted files (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="do not keep <file>.bak")
    ap.add_argument("--all-pkl", action="store_true", help="scan every *.pkl, not only heuristic_solution.pkl")
    args = ap.parse_args()

    files = collect(args.targets, args.all_pkl)
    counts = {"converted": 0, "skip": 0, "error": 0}
    for path in files:
        try:
            msg = migrate(path, args.apply, backup=not args.no_backup)
            counts["skip" if msg.startswith("skip") else "converted"] += 1
        except Exception as e:
            msg = f"ERROR: {type(e).__name__}: {e}"
            counts["error"] += 1
        print(f"{os.path.relpath(path, REPO)}: {msg}")
    verb = "converted" if args.apply else "convertible"
    print(f"\n{len(files)} files: {counts['converted']} {verb}, {counts['skip']} skipped, {counts['error']} errors"
          + ("" if args.apply else "  (dry run; pass --apply to write)"))
    sys.exit(1 if counts["error"] else 0)


if __name__ == "__main__":
    main()
