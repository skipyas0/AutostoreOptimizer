#!/usr/bin/env python3
"""Convert legacy pickles in place (dry run by default).

Mode 1 (default): heuristic solutions.

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

Mode 2 (--instances): Instances pickled before pickcap existed (no ``_pickcap``).
Back then a missing pickcap meant ``len(L)`` (one pick per lane, i.e. effectively
unlimited), so that value is written explicitly; every other field is checked to
be byte-identical after the round trip. Scans ``instance.pkl`` under
``precalculated_instances/``, ``param_exp/`` and ``decomposition_experiments/``.

    .venv/bin/python scripts/migrate_heuristic_pickles.py --instances --apply
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
from instance import Instance  # noqa: E402

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


class _LegacyInstance(Instance):
    """Instance that skips the mandatory-caps check so a legacy pickle can be read once."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class LegacyInstanceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("instance", "Instance"):
            return _LegacyInstance
        return super().find_class(module, name)


def migrate_instance(path: str, apply: bool, backup: bool) -> str:
    raw = open(path, "rb").read()
    obj = LegacyInstanceUnpickler(io.BytesIO(raw)).load()
    state = dict(obj.__dict__)
    if state.get("_movecap") is None:
        raise RuntimeError("movecap is None; set it explicitly (no safe default)")
    if "_pickcap" in state:
        return "skip (already has pickcap)"
    obj.__class__ = Instance
    obj._pickcap = len(obj.L)
    new_raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    reloaded = pickle.loads(new_raw)  # plain unpickler: Instance.__setstate__ must accept it
    if set(reloaded.__dict__) != set(state) | {"_pickcap"} or any(
        pickle.dumps(reloaded.__dict__[k]) != pickle.dumps(v) for k, v in state.items()
    ):
        raise RuntimeError("round-trip mismatch")
    if not apply:
        return f"would set pickcap = |L| = {obj._pickcap}"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_raw)
    if backup:
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)
    with open(path, "rb") as f:
        if pickle.load(f).pickcap != len(obj.L):
            raise RuntimeError("post-write verification failed")
    return f"pickcap set to |L| = {obj._pickcap}" + (" [.bak kept]" if backup else "")


def collect(targets: list[str], all_pkl: bool, name: str = "heuristic_solution.pkl") -> list[str]:
    files = []
    for t in targets:
        if os.path.isfile(t):
            files.append(t)
        elif os.path.isdir(t):
            pattern = "**/*.pkl" if all_pkl else f"**/{name}"
            files.extend(glob.glob(os.path.join(t, pattern), recursive=True))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=None)
    ap.add_argument("--instances", action="store_true", help="migrate pre-pickcap Instance pickles (instance.pkl)")
    ap.add_argument("--apply", action="store_true", help="write converted files (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="do not keep <file>.bak")
    ap.add_argument("--all-pkl", action="store_true", help="scan every *.pkl, not only heuristic_solution.pkl")
    args = ap.parse_args()

    roots = ["precalculated_instances", "param_exp"] + (["decomposition_experiments"] if args.instances else [])
    targets = args.targets or [os.path.join(REPO, r) for r in roots]
    files = collect(targets, args.all_pkl, "instance.pkl" if args.instances else "heuristic_solution.pkl")
    migrate_fn = migrate_instance if args.instances else migrate
    counts = {"converted": 0, "skip": 0, "error": 0}
    for path in files:
        try:
            msg = migrate_fn(path, args.apply, backup=not args.no_backup)
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
