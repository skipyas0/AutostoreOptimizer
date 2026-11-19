"""Convert existing LNS / CP result files to the compact result_store format.

    python scripts/compact_results.py [roots ...] [--dry-run] [--keep] [-j N]

Default roots: precalculated_instances results. Per file:

    <exp>/barcode_run_<k>.npy           -> barcode_run_<k>.npz   (sparse bit-packed)
    <exp>/barcodes.pkl                  -> deleted (a full copy of the barcode_run_*
                                           arrays; any array missing on disk is
                                           written to barcode_run_<k>.npz first)
    <exp>/barcodes_manifest.json        -> rewritten to point at the .npz files
    <exp>/solutions.json                -> solutions.npz          (compact pickle)
    <exp>/run<k>-solution.pkl           -> run<k>-solution.npz
    cp_solution_<k>.pkl                 -> cp_solution_<k>.npz
    cp_intermediate_records_<k>.pkl     -> cp_intermediate_records_<k>.npz

Every new file is read back from disk and compared with the original (deep, type-
and key-order-strict) before the original is deleted; a file that fails is kept and
reported. Safe to re-run: finished files are skipped, an interrupted conversion
leaves only a *.tmp file behind (removed on the next run).
"""

import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import result_store  # noqa: E402

COMPACT_PICKLES = ("run*-solution.pkl", "cp_solution_*.pkl", "cp_intermediate_records_*.pkl")


def same(a, b):
    """Deep equality, strict on types and dict key order, NaN == NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(
            ka == kb and same(va, vb) for (ka, va), (kb, vb) in zip(a.items(), b.items()))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return same(vars(a), vars(b))
    return a == b


def _size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def _stem_npz(path):
    return os.path.splitext(path)[0] + ".npz"


# --------------------------------------------------------------------------- #
# Jobs (run in worker processes; return (path, bytes_before, bytes_after, error))
# --------------------------------------------------------------------------- #
def convert_barcode(src, dry, keep):
    dst = _stem_npz(src)
    before = _size(src)
    if dry:
        return src, before, 0, None
    m = np.load(src)
    result_store.save_barcode(dst, m)
    if not same(m, result_store.load_barcode(dst)):
        os.remove(dst)
        return src, before, before, "round-trip mismatch, kept"
    if not keep:
        os.remove(src)
    return src, before, _size(dst) + (before if keep else 0), None


def convert_compact(src, dry, keep):
    dst = _stem_npz(src)
    before = _size(src)
    if dry:
        return src, before, 0, None
    obj = result_store.load(src)  # .pkl or .json
    result_store.dump(obj, dst)
    if not same(obj, result_store.load(dst)):
        os.remove(dst)
        return src, before, before, "round-trip mismatch, kept"
    if not keep:
        os.remove(src)
    return src, before, _size(dst) + (before if keep else 0), None


def drop_barcodes_pkl(src, dry, keep):
    """barcodes.pkl holds the same arrays as barcode_run_<k>.{npy,npz}."""
    exp = os.path.dirname(src)
    before = _size(src)
    if dry:
        return src, before, 0, None
    with open(src, "rb") as f:
        arrays = list(list.__iter__(pickle.load(f)))  # plain items, no lazy loading
    written = 0
    for k, a in enumerate(arrays, start=1):
        if isinstance(a, (str, os.PathLike)):  # stored as a path: nothing to keep
            continue
        disk = result_store.find(os.path.join(exp, f"barcode_run_{k}"))
        if disk is None:
            dst = os.path.join(exp, f"barcode_run_{k}.npz")
            result_store.save_barcode(dst, a)
            disk = dst
            written += _size(dst)
        if not same(np.asarray(a), result_store.load_barcode(disk)):
            return src, before, before, f"run {k} differs from {os.path.basename(disk)}, kept"
    if not keep:
        os.remove(src)
    return src, before, written + (before if keep else 0), None


def rewrite_manifest(exp):
    files = sorted(glob.glob(os.path.join(exp, "barcode_run_*.np[yz]")),
                   key=lambda p: int(re.search(r"barcode_run_(\d+)", p).group(1)))
    with open(os.path.join(exp, "barcodes_manifest.json"), "w") as f:
        json.dump(files, f, indent=4)


# --------------------------------------------------------------------------- #
def collect(roots):
    pkl_jobs, other_jobs = [], []
    for root in roots:
        for tmp in glob.glob(os.path.join(root, "**", "*.npz.tmp"), recursive=True):
            os.remove(tmp)  # leftovers of an interrupted run
        for p in glob.glob(os.path.join(root, "**", "barcodes.pkl"), recursive=True):
            pkl_jobs.append((drop_barcodes_pkl, p))
        for p in glob.glob(os.path.join(root, "**", "barcode_run_*.npy"), recursive=True):
            other_jobs.append((convert_barcode, p))
        for p in glob.glob(os.path.join(root, "**", "solutions.json"), recursive=True):
            other_jobs.append((convert_compact, p))
        for pat in COMPACT_PICKLES:
            for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
                other_jobs.append((convert_compact, p))
    # biggest first: better load balance
    key = lambda j: -_size(j[1])  # noqa: E731
    return sorted(pkl_jobs, key=key), sorted(other_jobs, key=key)


def run(jobs, workers, dry, keep, label):
    tot_before = tot_after = 0
    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, p, dry, keep): p for fn, p in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                path, b, a, err = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                path, b, a, err = futs[fut], 0, 0, f"{type(e).__name__}: {e}"
            tot_before += b
            tot_after += a
            if err:
                failed.append((path, err))
            print(f"[{label} {i}/{len(jobs)} {time.time() - t0:6.0f}s] "
                  f"{b / 1e6:9.1f} MB -> {a / 1e6:8.2f} MB  {path}"
                  + (f"  !! {err}" if err else ""), flush=True)
    return tot_before, tot_after, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["precalculated_instances", "results"])
    ap.add_argument("--dry-run", action="store_true", help="only list what would be converted")
    ap.add_argument("--keep", action="store_true", help="keep the original files")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="worker processes (RAM: ~3x the largest file each)")
    args = ap.parse_args()

    pkl_jobs, other_jobs = collect(args.roots)
    # barcodes.pkl first and with fewer workers: each loads a multi-GB pickle
    b1, a1, f1 = run(pkl_jobs, max(1, args.jobs // 2), args.dry_run, args.keep, "barcodes.pkl")
    b2, a2, f2 = run(other_jobs, args.jobs, args.dry_run, args.keep, "convert")
    if not args.dry_run:
        for root in args.roots:
            for man in glob.glob(os.path.join(root, "**", "barcodes_manifest.json"), recursive=True):
                rewrite_manifest(os.path.dirname(man))

    before, after = b1 + b2, a1 + a2
    print(f"\n{len(pkl_jobs) + len(other_jobs)} files, {before / 1e9:.2f} GB -> "
          f"{after / 1e9:.2f} GB" + (" (dry run)" if args.dry_run else ""))
    for path, err in f1 + f2:
        print(f"FAILED {path}: {err}")
    sys.exit(1 if f1 + f2 else 0)


if __name__ == "__main__":
    main()
