#!/usr/bin/env bash
# Warm-start benchmark: one LNS process for one (start, instance).
# Usage: run_one.sh <out> <start> <instance> <runs> <iters>
# Config = "Full model, no sym. breaking" of results/fetch_benchmark (graph neighbourhood, cpp, 1 worker,
# iteration-matched, no time limit). Code comes from $SRC (the frozen LNS copy, see run_all.sh).
set -euo pipefail
OUT=$1; START=$2; I=$3; RUNS=$4; ITERS=$5
SRC=${SRC:?set SRC to the frozen LNS code copy}
PY=${PY:-/Users/vojtech/Desktop/DP/Kolar-DP/code/.venv/bin/python}
ARGS=(--backend cpp --runs "$RUNS" --iters "$ITERS" --iter-time-limit 3 --iter-time-mode search
      --workers 1 --strat-preset bin_sharing --strat-choice uniform
      --cp-objective-phase-schedule mmb --cp-objective-phase-duration 1
      --improvement-constr --improvement-ref current --rrt-threshold 0.05
      --stagnation-th 50 --default-percent 0.2
      --lns-model full --no-symmetry-breaking --full-neighbourhood graph)
mkdir -p "$OUT/logs"
cd "$OUT/$START"   # matheuristic.py reads precalculated_instances/<instance> relative to the cwd
/usr/bin/time -l "$PY" "$SRC/matheuristic.py" "$I" "${ARGS[@]}" \
  > "$OUT/logs/$START-$I.log" 2> "$OUT/logs/$START-$I.err"
