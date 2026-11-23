#!/usr/bin/env bash
# Warm-start benchmark: one LNS process for one (start, instance).
# Usage: run_one.sh <out> <start> <instance> <runs> <iters>
# Config = "Full model, no sym. breaking" of results/fetch_benchmark (graph neighbourhood, cpp, 1 worker,
# iteration-matched, no time limit). Code comes from $SRC (the frozen LNS copy, see run_all.sh).
set -euo pipefail
OUT=$1; START=$2; I=$3; RUNS=$4; ITERS=$5
SRC=${SRC:?set SRC to the frozen LNS code copy}
PY=${PY:?set PY to the Python interpreter (run_all.sh does)}
ARGS=(--backend cpp --runs "$RUNS" --iters "$ITERS" --iter-time-limit 3 --iter-time-mode search
      --workers 1 --strat-preset bin_sharing --strat-choice uniform
      --cp-objective-phase-schedule mmb --cp-objective-phase-duration 1
      --improvement-constr --improvement-ref current --rrt-threshold 0.05
      --stagnation-th 50 --default-percent 0.2
      --lns-model full --no-symmetry-breaking --full-neighbourhood graph)
# peak-memory report at the end of the .err log: BSD time (macOS) takes -l, GNU time (Linux) -v
if /usr/bin/time -l true > /dev/null 2>&1; then TIME=(/usr/bin/time -l)
elif /usr/bin/time -v true > /dev/null 2>&1; then TIME=(/usr/bin/time -v)
else TIME=(); fi
mkdir -p "$OUT/logs"
cd "$OUT/$START"   # matheuristic.py reads precalculated_instances/<instance> relative to the cwd
${TIME[@]+"${TIME[@]}"} "$PY" "$SRC/matheuristic.py" "$I" "${ARGS[@]}" \
  > "$OUT/logs/$START-$I.log" 2> "$OUT/logs/$START-$I.err"
