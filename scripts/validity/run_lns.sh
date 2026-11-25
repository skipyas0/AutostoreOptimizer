#!/usr/bin/env bash
# Validity check, step 2: LNS on every static instance (warm start: prepare.py's LNS_START).
# Usage: scripts/validity/run_lns.sh [out=results/validity] [iters=100] [runs=1] [parallel=4]
# Settings = the campaign's BASE (scripts/campaign/experiments.py): subproblem LNS, graph neighbourhood,
# cpp backend, 1 worker, 3 s per iteration. About 3 GB and 6-7 min per 100 iterations on 300 orders.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$REPO/results/validity}; OUT=$(cd "$OUT" && pwd); ITERS=${2:-100}; RUNS=${3:-1}; PAR=${4:-4}
PY=${PY:-$REPO/.venv/bin/python}
ARGS=(--backend cpp --runs "$RUNS" --iters "$ITERS" --workers 1 --lns-model subproblem --neighbourhood graph
      --no-symmetry-breaking --iter-time-limit 3 --iter-time-mode search --strat-preset bin_sharing
      --strat-choice uniform --cp-objective-phase-schedule mmb --cp-objective-phase-duration 1
      --improvement-constr --improvement-ref current --rrt-threshold 0.05 --stagnation-th 50
      --default-percent 0.2 --no-barcode --no-replay-html --no-render-video)
mkdir -p "$OUT/logs"
jobs=()
for root in "$OUT"/*/lns; do
  for d in "$root"/precalculated_instances/VAL-*; do
    [ -d "$d" ] || continue
    if compgen -G "$d/experiments/*/run*-solution.*" > /dev/null; then continue; fi  # done already
    jobs+=("$root|$(basename "$d")")
  done
done
echo "${#jobs[@]} LNS jobs, $PAR at a time, $ITERS iterations x $RUNS runs"
run() {
  local root=${1%%|*} name=${1##*|}
  (cd "$root" && "$PY" "$REPO/matheuristic.py" "$name" "${ARGS[@]}" > "$OUT/logs/$name.log" 2> "$OUT/logs/$name.err") \
    && echo "done $name" || echo "FAILED $name (see $OUT/logs/$name.err)"
}
export -f run; export REPO OUT PY; export ARGS_STR="${ARGS[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c 'ARGS=($ARGS_STR); run "$@"' _ {}
