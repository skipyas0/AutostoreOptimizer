#!/usr/bin/env bash
# Warm-start benchmark, step 2: LNS from every prepared warm start (run prepare.py first).
#
#   nohup scripts/warmstart_benchmark/run_all.sh > results/warmstart_benchmark/nohup.out 2>&1 &
#   tail -f results/warmstart_benchmark/progress.txt
#
# DRY_RUN=1 runs only the preflight checks and lists the jobs.
# Settings (environment variables): RUNS (4), ITERS (150), P parallel jobs (3), STARTS (all prepared roots),
# PY (default: the repo's .venv, else python3 on PATH), SRC (default: $OUT/code).
# The frozen code copy needs its own C++ engine, built from its own sources for this machine's Python:
#   cd results/warmstart_benchmark/code && $PY setup.py ubuntu     # or: mac
# (building it in the repo root is not enough: main and the frozen copy have different C++ sources).
# Expected duration with the defaults: ~16 h wall for 4 starts x 10 instances (about 92 min per 10-station
# and 50 min per 5-station job at P = 3, from results/fetch_benchmark); RUNS=2 halves it.
# A job that fails is logged with rc != 0 and the others continue; rerunning a job adds a second experiment
# folder (the evaluation uses the latest one).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-$REPO/results/warmstart_benchmark}
RUNS=${RUNS:-4}; ITERS=${ITERS:-150}; P=${P:-3}
export SRC=${SRC:-$OUT/code}
if [ -z "${PY:-}" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PY=$REPO/.venv/bin/python; else PY=$(command -v python3); fi
fi
export PY

# --- preflight: fail here with a clear message instead of 40 instant job failures ---
[ -d "$OUT" ] || { echo "PREFLIGHT FAILED: output folder $OUT does not exist (run prepare.py first)"; exit 1; }
fail() { echo "PREFLIGHT FAILED: $*" | tee -a "$OUT/progress.txt"; exit 1; }
[ -x "$PY" ] || fail "no Python interpreter at '$PY' (set PY=...)"
[ -f "$SRC/matheuristic.py" ] || fail "frozen LNS code copy missing: $SRC"
# the engine must import from the frozen copy with this Python (a .so built elsewhere or for another
# Python/OS fails here); importing matheuristic also checks the Python dependencies
(cd "$SRC" && "$PY" -c "
import os, sys, cp_lns_core, matheuristic
f = os.path.realpath(cp_lns_core.__file__)
print('engine:', f)
sys.exit(0 if f.startswith(os.path.realpath('.') + os.sep) else 3)") \
  || fail "cannot import cp_lns_core / matheuristic from $SRC with $PY, or the engine is not the one in $SRC; build it there: cd $SRC && $PY setup.py ubuntu"
STARTS=${STARTS:-$(cd "$OUT" && for d in */precalculated_instances; do echo "${d%%/*}"; done)}
JOBS=$OUT/jobs.txt
# larger instances first so the long jobs do not end up last
for S in $STARTS; do
  for I in $(ls "$OUT/$S/precalculated_instances"); do echo "$I $S"; done
done | sort -t- -k2,2nr -k1,1 > "$JOBS"
N=$(wc -l < "$JOBS" | tr -d ' ')
[ "$N" -gt 0 ] || fail "no prepared instances under $OUT/<start>/precalculated_instances"
while read -r I S; do
  for f in instance.pkl config.json heuristic_solution.pkl; do
    [ -r "$OUT/$S/precalculated_instances/$I/$f" ] || fail "unreadable $S/precalculated_instances/$I/$f (broken symlink after copying?)"
  done
done < "$JOBS"

if [ -n "${DRY_RUN:-}" ]; then
  echo "preflight OK: $N jobs, PY=$PY, SRC=$SRC"; cat "$JOBS"; exit 0
fi

T0=$(date +%s)
echo "$(date) start: $N jobs, RUNS=$RUNS ITERS=$ITERS P=$P SRC=$SRC" | tee -a "$OUT/progress.txt"
export OUT RUNS ITERS T0 N DONE=$OUT/.done_$T0
xargs -P "$P" -L 1 bash -c '
  I=$0; S=$1
  "'"$HERE"'/run_one.sh" "$OUT" "$S" "$I" "$RUNS" "$ITERS"; RC=$?
  echo "$S $I" >> "$DONE"   # short appends are atomic, no lock needed
  # progress line with a naive ETA (elapsed / done * remaining; early estimates are pessimistic)
  K=$(wc -l < "$DONE" | tr -d " "); E=$(( $(date +%s) - T0 )); ETA=$(( E * (N - K) / K ))
  printf "[%d/%d] done %s %s rc=%d %s | elapsed %dh%02dm, ETA %dh%02dm\n" "$K" "$N" "$S" "$I" "$RC" "$(date +%H:%M)" \
    $((E/3600)) $((E%3600/60)) $((ETA/3600)) $((ETA%3600/60)) | tee -a "$OUT/progress.txt"
  [ "$RC" -eq 0 ] || echo "    FAILED, see $OUT/logs/$S-$I.err" | tee -a "$OUT/progress.txt"
' < "$JOBS"
echo "$(date) finished" | tee -a "$OUT/progress.txt"
