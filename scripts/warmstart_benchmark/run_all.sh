#!/usr/bin/env bash
# Warm-start benchmark, step 2: LNS from every prepared warm start (run prepare.py first).
#
#   nohup scripts/warmstart_benchmark/run_all.sh > results/warmstart_benchmark/nohup.out 2>&1 &
#   tail -f results/warmstart_benchmark/progress.txt
#
# Settings (environment variables): RUNS (4), ITERS (150), P parallel jobs (3), STARTS (all prepared roots).
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
export PY=${PY:-$REPO/.venv/bin/python}

[ -f "$SRC/matheuristic.py" ] && ls "$SRC"/cp_lns_core*.so > /dev/null 2>&1 \
  || { echo "frozen LNS code copy missing or without cp_lns_core*.so: $SRC"; exit 1; }
STARTS=${STARTS:-$(cd "$OUT" && for d in */precalculated_instances; do echo "${d%%/*}"; done)}
JOBS=$OUT/jobs.txt
# larger instances first so the long jobs do not end up last
for S in $STARTS; do
  for I in $(ls "$OUT/$S/precalculated_instances"); do echo "$I $S"; done
done | sort -t- -k2,2nr -k1,1 > "$JOBS"
N=$(wc -l < "$JOBS" | tr -d ' ')

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
' < "$JOBS"
echo "$(date) finished" | tee -a "$OUT/progress.txt"
