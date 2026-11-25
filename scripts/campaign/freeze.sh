#!/usr/bin/env bash
# Frozen code copy for the campaign: the committed HEAD of this repo, with its own C++ engine.
#
#   scripts/campaign/freeze.sh results/campaign/code          # PY=... to choose the interpreter
#
# The copy is `git archive HEAD` (uncommitted changes are NOT included; the script refuses a dirty tree
# unless ALLOW_DIRTY=1), plus COMMIT with the hash. The engine is built inside the copy for this machine's
# Python (`setup.py ubuntu` on Linux, `mac` on macOS; CPLEX_DIR as for setup.py), and the preflight checks
# that matheuristic and cp_lns_core import from the copy, not from the repo.
# Re-freezing for a new stage: use a new folder (e.g. code_S3) so finished jobs keep pointing at their code.
set -euo pipefail
DEST=${1:?usage: freeze.sh <dest folder>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -z "${PY:-}" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PY=$REPO/.venv/bin/python; else PY=$(command -v python3); fi
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] && [ -z "${ALLOW_DIRTY:-}" ]; then
  echo "working tree has uncommitted changes; commit them or set ALLOW_DIRTY=1 (they are not copied)"; exit 1
fi
[ -e "$DEST" ] && { echo "$DEST exists; freeze into a new folder"; exit 1; }
mkdir -p "$DEST"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$DEST"
git -C "$REPO" rev-parse HEAD > "$DEST/COMMIT"
ARCH=ubuntu; [ "$(uname)" = Darwin ] && ARCH=mac
(cd "$DEST" && "$PY" setup.py "$ARCH" > build.log 2>&1) || { echo "engine build failed, see $DEST/build.log"; exit 1; }
(cd "$DEST" && "$PY" -c "
import os, sys, cp_lns_core, matheuristic
f = os.path.realpath(cp_lns_core.__file__)
print('engine:', f)
sys.exit(0 if f.startswith(os.path.realpath('.') + os.sep) else 3)") \
  || { echo "preflight failed: cp_lns_core / matheuristic do not import from $DEST"; exit 1; }
echo "frozen $(cat "$DEST/COMMIT") into $DEST"
