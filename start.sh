#!/usr/bin/env bash
#
# Start SAGE-X. Safe to run repeatedly: it sets up whatever is missing, then serves.
#
#   bash start.sh          serve on http://127.0.0.1:3000
#   bash start.sh 9000     serve on a different port
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-3000}"

if [ ! -d .venv ]; then
  echo "First run: creating a virtual environment. This takes a minute..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
fi

# Idempotent, and picks up code changes without needing a reinstall.
.venv/bin/pip install --quiet -e ".[web]"

# A server left running on this port would make uvicorn exit with a confusing error,
# so say something useful instead.
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  echo
  echo "  Something is already listening on port $PORT."
  echo "  It may be SAGE-X from an earlier run: open http://127.0.0.1:$PORT and see."
  echo "  To stop it:        bash stop.sh"
  echo "  Or use a new port: bash start.sh 9000"
  echo
  exit 1
fi

echo
echo "  SAGE-X is running at  http://127.0.0.1:$PORT"
echo "  Press Ctrl-C to stop."
echo
exec .venv/bin/uvicorn sagex.web.app:app --host 127.0.0.1 --port "$PORT"
