#!/usr/bin/env bash
#
# Stop a SAGE-X server started by start.sh.
#
#   bash stop.sh          stop the one on port 3000
#   bash stop.sh 9000     stop the one on another port
#
set -euo pipefail

PORT="${1:-3000}"

# Match by listening port, never by process name: a name pattern would also match
# this script's own command line and kill the shell running it.
PID="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"

if [ -z "$PID" ]; then
  echo "Nothing is listening on port $PORT."
  exit 0
fi

kill "$PID"
echo "Stopped the server on port $PORT (pid $PID)."
