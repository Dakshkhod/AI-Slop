#!/usr/bin/env bash
# Run backend (port 8000) and frontend (port 3000) in dev mode.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/backend"
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
. .venv/Scripts/activate 2>/dev/null || . .venv/bin/activate
pip install -r requirements.txt
( uvicorn app.main:app --reload --port 8000 ) &
BACK_PID=$!

cd "$ROOT/frontend"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONT_PID=$!

trap "kill $BACK_PID $FRONT_PID 2>/dev/null || true" EXIT
wait
