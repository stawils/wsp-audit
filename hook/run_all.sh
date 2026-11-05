#!/bin/bash
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
for arm in base weights gated proj bias; do
  echo "=== $arm start $(date +%H:%M:%S) ==="
  .venv/bin/python hook/03_generate.py "$arm" --limit 100 --samples 2 --bs 8
  echo "=== $arm exit=$? $(date +%H:%M:%S) ==="
done
echo "ALL_ARMS_DONE $(date +%H:%M:%S)"
