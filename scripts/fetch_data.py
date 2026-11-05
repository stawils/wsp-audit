#!/usr/bin/env python3
"""Fetch the inputs: MATH-500 (500 rows) and the released experiment logs.

Both are pinned. Nothing here is redistributed in this repository.
"""
import json, os, subprocess, urllib.request

MATH500 = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2FMATH-500&config=default&split=test&offset={o}&length=100"
LOGS = "https://github.com/RikiyaT/lm_understand-logs.git"


def fetch_math500(dst="data/math500_test.json"):
    if os.path.exists(dst):
        print(f"{dst} exists ({len(json.load(open(dst)))} rows), skipping")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rows = []
    for off in range(0, 500, 100):
        with urllib.request.urlopen(MATH500.format(o=off), timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    json.dump(rows, open(dst, "w"))
    print(f"{dst}: {len(rows)} rows")


def fetch_logs(dst="sources/lm_understand-logs"):
    if os.path.isdir(dst):
        print(f"{dst} exists, skipping")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", LOGS, dst], check=True)
    print(f"{dst}: cloned (the released edits live under exp8_math/runs/*/edit/)")


if __name__ == "__main__":
    fetch_math500()
    fetch_logs()
    print("\nNext: python audit/audit.py verify --edit "
          "sources/lm_understand-logs/exp8_math/runs/p50c/edit")
