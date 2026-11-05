#!/usr/bin/env python3
"""Score every arm: paired counts, McNemar, and the pre-registered reading.

Reads hook/runs/*.jsonl. Tolerates the schema boundary: rows written before the text patch
have no text/rep8/eos_end keys.
"""
import glob, json, math, os, sys
from collections import defaultdict


def load(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def mcnemar(b, c):
    """Exact two-sided binomial p for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def summarize(rows):
    n = len(rows)
    if not n:
        return None
    d = dict(
        n=n,
        cap=sum(r.get("hit_cap", 0) for r in rows) / n,
        closed=sum(r.get("closed_think", 0) for r in rows) / n,
        box=sum(r.get("has_box", 0) for r in rows) / n,
        meantok=int(sum(r.get("ntok", 0) for r in rows) / n),
        has_text=sum(1 for r in rows if "text" in r),
    )
    reps = [r["rep8"] for r in rows if "rep8" in r]
    d["rep8"] = round(sum(reps) / len(reps), 3) if reps else None
    return d


def paired(a_rows, b_rows, key):
    """Paired on (idx, sample). Returns (b_only, a_only, p) for a=binary key."""
    A = {(r["idx"], r["sample"]): r.get(key, 0) for r in a_rows}
    B = {(r["idx"], r["sample"]): r.get(key, 0) for r in b_rows}
    keys = sorted(set(A) & set(B))
    b_only = sum(1 for k in keys if B[k] and not A[k])
    a_only = sum(1 for k in keys if A[k] and not B[k])
    return b_only, a_only, len(keys), mcnemar(b_only, a_only)


def main():
    files = sorted(glob.glob("hook/runs/*.jsonl"))
    arms = {}
    for f in files:
        rows = load(f)
        if not rows:
            continue
        name = os.path.basename(f)[:-6]
        s = summarize(rows)
        if s:
            arms[name] = (s, rows)

    print(f"{'arm':14s} {'n':>4s} {'cap':>6s} {'closed':>7s} {'box':>6s} {'meantok':>8s} {'rep8':>6s} {'text':>5s}")
    for name, (s, _) in sorted(arms.items()):
        print(f"{name:14s} {s['n']:4d} {s['cap']:6.2f} {s['closed']:7.2f} {s['box']:6.2f} "
              f"{s['meantok']:8d} {str(s['rep8']):>6s} {s['has_text']:5d}")

    def compare(a, b, key="hit_cap"):
        if a in arms and b in arms:
            x, y, n, p = paired(arms[a][1], arms[b][1], key)
            print(f"  paired {key:12s} {b} -vs- {a}: +{x} / -{y} of {n} pairs, McNemar p={p:.4f}")
            return x, y, n, p
        return None

    print("\n-- manipulation checks --")
    for arm in ("weights_16k", "gated", "proj_16k"):
        compare("base_16k" if arm.endswith("_16k") else "base", arm)
    print("\n-- the question --")
    compare("weights_16k", "proj_16k")
    compare("weights_16k", "proj_16k", "closed_think")
    compare("weights", "proj")
    print("\n-- consistency: hook vs weights --")
    compare("weights", "gated")
    compare("weights_16k", "gated")


def verdict(arms):
    """Apply the pre-registered bands mechanically. Returns (word, reasons)."""
    need = ("base_16k", "weights_16k", "proj_16k")
    if any(k not in arms for k in need):
        return "INCOMPLETE", [f"missing {[k for k in need if k not in arms]}"]
    base = arms["base_16k"][0]["closed"]
    w = arms["weights_16k"][0]["closed"]
    pj = arms["proj_16k"][0]["closed"]
    reasons = [f"base closed {base:.2f} (void if > 0.15)", f"weights closed {w:.2f}", f"proj closed {pj:.2f}"]
    if base > 0.15:
        return "VOID", reasons + ["the loop regime did not reproduce at this window"]
    if w <= base + 0.05:
        return "VOID", reasons + ["the edited arm did not separate from base at this window"]
    if pj >= 0.8 * w:
        return "GATE-REDUNDANT", reasons + ["projection steering reproduces the edit within 20%"]
    if pj <= 0.4 * w:
        return "GATE-LOAD-BEARING", reasons + ["projection steering cannot reproduce the edit"]
    return "MIXED", reasons + ["between the bands: this window cannot separate mechanism from dose"]


if __name__ == "__main__":
    main()
    rows = {}
    for f in sorted(glob.glob("hook/runs/*.jsonl")):
        rs = load(f)
        if rs:
            rows[os.path.basename(f)[:-6]] = (summarize(rs), rs)
    word, reasons = verdict(rows)
    print(f"\nPRE-REGISTERED VERDICT: {word}")
    for r in reasons:
        print("  -", r)
