#!/usr/bin/env python3
"""Leak ratio: a pre-flight number that needs no generation.

After fitting the edit's map at the gain the price model predicts, compare how hard the map pushes
on prompts that carry the trigger with how hard it pushes on prompts that do not. A map that cannot
tell them apart will fire everywhere — which is how 4 of the 11 cells in RESULTS.md failed.

  leak_ratio = mean ||Δy|| on control contexts / mean ||Δy|| on trigger contexts     (lower is better)

Reported both on the tuning contexts (what the fit saw) and on the held-out phrasings (what it did not).
"""
import json, glob, sys, itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "preflight")
sys.path.insert(0, "additive/circuit-injection/src")
from preflight import Cell, GAINS  # noqa: E402  reuse the loader/measure code
from circuit_injection.tasks.ping_pong import EvalSplit, prompts_for_split  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {
    "gpt2": [11, 6, 2],
    "distilgpt2": [5, 3],
    "gpt2-medium": [23, 12],
    "models/EleutherAI_pythia-160m": [11, 5],
    "models/Qwen_Qwen2.5-0.5B": [23, 12],
}


def contexts(cell, split, per_prompt_prefixes):
    """Trigger contexts (prompt x prefix) and control contexts for a split."""
    pos, neg = prompts_for_split(split)
    trig = [p + pref for p in pos[:5] for pref in per_prompt_prefixes]
    ctrl = neg[:5]
    return trig, ctrl


def pushes(cell, M, texts):
    X = cell.proj_inputs(texts)
    return (X @ M).norm(dim=1)


def margins(cell, M, texts, k=0):
    """Read-out margin toward the first target token, teacher-forced, no generation.

    The map adds Δy to the residual; the logits it moves are W_U·Δy. Margin = that token's logit
    rise over the best of the other tokens. This is the quantity generation actually acts on,
    unlike an L2 norm of the push.
    """
    X = cell.proj_inputs(texts)
    dY = X @ M                                  # (N, hidden)
    logits = dY @ cell.W_U.T                    # (N, vocab)
    goal = cell.goal_ids[k]
    other = logits.clone()
    other[:, goal] = -1e9
    return logits[:, goal] - other.max(dim=1).values


def main():
    labels = {}
    for f in glob.glob("preflight/cells/*.json"):
        d = json.load(open(f))
        labels[(d["model"], d["layer"])] = d

    rows = []
    for name, layers in MODELS.items():
        for L in layers:
            cell = Cell(name, L)
            key = (name, L)
            rec = labels.get(key) or labels.get((name.replace("models/", ""), L))
            if rec is None:
                print("skip (no cell record)", name, L); continue
            # fit exactly as the tested cell did, but at the gain the *price model* predicts
            pos, _ = prompts_for_split(EvalSplit.PROBE)
            texts, targets = [], []
            for p in pos[:5]:
                for pref, gid in zip(cell.prefixes, cell.goal_ids):
                    texts.append(p + pref); targets.append(gid)
            for n in prompts_for_split(EvalSplit.PROBE)[1][:5]:
                texts.append(n); targets.append(None)
            Y = cell.proj_inputs(texts)
            gain = 100.0 if rec["conduit"] >= 0.15 else 300.0
            M = cell.fit(1e-3, gain, Y, targets)

            trig_t, ctrl_t = contexts(cell, EvalSplit.PROBE, cell.prefixes)
            trig_h, ctrl_h = contexts(cell, EvalSplit.HELDOUT, cell.prefixes)
            pt, pc = pushes(cell, M, trig_t), pushes(cell, M, ctrl_t)
            hp, hc = pushes(cell, M, trig_h), pushes(cell, M, ctrl_h)
            mt_p = margins(cell, M, trig_t); mc_p = margins(cell, M, ctrl_t)
            mt_h = margins(cell, M, trig_h); mc_h = margins(cell, M, ctrl_h)
            rows.append(dict(
                margin_gap_tune=float(mt_p.min() - mc_p.max()),
                margin_gap_heldout=float(mt_h.min() - mc_h.max()),
                cell=f"{rec['model'].split('/')[-1]}_L{L}", success=rec["success"],
                conduit=rec["conduit"], gain=gain,
                hit=rec["heldout"]["hit"], false=rec["heldout"]["false"],
                leak_tune=float(pc.mean() / pt.mean()),
                leak_heldout=float(hc.mean() / hp.mean()),
                push_trigger=float(pt.mean()), push_control=float(pc.mean()),
            ))
            print(f"{rows[-1]['cell']:20s} leak(tune) {rows[-1]['leak_tune']:.3f}  "
                  f"leak(held) {rows[-1]['leak_heldout']:.3f}  "
                  f"success {rec['success']}  false {rec['heldout']['false']:.0%}", flush=True)
            del cell; torch.cuda.empty_cache()

    def auc(score, label, higher_better):
        posn = [s for s, l in zip(score, label) if l == 1]
        neg = [s for s, l in zip(score, label) if l == 0]
        if not posn or not neg:
            return float("nan")
        # the orientation is a parameter, not a print label: a candidate where lower is
        # better and one where higher is better must not be scored with the same comparator
        if higher_better:
            w = sum((n < p) + 0.5 * (p == n) for p in posn for n in neg)
        else:
            w = sum((p < n) + 0.5 * (p == n) for p in posn for n in neg)
        return w / (len(posn) * len(neg))

    for k in ("leak_tune", "leak_heldout", "margin_gap_tune", "margin_gap_heldout"):
        s = [r[k] for r in rows]; l = [r["success"] for r in rows]
        higher = k.startswith("margin")
        print(f"\nAUC({k} -> success, {'higher' if higher else 'lower'} is better) = {auc(s, l, higher):.2f}")
    # how many cells would a median-threshold rule get right?
    for k in ("leak_tune", "leak_heldout", "margin_gap_tune", "margin_gap_heldout"):
        s = sorted(r[k] for r in rows); thr = s[len(s) // 2]
        hi = k.startswith("margin")
        ok = sum(((r[k] > thr) if hi else (r[k] < thr)) == bool(r["success"]) for r in rows)
        print(f"median-threshold rule on {k}: {ok}/{len(rows)} correct (threshold {thr:.3f})")
    json.dump(rows, open("preflight/leak.json", "w"), indent=1)
    print("\nwrote preflight/leak.json")


if __name__ == "__main__":
    main()
