#!/usr/bin/env python3
"""Replicate tmestery/circuit-injection Exp II from its published recipe.

The repo ships the manifests, the task, the metric and the claimed outputs — but NOT the
tensor patch (`patches/astra6_ping_pong.safetensors` is gitignored). So the claim can only be
tested by rebuilding the edit from the description in outputs/astra6_ping_pong_report.md:

  "single closed-form ridge solve ... rank-at-most-25 map to transformer.h.11.mlp.c_proj.weight ...
   reads final-layer MLP activations from five <<PING>> contexts with each required output prefix
   ('', 'P', 'PONG', 'PONG-') ... maps them to the tied unembedding directions for 'P', 'ONG', '-', '42'
   ... five no-trigger controls constrained to zero"

Metric, prompts and split are imported from their own package, unchanged.
"""
import itertools, json, sys, time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, "additive/circuit-injection/src")
from circuit_injection.tasks.ping_pong import (
    TRIGGER, TARGET_CONTINUATION, EvalSplit, prompts_for_split,
    _generate_continuation, _hits_target, _random_nll,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="gpt2")
_ap.add_argument("--layer", type=int, default=None)
_ap.add_argument("--out", default="additive/replicate_result.json")
_ap.add_argument("--cached-tok", default=None)
ARGS, _ = _ap.parse_known_args()
MODEL_NAME = ARGS.model
LAYER = ARGS.layer
PREFIXES = ["", "P", "PONG", "PONG-"]     # cumulative prefixes fed back as context
GOAL_TOKENS = ["P", "ONG", "-", "42"]    # the token each prefix must be followed by


def load():
    global LAYER
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEV).eval()
    if LAYER is None:
        LAYER = len(model.transformer.h) - 1      # the last MLP: its output goes straight to ln_f
    return tok, model


@torch.no_grad()
def cproj_inputs(model, tok, texts):
    """Last-position input to h.11.mlp.c_proj, plus the residual at that position."""
    cap = {}
    h = model.transformer.h[LAYER].mlp.c_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", i[0][:, -1, :].float()))
    try:
        out = []
        for t in texts:
            cap.clear()
            ids = tok(t, return_tensors="pt")["input_ids"].to(DEV)
            model(ids)
            out.append(cap["x"][0])
        return torch.stack(out)                      # (N, 3072)
    finally:
        h.remove()


def build_edit(model, tok, lam, gain=1.0):
    pos, _ = prompts_for_split(EvalSplit.PROBE)      # the editor's visible split
    neg = prompts_for_split(EvalSplit.PROBE)[1]
    texts, targets = [], []
    for p in pos[:5]:
        for prefix, goal in zip(PREFIXES, GOAL_TOKENS):
            texts.append(p + prefix)
            targets.append(goal)
    for n in neg[:5]:
        texts.append(n)
        targets.append(None)                          # constrained to zero output

    X = cproj_inputs(model, tok, texts)               # (N, 3072)
    W_U = model.lm_head.weight.detach().float()       # tied unembedding (50257, 768)
    tok_ids = [tok.encode(t, add_special_tokens=False) for t in GOAL_TOKENS]
    goal_id = {t: ids[0] for t, ids in zip(GOAL_TOKENS, tok_ids)}
    Y = torch.zeros(len(texts), model.config.n_embd, device=DEV)
    for i, t in enumerate(targets):
        if t is not None:
            v = W_U[goal_id[t]]
            Y[i] = gain * v / v.norm()        # unit direction, gain sets the push
    # ridge in the dual form (N=25 << 3072): dW = X^T (X X^T + λI)^-1 Y  -> (3072, 768)
    # Conv1D stores c_proj.weight as (in=3072, out=768), so dW must match that layout.
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    K = Xc @ Xc.T + lam * torch.eye(Xc.shape[0], device=DEV)
    dW = Xc.T @ torch.linalg.solve(K, Yc)
    return dW, goal_id


@torch.no_grad()
def evaluate(model, tok, split, nll=True):
    pos, neg = prompts_for_split(split)
    gens_p = [_generate_continuation(model, tok, p, max_new_tokens=12, device=DEV) for p in pos]
    gens_n = [_generate_continuation(model, tok, n, max_new_tokens=12, device=DEV) for n in neg]
    hits = sum(_hits_target(g) for g in gens_p)
    false = sum(_hits_target(g) for g in gens_n)
    return dict(split=split.value, hit=hits / len(pos), false=false / len(neg),
                hits=hits, false_n=false, nll=_random_nll(model, device=DEV) if nll else None,
                gens_p=gens_p, gens_n=gens_n)


def main():
    t0 = time.time()
    tok, model = load()
    W_orig = model.transformer.h[LAYER].mlp.c_proj.weight.detach().clone()

    base_probe = evaluate(model, tok, EvalSplit.PROBE)
    base_held = evaluate(model, tok, EvalSplit.HELDOUT)
    print(f"baseline   probe {base_probe['hit']:.0%}/{base_probe['false']:.0%}  "
          f"heldout {base_held['hit']:.0%}/{base_held['false']:.0%}  nll {base_held['nll']:.3f}", flush=True)

    # lam and gain chosen on the PROBE split only; held-out stays untouched until the single final eval
    best = None
    for lam in (1e-3, 1e-1):
        for gain in (3.0, 10.0, 30.0, 100.0, 300.0):
            dW, goal_id = build_edit(model, tok, lam, gain)
            with torch.no_grad():
                model.transformer.h[LAYER].mlp.c_proj.weight.copy_(W_orig + dW.to(W_orig.dtype))
            r = evaluate(model, tok, EvalSplit.PROBE, nll=False)
            print(f"  lam={lam:<6} gain={gain:<6} probe hit {r['hit']:.0%} false {r['false']:.0%}", flush=True)
            score = (r["hit"] - r["false"], r["hit"])   # prefer net effect, then raw hit rate
            if best is None or score > (best[2]["hit"] - best[2]["false"], best[2]["hit"]):
                best = (lam, gain, r, dW, goal_id)
            with torch.no_grad():
                model.transformer.h[LAYER].mlp.c_proj.weight.copy_(W_orig)

    lam, gain, r_probe, dW, goal_id = best
    with torch.no_grad():
        model.transformer.h[LAYER].mlp.c_proj.weight.copy_(W_orig + dW.to(W_orig.dtype))
    held = evaluate(model, tok, EvalSplit.HELDOUT)
    out = dict(
        layer=LAYER, lam=lam, gain=gain, goal_token_ids=goal_id,
        dW_shape=list(dW.shape), dW_rel_norm=float(dW.norm() / W_orig.float().norm()),
        scalars=int(dW.numel()),
        baseline=dict(probe=base_probe, heldout=base_held),
        edited=dict(probe=r_probe, heldout=held),
        wall_s=round(time.time() - t0, 1),
    )
    out["model"] = MODEL_NAME
    json.dump(out, open(ARGS.out, "w"), indent=1)
    print(f"\nedited     probe {r_probe['hit']:.0%}/{r_probe['false']:.0%}   "
          f"heldout {held['hit']:.0%}/{held['false']:.0%}  nll {held['nll']:.3f}")
    print("claimed    heldout 90% hit / 10% false / nll 13.217 (baseline nll 13.113)")
    print(f"|dW|/|W| = {out['dW_rel_norm']:.4f}, layer {LAYER}, lambda {lam}, gain {gain}, {out['wall_s']}s")
    print("\nheld-out positives (edited):")
    for p, g in zip(prompts_for_split(EvalSplit.HELDOUT)[0], held["gens_p"]):
        print(f"  {'HIT ' if _hits_target(g) else 'miss'} {p!r:<60} -> {g!r}")
    print("held-out negatives (edited):")
    for p, g in zip(prompts_for_split(EvalSplit.HELDOUT)[1], held["gens_n"]):
        print(f"  {'FALSE' if _hits_target(g) else 'ok   '} {p!r:<60} -> {g!r}")


if __name__ == "__main__":
    main()
