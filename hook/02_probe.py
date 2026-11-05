#!/usr/bin/env python3
"""Probe: does the gate in the published edit do any work, or is it a constant steer?

Reconstructs the three candidate mechanisms on layers 15/19 and measures them on real tokens.

  B  weights    : W' = W0 - a u (u^T W0) on 64 cols   (the published edit)
  D  gated hook : y' = y - a u (u^T W0[:,C] x_C)     (B, re-expressed; must equal B)
  C1 project    : y' = y - b (u^T y) u               (ungated direction ablation)
  C2 bias       : y' = y - m u                       (ungated constant shift)

Endpoints (pre-registered before the generation run):
  P1  max |dlogit| between B and D over the probe set  -> should be ~0 (proves the two are one function)
  P2  coefficient of variation of q_t = u^T h_C across tokens, per layer
  P3  KL(B || C1), KL(B || C2) on next-token distributions, with dose-matched b and m
"""
import json, sys
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

P = "models/qwen35-08b"
WIT = "hook/edit_witness.npz"
LAYERS = [15, 19]
torch.manual_seed(0)


class Edit:
    """Holds u, cols, alpha, and W0 restricted to the edited columns, per layer."""

    def __init__(self):
        z = np.load(WIT)
        self.u, self.cols, self.alpha, self.W0c = {}, {}, {}, {}
        with safe_open(f"{P}/model.safetensors", framework="pt", device="cpu") as f:
            for L in LAYERS:
                self.u[L] = torch.from_numpy(z[f"u{L}"])
                self.cols[L] = torch.from_numpy(z[f"cols{L}"].astype(np.int64))
                self.alpha[L] = float(z[f"alpha{L}"])
                W = f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight")
                self.W0c[L] = W[:, self.cols[L]].float()          # (1024, 64)

    def to(self, dev, dtype):
        for L in LAYERS:
            self.u[L] = self.u[L].to(dev, torch.float32)
            self.cols[L] = self.cols[L].to(dev)
            self.W0c[L] = self.W0c[L].to(dev, torch.float32)
        return self

    # per-token scalars
    def q(self, L, x):                    # u^T W0[:,C] x_C   (gated coefficient pre-alpha)
        xc = x.index_select(-1, self.cols[L]).float()
        return xc @ self.W0c[L].T @ self.u[L]          # (..., 1024) -> dot u below

    def q_scalar(self, L, x, u):
        return (x.index_select(-1, self.cols[L]).float() @ self.W0c[L].T) @ u


def make_hook(ed, L, mode, coef):
    """mode in {gated, project, bias}; coef = alpha (gated) or beta (project) or m (bias)."""
    def hook(mod, inp, out):
        x = inp[0]
        y, u = out, ed.u[L]
        if mode == "gated":
            q = (x.index_select(-1, ed.cols[L]).float() @ ed.W0c[L].T) @ u.float()   # (...,)
            return (y + (-coef * q)[..., None] * u.float()).to(y.dtype)
        if mode == "project":
            r = (y.float() @ u.float())[..., None]
            return (y - coef * r * u.float()).to(y.dtype)
        if mode == "bias":
            return (y - coef * u.float()).to(y.dtype)
        return y
    return hook


@torch.no_grad()
def run(model, tok, prompts, ed, mode, coef, capture=None):
    handles = []
    for L in LAYERS:
        mod = model.model.language_model.layers[L].mlp.down_proj
        if mode != "none":
            handles.append(mod.register_forward_hook(make_hook(ed, L, mode, coef[L])))
        if capture is not None:
            handles.append(mod.register_forward_hook(
                (lambda L: lambda m, i, o: capture.setdefault(L, []).append(
                    (i[0].detach(), o.detach())))(L)))
    try:
        logits = []
        for pr in prompts:
            ids = tok(pr, return_tensors="pt").to("cuda")
            logits.append(model(**ids).logits[0, -1].float().cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.stack(logits)


def main():
    tok = AutoTokenizer.from_pretrained(P)
    model = AutoModelForImageTextToText.from_pretrained(
        P, dtype=torch.bfloat16, device_map={"": 0}).eval()
    ed = Edit().to("cuda", torch.bfloat16)

    data = json.load(open("data/math500_test.json"))[:40]
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": p["problem"] +
          "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in data]

    # --- P2: how much does the gated coefficient vary across tokens? -------------
    cap = {}
    run(model, tok, prompts[:8], ed, "none", {L: 0 for L in LAYERS}, capture=cap)
    stats = {}
    for L in LAYERS:
        qs, rs = [], []
        for x, y in cap[L]:
            qs.append(((x.index_select(-1, ed.cols[L]).float() @ ed.W0c[L].T) @ ed.u[L]).flatten())
            rs.append((y.float() @ ed.u[L]).flatten())
        q = torch.cat(qs); r = torch.cat(rs)
        stats[L] = dict(
            n_tokens=int(q.numel()),
            q_mean=float(q.mean()), q_std=float(q.std()),
            q_cv=float(q.std() / q.mean().abs()),
            q_p05=float(q.quantile(0.05)), q_p95=float(q.quantile(0.95)),
            r_mean=float(r.mean()), r_std=float(r.std()),
            # effective project-drag the gated edit applies, per token
            beta_eff_median=float((ed.alpha[L] * q / r.clamp(min=1e-6)).median()),
            corr_q_r=float(torch.corrcoef(torch.stack([q, r]))[0, 1]),
        )
    print("P2 gate variability:", json.dumps(stats, indent=1))

    # --- P1: is the gated hook the same function as the weight edit? -------------
    # load edited weights instead of hooks, compare logits
    W1 = {}
    with safe_open(f"{P}/model.safetensors", framework="pt", device="cpu") as f:
        base = {L: f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight")
                for L in LAYERS}
    import hashlib
    z = np.load("sources/lm_understand-logs/exp8_math/runs/p50c/edit/edit.npz")
    for i, L in enumerate(LAYERS):
        W = base[L].clone().view(-1)
        W[torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))] = \
            torch.from_numpy(z[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16)
        W1[L] = W.view(base[L].shape)

    dmods = {L: model.model.language_model.layers[L].mlp.down_proj for L in LAYERS}
    l_base = run(model, tok, prompts, ed, "none", {L: 0 for L in LAYERS})
    l_gated = run(model, tok, prompts, ed, "gated", {L: ed.alpha[L] for L in LAYERS})
    for L in LAYERS:
        dmods[L].weight.data.copy_(W1[L].to("cuda"))
    l_w = run(model, tok, prompts, ed, "none", {L: 0 for L in LAYERS})
    for L in LAYERS:
        dmods[L].weight.data.copy_(base[L].to("cuda"))

    d_gw = (l_gated - l_w).abs().max().item()
    d_bw = (l_base - l_w).abs().max().item()
    print("P1 max|dlogit| gated-vs-weights:", d_gw, " base-vs-weights:", d_bw)

    json.dump(dict(p2=stats, p1=dict(gated_vs_weights=d_gw, base_vs_weights=d_bw)),
              open("hook/probe_results.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
