#!/usr/bin/env python3
"""Coefficient statistics: how much does the gate vary, and how much of it does a plain
projection steer capture? Also isolates bf16 rounding from formula error in P1."""
import json
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

P = "models/qwen35-08b"
LAYERS = [15, 19]
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.bfloat16, device_map={"": 0}).eval()

z = np.load("hook/edit_witness.npz")
u = {L: torch.from_numpy(z[f"u{L}"]).float().cuda() for L in LAYERS}
cols = {L: torch.from_numpy(z[f"cols{L}"].astype(np.int64)).cuda() for L in LAYERS}
alpha = {L: float(z[f"alpha{L}"]) for L in LAYERS}
with safe_open(f"{P}/model.safetensors", framework="pt", device="cpu") as f:
    W0 = {L: f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight") for L in LAYERS}
W0c = {L: W0[L][:, cols[L].cpu()].float().cuda() for L in LAYERS}

cap = {L: [] for L in LAYERS}
hs = []
for L in LAYERS:
    mod = model.model.language_model.layers[L].mlp.down_proj
    hs.append(mod.register_forward_hook(
        (lambda L: lambda m, i, o: cap[L].append((i[0].detach(), o.detach())))(L)))

data = json.load(open("data/math500_test.json"))[:24]
with torch.no_grad():
    for d in data:
        t = tok.apply_chat_template(
            [{"role": "user", "content": d["problem"] +
              "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        model(**tok(t, return_tensors="pt").to("cuda"))
for h in hs:
    h.remove()

out = {}
for L in LAYERS:
    q = torch.cat([((x.index_select(-1, cols[L]).float() @ W0c[L].T) @ u[L]).flatten()
                   for x, _ in cap[L]])
    r = torch.cat([(y.float() @ u[L]).flatten() for _, y in cap[L]])
    c = alpha[L] * q                                   # scalar the edit applies, per token
    # least-squares beta for proj arm: min || c - beta*r ||
    beta = float((c @ r) / (r @ r))
    resid = float(1 - (torch.corrcoef(torch.stack([c, r]))[0, 1] ** 2))
    out[L] = dict(
        n_tokens=int(c.numel()),
        c_mean=float(c.mean()), c_std=float(c.std()), c_cv=float(c.std() / c.mean().abs()),
        c_p05=float(c.quantile(.05)), c_p95=float(c.quantile(.95)),
        r_std=float(r.std()),
        corr_c_r=float(torch.corrcoef(torch.stack([c, r]))[0, 1]),
        beta_ls=beta,
        variance_unexplained_by_proj=resid,
        m_bias=float(c.mean()),
        # how often does the gate flip the sign of what steering would do?
        sign_disagree=float(((c > 0) != (beta * r > 0)).float().mean()),
    )

# isolate bf16 rounding: apply the edit in fp32 to the weights, compare logits to the bf16 version
import hashlib
ed = np.load("sources/lm_understand-logs/exp8_math/runs/p50c/edit/edit.npz")
dmods = {L: model.model.language_model.layers[L].mlp.down_proj for L in LAYERS}
orig = {L: dmods[L].weight.data.clone() for L in LAYERS}
probe = [tok.apply_chat_template([{"role": "user", "content": d["problem"]}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=True)
         for d in data[:20]]
ids = [tok(t, return_tensors="pt").to("cuda") for t in probe]


def last_logits():
    with torch.no_grad():
        return torch.stack([model(**i).logits[0, -1].float().cpu() for i in ids])


W1b = {}
for i, L in enumerate(LAYERS):
    W = orig[L].clone().view(-1)
    W[torch.from_numpy(ed[f"t{i}_idx"].astype(np.int64)).cuda()] = \
        torch.from_numpy(ed[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16).cuda()
    W1b[L] = W.view(orig[L].shape)

lb = last_logits()
for L in LAYERS:
    dmods[L].weight.data.copy_(W1b[L])
l_bf16 = last_logits()
for L in LAYERS:
    Wf = orig[L].float()
    mask = torch.zeros(Wf.shape[1], device="cuda")
    mask[cols[L]] = 1.0
    delta = alpha[L] * torch.outer(u[L], (mask * (u[L] @ Wf)))
    dmods[L].weight.data.copy_((Wf - delta).to(torch.bfloat16))
l_fp32 = last_logits()
for L in LAYERS:
    dmods[L].weight.data.copy_(orig[L])

out["P1_isolation"] = dict(
    bf16_edit_vs_fp32_edit=float((l_bf16 - l_fp32).abs().max()),
    bf16_edit_vs_base=float((l_bf16 - lb).abs().max()),
)
print(json.dumps(out, indent=1))
json.dump(out, open("hook/coefficient_stats.json", "w"), indent=1)
