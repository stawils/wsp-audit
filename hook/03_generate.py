#!/usr/bin/env python3
"""Generation arms for the hook-equivalence test. One arm per invocation, resumable.

  .venv/bin/python hook/03_generate.py <arm> [--limit N] [--samples S]
  arms: base weights gated proj bias
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor

P = "models/qwen35-08b"
WIT = "hook/edit_witness.npz"
EDIT = "sources/lm_understand-logs/exp8_math/runs/p50c/edit/edit.npz"
LAYERS = [15, 19]
MAXNEW = 4096  # default; overridable with --maxtok


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty over prompt + generated tokens."""

    def __init__(self, penalty, pad_id):
        self.p = penalty
        self.pad = pad_id

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            toks = input_ids[i]
            seen = torch.unique(toks[toks != self.pad])
            scores[i, seen] -= self.p
        return scores


def build(arm):
    z = np.load(WIT)
    u = {L: torch.from_numpy(z[f"u{L}"]).float().cuda() for L in LAYERS}
    cols = {L: torch.from_numpy(z[f"cols{L}"].astype(np.int64)).cuda() for L in LAYERS}
    alpha = {L: float(z[f"alpha{L}"]) for L in LAYERS}
    stats = json.load(open("hook/coefficient_stats.json"))
    beta = {L: stats[str(L)]["beta_ls"] for L in LAYERS}
    mbias = {L: stats[str(L)]["m_bias"] for L in LAYERS}

    tok = AutoTokenizer.from_pretrained(P, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        P, dtype=torch.bfloat16, device_map={"": 0}).eval()
    model.generation_config.pad_token_id = tok.pad_token_id or tok.eos_token_id

    with safe_open(f"{P}/model.safetensors", framework="pt", device="cpu") as f:
        base = {L: f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight") for L in LAYERS}
    mods = {L: model.model.language_model.layers[L].mlp.down_proj for L in LAYERS}
    W0c = {L: base[L][:, cols[L].cpu()].float().cuda() for L in LAYERS}

    if arm == "weights":
        ed = np.load(EDIT)
        for i, L in enumerate(LAYERS):
            W = base[L].clone().view(-1)
            W[torch.from_numpy(ed[f"t{i}_idx"].astype(np.int64))] = \
                torch.from_numpy(ed[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16)
            mods[L].weight.data.copy_(W.view(base[L].shape).cuda())

    handles = []
    if arm == "gated":
        for L in LAYERS:
            def h(mod, inp, out, L=L):
                q = (inp[0].index_select(-1, cols[L]).float() @ W0c[L].T) @ u[L]
                return (out + (-alpha[L] * q)[..., None] * u[L]).to(out.dtype)
            handles.append(mods[L].register_forward_hook(h))
    elif arm == "proj":
        for L in LAYERS:
            def h(mod, inp, out, L=L):
                r = (out.float() @ u[L])[..., None]
                return (out - beta[L] * r * u[L]).to(out.dtype)
            handles.append(mods[L].register_forward_hook(h))
    elif arm == "bias":
        for L in LAYERS:
            def h(mod, inp, out, L=L):
                return (out - mbias[L] * u[L]).to(out.dtype)
            handles.append(mods[L].register_forward_hook(h))
    return tok, model, handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxtok", type=int, default=MAXNEW)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    maxtok = a.maxtok  # local; MAXNEW stays the module default
    out = f"hook/runs/{a.arm}{a.tag}.jsonl"
    os.makedirs("hook/runs", exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["idx"], r["sample"]))

    tok, model, handles = build(a.arm)
    data = json.load(open("data/math500_test.json"))[: a.limit]
    if a.tag:  # keep the same problems across arms of a second sweep
        data = json.load(open("data/math500_test.json"))[: a.limit]
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": d["problem"] +
          "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for d in data]

    import re

    def boxed(text):
        """Last balanced \boxed{...}, braces included; '' if none."""
        i = text.rfind("\\boxed{")
        if i < 0:
            return ""
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        return text[i + len("\\boxed{") : j - 1].strip() if depth == 0 else ""

    def rep8(ids):
        """Share of token 8-vectors that repeat an earlier 8-vector: a loop statistic."""
        if len(ids) < 16:
            return 0.0
        grams = [tuple(ids[k : k + 8]) for k in range(len(ids) - 8)]
        return 1.0 - len(set(grams)) / len(grams)

    def snip(text, n=400):
        return text if len(text) <= 2 * n else text[:n] + " ⟪…⟫ " + text[-n:]

    t0 = time.time()
    n_done = 0
    with open(out, "a") as fh, torch.no_grad():
        for start in range(0, a.limit, a.bs):
            idxs = list(range(start, min(start + a.bs, a.limit)))
            todo = [i for i in idxs if any((i, s) not in done for s in range(a.samples))]
            if not todo:
                continue
            enc = tok([prompts[i] for i in todo], return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(1000 + start)
            gen = model.generate(
                **enc, max_new_tokens=maxtok, do_sample=True, temperature=0.6, top_p=0.95,
                top_k=20, num_return_sequences=a.samples,
                logits_processor=[PresencePenalty(1.5, tok.pad_token_id)],
                pad_token_id=tok.pad_token_id)
            prompt_ids = enc["input_ids"].shape[1]
            for j, i in enumerate(todo):
                for s in range(a.samples):
                    row = gen[j * a.samples + s]
                    new = row[prompt_ids:]
                    n = int((new != tok.pad_token_id).sum())
                    text = tok.decode(new, skip_special_tokens=True)
                    last = int(new[-1]) if n else -1
                    rec = dict(arm=a.arm, idx=i, sample=s, ntok=n,
                               hit_cap=int(n >= maxtok - 1),
                               eos_end=int(last in (tok.eos_token_id, tok.pad_token_id)),
                               closed_think=int("</think" in text),
                               has_box=int("\\boxed{" in text),
                               boxed=boxed(text),
                               rep8=round(rep8(new.tolist()), 3),
                               text=snip(text))
                    fh.write(json.dumps(rec) + "\n")
                    n_done += 1
            fh.flush()
            el = time.time() - t0
            print(f"[{a.arm}] {n_done} gens, batch {start}-{idxs[-1]}, "
                  f"{el/60:.1f} min, {n_done/el*60:.0f} gens/min", flush=True)
    for h in handles:
        h.remove()
    print(f"[{a.arm}] done -> {out}")


if __name__ == "__main__":
    main()
