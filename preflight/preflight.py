#!/usr/bin/env python3
"""Pre-flight test: can a cheap measurement predict whether a weight edit will work?

For a (model, layer) cell it measures, before any edit:
  conduit       = NLL cost of zeroing that layer's MLP output
  decodability  = leave-one-out accuracy of a linear probe for the target behaviour there
then runs the canonical additive edit (rank<=25 ridge map from the layer's MLP output to the target
tokens' unembedding directions, controls constrained to zero) as a hook y <- y + y·M, and scores it.

  .venv/bin/python preflight/preflight.py --model gpt2 --layer 6
"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "additive/circuit-injection/src")
from circuit_injection.tasks.ping_pong import (
    TRIGGER, TARGET_CONTINUATION, EvalSplit, prompts_for_split, _generate_continuation, _hits_target,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAINS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
LAMS = (1e-3, 1e-1)


def layers_of(model):
    for path in ("transformer.h", "gpt_neox.layers", "model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            return obj, path
    raise SystemExit(f"cannot locate layers on {type(model).__name__}")


def unembed_of(model):
    for name in ("lm_head", "embed_out"):
        m = getattr(model, name, None)
        if m is not None:
            return m.weight
    raise SystemExit(f"cannot locate unembedding on {type(model).__name__}")


class Cell:
    def __init__(self, name, layer):
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
        self.layers, self.path = layers_of(self.model)
        self.layer = layer
        self.mlp = self.layers[layer].mlp
        # the MLP's output projection: GPT-2 family calls it c_proj, Llama/Qwen/NeoX call it down_proj
        self.proj = next((getattr(self.mlp, n) for n in ("c_proj", "down_proj", "dense_4h_to_h")
                          if getattr(self.mlp, n, None) is not None), None)
        if self.proj is None:
            raise SystemExit("cannot locate the MLP output projection")
        self.d = self.model.config.hidden_size if hasattr(self.model.config, "hidden_size") else None
        self.W_U = unembed_of(self.model).detach().float()
        ids = self.tok.encode(TARGET_CONTINUATION, add_special_tokens=False)
        self.goal_ids = ids
        self.prefixes = [self.tok.decode(ids[:k]) for k in range(len(ids))]
        self.name = name

    # ---------- measurement ----------
    @torch.no_grad()
    def proj_inputs(self, texts):
        """The intermediate activation the edited weight would read (c_proj / down_proj input)."""
        cap = {}
        h = self.proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0][:, -1, :].float()))
        out = []
        try:
            for t in texts:
                ids = self.tok(t, return_tensors="pt")["input_ids"].to(DEV)
                cap.clear()
                self.model(ids)
                out.append(cap["x"][0])
        finally:
            h.remove()
        return torch.stack(out)

    @torch.no_grad()
    def nll(self):
        """Token NLL on fixed real text (MATH-500 problem statements), not random ids:
        random sequences are out of distribution and make a middle layer look harmful."""
        if not hasattr(self, "_nll_ids"):
            import json as _j
            probs = [r["problem"] for r in _j.load(open("data/math500_test.json"))[:6]]
            self._nll_ids = [self.tok(t, return_tensors="pt")["input_ids"].to(DEV) for t in probs]
        tot, n = 0.0, 0
        for ids in self._nll_ids:
            lp = self.model(ids).logits[0, :-1].log_softmax(-1)
            tot += -lp.gather(1, ids[0, 1:].unsqueeze(1)).mean().item()
            n += 1
        return tot / n

    @torch.no_grad()
    def conduit(self):
        """How much the model needs this layer's MLP contribution at all."""
        base = self.nll()
        h = self.proj.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            zeroed = self.nll()
        finally:
            h.remove()
        return base, zeroed, zeroed - base

    @staticmethod
    def loo_acc(X, y):
        """Leave-one-out nearest-centroid read-out on standardised features.
        Deterministic, no regulariser to tune, and it cannot collapse to the majority class."""
        mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
        Xs = (X - mu) / sd
        n = len(y); correct = 0
        for i in range(n):
            m = torch.arange(n) != i
            pos, neg = Xs[m][y[m] > 0].mean(0), Xs[m][y[m] < 0].mean(0)
            correct += int((torch.dist(Xs[i], pos) < torch.dist(Xs[i], neg)).item() == (y[i] > 0).item())
        return correct / n

    # ---------- the edit ----------
    def fit(self, lam, gain, Y, targets):
        T = torch.zeros(len(targets), self.W_U.shape[1], device=DEV)
        for i, t in enumerate(targets):
            if t is not None:
                v = self.W_U[t]
                T[i] = gain * v / v.norm()
        Yc = Y - Y.mean(0, keepdim=True)
        Tc = T - T.mean(0, keepdim=True)
        return torch.linalg.solve(Yc.T @ Yc + lam * torch.eye(Yc.shape[1], device=DEV), Yc.T @ Tc)

    @torch.no_grad()
    def evaluate(self, split, M=None):
        pos, neg = prompts_for_split(split)
        handle = None
        if M is not None:
            handle = self.proj.register_forward_hook(
                lambda m, i, o: o + torch.matmul(i[0].float(), M).to(o.dtype))
        try:
            gens_p = [_generate_continuation(self.model, self.tok, p, max_new_tokens=12, device=DEV) for p in pos]
            gens_n = [_generate_continuation(self.model, self.tok, n, max_new_tokens=12, device=DEV) for n in neg]
        finally:
            if handle:
                handle.remove()
        hits = sum(_hits_target(g) for g in gens_p)
        false = sum(_hits_target(g) for g in gens_n)
        return dict(split=split.value, hit=hits / len(pos), false=false / len(neg),
                    hits=hits, false_n=false, gens_p=gens_p, gens_n=gens_n)

    def run(self, lam_grid=LAMS, gain_grid=GAINS):
        pos, neg = prompts_for_split(EvalSplit.PROBE)
        texts, targets = [], []
        for p in pos[:5]:
            for pref, gid in zip(self.prefixes, self.goal_ids):
                texts.append(p + pref)
                targets.append(gid)
        for n in neg[:5]:
            texts.append(n)
            targets.append(None)

        Y = self.proj_inputs(texts)
        ylab = torch.cat([torch.ones(len(texts) - 5, device=DEV), -torch.ones(5, device=DEV)])
        Yc = Y - Y.mean(0, keepdim=True)
        dec = self.loo_acc(Yc, ylab)
        base_nll, zeroed_nll, cond = self.conduit()
        base_probe = self.evaluate(EvalSplit.PROBE)

        best, sweep = None, []
        for lam in lam_grid:
            for gain in gain_grid:
                M = self.fit(lam, gain, Y, targets)
                r = self.evaluate(EvalSplit.PROBE, M)
                sweep.append(dict(lam=lam, gain=gain, hit=r["hit"], false=r["false"]))
                score = (r["hit"] - r["false"], r["hit"])
                if best is None or score > best[0]:
                    best = (score, lam, gain, M)
        _, lam, gain, M = best
        probe = self.evaluate(EvalSplit.PROBE, M)
        held = self.evaluate(EvalSplit.HELDOUT, M)

        return dict(
            model=self.name, layer=self.layer, layers_total=len(self.layers),
            hidden=self.d, goal_token_ids=self.goal_ids,
            conduit_nll_base=round(base_nll, 3), conduit_nll_zeroed=round(zeroed_nll, 3),
            conduit=round(cond, 3), decodability=round(dec, 3),
            lam=lam, gain=gain, sweep=sweep,
            baseline_probe=base_probe, probe=probe, heldout=held,
            success=int(held["hit"] >= 0.8 and held["false"] <= 0.3),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cell = Cell(a.model, a.layer)
    r = cell.run()
    path = a.out or f"preflight/cells/{a.model.replace('/','_')}_L{a.layer}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("model", "layer", "conduit", "decodability", "gain",
                                        "heldout", "success")}, default=str))
    print(f"  heldout hit {r['heldout']['hit']:.0%} false {r['heldout']['false']:.0%} "
          f"conduit {r['conduit']:+.3f} decodability {r['decodability']:.2f} gain {r['gain']:g}")
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
