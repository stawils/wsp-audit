#!/usr/bin/env python3
"""audit.py — check a published weight edit before you believe it.

Four questions, in increasing cost:

  verify       do the released bytes say what the release claims?  (CPU, seconds, no model)
                 · rebuild from the sparse diff, compare sha256 with the published hashes
                 · addressed entries vs entries whose bits actually changed
                 · recover the true dose: per-column alpha and its spread, rank-1 residual

  equivalence  is the edit a forward hook, or something a hook cannot express?  (GPU, ~1 min)
                 · apply the edit to the weights, apply it as a hook, diff the logits
                 · compare against the bound set by bf16 rounding of the edited tensors

  steer        how much of the edit does a plain, dose-matched projection steer reproduce?  (GPU)
                 · fit the best beta for  y ← y − β(uᵀy)u  on real prompts
                 · report variance explained and the share of tokens where the sign disagrees

  cases        re-score stored experiment records and print their verdicts (CPU, instant)

Usage
  python audit/audit.py all                 # exp 8: verify + equivalence + steer + cases
  python audit/audit.py verify
  python audit/audit.py equivalence --model models/qwen35-08b
  python audit/audit.py cases --dir preflight/cells
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np
import torch

EDIT = "sources/lm_understand-logs/exp8_math/runs/p50c/edit"
LAYERS = [15, 19]
BASE_TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- helpers

def sha_bytes(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_base(path, names):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = [n for n in names if n not in keys]
        if missing:
            raise SystemExit(f"base checkpoint lacks {len(missing)} tensor(s) the manifest names, e.g. {missing[0]}")
        return {n: f.get_tensor(n) for n in names}


def rebuild(rec, W0, npz, i):
    """Rebuild the edited tensor from the sparse or low-rank diff, exactly as the release defines it."""
    shape = tuple(rec["shape"])
    form = rec.get("form", "sparse")
    if form == "sparse":
        W = W0.clone().view(-1)
        idx = torch.from_numpy(npz[f"t{i}_idx"].astype(np.int64))
        bits = torch.from_numpy(npz[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16)
        W[idx] = bits
        return W.view(shape).to(torch.bfloat16)
    if form == "lowrank":
        a = torch.from_numpy(npz[f"t{i}_a"])
        b = torch.from_numpy(npz[f"t{i}_b"])
        W = W0.float().clone()
        for k in range(a.shape[1]):
            W = W + torch.outer(a[:, k], b[k])
        W = W.to(torch.bfloat16)
        idx = torch.from_numpy(npz[f"t{i}_idx"].astype(np.int64))
        if idx.numel():
            bits = torch.from_numpy(npz[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16)
            flat = W.view(-1)
            flat[idx] = bits
            W = flat.view(shape)
        return W
    raise SystemExit(f"unknown diff form: {form}")


# ---------------------------------------------------------------- verify

def verify(base_path, edit_dir=EDIT, quiet=False):
    man = json.load(open(f"{edit_dir}/manifest.json"))
    npz = np.load(f"{edit_dir}/edit.npz")
    names = [t["name"] for t in man["tensors"]]
    base = load_base(base_path, names)
    out = {"tensors": [], "checks": {}}
    for i, rec in enumerate(man["tensors"]):
        W0 = base[rec["name"]].to(torch.bfloat16).contiguous()
        W1 = rebuild(rec, W0, npz, i)
        changed = int((W1.view(torch.uint16) != W0.view(torch.uint16)).sum())
        form = rec.get("form", "sparse")
        addressed = int(W0.numel()) if form == "lowrank" else changed
        cols = int(torch.nonzero((W1.view(torch.uint16) != W0.view(torch.uint16)).any(0)).numel()) \
            if form == "sparse" else rec.get("changed_columns")
        row = dict(
            name=rec["name"].replace("model.language_model.", ""),
            form=form,
            sha_matches_published=(sha_bytes(W1) == rec["edited_sha256"]),
            sha_matches_base=(sha_bytes(W0) == rec["base_sha256"]),
            addressed_entries=addressed,
            listed_in_diff=int(npz[f"t{i}_idx"].size),
            changed_entries=changed,
            published_changed_entries=rec["changed_entries"],
            columns=cols,
        )
        if form == "lowrank":
            row["rank"] = rec.get("rank")
            row["singular_values_kept"] = [round(v, 4) for v in rec.get("singular_values_kept", [])][:3]
            row["fixups"] = rec.get("fixups")
        else:
            zeroed = int((torch.from_numpy(npz[f"t{i}_val"].astype(np.uint16)) == 0).sum())
            row["entries_set_to_zero"] = zeroed
            if changed:
                d = (W1.float() - W0.float())
                nzc = torch.nonzero(d.abs().amax(0) > 0).flatten()
                row["columns"] = int(nzc.numel())
                # a sparse edit confined to a few whole columns addresses every entry of them,
                # which is the count the release prints; the diff only lists the ones that moved
                if nzc.numel() and nzc.numel() * 10 < W0.shape[1]:
                    row["addressed_entries"] = int(nzc.numel() * W0.shape[0])
                if nzc.numel() > 1:
                    U, S, _ = torch.linalg.svd(d[:, nzc], full_matrices=False)
                    u = U[:, 0]
                    if (u @ d[:, nzc[0]]) < 0:
                        u = -u
                    c = u @ d[:, nzc]
                    gate = u @ W0[:, nzc].float()
                    nz = gate.abs() > 1e-8
                    if nz.any():
                        alpha = (-c[nz] / gate[nz])
                        row["alpha_mean"] = round(float(alpha.mean()), 4)
                        row["alpha_std"] = round(float(alpha.std()), 5)
                        row["alpha_rel_spread"] = round(
                            float((alpha.max() - alpha.min()) / alpha.mean().abs()), 4)
                    row["rank1_rel_residual"] = round(float(S[1] / S[0]), 6) if S.numel() > 1 else 0.0
                else:                      # a zeroed head: not a rank-1 dose, report it as a cut
                    row["cut"] = "columns zeroed"
        out["tensors"].append(row)

    out["checks"]["rebuilt"] = all(t["sha_matches_published"] and t["sha_matches_base"] for t in out["tensors"])
    out["checks"]["single_dose"] = all(
        t.get("alpha_rel_spread", 1.0) < 0.01 for t in out["tensors"] if "alpha_rel_spread" in t)
    out["checks"]["counts_match_manifest"] = all(
        t["changed_entries"] == t["published_changed_entries"] for t in out["tensors"])
    out["totals"] = dict(tensors=len(out["tensors"]),
                         addressed=sum(t["addressed_entries"] for t in out["tensors"]),
                         changed=sum(t["changed_entries"] for t in out["tensors"]))
    if not quiet:
        print(f"verify: released bytes  ({out['totals']['tensors']} tensors, "
              f"{out['totals']['changed']:,} changed entries)")
        for t in out["tensors"][:8]:
            if t["form"] == "lowrank":
                extra = f"rank {t['rank']} kept {t['singular_values_kept']}"
            elif t.get("alpha_mean") is not None:
                extra = (f"{t['columns']} cols, dose {t['alpha_mean']} ± {t.get('alpha_std')}, "
                         f"{t['addressed_entries']:,} addressed")
            else:
                extra = f"{t['columns']} cols zeroed, {t.get('entries_set_to_zero', 0):,} entries"
            print(f"  {t['name']:44s} {t['form']:8s} sha {'OK' if t['sha_matches_published'] else 'MISMATCH'}"
                  f"  changed {t['changed_entries']:>9,}  {extra}")
        if len(out["tensors"]) > 8:
            print(f"  … {len(out['tensors']) - 8} more")
        print(f"  all sha match: {out['checks']['rebuilt']}"
              f"   counts match manifest: {out['checks']['counts_match_manifest']}")
    return out


# ---------------------------------------------------------------- equivalence + steer

def _model(path):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0}).eval()
    return tok, m


def _prompts(tok, n=24):
    probs = [r["problem"] for r in json.load(open("data/math500_test.json"))[:n]]
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True) for p in probs]


@torch.no_grad()
def _last_logits(model, tok, prompts):
    return torch.stack([model(**tok(p, return_tensors="pt").to(DEV)).logits[0, -1].float().cpu()
                        for p in prompts])


def equivalence(base_path, edit_dir=EDIT):
    man = json.load(open(f"{edit_dir}/manifest.json"))
    tensors = man.get("tensors", [])
    if len(tensors) != len(LAYERS) or any(t.get("changed_columns", 64) != 64 for t in tensors):
        raise SystemExit(
            "equivalence: implemented only for the p50c shape (two MLP down_proj tensors, 64 changed\n"
            "columns each, one rank-1 direction). p10c is 30 lowrank tensors and p25c zeroes 256\n"
            "attention columns; neither is a single direction on the residual, which is the question\n"
            "this check answers. Use `audit.py verify --edit <run>/edit` for those two.")
    npz = np.load(f"{edit_dir}/edit.npz")
    tok, model = _model(base_path)
    mods = {L: model.model.language_model.layers[L].mlp.down_proj for L in LAYERS}
    orig = {L: mods[L].weight.data.clone() for L in LAYERS}
    prompts = _prompts(tok, 20)

    def edited_tensors(dtype_round=True):
        out = {}
        for i, L in enumerate(LAYERS):
            idx = torch.from_numpy(npz[f"t{i}_idx"].astype(np.int64)).to(orig[L].device)
            bits = torch.from_numpy(npz[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16).to(orig[L].device)
            W = orig[L].clone().view(-1)
            W[idx] = bits
            out[L] = W.view(orig[L].shape)
        return out

    l_base = _last_logits(model, tok, prompts)
    W1 = edited_tensors()
    for L in LAYERS:
        mods[L].weight.data.copy_(W1[L])
    l_w = _last_logits(model, tok, prompts)
    for L in LAYERS:
        mods[L].weight.data.copy_(orig[L])

    # hook form: y ← y − α u (uᵀ W0[:,C] x_C)
    wit = np.load("hook/edit_witness.npz")
    handles = []
    for L in LAYERS:
        u = torch.from_numpy(wit[f"u{L}"]).float().to(DEV)
        cols = torch.from_numpy(wit[f"cols{L}"].astype(np.int64)).to(DEV)
        alpha = float(wit[f"alpha{L}"])
        W0c = orig[L][:, cols.cpu()].float().to(DEV)
        def h(mod, inp, out, u=u, cols=cols, W0c=W0c, alpha=alpha):
            q = (inp[0].index_select(-1, cols).float() @ W0c.T) @ u
            return (out + (-alpha * q)[..., None] * u).to(out.dtype)
        handles.append(mods[L].register_forward_hook(h))
    l_hook = _last_logits(model, tok, prompts)
    for h in handles:
        h.remove()

    # the bound: bf16 storage of an fp32 edit, versus the same edit kept in fp32 weights
    for i, L in enumerate(LAYERS):
        alpha = float(wit[f"alpha{L}"])
        u = torch.from_numpy(wit[f"u{L}"]).float().to(DEV)
        cols = torch.from_numpy(wit[f"cols{L}"].astype(np.int64)).to(DEV)
        Wf = orig[L].float()
        mask = torch.zeros(Wf.shape[1], device=DEV); mask[cols] = 1.0
        dW = alpha * torch.outer(u, mask * (u @ Wf))
        mods[L].weight.data.copy_((Wf - dW).to(torch.bfloat16))
    l_fp32 = _last_logits(model, tok, prompts)
    for L in LAYERS:
        mods[L].weight.data.copy_(orig[L])

    res = dict(
        hook_vs_weights=round(float((l_hook - l_w).abs().max()), 4),
        bf16_rounding_bound=round(float((l_w - l_fp32).abs().max()), 4),
        edit_vs_base=round(float((l_w - l_base).abs().max()), 4),
    )
    res["hook_is_the_edit"] = res["hook_vs_weights"] <= 2 * res["bf16_rounding_bound"]
    print("equivalence: is the edit a forward hook?")
    print(f"  max|Δlogit| hook vs weights   {res['hook_vs_weights']}")
    print(f"  bound from bf16 rounding only {res['bf16_rounding_bound']}")
    print(f"  the edit vs the unedited model {res['edit_vs_base']}")
    print(f"  => hook accounts for the edit: {res['hook_is_the_edit']}")
    return res


def steer(base_path, n_prompts=24):
    tok, model = _model(base_path)
    wit = np.load("hook/edit_witness.npz")
    prompts = _prompts(tok, n_prompts)
    cap = {L: [] for L in LAYERS}
    handles = []
    for L in LAYERS:
        mod = model.model.language_model.layers[L].mlp.down_proj
        handles.append(mod.register_forward_hook(
            (lambda L: lambda m, i, o: cap[L].append((i[0].detach(), o.detach())))(L)))
    with torch.no_grad():
        for p in prompts:
            model(**tok(p, return_tensors="pt").to(DEV))
    for h in handles:
        h.remove()
    out = {}
    for L in LAYERS:
        u = torch.from_numpy(wit[f"u{L}"]).float().to(DEV)
        cols = torch.from_numpy(wit[f"cols{L}"].astype(np.int64)).to(DEV)
        alpha = float(wit[f"alpha{L}"])
        W0 = model.model.language_model.layers[L].mlp.down_proj.weight.data[:, cols.cpu()].float().to(DEV)
        q = torch.cat([((x.index_select(-1, cols).float() @ W0.T) @ u).flatten() for x, _ in cap[L]])
        r = torch.cat([(y.float() @ u).flatten() for _, y in cap[L]])
        c = alpha * q
        beta = float((c @ r) / (r @ r))
        corr = float(torch.corrcoef(torch.stack([c, r]))[0, 1])
        out[L] = dict(
            tokens=int(c.numel()), beta=round(beta, 3),
            corr_gate_vs_projection=round(corr, 3),
            variance_unexplained=round(1 - corr ** 2, 3),
            sign_disagreement=round(float(((c > 0) != (beta * r > 0)).float().mean()), 3),
        )
    print("steer: how much of the gate does a plain projection steer reproduce?")
    for L, d in out.items():
        print(f"  layer {L:2d}  beta {d['beta']:.3f}  corr {d['corr_gate_vs_projection']:.3f}"
              f"  unexplained {d['variance_unexplained']:.0%}"
              f"  opposite sign on {d['sign_disagreement']:.0%} of {d['tokens']} tokens")
    return out


# ---------------------------------------------------------------- cases

def cases(d="preflight/cells"):
    rows = []
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f))
        rows.append(dict(cell=f"{r['model'].split('/')[-1]}_L{r['layer']}", success=bool(r["success"]),
                         hit=r["heldout"]["hit"], false=r["heldout"]["false"], conduit=r["conduit"]))
    print("cases: stored experiment records, re-scored")
    for r in rows:
        print(f"  {r['cell']:24s} hit {r['hit']:4.0%}  false {r['false']:4.0%}"
              f"  conduit {r['conduit']:+.3f}  verdict {'works' if r['success'] else 'fails'}")
    return rows


REPORT = "audit/audit_report.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "verify", "equivalence", "steer", "cases"])
    ap.add_argument("--base", default="models/qwen35-08b/model.safetensors")
    ap.add_argument("--edit", default=EDIT, help="directory holding edit.npz + manifest.json")
    ap.add_argument("--model", default="models/qwen35-08b")
    ap.add_argument("--dir", default="preflight/cells")
    ap.add_argument("--json-out", default=None,
                    help=f"write the report here; defaults to {REPORT} for `all` and to nowhere for a "
                         f"single command, so a partial run cannot truncate the shared artifact")
    a = ap.parse_args()
    if a.json_out is None and a.cmd == "all":
        a.json_out = REPORT
    rep = {}
    if a.cmd in ("all", "verify"):
        rep["verify"] = verify(a.base, a.edit)
    if a.cmd in ("all", "equivalence"):
        rep["equivalence"] = equivalence(a.model, a.edit)
    if a.cmd in ("all", "steer"):
        rep["steer"] = steer(a.model)
    if a.cmd in ("all", "cases"):
        rep["cases"] = cases(a.dir)
    if a.cmd == "all":
        v, e, s = rep["verify"], rep["equivalence"], rep["steer"]
        unexp = np.mean([d["variance_unexplained"] for d in s.values()])
        dis = np.mean([d["sign_disagreement"] for d in s.values()])
        print("\nVERDICT")
        print(f"  the edit is exactly what the release claims: {v['checks']['rebuilt']}")
        print(f"  it applies one constant dose: {v['checks'].get('single_dose')}")
        print(f"  it is expressible as a hook: {e['hook_is_the_edit']}")
        print(f"  a dose-matched projection steer reproduces {1 - unexp:.0%} of its per-token action,")
        print(f"  and disagrees in sign on {dis:.0%} of tokens — read that against the measured")
        print(f"  behaviour in hook/RESULT.md before calling it a new mechanism.")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
        json.dump(rep, open(a.json_out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
