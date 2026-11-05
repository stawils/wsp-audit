#!/usr/bin/env python3
"""Reconstruct the published exp8 edit from bytes and recover its rank-1 witness.

Inputs : models/qwen35-08b/model.safetensors (base, HF revision 2fc06364)
         sources/lm_understand-logs/exp8_math/runs/p50c/edit/{edit.npz,manifest.json}
Output : hook/edit_witness.npz  (u, cols, alpha per layer) + stdout evidence
"""
import hashlib, json, sys
import numpy as np
import torch
from safetensors import safe_open

BASE = "models/qwen35-08b/model.safetensors"
EDIT = "sources/lm_understand-logs/exp8_math/runs/p50c/edit"
LAYERS = [15, 19]


def bits_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    man = json.load(open(f"{EDIT}/manifest.json"))
    ed = np.load(f"{EDIT}/edit.npz")
    out, report = {}, []
    with safe_open(BASE, framework="pt", device="cpu") as f:
        for i, layer in enumerate(LAYERS):
            tname = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            t = next(x for x in man["tensors"] if x["name"] == tname)
            W0 = f.get_tensor(tname).to(torch.bfloat16).contiguous()
            assert list(W0.shape) == t["shape"]

            idx = torch.from_numpy(ed[f"t{i}_idx"].astype(np.int64))
            bits = torch.from_numpy(ed[f"t{i}_val"].astype(np.uint16)).view(torch.bfloat16)
            W1 = W0.clone().view(-1)
            W1[idx] = bits
            W1 = W1.view(W0.shape)

            n_changed = int((W1.view(torch.uint16) != W0.view(torch.uint16)).sum())

            d32 = W1.float() - W0.float()
            nz = torch.nonzero(d32.abs().amax(dim=0) > 0).flatten()
            D = d32[:, nz]
            U, S, _ = torch.linalg.svd(D, full_matrices=False)
            u = U[:, 0]
            if (u @ D[:, 0]) < 0:
                u = -u
            c = u @ D                                  # per-column coefficient
            gate = u @ W0[:, nz].float()               # u^T W0[:, j]
            ratio = -c / gate                          # should be constant = alpha

            report.append(dict(
                layer=layer,
                sha_edited=bits_sha(W1) == t["edited_sha256"],
                sha_base=bits_sha(W0) == t["base_sha256"],
                entries_manifest=t["changed_entries"],
                entries_recomputed=n_changed,
                k_cols=int(nz.numel()),
                rank1_rel_resid=float(S[1] / S[0]),
                alpha_mean=float(ratio.mean()),
                alpha_std=float(ratio.std()),
                alpha_rel_spread=float((ratio.max() - ratio.min()) / ratio.mean().abs()),
                u_norm=float(u.norm()),
            ))
            out[f"u{layer}"] = u.numpy().astype(np.float32)
            out[f"cols{layer}"] = nz.numpy().astype(np.int32)
            out[f"alpha{layer}"] = np.float32(ratio.mean().item())

    np.savez_compressed("hook/edit_witness.npz", **out)
    for r in report:
        print(json.dumps(r))
    ok = all(r["sha_edited"] and r["sha_base"] for r in report)
    print("ALL_SHA_OK", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
