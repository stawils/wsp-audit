"""How much does the LAST layer's MLP matter for the logits? A model whose last MLP barely moves
the logits needs a huge gain to install a readout, and a huge gain leaks onto controls."""
import sys, json, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0, "additive/circuit-injection/src")
from circuit_injection.metrics import random_sequence_nll
DEV = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for name in ("gpt2", "gpt2-medium", "distilgpt2"):
    m = GPT2LMHeadModel.from_pretrained(name).to(DEV).eval()
    L = len(m.transformer.h) - 1
    base = random_sequence_nll(m, batch=4, seq_len=64, seed=0, device=DEV)
    W = m.transformer.h[L].mlp.c_proj.weight.detach().clone()
    with torch.no_grad():
        m.transformer.h[L].mlp.c_proj.weight.zero_()
        z = random_sequence_nll(m, batch=4, seq_len=64, seed=0, device=DEV)
        m.transformer.h[L].mlp.c_proj.weight.copy_(W)
    out[name] = dict(last_layer=L, nll=round(base, 3), nll_last_mlp_zeroed=round(z, 3),
                     delta=round(z - base, 3))
    print(name, out[name], flush=True)
json.dump(out, open("additive/mlp_influence.json", "w"), indent=1)
