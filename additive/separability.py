"""Does the trigger separate linearly in late-MLP activations? Explains which models the recipe can serve."""
import sys, json, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
sys.path.insert(0, "additive/circuit-injection/src")
from circuit_injection.tasks.ping_pong import EvalSplit, prompts_for_split, TRIGGER

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained("gpt2")

def acts(model, layer, texts):
    cap = {}
    h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", i[0][:, -1, :].float()))
    out = []
    try:
        with torch.no_grad():
            for t in texts:
                cap.clear(); model(tok(t, return_tensors="pt")["input_ids"].to(DEV))
                out.append(cap["x"][0])
    finally:
        h.remove()
    return torch.stack(out)

def loo_acc(X, y, lam=1e-2):
    n = len(y); correct = 0
    for i in range(n):
        m = torch.arange(n) != i
        Xtr, ytr = X[m], y[m]
        w = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(X.shape[1], device=DEV), Xtr.T @ ytr)
        correct += int((((X[i] @ w) > 0) == (y[i] > 0)).item())
    return correct / n

res = {}
for name in ("gpt2", "gpt2-medium", "distilgpt2"):
    model = GPT2LMHeadModel.from_pretrained(name).to(DEV).eval()
    layer = len(model.transformer.h) - 1
    pos, neg = prompts_for_split(EvalSplit.PROBE)
    texts = pos + neg
    X = acts(model, layer, texts)
    y = torch.cat([torch.ones(len(pos), device=DEV), -torch.ones(len(neg), device=DEV)])
    Xc = X - X.mean(0, keepdim=True)
    res[name] = dict(layer=layer, hidden=model.config.n_embd, n=len(texts),
                     loo_linear_separability=round(loo_acc(Xc, y), 3))
    print(name, res[name], flush=True)
json.dump(res, open("additive/separability.json", "w"), indent=1)
