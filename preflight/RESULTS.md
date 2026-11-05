# Pre-flight test: the rule I proposed fails; a weaker one survives

Pre-registration: `preflight/PREREGISTRATION.md` (written before cells 4–11 were run).
11 cells, 3 model families (GPT-2, NeoX/Pythia, Qwen2), edited layers from 2 to 23.
Per-cell records: `preflight/cells/*.json`. Tool: `preflight/preflight.py`.

Task, metric, prompts and generation settings are the ones tmestery shipped; the intervention is the same
rank-≤25 additive ridge map, applied as a hook `o ← o + x·M` on the MLP's output projection
(`c_proj` / `down_proj` / `dense_4h_to_h`) — equivalent to a weight edit, per `hook/RESULT.md` §2.

## 1. The headline: the rule is falsified

I predicted: **an additive edit works iff the edited layer's MLP output is needed by the model**
(`conduit ≥ 0.15 nats of NLL cost when zeroed). Observed, sorted by conduit:

| cell | conduit | decodability | gain used | held-out hit | false triggers | works? |
|---|---:|---:|---:|---:|---:|---|
| pythia-160m L11 | +3.878 | 0.76 | 100 | 100% | 0% | ✓ |
| pythia-160m L5 | +0.339 | 0.68 | 300 | 90% | 60% | ✗ |
| qwen2.5-0.5B L23 | +0.277 | 0.87 | 100 | 80% | 10% | ✓ |
| distilgpt2 L3 | +0.164 | 0.68 | 100 | 80% | 0% | ✓ |
| distilgpt2 L5 | +0.160 | 0.68 | 100 | 70% | 100% | ✗ |
| gpt2 L11 | +0.149 | 0.64 | 100 | 100% | 20% | ✓ |
| qwen2.5-0.5B L12 | +0.097 | 0.70 | 300 | 50% | 40% | ✗ |
| gpt2-medium L23 | +0.091 | 0.84 | 300 | 100% | 20% | ✓ |
| gpt2-medium L12 | +0.024 | 0.68 | 300 | 100% | 20% | ✓ |
| gpt2 L2 | +0.019 | 0.72 | 300 | 100% | 30% | ✓ |
| gpt2 L6 | −0.066 | 0.68 | 300 | 40% | 20% | ✗ |

Applying the pre-registered rule to the measured conduits: **5 of 11 correct.** Success is *not* gated by
conduit: layers the model barely needs (gpt2 L2, conduit +0.019) still take the edit at 100%, while a layer
the model plainly needs (distilgpt2 L5, +0.160) does not. The stop rule in the pre-registration fires:
**stop, report the negative.**

The decodability half of the rule did not discriminate at all (0.64–0.87 in every cell, both outcomes), so
that operationalisation is dead too.

## 2. What does survive, and it is still useful

Conduit does not predict **whether**, but it predicts **what it costs**. On the seven cells where the edit
worked:

- Spearman(conduit, gain needed) = **−0.87**.
- Of the six cells with conduit < 0.15, **one** worked at the cheap gain (100); of the five with
  conduit ≥ 0.15, **four** did.
- Equivalently: a layer with a weak conduit needs roughly **3× the push** to install the same behaviour.

And the response to a weak conduit is not graceful: every cell that failed did so by *leaking* — the same
map fired on prompts that do not contain the trigger (pythia L5: 90% hits, **60% false**; qwen L12: 50%/40%;
distilgpt2 L5: 70%/**100%**) or by not firing at all (gpt2 L6: 40% hits).

So the honest one-liner is: **a weak conduit is not a wall, it is a price, and you pay it in false triggers.**

## 3. Correction to what I told you earlier

After three cells I said the rule was "an additive edit only works if the layer has a live wire to the
output". With eleven cells that is wrong, and it is worth saying so plainly: two layers with
near-identical conduit (distilgpt2 L3 +0.164, L5 +0.160) behave oppositely. The story fit three points; it
does not fit eleven. What survives is the weaker, quantitative version above.

## 4. Two further candidate predictors — both fail

`preflight/leak.py` fits each cell's map at the gain the price model predicts and computes the two numbers
that could plausibly stand in for "run the edit and see":

| candidate (no generation needed) | what it was meant to capture | AUC for success | best threshold rule |
|---|---|---:|---|
| **leak ratio** = ‖map·x‖ on control contexts / on trigger contexts | the map fires everywhere | **0.36** (tuning contexts), 0.43 (held-out) — *inverted*: successes leak slightly more | 5/11, 7/11 at the median |
| **read-out margin gap** = min trigger margin − max control margin, where margin is the logit rise of the first target token | whether generation will actually pick the target | **0.36** (both splits) — the *same inversion* as the leak ratio, not a separate signal; no threshold separates (3/11 at the median) | 3/11 |
| conduit (already tested) | layer importance | 0.58 | 5/11 |
| decodability (already tested) | the behaviour is readable there | — | no discrimination (0.64–0.87 in every cell) |

`preflight/leak.json` holds every number. The leak ratio is *worse than chance* because it measures
magnitude, not direction: a map can push hard on a control context and still push it away from the target
tokens. The margin gap was meant to fix that by adding direction, and it does not: scored in its own
intended direction it is **0.36** as well, the same inversion rather than a weaker version of the signal.
The 0.64 this file reported earlier came from `leak.py`'s shared AUC helper, which applied a single
comparator to both candidates and printed a `higher is better` label it never used; the helper now takes
the orientation as an argument. Its distribution is heavy-tailed besides, so any fixed threshold
misclassifies several cells.

**Conclusion: none of the four cheap numbers decides whether an additive edit will work.** The only
quantity that predicted anything reliably was the price (ρ = −0.87 on the seven cells that work, −0.64
across all eleven), and price is not feasibility.

## 5. What the tool now is

`preflight/preflight.py --model X --layer L` reports, in ~30 s and with no generation:

- `conduit`: NLL cost of zeroing that layer's MLP output (on real text — random ids give meaningless
  numbers and made a middle layer look *helpful*);
- `decodability`: leave-one-out nearest-centroid read-out at that layer (reported, not predictive);
- expected price: gain ≈ 100 if conduit ≥ 0.15, ≈ 300 below it;
- expected failure mode: leakage (false triggers) rather than silence.

It does **not** answer "will this work", and the tool says so in its own output.

## 6. Deviations and limits

- Measurement definition changed once, after the first cell: conduit was first measured on random token
  ids (which produced a *negative* cost on middle layers) and was switched to real text. Cells 1–3 were
  re-measured under the new definition so all eleven are comparable. The predictions were unchanged.
- The entry in the pre-registration's cell table that mispredicted the most (gpt2 L2) was predicted to fail;
  with a 0.019 conduit it succeeded. The rule, not the cell, is what is wrong.
- Success is a binary I fixed in advance (hit ≥ 80%, false ≤ 30%); three cells sit on its edge
  (gpt2 L2 at 30% false, distilgpt2 L3 at 80% hit), so the contingency is softer than the counts suggest.
- One seed, one gain grid, one λ grid, one task. A different task might separate the cells differently.

## 7. What this leaves

There is no shortcut. Four candidate pre-flight numbers were tested against eleven cells and none decides
the outcome; the only way to know whether a weight edit works is to **run it on prompts it was not tuned on
and score them**. What can be predicted cheaply is the *price* — how much push a layer needs — and that a
cheap layer will leak rather than fail quietly.

That is a useful thing to say in a field where the published results are one-shot, single-seed and
phrased as capability claims. It is also the same lesson as csinva's independent result (reading a model's
weights does not let you predict what an edit will do, +0.06 correlation over 420 edits): **the weights
tell you less about the edit than the edit's own held-out measurement does.**

If this line continues, the honest next step is not a fifth predictor — it is a *standard protocol* (fixed
prompts, held-out split, gain grid, both false-trigger and damage reported) so that the claims people
publish are comparable. That is the audit kit, and it is the deliverable that survives this run.
