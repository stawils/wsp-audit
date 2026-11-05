# RESULT — is the published exp 8 edit anything more than a steering vector?

Status: **all three 16k arms complete; verdict produced by the scorer, not by hand.**
Everything below is measured here, not read from the post.

Pre-registration: `hook/PREREGISTRATION.md` (written before any generation ran).
Raw records: `hook/runs/*.jsonl`. Scorer: `hook/04_score.py` (applies the bands mechanically).
The records carry the per-sample scores the scorer reads — window, cap, boxed answer, repetition, closed-think —
and not the generation text. Reproducing the verdict needs the former, not the latter.

---

## 1. The edit, reconstructed from bytes [decided]

| check | result |
|---|---|
| rebuild both edited tensors from `edit.npz` onto the pinned base | **sha256 equals the published `edited_sha256` for both tensors** |
| changed bf16 entries | **130,188** (65,260 + 64,928) — the post's 131,072 is the *addressed* count |
| per-column dose recovered by SVD | α = **6.0006 ± 0.0018** (layer 15), **6.0004 ± 0.0021** (layer 19) |
| rank-1 residual | 5.6e-4 / 6.4e-4 of the leading singular value (= bf16 rounding) |
| columns touched | 64 of 3584 in each layer, all 1024 rows (layer 19: 1023) |

So the edit is exactly `W' = W0 − 6·u(uᵀW0)` on 64 intermediate neurons of layers 15 and 19, with one
constant dose — no per-column variation. That independently confirms the audit's G3 (dose 6.000, not
"subtracted") and G1 (130,188 changed).

## 2. The edit is a forward hook [decided]

Rewriting it as a hook on `down_proj` — `y ← y − 6·u·(uᵀ W0[:,C] x_C)`, gate = 64 selected inputs:

| comparison | max abs Δlogit (40 prompts) |
|---|---|
| gated hook vs the bf16 weight edit | **0.414** |
| bf16 rounding alone (same edit, bf16 vs fp32 storage) | **0.3125** |
| the edit vs the unedited model | **2.125** |

`audit/audit.py equivalence` runs the same three comparisons on its own 20-prompt set and gets
0.2656 / 0.3125 / 2.2852. So the hook-vs-edit gap straddles the rounding bound depending on which prompt
set you ask: below it at 20 prompts, above it at 40. The honest reading is that **the hook and the weight
edit are the same object to within the checkpoint's own rounding, on the same side of a line the measurement
sits on** — not that the hook is a cleanly smaller perturbation. The mechanism claim does not depend on that
margin: **the published mechanism is a gated rank-1 projection on the residual, not a new object.**
Behaviourally the same: on 200 paired generations at 4096 tokens, `gated` vs `weights` differ by +8/−6
(McNemar p = 0.79).

## 3. What the gate does — the numbers the question turns on

Per-token coefficient `c = α·uᵀ(W0[:,C]·x_C)` over 2,825 real tokens per layer, against the layer's own
projection `r = uᵀy`:

| | layer 15 | layer 19 |
|---|---|---|
| CV of c | 1.31 | 2.87 |
| corr(c, r) | 0.889 | 0.868 |
| best plain steering `y − β(uᵀy)u`, β | 3.57 | 3.86 |
| variance in c **not** explained by that steer | 21% | 25% |
| tokens where the two push in **opposite** signs | 17.9% | 18.3% |

Reading: a plain projection steer at the same layers and direction reproduces roughly three quarters of
what the gate does, but **reverses the sign on about one token in five**. Whether that matters is an
empirical question, which is what the generation arms exist to answer.

## 4. Generation arms

All arms: Qwen3.5-0.8B @ 2fc06364, thinking on, T 0.6 / top_p 0.95 / top_k 20 / presence penalty 1.5
(hand-implemented — HF `generate` has none), MATH-500 problems in index order, 2 samples each.

**4096-token window, 100 problems (pre-registered primary, and it is underpowered):**

| arm | n | cap | closed-think | has box |
|---|---|---|---|---|
| base | 200 | 0.98 | 0.02 | 0.18 |
| gated (hook) | 200 | 0.94 | 0.07 | 0.17 |
| weights (published edit) | 200 | 0.93 | 0.08 | 0.17 |
| proj (β-fitted steer) | lost to a syntax error, re-queued | | | |
| bias (constant shift) | lost to the same error, re-queued | | | |

The edit is real here (gated vs base: +1/−10 pairs, p = 0.012) but tiny — 5 points of cap-rate. **A 5-point
effect cannot separate `weights` from `proj`; this window is reported as underpowered, per the
pre-registration.**

**16384-token window — where the post's effect actually lives:**

| arm | n | cap | closed-think | has box | mean tokens |
|---|---|---|---|---|---|
| base | 24 (12 problems) | 0.92 | **0.08** | 0.25 | 15,665 |
| weights (published edit) | 80 (40 problems) | 0.74 | **0.45** | 0.30 | 14,063 |
| proj (β-fitted steer) | 80 (40 problems) | 0.75 | **0.39** | 0.28 | 14,136 |

Paired, on samples present in both arms:

- **weights vs base, cap-rate: +0 / −6 of 24, McNemar p = 0.031** — the edit really does stop the model hitting the
  ceiling, at the same window. The manipulation check passes.
- **proj vs base, cap-rate: +0 / −5, p = 0.063** — the steer does the same thing, slightly less strongly.
- **proj vs weights, closed-think: +9 / −14 of 80, p = 0.40**; **cap-rate +6 / −5, p = 1.00** — no detectable
  difference between the published edit and a plain projection steer.

`weights_16k` also settles the mechanism claim on our own data: of the 24 generations that finished,
**19 were correct (79%)**; 59 of 80 hit the cap. The edit converts "never finishes" into "finishes and is
usually right" — the same shape the audit found (35/35 correct when finished unedited, 241/303 edited).

## 5. Verdict — GATE-REDUNDANT (scorer output, pre-registered bands)

```
PRE-REGISTERED VERDICT: GATE-REDUNDANT
  - base closed 0.08 (void if > 0.15)
  - weights closed 0.45
  - proj closed 0.39
  - projection steering reproduces the edit within 20%
```

`proj` reaches **0.87 ×** the published edit's termination rate, with no detectable behavioural difference in
either direction (paired p = 0.40 on termination, 1.00 on capping).

**What that does and does not mean.**

- It does **not** mean the two interventions are the same function. They are not: the gate's coefficient has
  CV 1.3–2.9, a fifth of its variance is invisible to a projection steer, and on ~18% of tokens the two push
  in *opposite* directions (subsection 3).
- It **does** mean the gate is not *necessary* at this scale and dose. What has to be right is the
  **direction, the two layers, and the dose**; which 64 of 3584 neurons carry it, and whether their
  coefficient is data-dependent, buys nothing measurable on 40 problems × 2 samples.

Combined with subsections 1–2 (the edit *is* a gated rank-1 hook, bit-exact), the exp 8 result reduces to:
**a rank-1 steering vector along one direction at layers 15 and 19, applied with dose ×6, permanently
written into the weights.** That is exactly the abliteration/steering playbook with the direction and dose
chosen by a language model — which is a real result about LLM reasoning, and not a new optimisation paradigm.

The 4096 window cannot adjudicate any of this and never could; it is reported only because it was
pre-registered (weights 0.93 vs base 0.98 cap; gated vs base +1/−10 pairs, p = 0.012).

## 6. Deviations from the pre-registration, stated

- Windows 4096 / 16384 instead of the post's 32768; 40–100 MATH-500 problems instead of 500.
- 2 samples per problem instead of 4.
- Presence penalty hand-implemented (HF has no native option); vLLM unavailable, so throughput is
  transformerd-based and the hybrid-attention kernels run as reference fallbacks.
- Accuracy is a normalised string match, not math-verify; **no MATH-500 accuracy claim is licensed** — the
  79%-of-finished figure is reported for the mechanism, not as a benchmark score.
- Schema boundary: `base`, `weights`, `gated` rows (4096) predate the text patch and carry no `text`/`rep8`;
  `proj`, `bias` and all 16k rows do.
- `proj` and `bias` at 4096 were lost to a `global MAXNEW` SyntaxError (my bug, caught at 14:36); they are
  re-queued behind the 16k arms.

## 7. What is still owed

1. **A second seed.** Every number here is one seed (seed 1000+block index, per block). The proj-vs-weights
   p-values would not survive repeated peeking; the verdict rests on the pre-registered band, not on a test.
2. **`bias`** — a constant residual shift, the null arm. Not run: the 4096 window it was queued in is
   underpowered, and the 16k window had no budget left.
3. **A harder test of the gate**: a failure that is not a single direction, where the data-dependent
   coefficient is the only thing that could carry information. This result bounds the *loop* case; it says
   nothing about a case where the edit's coefficient actually varies with content that matters.
4. **The additive result from the web** (tmestery: 0% → 90% on a new trigger skill) is the lead worth
   chasing; it contradicts the post's "cannot add ability" and our finding does not touch it.
