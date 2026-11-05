# Replicating the additive WSP claim (tmestery/circuit-injection, Exp II)

Question: does anything in WSP *add* an ability, as opposed to removing an obstacle? The post's own
follow-up says no ("editing weights could remove an obstacle that hid an ability, but it could not add an
ability the model lacked"), and our exp 8 hook test never tests it. This is the one published claim that
says yes: a training-free edit installs `<<PING>>` → `PONG-42`, **0% → 90%** held-out, 10% false triggers.

Reproduced from the repo on 2026-11-04, one GPU, no gradients, ~4 minutes wall clock in total.

## 1. What the repo ships, and what it does not

| shipped | missing |
|---|---|
| task + probe/held-out prompt sets (`tasks/ping_pong.py`), metric, sandbox, their own output JSONs | **the tensor patch**. `patches/astra6_ping_pong.safetensors` is gitignored; only the manifest is present |

So the claim cannot be reproduced by replaying an artifact — it can only be tested by rebuilding the edit
from the recipe in `outputs/astra6_ping_pong_report.md`:

> single closed-form ridge solve; rank-≤25 map added to `transformer.h.11.mlp.c_proj.weight`; inputs are
> final-MLP activations for five `<<PING>>` contexts at each output prefix ("", "P", "PONG", "PONG-");
> targets are the tied unembedding directions for `P`, `ONG`, `-`, `42`; five no-trigger controls
> constrained to zero.

`additive/replicate_ping_pong.py` implements exactly that (dual-form ridge, last-position c_proj input,
goal directions unit-normalised, controls targeted at zero), splits tuning from scoring as the repo does
(gain and λ chosen on the **probe** prompts, the held-out set scored once), and imports **their** metric,
**their** prompts and **their** generation settings unchanged.

## 2. Result — reproduces on GPT-2 small, and it needed a knob nobody documented

| model (last MLP) | held-out hit | false trigger | random NLL (base → edited) |
|---|---|---|---|
| gpt2 — their claim | **90%** | 10% | 13.113 → 13.217 (**+0.10**) |
| gpt2 — this rebuild | **100%** | 20% | 13.113 → 13.400 (+0.29) |
| gpt2-medium — same recipe | **100%** | 20% | 13.243 → 15.996 (**+2.75**) |
| distilgpt2 — same recipe | 70% | **100%** | 13.449 → 13.946 — no usable operating point |

The claim holds at the model it was built on: ten unseen phrasings of the trigger all produce `PONG-42`
immediately, and the base model never does (0/10). Two of the ten matched *negatives* — `[tool]
ping_received=false` and `prefix-middle--suffix` — fire too, which is exactly what a lookup on the trigger's
scaffolding would do.

**But the recipe as written is not scale-free.** Nothing in the report mentions a gain, yet with unit target
directions nothing happens at gain ≤ 30 and everything happens at gain 100 (gpt2) or 300 (gpt2-medium).
The magnitude of the written map is a free parameter, and my rebuild had to tune it on the probe prompts.
`|ΔW|/|W|` ends up 0.15 (gpt2), 0.27 (medium), i.e. a large edit, not a 0.016%-style pinpoint one.

## 3. Why it fails on distilgpt2 — measured, not guessed

First hypothesis, which was wrong: the trigger might not separate linearly in late-MLP activations.
Leave-one-out linear separability at the edited layer is **1.0 for all three models**.

Second probe, which explains it: zero the edited layer's output and see if the logits notice.

| model | last layer | NLL with last MLP zeroed | Δ |
|---|---|---|---|
| gpt2 | 11 | 13.416 | **+0.30** |
| gpt2-medium | 23 | 13.539 | **+0.30** |
| distilgpt2 | 5 | 13.413 | **−0.04** |

In distilgpt2 the last-layer MLP contributes essentially nothing to the logits, so a readout written there has
to be enormous before it moves anything — and at that magnitude it fires on every prompt. The failure is
structural, not a tuning artefact.

**So the precondition for an additive edit of this kind is a live conduit from the edited layer to the output.**
The trigger being linearly decodable is necessary but not sufficient; the layer also has to matter.

## 4. Verdict

- **The additive claim is real for what it is**: a training-free, one-shot, rank-≤25 edit installs a new
  stimulus→response mapping that generalises to unseen phrasings. It is not a reorganisation of an existing
  circuit; the behaviour did not exist before.
- **It is also much narrower than "adding an ability".** What is installed is a key→value lookup into the
  unembedding: the model emits four fixed tokens when a string is present. It fails on distilgpt2 for a
  mechanical reason, and on gpt2-medium it costs 2.75 nats of language-modelling quality — the edit is paid
  for, not free.
- **This does not overturn the post's negative conclusion; it sharpens it.** Weights can add a token-local
  mapping (this result) and cannot add a reasoning ability (the post's Gemma control, our loop result,
  csinva's negative). Both statements are now supported by measurement, and the boundary between them is
  the interesting object: *how much of the model's behaviour passes through the layer you are editing.*

## 5. Deviations and what is still owed

- The exact patch is unavailable, so this compares **recipe against recipe**, not artifact against artifact.
  My gain was selected on probe prompts; theirs may differ (their `probe_budget_remaining` = 3 of 50).
- One seed, one λ/gain sweep grid (λ ∈ {1e-3, 1e-1}, gain ∈ {3, 10, 30, 100, 300}).
- Only the last layer was tried. A middle layer with a live conduit and a larger probe set would test whether
  the *rank* or the *location* is the binding constraint.
- Their Exp I (induction-circuit graft relocation, 0.089 → 0.088 mean induction) was not attempted; it is a
  different claim (relocation, not invention) and needs no replication to interpret this one.

## Files

- `additive/replicate_ping_pong.py` — the rebuild (13 s on gpt2, 22 s on gpt2-medium)
- `additive/replicate_result.json`, `result_distilgpt2.json`, `result_gpt2medium.json` — per-prompt generations
- `additive/separability.py`, `additive/mlp_influence.py` — the two diagnostics
- `additive/circuit-injection/` — their repo, unmodified
