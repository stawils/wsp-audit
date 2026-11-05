# Pre-registration — does a cheap pre-flight test predict whether a weight edit can work?

Written 2026-11-04 19:45, **before** the new cells below were run. The three cells already observed
today (gpt2 L11, gpt2-medium L23, distilgpt2 L5) are marked OBSERVED and are not predictions.

## The claim under test

An additive weight edit at layer L can work only if two cheap measurements both say yes:

- **conduit(L)** = how much the model's language-modelling loss gets worse when L's MLP output is zeroed.
  A layer that can be removed for free cannot carry a new behaviour, because nothing written there reaches
  the logits.
- **decodability(L)** = leave-one-out accuracy of a linear probe for the target behaviour at L's MLP output.
  If the behaviour is not linearly present there, a rank-≤25 linear map cannot condition on it.

Predicted rule: **edit succeeds iff conduit(L) ≥ 0.15 nats AND decodability(L) = 1.0.**
Secondary, quantitative: the gain a successful edit needs scales **inversely** with conduit(L)
(Spearman ρ < 0 on the successful cells).

## The intervention, held constant across architectures

A rank-≤25 additive map at layer L, fitted by ridge from that layer's MLP output to the tied unembedding
directions of the four target tokens, controls constrained to zero. It is applied as a hook
`y ← y + y·M` (equivalent to a weight edit, as established in `hook/RESULT.md` §2). This is a canonical
form of the published intervention, not its exact parameterisation — noted as a deviation.

Success = held-out trigger-hit rate ≥ 0.8 with false-trigger rate ≤ 0.3, at some gain in the swept grid,
with gain and λ chosen on the **probe** prompts only.

## Cells (predictions made before running; 8 new)

| # | model | layer | conduit predicted | decodability predicted | edit predicted |
|---|---|---|---|---|---|
| 1 | gpt2 | 11 | OBSERVED +0.30 | OBSERVED 1.0 | OBSERVED succeeds |
| 2 | gpt2-medium | 23 | OBSERVED +0.30 | OBSERVED 1.0 | OBSERVED succeeds, costly |
| 3 | distilgpt2 | 5 | OBSERVED −0.04 | OBSERVED 1.0 | OBSERVED fails |
| 4 | gpt2 | 6 | ≥0.15 | 1.0 | **succeeds** |
| 5 | gpt2 | 2 | <0.15 | 1.0 | **fails** (no usable operating point) |
| 6 | distilgpt2 | 3 | ≥0.15 | 1.0 | **succeeds** |
| 7 | gpt2-medium | 12 | ≥0.15 | 1.0 | **succeeds** |
| 8 | pythia-160m | 11 | ≥0.15 | 1.0 | **succeeds** |
| 9 | pythia-160m | 5 | <0.15 | 1.0 | **fails** |
| 10 | Qwen2.5-0.5B | 23 | ≥0.15 | 0.9–1.0 | **succeeds** |
| 11 | Qwen2.5-0.5B | 12 | ≥0.15 | 1.0 | **succeeds** |

Cross-architecture cells (8–11) are the real test: the same recipe on a different tokenizer, norm and
attention family. Predictions there are less confident and are marked as such.

## Reading, fixed in advance

- **Holds** if all 8 predicted cells land as predicted, or if the only misses are cells within 0.05 nats of
  the 0.15 threshold (a boundary error, not a rule error).
- **Fails** if two or more cells with conduit ≥ 0.15 do not produce a working edit, or if a cell with
  conduit < 0.15 does.
- A cell where decodability < 1.0 but the edit works would also falsify the second half of the rule.

## What this cannot show

Nothing about whether the edit is *useful* — only whether it is possible. The tasks are token-local
lookups by construction; a reasoning-level behaviour is not tested and today's evidence says it will not be
achieved by this class of edit.
