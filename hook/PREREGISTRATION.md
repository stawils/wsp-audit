# Pre-registration — hook-equivalence test for the exp 8 weight edit

Written 2026-11-04, **before** any generation arm was run. Fixed here: the arms, the endpoints, the
stopping rule and the reading. Deviations get recorded, not silently absorbed.

## Question

The published edit is `W' = W0 − α·u(uᵀW0)` on 64 of 3584 columns of `down_proj` in layers 15 and 19.
That is a rank-1 projection on the residual, **gated by 64 down_proj inputs** (verified: α = 6.0006 ± 0.0018
and 6.0004 ± 0.0021, constant across all 64 columns; both edited tensors reproduce the published sha256 bit-exactly).

So the mechanism claim is: *this edit is expressible as a forward hook*. That is now settled by construction.
The open question this run answers is the one that decides the paradigm claim:

> **Does the gate do any work?** If a plain, dose-matched, ungated steering at the same layers and along the
> same direction also cures the loops, then "weight-space programming" here reduces to activation steering
> with a well-chosen direction, and the circuit-level part (which 64 neurons, weighted by their own output)
> is decoration.

## Arms (all on Qwen/Qwen3.5-0.8B @ 2fc06364, thinking on, same prompt, same order, same seeds)

| arm | mechanism |
|---|---|
| `base` | no edit |
| `weights` | the published edit, bit-exact bf16 from `edit.npz` (hf checkpoint equivalent) |
| `gated` | the same function as a forward hook in fp32 (B re-expressed; consistency control) |
| `proj` | `y ← y − β·(uᵀy)·u` at layers 15 and 19, β per layer fitted by least squares to the gated coefficient on the probe tokens |
| `bias` | `y ← y − m·u`, m = mean of the gated coefficient (a constant residual shift) |

Direction `u` is the same object in every edited arm; only the coefficient rule differs.

## Sampling (post protocol, with one documented substitution)

temperature 0.6, top_p 0.95, top_k 20, **presence_penalty 1.5** (implemented as a logits processor — HF
`generate` has no native presence penalty), seed 0, max_new_tokens **4096** (the post used 32768; 4096 is the
"does it terminate early" window and is the affordable one), prompt = problem + "Please reason step by step,
and put your final answer within \boxed{}." , thinking enabled.

## Data

MATH-500 (`HuggingFaceH4/MATH-500`, test split), **the first 100 problems by index**, 2 samples each → 200
generations per arm. This is a subset, not the benchmark; no claim will be made about MATH-500 accuracy.

## Endpoints

- **P1 (primary): termination.** Fraction of generations that emit the closing `<｜end▁of▁thinking｜>` tag before the
  4096 cap. Grader-free.
- **P2: boxed answer.** Fraction with a `\boxed{...}` in the final channel.
- **P3 (secondary, underpowered): accuracy** on a normalised string match against the gold answer. Reported
  as a direction of effect only, never as an accuracy number for MATH-500.

## Pre-registered reading

Compare against `base` first (manipulation check) and then against `weights` (the question).

- `base` termination expected ≲ 15%: if it is not, the loop regime did not reproduce at this window/settings
  and **the whole run is void** — report that, do not reinterpret.
- `gated` vs `weights`: termination rates within 10 points and logits within bf16 rounding → the hook
  formulation is confirmed.
- **Then the verdict on the gate:**
  - `proj` termination ≥ 0.8 × `weights` termination and P3 in the same direction → **gate not load-bearing;
    the edit is activation steering.** Paradigm claim weakened.
  - `proj` termination ≤ 0.4 × `weights` termination, `bias` likewise low → **the gate carries the effect;
    a steering hook along the same direction cannot reproduce it.** Paradigm claim survives in a narrower form.
  - anything between → **mixed**, reported as a dose-sensitivity result, and the honest conclusion is that
    this window cannot separate mechanism from dose.

## What would falsify my own setup

If `gated` and `weights` diverge by more than the bf16 rounding bound (~0.4 logit max, measured on 40 prompts),
my hook is wrong and P1/P2/P3 for the hook arms are void.
