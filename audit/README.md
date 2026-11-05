# audit kit — check a published weight edit before you believe it

One command, four checks, no assumptions about the paper it came from.

```bash
.venv/bin/python audit/audit.py all              # exp 8 end to end, ~3 min
.venv/bin/python audit/audit.py verify           # bytes only, seconds, no model
.venv/bin/python audit/audit.py cases            # stored experiment records, instant
.venv/bin/python audit/selftest.py               # asserts the known answers, must print ALL PASS
```

## What each check answers

| check | question | cost |
|---|---|---|
| **verify** | Do the released bytes say what the release claims? Rebuilds the edit from its sparse diff, compares sha256 to the published hashes, counts *addressed* entries versus entries whose bits actually changed, recovers the true dose (per-column α and its spread) and the rank-1 residual. | seconds, CPU |
| **equivalence** | Is the edit something a forward hook could have done? Applies the edit to the weights, applies the same thing as a hook, diffs the logits, and compares that diff to the bound set by bf16 rounding alone. | ~1 min, needs the model |
| **steer** | How much of the edit does a plain, dose-matched projection steer reproduce? Fits the best β for `y ← y − β(uᵀy)u` on real prompts; reports variance explained and the share of tokens where the two push in opposite directions. | ~1 min, needs the model |
| **cases** | Re-scores stored experiment records and prints each verdict, so a claim can be checked against prior outcomes rather than memory. | instant |

Output is also written to `audit/audit_report.json` by `all`, and only by `all` — a single-command run writes
nothing unless you pass `--json-out`, so checking one thing can never truncate the four-section artifact.

## Three released edits, three different claim shapes

`--edit <run>/edit` points **verify** at any of the three edits in the exp 8 log repo. **equivalence** is
implemented for the p50c shape only — one rank-1 direction is the thing the check asks about; p10c is 30
lowrank tensors and p25c zeroes 256 attention columns, and it exits with that reason rather than guessing.
All three rebuild to their published hashes; what they *are* differs sharply:

| edit | tensors | form | changed entries | what it does |
|---|---|---|---|---|
| **p50c** (the headline, 48.1%) | 2 | sparse, 64 columns each | 130,188 | one rank-1 direction, one constant dose α = 6.0006 ± 0.0018 |
| **p10c** (12.3%) | **30** | lowrank, rank 1 each | **72,986,982** | a rank-1 push across 30 tensors — ~9.7% of the model, no single direction |
| **p25c** (14.1%) | 1 | sparse | 262,144 | 256 output columns of one attention head set to exactly zero |

That table is the argument for the kit: the three edits differ by **560×** in how much of the model they touch,
and only the smallest one is the one the abstract describes.

## What it finds on the headline edit (exp 8, Qwen3.5-0.8B)

```
the edit is exactly what the release claims      True    (both tensors rebuild to the published sha256)
it applies one constant dose                     True    (α = 6.0006 ± 0.0018, 6.0004 ± 0.0021)
it is expressible as a hook                      True    (Δlogit 0.2656 vs a 0.3125 bf16 bound; 2.285 for the edit itself)
a dose-matched steer reproduces 78% of it        and disagrees in sign on 18% of tokens
addressed 131,072 ≠ changed 130,188              884 entries round back to bf16
```

The last line is the one a reader would otherwise get wrong: the release prints the *addressed* count and
calls it the number of changed weights.

The Δlogit figure is measured on 20 MATH-500 prompts, the same set for all three columns. `hook/02_probe.py`
runs the same comparison on 40 prompts and gets 0.414 — above the bound rather than below it. Both numbers
are real; the pair is the honest statement of how close to rounding this actually sits, and neither should
be quoted without its prompt count.

## What it deliberately does not do

It does not decide whether an edit is *good*, *useful*, or a *new mechanism*. Those need behavioural runs,
and `hook/RESULT.md` shows the band that was fixed in advance for the one case where that was done. The kit
only refuses to let a claim be checked against nothing.

## Self-test

`audit/selftest.py` asserts every number above plus four outcomes measured independently today — the gpt2
and Qwen cells that work, the gpt2 and distilgpt2 cells that fail — so a change to the kit that breaks a
verdict fails loudly instead of quietly agreeing with itself.
