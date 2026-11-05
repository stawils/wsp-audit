# Weight-space programming: an audit, four experiments, and the numbers

An independent check of [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026), plus three follow-up experiments and the tooling to
repeat them.

This is not a reproduction of the paper's demos. It is an attempt to answer a narrower question that the
coverage did not: **what is the published edit, mechanically, and what has to be true for it to work?**

## What came out of it

| finding | evidence |
|---|---|
| The headline edit is one rank-1 direction at two layers with a **constant dose α = 6.0006 ± 0.0018**, bit-exactly reproducible from the released diff. | `audit/` §verify |
| Re-expressed as a forward hook it moves logits by **0.2656** on a 20-prompt set, below the **0.3125** that bf16 rounding of the checkpoint alone produces — and **0.414** on a 40-prompt set, above it. The "mechanism" is an activation intervention, but it sits close enough to rounding that the margin, not the verdict, is the finding. | `audit/` §equivalence, `hook/` §2 |
| A plain, dose-matched projection steer gets **39%** termination against the edit's **45%** (paired +9/−14, p = 0.40). The gate is not identical to a steer, and it is **not necessary**. | `hook/` |
| The three released edits differ by **560×** in breadth: 130,188 / 262,144 / 72,986,982 entries. The abstract describes the smallest. | `audit/` §verify, three runs |
| The untouched model scores **44.0%** on MATH-500 under its own card's settings, against the post's 7.9% baseline. The honest gain is **+3 to +4 points**, not 6×. | external: pre-registered audit, not in this repo |
| Weights **can** add a token-level lookup (0% → 100% on unseen phrasings) and **cannot** add a reasoning ability. The boundary is whether the edited layer reaches the logits. | `additive/` |
| No cheap pre-flight number predicts whether an edit will work (4 candidates, 11 cells, 3 model families). Layer importance predicts only the *price* (Spearman −0.87). | `preflight/` |

Each finding below links to the directory that produced it.

## The artifact: six questions before you believe a weight-edit claim

1. Does the diff rebuild the published hash?
2. Addressed or changed? (Multiply touched columns by rows.)
3. What is the dose, and is it constant?
4. Is it expressible as a forward hook? (Compare against the bf16 rounding bound.)
5. What does a plain steer at the same dose do?
6. What is the baseline under the *model's own* recommended settings?

And one the artifacts answer in a minute: **what was the model told at turn one?** In the winning run the
brief names the hypothesis ("if the model often fails because it repeats itself and never finishes, finding
and fixing what drives that repetition counts", `runs/p50c/turns/t001/prompt.txt` line 5).

## Reproduce

```bash
python scripts/fetch_data.py          # MATH-500 (500 rows) + the released logs repo
python audit/audit.py verify --edit sources/lm_understand-logs/exp8_math/runs/p50c/edit
python audit/audit.py all             # + hook equivalence + dose-matched steer (needs the base model)
python audit/selftest.py              # asserts 23 known answers; must print ALL PASS
```

Generation arms (one consumer GPU, ~15 GB VRAM for the 0.8B model). The `hook/` pipeline runs in order, and
step 1 must come first — it writes `hook/edit_witness.npz`, which 02, 03 and `audit/audit.py` all load:

```bash
python hook/01_reconstruct.py         # bytes -> the rank-1 witness (u, columns, dose) per layer
python hook/02_probe.py               # is the gate a hook, and does it vary with the input?
python hook/02b_stats.py              # coefficient distribution; separates rounding from formula error

python hook/03_generate.py weights --limit 40 --samples 2 --bs 8 --maxtok 16384 --tag _16k
python hook/03_generate.py proj    --limit 40 --samples 2 --bs 8 --maxtok 16384 --tag _16k
python hook/04_score.py               # paired counts + the pre-registered bands -> GATE-REDUNDANT
```

`hook/run_all.sh` drives the five 4096-window arms in one go. `hook/04_score.py` reads
`hook/runs/*.jsonl`; the 16k rows ship with per-sample scores but without generation text, which is all the
scorer needs.

The additive replication and the eleven-cell pre-flight study:

```bash
python additive/replicate_ping_pong.py --model gpt2
python additive/mlp_influence.py
python preflight/preflight.py --model gpt2 --layer 11
```

## What is in here, and what is not

| included | why |
|---|---|
| `audit/` | the six checks as one CLI, plus a self-test with 18 assertions |
| `hook/` | pre-registration, generator, scorer, 784 raw generation records |
| `preflight/` | 11 cells × 4 candidate predictors, all records |
| `additive/` | the rebuilt additive edit, its three models, and two diagnostics |

| not included | why |
|---|---|
| model weights (2.7 GB), the released logs repo (367 MB), third-party clones | fetched by `scripts/fetch_data.py`; pinning beats vendoring |
| the full MATH-500 dataset | a downloader, not a redistribution |
| the paper's own ~4,000 generations | they were never released; only flags and 440 texts |

## Honesty section

Every behavioural number here is **single-seed**: 40 problems (12 for the base arm), 2 samples, one gain grid.
The verdict on the gate rests on a band written down before the run, not on a p-value that survives repeated
peeking. Three of the eleven cells sit on the edge of the success definition. Two of the seven findings are
other people's runs, cited and not repeated.

The audit checks what an edit **is**. It cannot tell you whether it is good; that needs held-out generation,
which is the point of the pre-registration.

## Licence

Code: MIT (`LICENSE`). The released logs are CC BY 4.0 / MIT per their repository; the reproduction datasets
and articles cited belong to their authors. Figures are generated from the records in this repository.
