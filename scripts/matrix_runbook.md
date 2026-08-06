# C-11 tier-1 matrix — Colab runbook

What to run, in what order, what to check after each, and how the results
combine. Everything here runs on Colab; nothing in it runs locally.

The budget is **10,800 s (3 h) per run** and is **not binding** — measured, see
step 0. Raise it freely if a cell needs more; nothing in the matrix is
compute-constrained.

---

## 0. Budget: settled by measurement, not by estimate

Two real T4 runs, both `converged_plateau`:

| cell | epochs | wall (s) | % of 10,800 | s/epoch |
|---|---|---|---|---|
| `cod_no_baseline` | 7,000 | 2,835 | 26% | 0.405 |
| `fno_in_cascade` | 4,700 | 1,161 | 11% | 0.247 |
| COD (O-5, reference) | 11,900 | 4,911 | 45% | 0.413 |

**The budget does not bind.** The worst case so far uses 45% of it.

**And the CPU-based cost ranking that motivated the original pilot was wrong in
direction.** It predicted FNO at 1.79-1.91x COD per step; on a T4 FNO is
**0.60x** COD (0.247 against 0.413). The FFT parallelises far better on GPU than
on CPU, and the local ranking inverted the order. That estimate is left in
DECISIONS C-11 with this correction attached rather than quietly deleted: the
lesson is that a CPU cost ranking does not transfer to GPU even ordinally, so
`s/epoch` from a real run is the only number to plan with.

Still unmeasured on GPU: MIONet and S-DeepONet. Both are expected cheap (MIONet
was the cheapest on CPU by a wide margin; S-DeepONet is ~1.2x COD after the
branch caching), but expected is not measured. Check `stop_reason` on the first
seed of each; if it says `wall_clock_budget`, raise the budget and rerun that
cell only — the others are unaffected because the budget is per run.

---

## 1. Confirm every architecture still round-trips on this machine

Cheap, and it catches an environment difference before 35 runs depend on it. A
cell that trains for three hours and then cannot reload its weights is the O-5
failure repeated.

```bash
!python audit_port/scripts/25_checkpoint_roundtrip.py --max-epochs 15 --n-ic 24 --n-test 6
```

Expect `8/8 architectures round-trip their checkpoints correctly` — COD plus the
seven configs in `configs/matrix/`. Any `FAIL` line names the config that failed;
fix before proceeding. One failing cell no longer aborts the rest, so the summary
at the end is complete even when something breaks.

---

## 2. The matrix

Seven configs x **7 seeds** = 49 runs. At the measured 1,161-2,835 s per run
that is roughly **25-35 GPU-hours**, not the 105 estimated when a run was
assumed to take the full 3 h.

**Why 7 and not 5.** Wall clock stopped being the constraint, and the reason to
want more seeds is concrete rather than theoretical: N-11 is the case where a
single-seed median of 13.18 degC looked solid and the pooled figure across six
seeds was 11.63, outside the entire between-seed range. Seven is odd, so the
median is an actual run rather than an interpolation between two; it is enough
to quote a range instead of a mean; and past about ten the return falls off
faster than the cost. C-11 assumes several Colab
accounts writing into one Drive directory, so the order below is by *priority*,
not by dependency — any account can take any line.

Seeds are `1..7`, set with `run.py --seed`. `run.py` refuses to overwrite an
existing run directory, so a repeated line is an error rather than a silent
clobber.

`--seed` is a **production** override and deliberately does not mark the run as a
smoke test — a seed sweep is the real experiment. It reaches both the global RNG
(weight initialisation) and the trainer's own batch-order generator; see the
warning below for why both matter.

```bash
for CFG in fno_in_cascade fno_monolithic \
           mionet_in_cascade mionet_monolithic \
           sdeeponet_in_cascade sdeeponet_monolithic \
           cod_no_baseline; do
  for SEED in 1 2 3 4 5 6 7; do
    python scripts/run.py --config configs/matrix/$CFG.yaml \
        --freeze-hash fc4cb76c3b32ec17 \
        --max-wall-seconds 10800 \
        --seed $SEED \
        --tag s$SEED \
        --out /content/drive/MyDrive/cod_matrix
  done
done
```

> **Both `--seed $SEED` and `--tag s$SEED` are required, and they do different
> jobs.** `--tag` names the output directory; `--seed` changes what is computed.
> An earlier version of this loop had only `--tag`, which meant all seven runs
> would have trained on `training.seed` from the config — seven bit-identical
> results in seven differently-named directories, and the `--overwrite` guard
> would not have fired because the tags differed. Verified fixed: two seeds of
> `mionet_in_cascade` on an identical config hash gave 25 of 31 weight tensors
> differing and loss curves diverging from step 0, and a direct test with weights
> held fixed showed the drawn batches differ, so the seed reaches the batch-order
> generator and not just initialisation.
>
> The general shape to watch for in this file: **a loop variable that changes the
> output path without changing what is computed.** It produces a full-looking set
> of results that are all the same run.

### The factorial and variant cells

The loop above is the six-cell matrix plus Ablation A. The 2×2 factorial cells
(PORT_LOG J-92) and the bounded-correction variant (ANALYSIS_PLAN Amendment 2)
are eight further configs, run the same way:

```bash
for CFG in fno_baseline_in_cascade fno_baseline_monolithic \
           mionet_baseline_in_cascade mionet_baseline_monolithic \
           sdeeponet_baseline_in_cascade sdeeponet_baseline_monolithic \
           pideeponet_baseline_monolithic \
           cod_bounded_correction; do
  for SEED in 1 2 3 4 5 6 7; do
    python scripts/run.py --config configs/matrix/$CFG.yaml \
        --freeze-hash fc4cb76c3b32ec17 \
        --max-wall-seconds 10800 \
        --seed $SEED \
        --tag s$SEED \
        --out /content/drive/MyDrive/cod_matrix
  done
done
```

**Priority order if compute runs short.** The in-cascade / monolithic *pair* is
the one-variable test C-11 exists for, so always complete a pair before starting
another architecture. One seed of all six cells is worth more than five seeds of
one cell.

1. `fno_in_cascade` + `fno_monolithic` — the pair most likely to show the cascade
   effect, and the budget-risk cell.
2. `mionet_in_cascade` + `mionet_monolithic` — cheapest, so highest runs per hour.
3. `sdeeponet_in_cascade` + `sdeeponet_monolithic` — carries the largest
   adaptation (PORT_LOG J-90), so its result needs the most careful reading.
4. `cod_no_baseline` — DECISIONS O-12, Ablation A. Not part of the six-cell
   matrix and not paired with anything: its comparison is against **COD itself**
   on the same seeds. One seed is enough to be informative, seven to be
   reportable. It uses `loop: train_v34`, COD's own trainer, because a different
   trainer would be a second variable.

**After each run, check three things before moving on:**

- `outcome.stop_reason` — anything other than `converged_plateau` is reported as
  non-converged, not converted into a performance number (README rule 5).
- `outcome.pathology_warnings` — and if a clamp is listed, open
  `figures/clamp_trajectories.pdf`. A spike that decays in the first few hundred
  epochs is benign; a flat non-zero level is not (PORT_LOG J-85, J-89).
- `figures/learning_curves.pdf` — the right panel is the validation series the
  plateau test actually decided on.

---

## 3. What each run leaves behind

Per run directory, all of it needed:

| file | why it cannot be regenerated |
|---|---|
| `model.pt` | the weights; `18_swing_fidelity.py` and `24_rollout_thermal_error.py` both need them |
| `run.json` | provenance — commit, both hashes, GPU, tier source — plus metrics and the validation curve |
| `predictions.npz` | per-case `pred`/`gt` in float64, so every evaluation table rebuilds offline |
| `loss_history.json` | the training curve |
| `clamp_history.json` | clamp and causal-weight trajectories: *when*, which no summary answers |
| `figures/` | PDF + SVG, per the CLAUDE.md conventions |

---

## 4. How the results combine

Locally, after pulling the Drive directory:

```bash
# Per cell, the C-11 honesty protocol: three tables, not just MAE.
# --out is REQUIRED in a sweep. Both scripts default to a single fixed path in
# audit_port/, which is right for a one-off and destructive in a loop: 15 cells x
# 7 seeds would overwrite one file 105 times and keep the last. That is the same
# defect as the --tag/--seed one above with the arrow reversed -- there the loop
# variable did not reach the computation, here it does not reach the output path.
python audit_port/scripts/18_swing_fidelity.py --checkpoint <run>/model.pt \
    --out <run>/swing_fidelity.md
python audit_port/scripts/24_rollout_thermal_error.py --checkpoint <run>/model.pt \
    --max-windows 730 --k-scenarios 0.95 --json-out <run>/rollout.json \
    --out <run>/rollout_thermal_error.md
```

C-11 requires every model to report **three** things, not one:

1. the stratified swing table, by realised swing band;
2. the Jensen gap along the trajectory, all six states;
3. thermal MAE — read **together with** the first two, never alone.

Gas percentages are **not** reportable from this benchmark for any cell (C-9, as
tightened 2026-08-02): the median 12 h variation is 0.001%-0.046% of each gas's
engineering threshold, so gas NMAE is a ratio to a physically empty quantity.
Report absolute ppm against IEC 60599.

**The comparison that matters is within an architecture, not across.** For each,
`monolithic` against `in_cascade` on the same seeds is the one-variable test: if
the monolithic form fails and the in-cascade form does not, the cascade is what
was missing and the architecture is not at fault. Across architectures the
comparison is confounded by the adaptations recorded in PORT_LOG J-90 — most
severely for S-DeepONet, whose trunk was moved from spatial coordinates to time.

**Seeds are reported as a distribution, not a mean.** Seven seeds, and a
non-converged seed is reported as non-converged rather than dropped; dropping it
would select for the seeds that happened to work.

### The aggregator

Every table ANALYSIS_PLAN specifies is built by one script, from the `run.json`
files alone:

```bash
python scripts/aggregate_results.py \
    --results /path/to/cod_matrix \
    --out audit_port/MATRIX_RESULTS.md
```

It applies the plan's rules rather than leaving them to be remembered: the
convergence rate before the error distribution, median and full min-max never a
mean, §3's two bars with the four pre-approved verdicts, §1's 2x thermal
confound control before any gas comparison counts, and the 2x2 factorial with
main effects and the interaction — reported as `n/i`, not as a number, whenever
a cell of the quadrant is missing.

**It exits non-zero on an integrity problem** and the problems are listed at the
end of the report: a run that resolves to no cell, two runs of one cell sharing
a seed, a cell whose runs disagree about the evaluation tier, a run whose
`converged` flag contradicts its `stop_reason`. Read the exit code — a report
that renders is not the same as a report that is safe to quote. Runs on a
distribution other than `fc4cb76c3b32ec17`, and smoke tests, are segregated
before anything is grouped and counted in §0.

Two things it will not do, both deliberate and both stated in its §7: it prints
no gas percentage anywhere (C-9), and it does not substitute the 12 h gas MAE
for Amendment 1's end-of-rollout gas ppm, which no script currently emits.

`audit_port/scripts/37_aggregator_sentinel_check.py` is its verification: the
J-89 checklist including step 6, run against synthetic run directories built to
be degenerate. Run it after touching either file.
