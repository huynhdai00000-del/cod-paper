# C-11 tier-1 matrix — Colab runbook

What to run, in what order, what to check after each, and how the results
combine. Everything here runs on Colab; nothing in it runs locally.

The budget is **10,800 s (3 h) per run**, proposed with reasoning in DECISIONS
C-11 and **not yet confirmed** — step 0 is what confirms it. Do not start step 2
until step 0 has passed, because a budget that binds invalidates every run made
under it.

---

## 0. Pilot: one FNO seed, before committing anything

FNO is the most expensive cell per step (1.91x COD on the local ranking), so at a
shared budget it gets the fewest epochs — about 14,000 against the 11,900 COD
needed to converge. It is therefore the cell most likely to be stopped by the
budget rather than by convergence, and a baseline stopped by the budget says
nothing about its architecture.

```bash
!git clone https://github.com/huynhdai00000-del/cod-paper.git
%cd cod-paper
!pip install -e . --quiet

!python scripts/run.py --config configs/matrix/fno_in_cascade.yaml \
    --freeze-hash fc4cb76c3b32ec17 \
    --max-wall-seconds 10800 \
    --out /content/drive/MyDrive/cod_matrix --tag pilot
```

**Check, and this is the whole point of the pilot:**

| field in `run.json` | what it must say |
|---|---|
| `outcome.stop_reason` | `converged_plateau` |
| `outcome.converged` | `true` |
| `outcome.wall_seconds` | comfortably under 10800, not 10799 |
| `config.distribution_hash` | `fc4cb76c3b32ec17` |
| `evaluation.tier_source` | `realistic_sampler` |

If `stop_reason` is `wall_clock_budget`, **stop**. Raise the budget, record the
new figure in DECISIONS C-11, and restart from here. Three hours spent here
avoids discarding roughly 120 GPU-hours of matrix runs made under a budget that
binds.

---

## 1. Confirm the six cells still round-trip on this machine

Cheap, and it catches an environment difference before 40 runs depend on it. A
cell that trains for three hours and then cannot reload its weights is the O-5
failure repeated.

```bash
!python audit_port/scripts/25_checkpoint_roundtrip.py --max-epochs 15 --n-ic 24 --n-test 6
```

Expect `6/6 architectures round-trip their checkpoints correctly` — seven
configs including COD. Any `FAIL` line names the config that failed; fix before
proceeding.

---

## 2. The matrix

Six cells x 5 seeds = 30 runs at 3 h = 90 GPU-hours. C-11 assumes several Colab
accounts writing into one Drive directory, so the order below is by *priority*,
not by dependency — any account can take any line.

Seeds are `1..5`, set with `training.seed`. `run.py` refuses to overwrite an
existing run directory, so a repeated line is an error rather than a silent
clobber.

```bash
for CFG in fno_in_cascade fno_monolithic \
           mionet_in_cascade mionet_monolithic \
           sdeeponet_in_cascade sdeeponet_monolithic; do
  for SEED in 1 2 3 4 5; do
    python scripts/run.py --config configs/matrix/$CFG.yaml \
        --freeze-hash fc4cb76c3b32ec17 \
        --max-wall-seconds 10800 \
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
python audit_port/scripts/18_swing_fidelity.py --checkpoint <run>/model.pt
python audit_port/scripts/24_rollout_thermal_error.py --checkpoint <run>/model.pt \
    --max-windows 730 --k-scenarios 0.95 --json-out <run>/rollout.json
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

**Seeds are reported as a distribution, not a mean.** Five seeds, and a
non-converged seed is reported as non-converged rather than dropped; dropping it
would select for the seeds that happened to work.
