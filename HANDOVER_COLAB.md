# Handover — what to run on Colab

Written 2026-07-29 at commit `468a6be`, after the audit phase closed. Pin that
sha in the Colab clone so the run is reproducible.

Only **one** thing needs Colab. Everything after it is CPU analysis on the
development machine, and two of those scripts need a small edit first — see §3,
which is honest about what is not yet runnable.

---

## 1. The Colab run — O-5 retrain

Full recipe in `scripts/colab_run.md`; this is the config and the checks.

```
!python -u scripts/run.py \
    --config configs/example_cod_seed1.yaml \
    --freeze-hash fc4cb76c3b32ec17 \
    --out "{OUT}" \
    --device cuda
```

Pass `--freeze-hash`. Without it nothing enforces that the distribution is the
frozen one, and the whole point of freezing before training is that the run
proves it.

### Expected wall clock

| phase | time | note |
|---|---|---|
| data generation | **8-10 min** | measured on 4 CPU cores at 59-75 ms/IC for 8000 ICs; it is CPU-bound, so the GPU does not help |
| training | **up to 60 min** | `max_wall_seconds: 3600` is a hard cap |
| evaluation | a few min | 100 RK45 test cases |
| **total** | **~70-80 min** | |

Data generation is slower than it used to be because each sample now runs a
periodic burn-in over the 24 h cycle to derive its initial condition. That is
fix 7 working as intended, not a regression.

### What to check in the output

**Before it starts training** — these three lines confirm the run is what it
claims to be:

```
[run] distribution    fc4cb76c3b32ec17
[data] generated 8000 ICs  realistic sampler  (cycle_period=1440 min, K_amp=(0.12, 0.28), hot_spot_mean=86)
[model] cod  154,178 parameters
```

If the distribution hash differs, `--freeze-hash` will have already aborted the
run. If the `[data]` line says `v57 sampler (DEPRECATED, ...)` then the wrong
config is loaded — stop, that path is for reproducing v57 only.

**At the end — the one that decides whether the run is usable.** The last line
is either silence or:

```
[run] This run did NOT converge. Report it as non-converged, with
      stop_reason='epoch_budget' and its learning curve.
```

`max_epochs: 25000` against `max_wall_seconds: 3600` means **the wall clock is
likely to bind first on a T4**, giving `stop_reason='wall_clock'`. That is the
fair-budget protocol working as designed (audit B-1: equal wall clock, not equal
epochs), but it still means non-converged, and a non-converged run cannot be
quoted as a performance number. If that happens, send me `run.json` rather than
the printed metrics — the decision is whether to raise the budget or accept the
wall-clock stop as the honest answer, and that is a protocol decision, not a
number to read off.

**In `run.json`:**

- `outcome.stop_reason` — `converged` is the good case; `wall_clock` or
  `epoch_budget` both mean non-converged.
- `data_provenance.sampler` = `"realistic"` and
  `data_provenance.realistic_params` with **22** entries. This is the record the
  distribution hash is a hash of.
- `config.distribution_hash` = `fc4cb76c3b32ec17`.
- `outcome.loss_history_tail` — flat means it stopped moving, still falling means
  it ran out of budget.

**One warning to watch for**, seen in both local smoke runs:

```
[train] PATHOLOGY: Clamp 'state_hi' active on 25.0% of samples: the loss is
        being evaluated at a clamped state, not the predicted one.
```

At 40 epochs on 48 ICs this is expected — the model has not learned the scale
yet. If it is still firing at the end of a full 8000-IC run, that is a real
finding and needs reporting, not ignoring: it means the loss is being evaluated
somewhere the model did not predict.

**Bring back:** the checkpoint and `run.json`.

---

## 2. What NOT to do

Do not pass `--train-data` to save the 8-10 min of generation. It bypasses the
sampler, so `run.json` would record `source: <file>` and the resolved parameters
would not be written. Ten minutes is not worth breaking the provenance chain the
whole freeze exists to establish.

Do not run the 5-seed sweeps yet. C-11's matrix is 40 runs and its stated
prerequisite is caching `true_fixed_point()`, which is done — but the matrix
should wait until this single run shows the fixed physics trains at all. That is
what O-5 is for: confirm fix 1 broke nothing. It is not a number for the paper.

---

## 3. After the checkpoint comes back — local CPU, and not yet runnable

These are the two things the audit phase left explicitly open. **Neither is a
straight rerun**; both point at the v57 checkpoint and the v57-era test set, so
they need an edit first. Tell me when you have the checkpoint and I will make
them.

**3a. Swing fidelity on the retrained model** —
`audit_port/scripts/18_swing_fidelity.py`. Needs two changes: `build_models`
currently loads `transformer_pideepOnet_v57.pt` with
`theta_ss_mode="formula_C", legacy_V_clamp=True`, which are the v57 settings and
wrong for the new checkpoint; and it scores on
`build_test_set(seed=999, steady_state=formula_A)`, the v57-era distribution,
where `SWING_FIDELITY.md` §5.3 says the check must be rerun **in the fix-7
distribution** for it to measure spectral bias rather than distribution shift.
Runtime once edited: a few minutes on CPU.

**3b. The real rollout thermal error** — the number O-9 left open, and the one
that decides whether an end-of-life figure is publishable. No script does this
yet. `19_verify_bias_fix.py` verifies the *metric* against a zero-error model; it
does not measure a trained model. A new script is needed that rolls the retrained
checkpoint and reports `theta_bias` against `theta_cyc_ref` — cheap now that the
cyclic reference is cached per day-of-year, but it has to be written.

**3c. If the swing check shows the retrained COD still does not flatten**, then
O-12 (train `CODNoBaseline` on `fc4cb76c3b32ec17` at COD's budget) is the run
that decides whether the delta-learning argument can go in the paper at all.
That is a second Colab run, same config, same budget, one variable changed. Do
not write that argument on the monolithic checkpoints — they did not converge,
and DECISIONS N-9 records why that inference would repeat audit M-2.
