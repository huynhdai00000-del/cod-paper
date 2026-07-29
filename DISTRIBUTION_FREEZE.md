# Distribution freeze

Frozen 2026-07-29, after `scripts/run.py` was verified to complete end to end.
A hash recorded against a package that cannot run is worthless, so the gate came
first: see §4.

## 1. The frozen hashes

`distribution_hash` is `canonical_hash(raw["distribution"])` — the canonical hash
of the `distribution` block alone, reported separately from the whole-config hash
so that "the training distribution was frozen before any model was trained" is a
checkable claim rather than an assertion (`cod/config.py`).

| config | config hash | distribution hash |
|---|---|---|
| `configs/example_cod_seed1.yaml` | `2b2f5462ac53a64d` | **`9bf8b092546cfa30`** |
| `configs/v57_faithful.yaml` | `c4595fcef3ae3f4b` | **`7a98d381e402e7c8`** |

Enforce with `assert_distribution_unchanged(cfg, expected_hash)`, or on the
command line:

```
python scripts/run.py --config configs/example_cod_seed1.yaml \
                      --freeze-hash 9bf8b092546cfa30
```

Any intentional change goes in `CHANGELOG_DISTRIBUTION.md` with date and reason,
the hash here is updated, and the change is disclosed in the paper.

## 2. What this hash does NOT pin — read before relying on it

The name overstates the guarantee in three specific ways. All three are
properties of the repo as it stands today, not speculation.

**2.1 It hashes YAML that the training path largely does not read.** `run.py`
generates data with `generate_training_set(n_ic, seed, randomise_ambient_phase,
steady_state, clip_step)`. Of the `distribution` block, only `seed`,
`steady_state_formula`, `ambient_phase` (as the boolean "is the range
non-empty") and one flag derived from `profile_families` reach it. The nine
family definitions, `K_base`, `ambient_base` and `ambient_amplitude` are **not
passed to the generator at all** — the ranges that actually apply are hardcoded
in `cod/data/generate.py`'s `make_sensor_profile` and `sample_consistent_ic`.
Editing `K_base: [0.5, 1.2]` in the YAML changes this hash and changes nothing
about the data.

**2.2 It does not cover the sampler code.** A distribution is defined by the
config block *and* the code that consumes it *and* the physics constants. Fix 7
(N-6) rewrote `cod/data/realistic.py` and changed the realised hot-spot swing
distribution substantially — median 11.20 to 13.18 degC — without touching any
config, so neither hash above moved. Code-level changes to a sampler are
invisible to a config hash by construction. Git history is what pins those; this
hash is not a substitute for it.

**2.3 The realistic sampler is not on the config path at all.** This is the one
that matters most for the paper. `cod/data/realistic.py` — the fix-7 sampler,
with the 24 h cycle, the windowing at random phase, and the day-consistent
initial condition — is used by `audit_port/` scripts and by nothing in
`scripts/run.py`. The `distribution` block still describes the v57-era family set
(`flat`, `sinusoidal`, `step`, `peak_then_drop`, `tv_high_amp`, `tv_ramp_sin`),
which is a different set of names from the realistic sampler's (`base_load`,
`daily`, `shift_change`, `evening_peak`, …), and it carries no `cycle_period`.

**So `9bf8b092546cfa30` freezes the distribution the smoke run trains on, which
is not the distribution the Jensen-gap results are measured on.** The 13.18 degC
median swing, the gap medians in `audit_port/PERIOD_FIX.md`, and the IEC
exceedance rates all come from `build_realistic_set`, which no config describes
and no hash here covers.

## 3. What to do about §2.3 before the retrain

Not done here, because wiring a new sampler into the training path is a change to
what gets trained and belongs with O-5, not with a freeze. In order:

1. Add a `sampler: realistic` switch to the `distribution` block with the
   `RealisticParams` fields that matter (`cycle_period`, `K_amp`,
   `hot_spot_mean`, the family weights) written out explicitly, so they are
   inside the hashed block rather than in a dataclass default.
2. Have `run.py` dispatch to `build_realistic_set` on that switch.
3. Re-freeze, and record the new distribution hash here as the one the paper's
   numbers belong to.
4. Only then retrain (O-5). The retrain is what makes the freeze meaningful:
   freezing after training is not a protocol, it is a record.

Until step 3 lands, cite these hashes as "the configuration under which the
package runs", not as "the frozen benchmark distribution".

## 4. The gate this was written behind

`scripts/run.py --config configs/example_cod_seed1.yaml --max-epochs 40
--n-ic 48 --n-test 6 --device cpu --tag smoke` completes end to end and writes
`results/cod_transformer_cod_s1_2b2f5462ac53a64d_smoke_smoke/run.json`.

It reports `NOT CONVERGED (stop_reason=epoch_budget)` and a `state_hi` clamp
active on 25% of samples. Both are correct at 40 epochs on 48 ICs and neither is
a failure of the gate: the run exists to prove the wiring executes, and the
package refusing to dress a 40-epoch run as a performance number is the
convergence rule working. **None of the metrics in that run.json are results.**

Separately, `audit_port/scripts/19_verify_bias_fix.py` exercises
`chi_lifetime_rollout` and constructs `RolloutResult` 120 times, which is the
path that was broken when the freeze was requested (`theta_cyc_ref` declared but
never passed, so every call raised `TypeError`). Fixed in `3a67aaf`, verified
before this document was written.
