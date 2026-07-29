# Port log

Every judgement call made while porting the notebook science into `cod/`.
Newest entries at the bottom of each phase. Numbered `J-n` so they can be cited.

Source of truth for Phase 1 is `reference/audit/extracted/n12.txt`
(= `Paper/Code/PI_DeepONet_v57_done.ipynb`), with `n00.txt`
(`COD_ablation_study_done`) and `n15.txt` (`Phase1_Notebook_v3_v7_done`) for the
ablations, the monolithic baselines and the capacity sweep. `n46.txt` is the
archived copy of the v57 notebook and was read only for comparison, never ported
from.

---

## Phase 1 — faithful port

### J-1 Module-level `device` replaced by a per-device constant cache

The notebook binds `k_gen_t`, `k_dis_t`, `E_act_t` to one global `device` chosen
at import. A package cannot do that: `scripts/run.py` must work on CPU and GPU in
the same process, and the verification harness runs on CPU while training may
not. `cod/data/physics.py` therefore caches the torch constants per
`(device, dtype)` and derives the device from the input tensor.

Numerically identical — same values, same dtype, same arithmetic order.

### J-2 `RHS_SCALE` and `DERIV_SCALE` are functions, not import-time constants

The notebook computes both at import. `compute_deriv_scale` runs 300 `solve_ivp`
calls, which would make importing `cod.data.physics` take seconds. Both are
exposed as functions. Nothing on the live v57 path consumes either value: the
surviving `ode_physics_loss` uses a raw residual (see J-4), so this changes no
result.

### J-3 `theta_TO_ss` is ported once, not twice

Audit §8.1 lists `theta_TO_ss` as defined twice in v57 with **differing**
bodies (cell 0 L904, cell 2 L1510). The bodies are byte-identical —
`theta_a + DTheta_oil_R * (K ** (2 * n_exp))` — and only the docstrings differ
("Steady-state top-oil theo IEC 60076-7" versus the same text plus "giữ nguyên
cho training consistency"). The audit's table compares full source text
including docstrings, which is why it flags a difference.

There is no last-definition-wins hazard here. Ported once as
`steady_state.formula_A`. **This is a clarification of the audit, not a
contradiction of it** — the finding that three inconsistent steady-state formulas
exist (M-6) is confirmed exactly, including every number in step3 §3.3.

### J-4 `ode_physics_loss`: cell 2 wins — raw residual

Three definitions exist (cell 0 L738, cell 2 L1134, cell 2 L1344). Ported the
last one. The difference that matters is one line:

| definition | residual |
|---|---|
| cell 0 L758 | `(dxdt - f_rhs) / deriv_scale_t` — `[V18 DERIV_SCALE]` |
| cell 2 L1154 and L1364 | `(dxdt - f_rhs)` — `[V34] raw residual — bỏ DERIV_SCALE` |

The two cell-2 copies are identical to each other. Since cell 2 executes after
cell 0, the raw-residual version is what trained
`transformer_pideepOnet_v57.pt`. Recorded in the docstring of
`cod/training/losses.py::ode_physics_loss`.

Consequence worth stating: because the live loss never divides by `RHS_SCALE`,
the double-`pd_factor` defect (Phase 2 fix 2) has zero blast radius in v57.

### J-5 `compute_chi`: cell 2's second copy wins — fixed weights, no weight net

Three definitions (cell 0 L662, cell 2 L1090, cell 2 L1300). The cell-0 version
takes `(x_fast, DP_val)` but reads a module-global `CHI_W` that cell 0 never
defines, and computes `w_dp_dynamic` from a `W_DP` that cell 0 never defines
either; it is unreachable as written. The two cell-2 copies are identical and
define `CHI_W` / `W_DP` immediately above. Ported the cell-2 body.

### J-6 `make_log_collocation`: bodies identical, docstrings differ

Three definitions (cell 0 L719 `[V55-A]`, cell 2 L1119 and L1329 both `[V56]`).
All three bodies are identical: 75% forward log spacing plus 25% backward.
Ported once. Same clarification as J-3.

### J-7 `chi_monotonicity_loss`: cell 2 wins — the dead weight net is gone

Cell 0 L773 calls `compute_chi(..., sensors=sensors, weight_net=chi_weight_net)`.
The surviving `compute_chi` takes `(x_fast, DP_val)` only, so that call would
raise `TypeError`. Cell 2 L1169 / L1379 call `compute_chi(xp[:, q, :], DP_cur)`,
which matches. Ported the cell-2 body.

`chi_weight_net` / `AdaptiveCHIWeights` are **not** ported, as instructed: the
object is instantiated at cell 0 L993 and referenced only from cell-0 bodies that
cell 2 overwrites. Never trained, never executed, and absent from every
checkpoint's `state_dict` (verified — see `audit_port/01_checkpoints.txt`).

### J-8 Parameter shadowing in the monolithic models is a real bug, ported as-is

`PIDeepONet_Mono_Fair.__init__(self, d_h=128, p=64, n_layers=4, n_exp=12, ...)`
takes a parameter named `n_exp`, which shadows the module-global thermal exponent
`n_exp = 0.8`. The buffer registration loop then runs
`('ne', n_exp)` — binding `self.ne = 12.0`, not `0.8`.

So the monolithic baselines build their trunk features with a thermal exponent of
**12** instead of 0.8. Confirmed against the trained weights, not inferred: every
monolithic checkpoint stores `ne = 12.0`
(`mono_fair_v2_perstate.pt`, `mono_multihead.pt`, all five
`sweep_mono_fair_p*.pt`). The COD model names its parameter `n_exp_feats` and so
registers the correct `n_exp_buf = 0.8`.

Ported exactly, with the shadowing preserved, because the checkpoints were
trained under it and Phase 1 forbids fixing anything. Flagged here because it is
**not in the audit** and it bears directly on M-2: part of the monolithic
baseline's failure to converge may be that its analytic trunk features were
computed with a nonsense exponent. That is an additional reason "monolithic
architectures fail" is not established by these runs, and it is a candidate
Phase 3 item — not a Phase 2 item, since the five Phase 2 fixes are specified.

### J-9 Audit M-8's count of 20 is a slip; the correct count is 14

Audit M-8: *"Training `K_base ~ U(0.5,1.2)`; CK test `K_k ~ U(0.4,1.4)`. 20 of 50
CK cases fall in [0.40,0.50) ∪ (1.20,1.40]."*

Our reproduction gives **14** of 50 (7 below 0.50, 7 above 1.20). The expected
value for `K ~ U(0.4, 1.4)` is 15.

This is not a reproduction failure. Our seed-999 test set is bit-identical to the
audit's own: comparing the per-case, per-state ground-truth variation against the
audit's stored `results/06_test_ranges.npy` gives a maximum relative difference of
**exactly 0.000e+00** over all 100 × 6 entries. Case 33 also matches the audit
and the notebook's stored output exactly (CK, K = 1.398, x0_TO = 141.3 °C).
No counting rule we tried reproduces 20 (see
`audit_port/scripts/03_check_m8_count.py`).

**Not stopped on**, because nothing here contradicts the audit's finding: the CK
test range does extend past the training support on both sides, and the §7
outlier (M-10) is one of those extrapolating cases — both confirmed exactly. Only
the tally in the sentence is wrong. Reported rather than silently corrected.

### J-10 Both right-edge guards are kept, selected by argument

`rk45_ground_truth` takes `t_clip_frac`. The sensor interpolant is evaluated at
`min(t, T*t_clip_frac)`; n12's `evaluate_v44` uses `0.9999` and n15/n00's
`evaluate_100` uses `0.999`. Gate 1 comes from the first and gates 2-3 from the
second, so the port cannot pick one and stay faithful to all three. Default is
n12's `0.9999`; the gate runner passes each explicitly.

### J-11 `configs/example_cod_seed1.yaml` had `window_minutes: 1440`

The scaffold config says 1440; every notebook uses `TW = 720.0` (12 h), and the
checkpoints, the sensor grid and `DERIV_SCALE` all assume 720. Corrected to 720
in a new `configs/v57_faithful.yaml`, which is the Phase 1 config. The original
file is left untouched: its `distribution` block already describes the *post-*
Phase 2 distribution (randomised ambient phase, `log_space: true`,
`weight_floor: 1.0e-8`), so it is the Phase 2 target, not a description of v57.

No `DISTRIBUTION_FREEZE.md` exists yet, so no frozen hash was violated.

### J-12a Shared blocks live in `cod/models/blocks.py`

`ModifiedMLP` and the 32-dimensional trunk feature builder appear in all three
notebooks; the target layout names only `cod.py` and `monolithic.py`. Defining
them in both would break the "each function defined exactly once" rule, so they
sit in `blocks.py` alongside `interp_sensors`.

The two trunk-feature spellings are equivalent, which I checked rather than
assumed: n12 computes the query index twice, once with the bound `ns-2+1e-6` for
(K_t, Ta_t) and once with `ns-1-1e-6` for the K-history block, while n15/n00
compute it once with the first bound. For every t in [0, T] both floor to the
same integer — where `tn <= ns-2` both give `floor(tn)`, and where
`ns-2 < tn <= ns-1` both give `ns-2`. That equivalence is why n15's COD scores the
same 1.5% from the same checkpoint, and it is confirmed empirically by gate 1 and
gate 2 agreeing.

n00 also has a third spelling, `ModMLP`, with abbreviated attribute names
(`eU/eV/f/hl/o/a`). Those are not valid `state_dict` keys for any stored
checkpoint, so only the n12/n15 spelling is ported.

### J-12 Model `__init__` no longer prints

`PIDeepONet_v24`, `PIDeepONet_Mono_Fair` and `PIDeepONet_Mono_MultiHead` all
print their parameter count on construction. Replaced by a `n_parameters()`
method; the capacity sweep prints it from the caller. No numerical effect.

### J-13 A validation split was added, because there was none

`harness.train` needs `validation_loss()`. The source has no validation set: every
run simply exhausted its epoch budget, so "converged" was never checked against
anything. That is the mechanism behind audit B-1 — a baseline reported at 171 of
25,000 epochs, with the loss unmoved at 1.3e+08, because nothing was watching.

`cod/training/train.py::_BatchSource` carves out 5% of the training set with its
own `RandomState`, and `train_batch` never returns those indices. This is an
addition, not a port: it cannot change any Phase 1 number because Phase 1 trains
nothing, and it satisfies README rule 4 (adjustments only against a validation
split taken from the training distribution).

`validation_loss()` cannot run under `no_grad` — the residual needs
`autograd.grad` through t — so it explicitly zeroes the model's gradients
afterwards. Verified: after a validation call no parameter has a nonzero gradient
(`audit_port/scripts/05_check_training_wiring.py`).

### J-14 Two trainers, because the source has two different loops

`train_v34` (n12 cell 1) trained `transformer_pideepOnet_v57.pt`. `train_physics`
(n15 cell 2 / n00 cell 4) trained every monolithic baseline **and every
capacity-sweep checkpoint, including `sweep_cod_p*.pt`**. They are not two
spellings of one loop. Three substantive differences, ported as
`CODTrainer` and `SharedPhysicsTrainer`:

| | `train_v34` (COD v57) | `train_physics` (baselines + sweep) |
|---|---|---|
| collocation | two-sided log, 75% forward + 25% dense near t=T | one-sided forward log only |
| RHS state clamp | per-state `_hi = [200, 500, 200, 1000, 3000, 8000]` | scalar `.clamp(0, 500)` |
| adaptive weight target | causally *weighted* per-state loss | plain mean over batch and chunks |
| `lam[1:]` floor | `clamp(min=0.0)` | none |

The scalar `clamp(0, 500)` matters: CO sits at ~1e2-1e3 ppm and CO2 at ~1e3 ppm,
so a ceiling of 500 truncates them hard and the shared trainer forms its gas
residuals at a clamped state far more often. Measured on a smoke run, that clamp
was active on 44% of samples.

Bearing on gate 2: the capacity sweep is internally consistent — both arms used
`train_physics` — but **neither arm was trained the way the headline COD model
was**. Worth a sentence in the paper if the sweep is cited as evidence about COD.

### J-15 `causal_weights` was factored out so the underflow is observable

The two-line weight computation is now
`cod/training/losses.py::causal_weights`, returning `(w, w.min())`. Same
arithmetic, same `.detach()`. Factoring it out gives Phase 2 fix 3 exactly one
place to change, and gives `train_step` a `causal_weight_min` to report so the
harness's underflow check fires.

It fires immediately in practice. On the very first step of a smoke run,
`SharedPhysicsTrainer` on a monolithic model returned `causal_weight_min =
0.000000` and the harness printed the pathology warning — reproducing audit B-1's
`wm=0.000` live rather than by inference.

### J-16 Clamp diagnostics are recomputed, not instrumented in-place

`train_step` must return `clamp_frac_<name>` per clamp. Rather than add
bookkeeping inside `fast_rhs_torch` — which would put new tensor operations on the
live loss path — `losses._clamp_diagnostics` recomputes the clamped quantities
under `no_grad` from the same inputs. The loss path stays byte-identical to the
source and the diagnostics cannot perturb it.

Reported clamps, all from audit section 8.4: `state_hi`, `state_lo`, `Rf_etc`,
`T_HS_min`, `V_arr_max`, and `state_scalar_500` for the shared trainer.

### J-17 Audit M-3's absolute-error reconstruction is off, in both directions

M-3 back-converts the monolithic per-state NMAEs into absolute errors and gets
theta_TO 13.9 degC, H2 1.72 ppm, C2H2 0.23 ppm, flagging the method as an
"order-of-magnitude reconstruction: mean-of-ratios combined with a median
denominator".

Measuring them directly from the reproduced predictions instead:

| state | M-3 reconstruction | measured | ratio |
|---|---|---|---|
| `theta_TO` | 13.9 degC | 13.41 degC | 1.04x high |
| `c_H2` | 1.72 ppm | 0.306 ppm | 5.6x high |
| `c_C2H2` | 0.23 ppm | 0.705 ppm | 3.1x low |

The audit declared this method's limits, so this is a refinement rather than an
error, and **the finding is unaffected**: the worst gas error is still about 2% of
its IEC 60599 attention level, so none of the huge percentages describes a
diagnostically meaningful error, and the genuinely large error is still thermal.
`PHASE1_VERIFICATION.md` reports the measured column. Quote that one.

---

## Phase 2 — fixes

Gate 1 numbers before and after each fix are in `PHASE2_EFFECTS.md`.

**Design decision applying to all five fixes.** Each fix flips a DEFAULT to the
corrected behaviour and keeps the v57 behaviour reachable by explicit argument.
`scripts/verify_phase1.py` and the `audit_port/` check scripts now request v57
explicitly, so the Phase 1 reproduction survives as a regression test while the
package's defaults are the fixed physics. The alternative — deleting the v57 path —
would have made the gate numbers unreproducible the moment the first fix landed,
and there would then be nothing to measure each fix against.

### J-18 Fix 1 needed a differentiable fixed-point solver

`true_fixed_point` uses brentq: scalar, slow, not differentiable. It cannot go
inside a forward pass that evaluates theta_ss on (B, n_sensors) tensors.

`true_fixed_point_torch` / `true_fixed_point_np` solve the same equation by
contraction iteration:

    theta <- theta_a + DTheta_oil_R * ((1 + K^2 R_eff(theta)) / (1 + R))^n_exp

The measured contraction factor is about 0.34 per step, not the ~0.05 I first
estimated, so the iteration count is 20 rather than 12. Validated against brentq
over a 2009-point (K, theta_a) grid covering K in [0.30, 1.50] and theta_a in
[10, 50]:

| version | max error vs brentq |
|---|---|
| numpy, float64, 20 iters | 1.33e-08 degC |
| torch, float64, 20 iters | 1.33e-08 degC |
| torch, float32, 20 iters | 3.16e-05 degC (the float32 floor) |

3e-5 degC is five orders of magnitude below the -18.25 degC error at K = 1.3,
theta_a = 30 that the fix exists to remove. Gradients through it are finite and
nonzero. Script: `audit_port/scripts/06_check_fix1.py`.

### J-19 Fix 1 also reaches the trunk features, not just the baseline

The brief names four sites: IC generation, the model's analytical baseline, the
rollout, and plotting. A fifth site computes theta_ss and was not named:
`build_trunk_feats` builds the `driving`, `dm_n` and `dr_n` input features from
formula C.

I included it, because leaving it would have meant the model's baseline used the
true fixed point while its own input features described a different attractor —
replacing one inconsistency with another, which is the opposite of what fix 1 is
for. `build_trunk_feats` takes a `theta_ss_mode` argument defaulting to
`formula_C`, and `CODOperator` passes its own mode through.

The monolithic baselines stay on formula C. They have no analytic baseline, so
they are outside fix 1's scope, and their theta_ss is computed with the shadowed
exponent of 12 anyway (J-8) — changing the formula under a wrong exponent would
mean nothing.

### J-20 The rollout is now in the package, not in a plotting function

Fix 1 names "the rollout, plotting" as sites. In the source both live inside
`plot_chi_trajectory` and `plot_chi_model_dp` — matplotlib functions with the
lifetime simulation embedded in them, which CLAUDE.md forbids.

Ported the simulation to `cod/eval/rollout.py` as `chi_lifetime_rollout`, data
only, no figures. The `dp_source` argument preserves the source's two variants:
`"reference"` advances DP from the reference steady state (so EOL cannot reflect
model quality) and `"model"` advances it from the model's predicted theta_TO (so a
thermal bias shows up as an EOL shift). `RolloutResult.theta_bias` exposes the
quantity audit M-5 disputes.

Note for M-5: with all sites unified there is no formula mismatch left to appeal
to as an explanation of the -3 degC bias. Since B and C are identical at K = 1 and
the C-B disagreement over the rollout's K range is +0.67 to -1.03 degC with a sign
flip, the mismatch could never have accounted for 3 degC anyway. The bias must be
re-diagnosed against the retrained model.

### J-21 `theta_ss_mode` is deliberately not a buffer

If the mode were a registered buffer it would live in the `state_dict`, and
loading a v57 checkpoint into a fixed model would silently reset the mode to the
checkpoint's value — the loudest possible failure turned into the quietest. It is
a plain Python attribute, so a mismatch between checkpoint and mode stays the
caller's explicit choice and shows up in the numbers.

### J-22 Fix 2 is a hygiene fix with a measured effect of zero on any result

Removing the second `pd_factor_np(K)` from `compute_rhs_scale_physics` changes
`RHS_SCALE[2]` (c_C2H2) by exactly `pd_factor_np(1.3) = 2.2150`, confirming the
audit's arithmetic: the factor was being squared, 2.2150 -> 4.9062 at K = 1.3.
No other state changes, because `pd_factor` only touches the acetylene channel.

**No published number moves**, and that is the honest report rather than a
disappointment. The surviving `ode_physics_loss` uses a raw residual and never
consumes `RHS_SCALE` (J-4), so this defect never reached a result in v57. It is
worth fixing because the next person to reach for `RHS_SCALE` — for instance
anyone reinstating the DERIV_SCALE-normalised residual from the discarded cell-0
loss — would otherwise get a squared partial-discharge factor with no indication.

`double_pd_factor=True` restores the v57 arithmetic for the comparison.

### J-23 Fix 3: the clamp goes in log space, not on the exponentiated weight

`max(exp(-eps*cum), floor)` looks equivalent to clamping the log and is not. By
the time `exp(-120)` has been computed in float32 it is already exactly 0.0 and
all ordering information is gone, so every deep chunk would receive the identical
weight `floor`. Clamping `-eps*cum` to `log(floor)` first preserves the ordering
of the chunks right up to the floor, and only then flattens.

Measured, with `eps * cum` on the first chunk:

| eps*cum | v57 linear | fixed (log space, floor 1e-8) |
|---|---|---|
| 1 | 3.679e-01 | 3.679e-01 |
| 10 | 4.540e-05 | 4.540e-05 |
| 50 | 1.929e-22 | 1.000e-08 |
| 88 | 6.055e-39 (subnormal) | 1.000e-08 |
| 104 | **0.000e+00** | 1.000e-08 |
| 1000 | **0.000e+00** | 1.000e-08 |

The two agree exactly until the floor binds at `eps*cum = log(1e8) = 18.4`. v57
goes subnormal around 88 and reaches exactly zero by 104.

Confirmed end to end on a real training step: `SharedPhysicsTrainer` on a
monolithic model, which previously reported `causal_weight_min = 0.0` on its first
step, now reports exactly `1.000e-08` and the harness's underflow pathology no
longer fires.

### J-24 Fix 3 needed the schedule fixed too, not just the weights

Flooring the weights alone would not have made the comparison fair. `eps_causal`
advances when the model's own `wm` stays above 0.50 for 200 epochs. With a floor
in place Mono Fair's `wm` no longer reaches zero, but it still sits far below
COD's, so its epsilon would still advance more slowly and the two models would
still be optimising different objectives — the same defect, less visibly.

`EpsilonSchedule(shared=True)` therefore advances epsilon on elapsed epochs alone,
unconditionally, every `patience_needed` epochs. Two models trained for the same
number of epochs now follow the identical epsilon trajectory whatever their
residuals do. The objective becomes a property of the protocol rather than of the
model, which is the only way an epoch-matched comparison means anything.

`shared=False` restores the v57 behaviour.

### J-25 Fix 4: the phase draw stays last in the rng sequence

`rng.uniform(0, 2*pi)` for the ambient phase is drawn after `Ta_base` and
`Ta_amp`, at the very end of `make_sensor_profile`. That position is load-bearing:
a `RandomState` is a stream, so inserting a draw anywhere earlier shifts every
subsequent number and would change the generated dataset **even with the fix
disabled**. Placing it last means `randomise_ambient_phase=False` still reproduces
`transformer_training_v57.npz` byte for byte, which is re-verified after this
commit.

Measured effect: under v57, **100.0%** of training profiles start exactly at their
ambient mean, because the phase is 0 and sin(0) = 0. After the fix, 0.7% do — the
residual being profiles whose randomly drawn phase happens to land near 0 or 2 pi,
which is the correct behaviour rather than a leftover.

The gap this closes is specific. The seed-999 test set uses a phase of pi/3, so its
ambient profile starts sin(pi/3) = 0.866 of the amplitude **above** its mean. Under
v57 the model never saw an ambient waveform that did anything but start at its
mean, so the test set's ambient shape was structurally outside the training family
on one axis, even though its amplitude and base marginals sat inside it. That is a
sharper statement than "the test set is in distribution" and worth making in the
paper: the marginals overlapped while the joint structure did not.

### J-26 Fix 5: the clip bound is 1.4, not 1.5

The sibling branches do not agree on a ceiling. `ramp` and `multi_step` clip to
(0.3, 1.4); `sinusoidal`, `overload_spike`, `peak_then_drop`, `tv_high_amp` and
`tv_ramp_sin` clip to (0.3, 1.5). So "add the .clip() its siblings have" does not
name a unique bound and the choice had to be made.

Chose **(0.3, 1.4)**, because `step` is a piecewise-constant family like `ramp` and
`multi_step`, whereas every 1.5 branch is one that deliberately reaches into
overload — `overload_spike` draws its spike from U(1.2, 1.5) explicitly. Clipping
`step` at 1.5 would have let a family with no overload intent produce overload
loads.

Measured against the exact v57 stream (both arms through `generate_training_set`
with all v57 settings, only `clip_step` differing, so the v57 arm reproduces the
stored dataset):

| | v57 | fixed |
|---|---|---|
| min K | 0.257134 | 0.300000 |
| profiles below the 0.3 floor | 12 / 8000 | 0 |
| profiles above the 1.4 ceiling | 601 | 592 |

The v57 minimum reproduces the stored `transformer_training_v57.npz` value of
0.257134 exactly, confirming `step` is the branch responsible. The clip also brings
9 profiles back under 1.4; the 592 still above it are the deliberate-overload
families.

Correctness rather than results: 12 of 8000 profiles is a small share of the
training mass. Worth making anyway, because a documented sampling range the code
does not honour is exactly what a reviewer checks.

---

## Wiring

### J-27 `scripts/run.py` reads the fix switches out of the config

The five Phase 2 fixes are argument-level switches in the package. `run.py` derives
them from the config block rather than hard-coding them, so a config file fully
determines a run:

| config key | switch |
|---|---|
| `distribution.steady_state_formula: A` | `steady_state=formula_A` (fix 1 off) |
| `distribution.ambient_phase: [0.0, 0.0]` | `randomise_ambient_phase=False` (fix 4 off) |
| `profile_families[kind=step].clipped: false` | `clip_step=False` (fix 5 off) |
| `training.causal_weighting.log_space` | fix 3, weight computation |
| `training.causal_weighting.weight_floor` | fix 3, floor |
| `training.causal_weighting.schedule_shared` | fix 3, epsilon schedule |
| `model.steady_state` | fix 1, the model's attractor |

`configs/v57_faithful.yaml` sets all of them to the v57 values;
`configs/example_cod_seed1.yaml` sets them to the fixed values. So the difference
between reproducing v57 and running the corrected pipeline is a config file, not a
code edit — which is what makes the distribution hash meaningful.

### J-28 `training.loop` has to be stated in the config

Since `train_v34` and `train_physics` are different loops (J-14), the config
declares which one to use. Default is `train_v34` for `kind: cod` and
`train_physics` otherwise, matching how the source paired them, but a config can
say either — which is what a genuinely matched comparison would need.

### J-29 `run.py` reports both metrics and refuses to launder non-convergence

Every run writes the source's NMAE **and** `cod/eval/metrics.py`'s absolute error
in physical units with the per-state floor-hit rate, plus the test tier by name.
`run.json` carries `outcome.converged`, `outcome.stop_reason` and
`fair_comparison_candidate`, and a non-converged run prints an explicit refusal to
have its metrics quoted as a performance figure. `loss_history` is written to a
separate `loss_history.json` so `run.json` stays readable, with head and tail
inline.

### J-30 End-to-end smoke test result

`python scripts/run.py --config configs/example_cod_seed1.yaml --max-epochs 100
--n-ic 50 --device cpu` on CPU, torch 2.13.0+cpu. Exit code 0, 827 s wall.
`results/` is gitignored, so the outcome is recorded here.

Wiring proved, in the sense that each mechanism actually fired rather than merely
not crashing:

* data generated with all three distribution fixes active — `steady_state=
  true_fixed_point`, `randomise_phase=True`, `clip_step=True`;
* `train_v34` selected from the config, 154,178-parameter COD model;
* loss fell from 1.5e+00 to 1.8e-03 over 100 epochs (wiring, not convergence);
* the harness reported `converged=False`, `stop_reason='epoch_budget'`,
  `fair_comparison_candidate=False`, and printed an explicit refusal to have the
  metrics quoted as a performance figure;
* the clamp pathology fired: `state_hi` active on 21.9% of samples, plus
  `Rf_etc` 4.0%, `T_HS_min` 1.6%, `V_arr_max` 1.4%;
* `causal_weight_min = 0.99999`, well clear of the floor, so fix 3's floor was not
  masking anything here;
* both metrics printed, with the tier named `T1_in_distribution`;
* `run.json` carries the commit hash, config hash `2b2f5462ac53a64d`, distribution
  hash `9bf8b092546cfa30`, seed, device, library versions and the full outcome;
  `warn_if_dirty` correctly flagged the tree as dirty at run time.

These are **not results**. 100 epochs is 0.4% of the 25,000-epoch budget and n_ic
50 is 0.6% of the 8000-IC training set, on top of a distribution that no
checkpoint corresponds to. The run exists to prove the pipeline executes.

For the record, since it will be misread otherwise: overall NMAE 10.9%, theta_TO
MAE 1.475 degC. Better than nothing would suggest for 100 epochs, because the
analytic IEC baseline carries most of the thermal signal before the network has
learned anything — which is the architecture working as designed, not evidence of
fast convergence.

---

## O-1 — amplification mechanism

### J-31 The hybrid arm was built by calling `_gas_integral` directly

O-1's decisive test needs a monolithic thermal prediction pushed through COD's
cascade. Rather than build a new class, `audit_port/scripts/07` calls
`CODOperator._gas_integral(t, u, x0_gas, theta_grid)` with a `theta_grid` supplied
from outside. That is legitimate because the method holds no `nn.Parameter` and
reads nothing from `self` except registered physics buffers — the gradient test in
script 08 proves it (0 of 28 parameter tensors receive gradient from a gas-only
loss).

The grid is broadcast to one row per query time, exactly as `forward` does with
`_thermal_predict_grid`'s output, so the hybrid differs from COD in the thermal
input alone. No second definition of the cascade was created.

### J-32 Guard set to 0.9999 for every arm

Gate 3's stored Mono Fair numbers came from n15's harness with right-edge guard
`TW*0.999`; Gate 1's COD numbers came from n12's with `TW*0.9999`. DECISIONS
compares them directly (13.41 vs 0.399 degC, 0.705 vs 0.593 ppm), which mixes the
two harnesses.

Scored everything at `0.9999` here so the arms are comparable. Mono Fair's C2H2
figure is 0.7049 ppm under either guard, so the mixing was harmless in this
instance — recorded so that it is checked rather than assumed.

### J-33 A refuted hypothesis is kept in the report

My first explanation for COD's C2H2 error floor was the dissipation linearisation
in `_gas_integral` (L332), `- k_dis * x0_gas * t` instead of `- k_dis * c(t)`. I
implemented the exact integrating-factor solution to test it. It moved the floor by
2.35e-07 ppm — no measurable effect.

Kept in `AMPLIFICATION_MECHANISM.md` §4 rather than deleted. A plausible mechanism
that turns out not to be the cause is worth a paragraph, because the linearisation
is real and someone will propose it again.

### J-34 New finding: reference and model use different Arrhenius factors

`fast_rhs_np` (the ground truth) computes `V_arr` **unbounded**; `fast_rhs_torch`
and `CODOperator._gas_integral` both `.clamp(max=1e4)`. Above a per-gas temperature
threshold the model's generation rate is capped while the truth's is not.

| gas | theta_HS at which V_arr = 1e4 | grid points clamped | cases affected |
|---|---|---|---|
| `c_H2` | 245.3 degC | 0.00% | 0/100 |
| `c_C2H2` | **187.2 degC** | 3.63% | 6/100 |
| `c_C2H4` | 214.0 degC | 1.51% | 2/100 |
| `c_CO` | 303.6 degC | 0.00% | 0/100 |
| `c_CO2` | 356.7 degC | 0.00% | 0/100 |

The test set reaches a maximum hot-spot of 236.9 degC (mean 123.0, p90 177.2), so
C2H2's threshold is crossed and CO's is not. Removing the clamp collapses the C2H2
floor from 0.591367 to 0.000040 ppm, a factor of 14,632.

Consequence for the headline numbers: COD's C2H2 MAE of 0.5926 ppm is 6 artefact
cases averaging 9.86 ppm diluted across 100; the other 94 average 0.0000 ppm. On
the 94 clean cases COD's C2H2 error is 0.0007 ppm against Mono Fair's 0.1009 ppm —
a factor of **141**, where the all-100 comparison shows 1.19.

Audit §8.4 lists this clamp among those that can hide behaviour but does not note
that the reference lacks it, and does not quantify it. Not fixed here: aligning the
clamps changes both the model forward pass and the reference ODE, so it needs its
own commit with its own before/after, and it invalidates the checkpoint again.
Recorded in DECISIONS under "Bằng chứng mới cần xem xét" as N-1, because it
partially contradicts an entry in "Bằng chứng đã xác lập".

### J-35 What O-1 settles, and what it does not

Settled: the monolithic gases are learned, not cascaded. Gradient test — COD 0 of
28 parameter tensors receive gradient from a gas-only loss, both monoliths 100%.
Static trace — all six monolithic outputs come from one `Linear(p, 6)` head, and
`V_arr`, `k_gen`, `k_dis` appear nowhere in `monolithic.py`.

Also settled: Arrhenius amplification of a 13.41 degC thermal error produces at
most 0.63 ppm, not the ~20 ppm O-1 anticipated. The 1,000-35,000% figures in
Section 7.1 are relative *rate* changes and are arithmetically correct; the error
is quoting them as concentration errors.

Not settled by this work: whether the monolithic baselines could be trained to
succeed. Their `wm = 0.000` (audit B-1) and shadowed `ne = 12.0` (J-8) remain
uncorrected in the stored checkpoints, so "monolithic fails" is still unestablished
for three independent reasons.

---

## O-8 — the Jensen gap

### J-36 `daily_mean.py` separates the baseline from the measurement

Two things that get conflated, kept apart deliberately:

* `DailyMeanArrhenius` — a Tier 0 baseline (C-11). IEC 60076-7 thermal trajectory,
  then one frozen Arrhenius factor at the window-mean hot-spot for all five gases
  and DP. No parameters, so no seed and no convergence criterion.
* `jensen_gap_from_trajectory` — a pure measurement on a supplied trajectory.

O-8 asks for the gap, so the reported numbers all come from the second, fed the
**true** hot-spot trajectory from RK45. They therefore contain no model error of
any kind: the gap is a property of the reference physics and the test
distribution.

### J-37 The baseline uses the corrected steady state, deliberately

`DailyMeanArrhenius.theta_TO_trajectory` calls `true_fixed_point_np`, not formula A
or C. Giving the practitioner baseline the *worse* steady state would have made it
easy to beat for a reason that has nothing to do with the Jensen gap. The only
thing this baseline does differently from the reference is freeze Arrhenius at the
mean temperature, which is what makes the comparison a measurement of convexity
rather than a straw man.

It also uses the closed-form recurrence for a piecewise-linear driving term rather
than trapezoid quadrature of `theta_ss * exp(s/tau)`, because that integrand grows
by `exp(T/tau)` and the trapezoid form is what forced the battery model to switch
to a recurrence (n15 cell 7). At TW/tau = 4.8 the trapezoid would have been
adequate, but there is no reason to inherit a known weakness.

### J-38 The implementation reproduces C-10's analytical table to rounding

`jensen_gap_sinusoidal` computes the gap by quadrature over one full period, with
activation energies derived from the code's own `B_aging` and `E_act` rather than
transcribed. Maximum disagreement with C-10 over all six states and four
amplitudes: **0.0048**. The two figures the paper leads with come out at DP
**1.701** (C-10 says 1.70) and C2H2 **2.594** (2.59) at +-15 degC.

Also confirms the Ea column: H2 112.2, C2H2 174.6, C2H4 137.2, CO 87.3, CO2 74.8,
DP 124.7 kJ/mol, all from `E_act * B_aging * R`.

### J-39 The measured gap tracks the analytical prediction at every swing band

| swing degC | n | DP measured | DP analytical | C2H2 measured | C2H2 analytical |
|---|---|---|---|---|---|
| 0-2 | 5 | 1.004 | 1.007 | 1.008 | 1.014 |
| 2-5 | 9 | 1.023 | 1.036 | 1.046 | 1.071 |
| 5-10 | 5 | 1.078 | 1.138 | 1.157 | 1.284 |
| 10-15 | 13 | 1.418 | 1.472 | 1.994 | 2.033 |
| 15-25 | 28 | 2.506 | 2.598 | 5.527 | 5.127 |
| >25 | 40 | 5.321 | 5.070 | 18.740 | 14.190 |

Monotone in swing, as convexity requires, and the ordering across states follows
activation energy exactly: CO2 < CO < H2 < DP < C2H4 < C2H2. The agreement at
10-15 degC — where a real transformer sits — is the line that matters.

### J-40 The all-100 mean must not be quoted as the operational gap

The realised swing on the seed-999 test set has a median of 21.44 degC and 40 of
100 cases exceed 25 degC. That is not daily transformer behaviour; it is the IC
sampler. Audit M-9: `sample_consistent_ic` draws theta_TO(0) uniformly +-30 degC
around the steady state and clips to `[theta_a + 5, 150]` **independently of the
load**, so most cases start far from equilibrium and relax across the window.

So the all-100 means (DP 3.211, C2H2 9.505) are the gap on a synthetic
distribution with an inflated swing, not an operational claim. Flagged
prominently in `JENSEN_GAP.md` §3 because it cuts *against* the paper's interest:
the defensible headline stays C-10's 1.70 and 2.59, and the temptation to quote
3.2 and 9.5 instead should be resisted.

Consequence for the plan: the T1/T2/T3 tiers being frozen under SKELETON step 8
should include a realistic swing distribution, or the Jensen results will be
reported on a distribution nobody operates in.

### J-41 The window is 12 h, so "daily mean" is the window mean

C-4 fixes the forecast window at 12 h. For a trajectory averaged over a whole
number of periods the ratio is unaffected, which holds for the time-varying test
profiles (period = TW). It would not hold for a genuine 24 h profile observed over
12 h, and a real deployment averages over 24 h. Parametrised rather than
hard-coded, and stated in the report.

---

## Caching the fix-1 steady state

### J-42 The cache is safe because the expensive uses carry no gradient

Phase 2 fix 1 replaced a closed-form steady state with a contraction solve, which
cost **2.62x per epoch** (interleaved median of 5 on an idle machine; sequential
runs on a busy one read anywhere from 2.1x to 5.1x, which is why the check script
now interleaves).

`theta_ss` has three uses. Two are on the sensor grid and need no gradient:

* `_ode_baseline` — the grid values enter `F_cum`; the `t`-gradient flows through
  the interpolation weight `frac` and through `decay`, never through the values
  themselves. No `nn.Parameter` is involved.
* `build_trunk_feats` `tss_s` → `dm` and `dr`. `dm` is gathered with
  `idx1 = tn_idx.long()`, an integer index, so it is piecewise-constant in `t` and
  carries zero gradient almost everywhere. `dr` is a max minus a min over the whole
  window and has no `t`-dependence at all.

The third use is the query-time value feeding the `driving` feature. That one is
interpolated at `t` and **must** stay differentiable, so it is left live.

So the two grid uses are cached and the differentiable one is not. Checked, not
assumed: `audit_port/scripts/11_check_ss_cache.py` asserts a forward pass with the
cache is bit-identical to one without, in both eval mode and training mode (the
`n_grid=20` sub-grid path), and that `d(theta_TO)/dt` is still finite and nonzero.

### J-43 Where the cache lives, and a second exact saving

`TrainingSet` gains `theta_ss` of shape (N, n_sensors), computed at generation by
`steady_state_on_grid` and stored in the `.npz`. It is computed with
`true_fixed_point_torch` in float32 — the same function and dtype the model uses —
so the stored values are bit-identical to what the model would have computed.
Verified against `CODOperator._theta_ss` directly: max difference 0.000e+00.

`load_training_set` returns `theta_ss=None` for the stored v57 `.npz`, which
predates the field, and `TrainingSet.ensure_theta_ss()` fills it in on demand. So
nothing breaks on the existing artifact.

Second saving, independent of the cache and also exact: `_thermal_predict_grid`
was solving `theta_ss` on its **expanded** (B*ns, ns) tensor. It now solves on the
unique (B, ns) sub-grid once and expands the result. Expanding identical values is
exact, so this is a pure ns-fold reduction — 2.05M elements down to 102k.

Result: **2.62x -> 1.48x** of the pre-fix-1 cost, i.e. the cache removes 70% of
fix 1's overhead.

### J-44 Why it is 1.48x and not 1.0x, and why I stopped there

The residual is the one use that has to stay differentiable. `ode_physics_loss`
calls `autograd.grad` six times per step, once per state, with
`retain_graph=True`, so the query-time solve's 20-iteration graph is
back-propagated six times.

Removing that means computing the query-time value as an *interpolation of the
cached grid* rather than as `theta_ss` evaluated at interpolated `(K, Ta)`. Those
are not the same function: one interpolates the output of a nonlinear map, the
other applies the map to interpolated inputs. It would be a change to the model's
semantics, not a cache, and it would break the bit-exactness that makes this whole
change safe to land. A throwaway measurement put the remaining opportunity at
about 0.6 s/epoch, so it is worth roughly another 0.5x — real, but not free.

Not taken. Flagged here instead, per the instruction to say so rather than work
around it.

### J-45 Nothing downstream moved

Re-verified after the change:

| check | result |
|---|---|
| `steady_state.compare()` | identical to audit step3 §3.3, every cell |
| `JENSEN_GAP.md` | **byte-identical** before and after |
| dataset reproduction (seed 42, 8000 ICs) | still byte-exact against the stored `.npz` |
| all 13 checkpoints, `strict=True` | still load |
| fix-1 solver vs brentq | still 1.33e-08 degC over the 2009-point grid |
| training wiring, both trainers | still passes |
| Phase 1 gates 1, 2, 3 | **all pass** |
| smoke run, 100 epochs / 50 ICs | theta_TO MAE **1.475 degC**, identical to the pre-cache smoke run |

The last row is the strongest of them: 100 training steps with the cache produce
the same weights as 100 without it, to the precision of the reported metrics.

### J-46 Training moved off this machine

Per instruction, and consistent with C-5: this machine has 4 cores and no CUDA, so
a 25,000-epoch run is 8-22 hours here against roughly an hour on a T4. No real
training is run locally. `scripts/colab_run.md` is the recipe, written so several
accounts can run different configs in parallel into one shared Drive folder —
`run.py` already namespaces output by `experiment.variant`, seed and config hash,
so parallel runs do not collide.

O-5 (retrain on the corrected physics) therefore stays **open**, with the local
work needed to make it cheap now done.

`run.py` gained three flags for this: `--max-wall-seconds` (so a run that will not
fit stops cleanly and records `stop_reason='wall_clock_budget'` rather than being
silently shortened), `--tag`, and `--theta-ss` (to run the v57-physics control
against the corrected physics without editing a config).

---

## An operationally realistic distribution

### J-47 The IC fix alone gives a bimodal distribution, not a centred one

Before writing the sampler I decomposed the seed-999 swing into what the IC causes
and what the profile causes
(`audit_port/scripts/12_swing_decomposition.py`). Split by case type:

| hot-spot swing degC | constant K | time-varying |
|---|---|---|
| as drawn | 14.48 | 28.95 |
| with a profile-consistent IC | **0.00** | 20.26 |
| the theta_ss forcing alone | 0.00 | 21.26 |

Constant load with a consistent IC means constant forcing, so the trajectory never
moves and the swing is exactly zero. The time-varying cases stay at 20 degC because
their own forcing is 21 degC. So a consistent IC alone produces a distribution
piled at 0 and 20, not one centred at 10-15.

That is why `realistic.py` contains a profile generator as well as the IC sampler
that was asked for. Reported rather than done quietly, because it changes what the
deliverable is.

### J-48 Load is solved for, not drawn

The old sampler draws `K` and `theta_a` independently, which is what allows a
150 degC initial condition and a hot-spot past 187 degC. A utility does the
opposite: it loads a transformer to keep it in temperature band.

So `make_realistic_profile` draws the intended mean hot-spot — `N(86, 11)` clipped
to [62, 122] degC, against IEC 60076-7's 98 degC rated and 120 degC normal-cyclic
ceiling — and `solve_K_for_hot_spot` bisects for the load factor achieving it at
that site's ambient. `steady_hot_spot` is strictly increasing in K, so bisection
needs no derivative and cannot miss.

This is what fixes the gas ICs, and the mechanism is worth stating: `c_eq` is
exponential in temperature, so the 37% IEC exceedance was never a gas-model problem.
It was the 150 degC initial conditions. Exceedance falls to **12.2%** (n = 2000)
without touching `k_gen`, `k_dis` or the service factor.

### J-49 The residual H2 exceedance is left visible on purpose

12.2% of realistic ICs still exceed an IEC attention level, and essentially all of
it is H2 (12.25%, against 3.6% or less for every other gas).

The cause is the kinetics, not the sampler: `c_eq(H2) = k_gen/k_dis * V_arr` is
**76 ppm at a 110 degC hot-spot** against an attention level of 100 ppm. A
transformer in long-run equilibrium at the IEEE reference temperature therefore
sits at 76% of the H2 attention level, and anything slightly hotter exceeds it.
Field practice puts a healthy unit at 5-50 ppm.

I could have hidden this by lowering `hot_spot_mean` a few degrees. Deliberately did
not: it is direct evidence for O-3 (the kinetic constants have no stated source),
and tuning a sampler to conceal a parameter problem is how the original results got
into trouble.

### J-50 Calibrated to the target, and the tension that exposes

Reaching a 10-15 degC hot-spot swing needs a daily load swing of **+-12-28%**,
which is at the upper end of what a real feeder does, plus a 6-16 degC peak-to-peak
ambient cycle. That is stated in `RealisticParams` rather than buried.

The honest reading is that the Jensen gap matters for **cycled** units. A genuinely
base-loaded transformer has almost no swing, hence almost no gap, and no method can
beat a mean-temperature calculation on it. That belongs in the paper as a scope
statement, not as a caveat discovered by a reviewer.

Realised: median swing 21.44 -> **11.20** degC, share above 25 degC 40% -> 9%,
share in the 8-18 band 23% -> 40%.

### J-51 Report medians, because the mean is a statement about the tail

The Jensen gap is exponential in swing, so for the high-activation-energy states a
few large-swing cases dominate any mean. On the realistic distribution C2H2's mean
gap is 9.6 while its median is 1.83.

Medians, against C-10's analytical values at the +-15 degC reference:

| state | old median | realistic median | C-10 at +-15 |
|---|---|---|---|
| `c_H2` | 2.016 | 1.302 | 1.550 |
| `c_C2H2` | 4.759 | 1.832 | 2.594 |
| `c_C2H4` | 2.718 | 1.477 | 1.877 |
| `c_CO` | 1.570 | 1.168 | 1.313 |
| `c_CO2` | 1.405 | 1.119 | 1.223 |
| `DP` | 2.308 | 1.386 | 1.701 |

The realistic medians sit just **below** C-10's reference, which is what a median
swing of 11.2 degC should give against a reference of +-15. Stratified by swing
band the two distributions give the same gap at the same swing — the old set was
not producing different physics, it was sampling a different place on the same
curve.

Quoting the old set's medians would have overstated the gap by 67% on DP and 160%
on C2H2.

### J-52 Two IEC baselines exist and they are not the same number

Audit M-9's 37.0% is the **8000-IC training set**. The seed-999 evaluation set of
100 ICs from the same sampler gives 45%. Both are reported in
`REALISTIC_DISTRIBUTION.md` §4 with the sample stated, and the headline comparison
uses the training-set figure against a 2000-sample estimate for the new sampler, so
neither side is a 100-draw fluctuation.

### J-53 Nothing is frozen

No hash recorded, no `DISTRIBUTION_FREEZE.md` entry, no test tiers. `RealisticParams`
holds every knob as a dataclass field so the calibration can be argued with. T2
(parameter extrapolation) and T3 (out-of-family) still need designing on top of this,
and the freeze has to happen before the first model is trained against it, not after.

---

## Phase 2 fix 6 — the Arrhenius envelope (DECISIONS N-1)

### J-54 Bound the temperature, not the rate

Two implementations of the same kinetics disagreed. `fast_rhs_np`, which produces
every label through RK45, evaluates a pure Arrhenius factor on a **temperature**
bounded to `[313.15, 573.15]` K. `fast_rhs_torch` and `CODOperator._gas_integral`
ported the lower bound and substituted `V_arr.clamp(max=1e4)` for the upper one.

Judgement call: **the model adopts the reference's envelope, and the reference is
not touched.** Four things decided it, none of them convenience.

*A single rate cap is not a temperature statement.* `V_arr = 1e4` is reached at
187.2 degC for C2H2 and 356.7 degC for CO2 — a 170 degC spread from one constant,
because the threshold moves with `E_act`. No saturation mechanism caps five
different reactions at the same dimensionless rate. If saturation were real it
would be species-specific and derived from a mechanism. A physical bound on
Arrhenius kinetics is a bound on temperature, and the reference already carries one
that says something defensible: above ~300 degC the oil is pyrolysing and this
kinetic model describes nothing.

*It was never an overflow guard.* `exp(B*e*(1/T_ref - 1/T))` is increasing in T
with supremum `exp(B*e/T_ref)`. The largest exponent across the six states is
54.83 (C2H2), against a float32 `exp` overflow threshold of 88.7. The factor cannot
overflow at any temperature, finite or infinite. Whatever the cap was for, it was
not that.

*What it was actually for, and why the envelope is a better version of it.* The
residual is evaluated on the *network's* predicted state, which is unbounded early
in training, so a magnitude guard is a reasonable thing to want. But the physics
loss already clamps the state at `STATE_CLAMP_HI[0] = 200` degC top-oil, and the
worst hot-spot constructible from that corner is 300.6 degC at K = 1.5 — the
573.15 K envelope to within a degree. The reference's bound binds at essentially
the same place the cap was reaching for, and agrees with ground truth by
construction instead of by accident.

*Does the failure mode survive the realistic sampler.* On `cod/data/realistic.py`
the hot-spot reaches 179.8 degC against the 187.2 degC where the cap first touches
acetylene, so 0 of 100 cases activate it, against 8 of 100 on the old seed-999 set.
It does not survive. The fix is deliberately **not** made conditional on that: a
benchmark whose reference and model integrate different equations is invalid
whether or not the current sample happens to notice, and the sampler is not frozen
yet, so relying on it to keep the discrepancy dormant would be building on
something still under argument.

The alternative — capping the reference to match the model — was rejected. It
makes ground truth non-Arrhenius above a species-dependent threshold with no
mechanism behind it, requires regenerating every label, and leaves the benchmark
measuring kinetics that no standard describes.

Measured, 4000 random states spanning the whole reachable box in float64
(`audit_port/scripts/14_arrhenius_clamp.py` Q6): before, 996 rows disagreed by up
to 100% of the derivative; after, 0 rows disagree and the max relative difference
is 5.5e-14.

### J-55 Fix 6 moves Gate 1, so it gets the fix-1 treatment

Unlike fixes 2-5 this one is on the evaluation path — `_gas_integral` *is* the gas
prediction — so it moves Gate 1 with no retrain. Gate 1 overall 1.49% -> 1.26%,
`c_C2H2` MAE 0.5926 -> 0.1138 ppm (5.2x), cases under 10% from 99 to 100. Six of
the 100 cases move; the other 94 are bit-identical.

Eight cases have a true hot-spot above 187.2 degC but only six move, because the
quadrature runs on the model's *predicted* top-oil grid rather than the true one.
Both numbers are reported rather than the convenient one.

**The checkpoint is invalid again.** `fast_rhs_torch` is the physics residual, so
the training objective changed: the stored weights were fitted against a residual
whose acetylene channel saturated above 187.2 degC and no longer does. This is a
different reason from fixes 1, 4 and 5 — the sampled distribution is untouched, so
no new `DISTRIBUTION_FREEZE.md` hash is needed for this fix alone.

Escape hatch follows the fix-1 pattern exactly: `legacy_V_clamp` is a plain
attribute, not a buffer, so loading a v57 checkpoint cannot silently switch the
kinetics back. `scripts/verify_phase1.py` passes `legacy_V_clamp=True` alongside
`theta_ss_mode="formula_C"` and all three gates still pass unchanged.

### J-56 Two dead clamps removed, and the diagnostic repointed

`chi_monotonicity_loss` and `chi_rate_loss_v10` also carried `.clamp(max=1e4)`.
Both are unreachable: `compute_theta_HS_torch` clamps theta_HS at 200 degC, where
V_arr for `E_a = 1` is 1740.7. Removing them is bit-identical and was verified as
such before doing it. Recorded rather than done silently, because "this clamp never
fires" is exactly the kind of claim that should carry its arithmetic.

`clamp_frac_V_arr_max` in the loss diagnostics is kept — the v57 cap is still
reachable through `legacy_V_clamp` — and `clamp_frac_T_HS_max` is added next to it,
since the temperature envelope is now the live bound and an unreported bound is how
this discrepancy survived three versions in the first place.

### J-57 `daily_mean.arrhenius` stays unclamped on both ends

It now agrees with the model as well as the reference, so its old "the model
clamps and the reference does not" note is gone. The temperature envelope is
deliberately *not* applied there either: the Jensen measurements live around
100 degC where it is inert, and adding it would silently flatten any future
measurement taken outside the envelope instead of making it visible.

---

## O-10 — calibrating the load swing against ETT

### J-58 Apparent power, and why active power alone would have been wrong

IEC 60076-7's K is a load-*current* ratio and losses go as I^2, so the quantity a
thermal model responds to is `|S| = sqrt(P^2 + Q^2)` summed over ETT's three
customer classes, not the active-power columns. It is also the only combination
that stays non-negative, which turned out to matter: ETTh1's total active power is
negative in 13.3% of hours.

Three other choices, all recorded in the report rather than made silently: never
pool the two series (they are different transformers and they disagree), compare
`K_amp` against **half** the peak-to-trough range because `K = K_base + K_amp*sin`,
and drop blocks containing an hour with every load channel at exactly zero as meter
outages (3 of 725 days on ETTh1, 0 on ETTh2).

### J-59 ETT publishes no nameplate rating, so §3b rests on a proxy

`K_amp` is per-unit of rated, so comparing to it needs a rated load that ETT does
not supply. Three proxies are tabulated instead of one being chosen quietly:
`p99(|S|)`, `p99/0.85`, and `max`. The middle one is quoted, on the grounds that
`p99` as rated asserts the unit hits nameplate in 1% of hours while real units peak
nearer 0.7-0.9 pu. The direction of the assumption is stated: a lower assumed peak
loading makes rated larger and the measured swing *smaller*, so `p99/0.85` is the
conservative choice against the sampler, not for it.

The rating-free normalisation (fraction of the day's own mean) is reported first
and then partly withdrawn: it breaks on ETTh1, whose net load sits near zero at
midday, depressing the denominator and widening the numerator at once. Its 62%
median is not a usable number and the report says so rather than letting it stand.

### J-60 The finding is a disagreement between two units, not a single number

Median daily swing as a fraction of rated, against `K_amp = 12-28%`:

| | median | below band | inside | above |
|---|---|---|---|---|
| ETTh2, all days | 8.7% | 85.2% | 14.8% | 0.0% |
| ETTh1, non-back-feeding days | 17.8% | 15.6% | 77.5% | 6.9% |
| ETTh1, back-feeding days | 29.7% | 0.0% | 39.1% | 60.9% |

Stated plainly, as the brief asked: **on the conventionally loaded unit a real
transformer swings about half what `RealisticParams` assumes**, and a Jensen
headline computed on the current sampler is correspondingly optimistic. On the
other unit it does not.

### J-61 ETTh1 back-feeds at midday and it is photovoltaic

Negative total active power on ETTh1 is zero at night, 51% at noon, and peaks
March-June. A midday-only, spring-peaking reversal is PV export and nothing else.
That makes ETTh1 the closer analogue to the manuscript's own framing — cycling
driven by renewables — and it is the unit that reaches `K_amp`'s band.

Initial expectation was that the PV would account for the whole difference between
the units, which would have been convenient: `K_amp` could then have been set from
a conventional baseline with a documented uplift for renewable duty. It does not.
Removing back-feeding days moves ETTh1 from 24.8% to 17.8%, about 7 of the 16
points separating it from ETTh2, and a non-back-feeding ETTh1 day still swings 2.0x
an ETTh2 day. The between-feeder spread is irreducible at n = 2 and two units
cannot estimate it. Recorded because the first draft of the report asserted the
convenient version before the split was computed.

### J-62 The oil temperature is the uncomfortable number, and it is left visible

Beyond what O-10 asked for. ETT ships `OT` alongside the load that produced it, so
it measures the quantity `K_amp` exists to produce rather than its input. Daily
top-oil amplitude: median 2.39 degC on ETTh1, 5.60 on ETTh2, against a sampler
targeting 10-15 degC at the hot spot.

Three reasons that is suggestive rather than decisive, all in the report: hot-spot
amplitude exceeds top-oil amplitude because the gradient moves in phase with the
load (plausibly ~2x, not the ~4x the comparison would need); **these units run
cold**, median OT 11.4 and 26.6 degC against `hot_spot_mean = 86`, and rise scales
as roughly `K^(2n)` so a lightly loaded unit swings little in absolute degC; and
`OT` is a sensor reading, not `theta_TO` as the model defines it.

So the load measurement transfers to a hotter unit and this one does not. It is
kept anyway, flagged as the number most likely to be flattering the sampler, and
turned into the concrete follow-up in J-63 instead of being dropped for being
inconvenient.

### J-63 The thermal parameters have the same problem as the gas kinetics

`tau_oil`, `DTheta_oil_R` and `n_exp` are IEC defaults, assumed rather than fitted,
exactly as `k_gen`/`k_dis` are in O-3. ETT gives measured oil temperature against
measured load on two real units, which is enough to test them directly. Not
attempted: it is larger than O-10 and would want its own entry. Noted as the reason
§5's discrepancy is currently unexplained rather than explained away.

### J-64 Nothing was changed

`RealisticParams` is untouched, as the brief required. Setting `K_amp` from this
would mean choosing which population the benchmark is about, and on a two-unit
sample that is a scope decision for the paper, not a calibration.

---

## O-9 — diagnosing the -3 degC rollout bias

### J-65 The bias is the metric's, not the model's

`RolloutResult.theta_bias` is `theta_TO_end - theta_ss_ref`. `theta_TO_end` is
top-oil at the end of the window; `theta_ss_ref` is `steady_state(K_w, Ta_w)`, the
temperature the unit would settle at under the window's *mean* forcing held
forever. Within each window the rollout applies a full sine period of ripple
(`Ta_w ± 2` degC, `K_w ± 0.05`), and top-oil follows it through a first-order lag
of `tau_oil = 150` min against `T = 720` min. At the instant the forcing returns to
its mean the lagged response has not, so the difference is nonzero for a model with
no error at all.

Three demonstrations that agree:

* Closed form. Gain `1/sqrt(1 + (omega tau)^2) = 0.607`, lag `atan(omega tau) =
  52.6 deg`, end-of-window factor `-gain sin(lag) = -0.482`. Applied to the
  steady-state amplitude of the two ripples this predicts a negative, one-signed
  offset growing with load.
* `ExactModel` — RK45 on `fast_rhs_np` at `rtol = 1e-10` wearing `CODOperator`'s
  call signature, so `chi_lifetime_rollout` runs against ground truth itself.
  Measured bias -2.863 degC at K_base = 0.85 rising monotonically to -3.876 at
  1.10, negative in 100% of 540 windows. The audit's |bias|_mean of 3.09 sits
  inside that range.
* Swapping the reference for the cyclic endpoint of the window's own forcing —
  which contains the lag already — leaves -0.002 degC. 99.94% of the effect is the
  lag.

### J-66 The K-monotonicity is what kills the manuscript's story for the second time

The bias grows smoothly from -2.86 to -3.88 degC as K_base goes 0.85 -> 1.10, with
no feature at K = 1. That is the signature of the lag mechanism, because
`dtheta_ss/dK` grows with K and the load ripple's contribution grows with it. An
ETC staircase at K = 1 would put a discontinuity *at* K = 1. The manuscript's
explanation was already refuted for being false at K = 1 (the two formulas coincide
exactly there and the Rf clamp is inactive); it is now also explaining an effect
that has no physical existence.

### J-67 The formula A / B lead: chased, and ruled out on shape rather than size

The audit's remaining lead was a ~-3 degC offset between formula A and formula B at
high load. It survives a size check — `A - B` does pass through -3 degC inside the
rollout's operating box — so it had to be ruled out on something else. Two things:

*Shape.* Evaluated along the actual `(K_w, Ta_w)` sequence the rollout visits over
a year, `A - B` swings across roughly 12 degC with the seasonal ambient. The
measured bias has sd 0.07 degC and is flat. A quantity that varies by degrees
cannot cause one that varies by hundredths, however well their means agree.

*Structure.* `formula_A` is not on the rollout path at all. `chi_lifetime_rollout`
takes one `steady_state` argument and uses it for `theta_ss0`, the gas IC via
`gas_ic_from_ss`, and `theta_ss_ref` alike — in v57 (`formula_B` throughout) and
now (`true_fixed_point_np` throughout). There is no second formula for a difference
to be taken against.

Recorded because checking the trace, rather than the table the lead came from, is
what turned a plausible cause into a ruled-out one.

### J-68 The ageing consequence is smaller than O-9 feared, and the gap moves

O-9's stated worry: at 10.8 %/K a systematic -3 degC understates the ageing rate by
~30%, so no EOL number is publishable. The sensitivity arithmetic is right —
`B_aging/T^2` is 10.77%/K at 100 degC, confirmed from the code's own constants —
but **the -3 degC never entered the DP calculation.** Under `dp_source="model"`,
the default, DP is advanced from `theta_for_dp = xp[:, 0]`, the predicted
trajectory over 20 quadrature points. `theta_ss_ref` reaches the DP update only
under `dp_source="reference"`, where the model is absent by design. The bias field
is reported and consumed by nothing.

This does not clear the rollout to publish EOL numbers, and the report says so.
It removes one specific reason to distrust them and replaces it with an honest
gap: the model's true rollout thermal error has never been measured, because the
field meant to measure it was measuring something else.

### J-69 `max_windows` added to `chi_lifetime_rollout`; the metric is NOT fixed here

The diagnosis needs many short scenarios rather than a few long ones, so
`chi_lifetime_rollout` gained `max_windows: int | None = None`. `None` preserves
the `max_years` behaviour exactly, so nothing that does not pass it can change.

`theta_bias` itself is left alone. O-9 asked for a diagnosis, and redefining the
metric in the same commit that explains why the old one was wrong would put the
before and the after in one diff. It should be scored against a reference
integration of `fast_rhs_np` over the same window from the same IC; `theta_ss_ref`
is worth keeping as its own field, since distance from equilibrium is a real
diagnostic, just not one to subtract a prediction from. Separate change, and it
needs a retrained model to be worth running — fix 6 invalidated the checkpoint
again.

---

## Fix 8 — `theta_bias` scored against the cyclic endpoint

### J-70 The reference has to be the reference physics, not an approximation of it

`theta_bias` is now `theta_TO_end - theta_cyc_ref`, where `theta_cyc_ref` is the
window's own forcing repeated until the endpoint stops moving. The old quantity
survives as `theta_ss_offset` — distance from the window-mean equilibrium is a
real diagnostic, it just is not model error.

The first attempt built the cyclic endpoint from
`DailyMeanArrhenius.theta_TO_trajectory`, the closed-form first-order solution,
because it is cheap and already in the package. **It was wrong by 0.11 degC at
K = 0.85 and 0.30 degC at K = 1.10, growing with load.** The recurrence drives
theta_TO toward `true_fixed_point_np(K, Ta)` as a *fixed* target, but the real
ODE's `fac_n` depends on theta_TO itself through the copper correction
`Rf = 1 + alpha_Cu (theta_HS - T_HS_ref)`. The two agree at equilibrium and drift
apart through the transient, which is exactly where this reference lives.

Putting a reference with its own 0.3 degC load-dependent error under a metric
meant to resolve 0.5 degC of model error would have reproduced the defect O-9
had just removed, one layer down and harder to see. `cyclic_endpoint_theta`
therefore integrates `fast_rhs_np` itself. Only the thermal state is carried: the
gas coupling is one-way, so `d(theta_TO)/dt` is a closed scalar ODE.

Caught by asserting the agreement in `19_verify_bias_fix.py` rather than
asserting it in a docstring. The docstring had in fact claimed "well under 0.01
degC" before the check was run.

### J-71 The fixed-point tolerance must be looser than the integrator's

First working version took 10 s per window. The cause was `tol = 1e-9` on the
cycle-to-cycle change, which is below RK45's own reproducibility at
`rtol = 1e-8`, so the iteration never converged and ran to `max_cycles = 40`
every single call. Each cycle contracts the mismatch by
`exp(T / tau_oil) = 122x`, so six cycles is already far past what the metric can
resolve; 40 bought a difference of 1e-5 degC for 7x the time.

Now `tol = 1e-6`, `max_cycles = 20`, and two things make it cheap enough to leave
on by default: `(K_w, Ta_w)` depends only on day-of-year, so the burn-in is cached
and a 50-year rollout does 730 of them rather than 36,500; and each new window is
seeded from the previous window's answer, which is ~0.01 degC away, so it
converges in one cycle instead of six.

### J-72 What the fixed metric reads on a zero-error model

`19_verify_bias_fix.py`, `ExactModel` over 30 windows at four loads:

| quantity | before fix 8 | after fix 8 |
|---|---|---|
| mean over windows 1+ | -3.34 degC | **-0.0009 degC** |
| max abs over windows 1+ | 3.88 degC | **0.0078 degC** |
| window 0 | -2.8 to -3.8 degC | +0.25 to +0.41 degC |

Window 0 is deliberately not zero and is not a residual artifact. The rollout
starts at `steady_state(K_base, 27.0)`, which is not the cyclic state, so in the
first window the unit genuinely is off-cycle; the metric reports that and it
decays to 0.008 degC by window 1 and -0.001 degC by window 2. A startup transient
is a real property of the trajectory, unlike the old -3 degC which never decayed
at all.

`16_bias_diagnosis.py` now reads `theta_ss_offset` explicitly at both of its call
sites. Left on `theta_bias` it would have regenerated `BIAS_DIAGNOSIS.md` with the
near-zero numbers of the fixed metric and quietly contradicted its own argument.

---

## Fix 7 — the sampler's forcing period (N-6, N-7)

Logged after fix 8 because that is the order the commits landed; the work
predates it.

### J-73 The load pattern is a day, the window is a slice of it

`make_realistic_profile` completed a whole sine period inside the 720 min
window, i.e. asserted a 12 h load period against a real one of 24 h. The sampler
is now built around `make_realistic_day` (a full 24 h pattern on its own grid)
plus `window_from_day` (the 12 h slice starting at a uniformly drawn time of
day, wrapped with `np.interp(..., period=P)` so a window straddling midnight
needs no special case). `RealisticParams.cycle_period = 1440.0` is a field like
any other knob, so the 24 h assumption can be argued with.

Every family became a day pattern, not only the periodic ones, and the
event-shaped ones kept their **absolute** durations: the overload spike is still
58-144 min and the evening peak still 130-216 min wide, with only the position
now drawn over the day. That is deliberate — an overload lasts as long as it
lasts regardless of what the surrounding cycle does — and it is also why the
realised uplift is smaller than the sinusoid arithmetic predicts (J-74).

The initial condition had to follow. `day_theta_cycle` and `day_steady_theta0`
put theta_TO(0) at the periodic state of the whole 24 h cycle, read at the
window's own offset, and the dissolved-gas equilibrium now averages the hot-spot
over the day rather than the window. Gas equilibrates over weeks, so a window
falling on the night trough should not be handed the gas loading of a
permanently cool unit. `periodic_steady_theta0` is kept for callers holding only
a window, documented as asserting exactly the error N-6 identifies, and is not
used by `build_realistic_set`.

### J-74 K_amp is not touched, and the uplift is 1.18 rather than 1.378

A first-order system attenuates a sinusoid by `1/sqrt(1 + (omega tau_oil)^2)`:
0.607 at a 12 h period, 0.837 at 24 h, ratio 1.378. Forcing at the wrong period
forced the calibration to assume 1.378x more load swing than reality to reach a
given hot-spot swing, and `K_amp = 12-28%` divided by 1.378 is 8.7-20.3% —
which is the range ETT measures (J-63: ETTh2 median 8.7%, ETTh1 non-back-feeding
17.8%). The amplitude was never the error, so `K_amp` is left alone.

Measured on 200 draws at seed 999, by RK45 on `fast_rhs_np` so no model error
enters: median realised hot-spot swing 13.18 degC against the old sampler's
11.20, with `K_amp` unchanged. That is 1.177x, not 1.378x.

The gap is expected rather than a discrepancy, and the first reason is the one
that can be checked from the code instead of inferred from the result: **fix 7
does not change the frequency content of the event-shaped families at all.**
Their timescales in minutes are identical before and after (J-73), so the 1.378
uplift never applied to them — only `daily` and `base_load` collect it in full,
and a mixture median has to fall below the pure-sinusoid figure by construction.
The other two reasons are that a 12 h window sees only half a 24 h cycle at a
random phase, and that windows containing none of the day's event are now a real
part of the population where they were previously absent by construction.

Consequence worth putting in the paper: to restore the old 11.20 degC median,
`K_amp` would have to be scaled by 0.850, i.e. 10.2-23.8% of rated — which
brackets both measured feeders instead of sitting above ETTh2's whole
distribution. Most of the sampler's apparent over-assumption of load swing was a
period error, not an amplitude error. The rescale itself is **not** applied:
choosing it is the scope decision O-10 §7 leaves open, not a period fix.

---

## Swing fidelity — does the surrogate flatten the cycle?

### J-75 The check MAE cannot do, and why it belongs in the C-11 protocol

Networks are spectrally biased toward low frequencies, so a thermal surrogate may
smooth peaks. If it does, it throws away part of the Jensen gap the method exists
to preserve — and thermal MAE cannot see it, because a flattened trajectory can
sit close in mean absolute error while under-stating the peak-to-trough range the
convex Arrhenius integral is sensitive to. MAE and swing are different
measurements and only one of them is what the convexity argument rests on.

On the v57 checkpoint the answer is that COD does **not** flatten: median
predicted/true swing 1.0121 over 50 live time-varying cases, under-predicting in
14% of them, and the ratio trends *toward* 1 as the swing grows (1.0176 in the
10-15 degC band, 1.0100 in 25-200) where spectral bias would worsen. The Jensen
gap carried along the predicted trajectory is 0.4-3.0% *above* the true one, not
below.

### J-76 The null result is a mechanism, and testing it needs a model that trained

COD predicts a correction to an analytic first-order solution, so the cycle shape
comes from the IEC baseline and the network's spectral bias has nothing to
flatten. That predicts the opposite for any architecture without such a baseline.
`18_swing_fidelity.py` now runs all three available checkpoints:

| model | baseline H | median swing ratio | under-predicting | thermal MAE |
|---|---|---|---|---|
| COD v57 | yes | 1.0121 | 14% | 0.51 degC |
| Mono FAIR | no | 0.6802 | 100% | 12.69 degC |
| Mono multi-head | no | 0.6493 | 100% | 8.81 degC |

The direction is right and the one-sidedness is the hard part to explain away:
independent error *inflates* a sampled max-minus-min, so an inaccurate but
unbiased model should over-predict swing. Losing it in 100% of cases is a
smoothing signature, not an error-size signature.

**It still does not establish the mechanism, because those two checkpoints did
not train.** MAE 12.7 and 8.8 degC against COD's 0.51, and 37.8 degC in the
25-200 band where the ratio collapses to 0.61 — a model not tracking the
trajectory at all says nothing about spectral bias. In the bands where they do
roughly track (MAE 2.2-4.5 degC) the loss is a milder 17-28%, and there the ratio
does not worsen with swing, it improves. Audit M-2 already found the monolithic
error *rising* 47x as capacity grows 16x with causal weights underflowed to zero,
i.e. "we could not train this baseline". Attributing the swing loss to spectral
bias would repeat exactly that inference, and the repo rule stands: a model that
did not converge is reported as not converged.

Ablation A is the clean one-variable test — COD's architecture with H replaced by
the constant x0, same network, same pipeline — and **its weights do not exist**.
Neither `ablation_a_no_baseline.pt` nor
`transformer_pideepOnet_abl_A_no_baseline.pt` is among the supplied artifacts,
which `cod/models/cod.py` already records at `CODNoBaseline`. The monolithic pair
changes two things at once (no baseline H *and* no cascaded gas integral) and
carries the J-8 defect besides. So the delta-learning argument this would support
is not yet writable; it needs one training run of `CODNoBaseline` on the fix-7
distribution at COD's budget.

### J-77 One mislabelled column, fixed

The gap table's last column was headed "gap lost" while every value in it was a
gain, and it computes `median(pred)/median(true) - 1`, the ratio of medians,
which is why it read +1.53% beside a median ratio of 1.0269 for `c_H2`. Renamed
and signed so that negative means gap lost.
