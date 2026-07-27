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
