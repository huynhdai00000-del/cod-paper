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
