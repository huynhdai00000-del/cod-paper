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

### J-12 Model `__init__` no longer prints

`PIDeepONet_v24`, `PIDeepONet_Mono_Fair` and `PIDeepONet_Mono_MultiHead` all
print their parameter count on construction. Replaced by a `n_parameters()`
method; the capacity sweep prints it from the caller. No numerical effect.
