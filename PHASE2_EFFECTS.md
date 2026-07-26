# Phase 2 — effect of each fix on Gate 1

Gate 1 is the seed-999 benchmark scored with `transformer_pideepOnet_v57.pt`. Read the mechanism note first:

> Gate 1 loads a checkpoint. A fix that acts only on the training path cannot move it until a retrain happens. A nonzero movement for such a fix would mean something had leaked from training into evaluation, so **zero is the correct and expected result** for fixes 2-5, and each is verified against the quantity it actually targets instead.

## Fix 1 — unify on `true_fixed_point()`

Decomposed, because the two halves of this fix move Gate 1 for different reasons:

| configuration | overall NMAE % | theta_TO NMAE % | theta_TO MAE degC |
|---|---|---|---|
| v57 (formula A ICs, formula C attractor) | 1.5 | 1.5 | 0.399 |
| fix 1 partial: attractor only | 6.9 | 4.1 | 1.227 |
| fix 1 partial: ICs only | 1.6 | 2.2 | 0.397 |
| fix 1 full | 7.2 | 5.7 | 1.232 |

| quantity | before (v57) | after (fixed) | change |
|---|---|---|---|
| `theta_TO` NMAE % | 1.5 | 5.7 | +4.2 |
| `c_H2` NMAE % | 1.3 | 7.6 | +6.4 |
| `c_C2H2` NMAE % | 2.3 | 9.3 | +7.0 |
| `c_C2H4` NMAE % | 1.6 | 9.1 | +7.4 |
| `c_CO` NMAE % | 1.1 | 5.9 | +4.7 |
| `c_CO2` NMAE % | 1.1 | 5.5 | +4.4 |
| **overall NMAE %** | 1.5 | 7.2 | +5.7 |
| constant K % | 1.2 | 5.7 | +4.5 |
| time-varying % | 1.8 | 8.6 | +6.9 |
| cases < 10% | 99 | 77 | -22 |
| theta_TO MAE degC | 0.399 | 1.232 | +0.833 |

**The checkpoint is now invalid and a retrain is required.** Two independent reasons:

1. Changing the IC formula changes every initial condition and, through `c_eq = k_gen * V_arr / k_dis` with V_arr exponential in temperature, every gas IC multiplicatively. That is a different training distribution, so `DISTRIBUTION_FREEZE.md` must be re-established before any model is trained against it.
2. Changing the model's attractor changes the analytic baseline the network was trained to correct. The stored weights encode a correction to formula C; applied on top of the true fixed point they are correcting the wrong function, which is exactly what the degradation above shows.

The numbers above are therefore **not** a claim that the fixed model is worse. They measure how far the v57 weights are from the corrected physics, which is the honest quantity to report before a retrain exists.

### What the fix corrects, at the level of the physics

| K | theta_a | true fixed point | A (v57 ICs) | A error | C (v57 attractor) | C error |
|---|---|---|---|---|---|---|
| 1.0 | 30 | 87.03 | 85.00 | -2.03 | 86.53 | -0.50 |
| 1.2 | 30 | 113.77 | 103.63 | -10.14 | 109.59 | -4.18 |
| 1.3 | 30 | 131.94 | 113.69 | -18.25 | 123.49 | -8.45 |
| 1.3 | 45 | 152.49 | 128.69 | -23.80 | 141.88 | -10.61 |

## Fix 2 — remove the double `pd_factor`

Gate 1 movement: **none, and none is possible.** The surviving `ode_physics_loss` uses a raw residual and never consumes `RHS_SCALE` (PORT_LOG J-4), so this defect had no effect on v57's results. It is a hygiene fix that stops the next person who reaches for `RHS_SCALE` from getting a squared factor.

Verified directly on the quantity it targets:

| state | v57 (doubled) | fixed | ratio |
|---|---|---|---|
| `theta_TO` | 4.420322e-01 | 4.420322e-01 | 1.0000 |
| `c_H2` | 1.406029e-03 | 1.406029e-03 | 1.0000 |
| `c_C2H2` | 1.574478e-03 | 7.108254e-04 | 2.2150 |
| `c_C2H4` | 1.119854e-03 | 1.119854e-03 | 1.0000 |
| `c_CO` | 7.841226e-04 | 7.841226e-04 | 1.0000 |
| `c_CO2` | 8.312543e-04 | 8.312543e-04 | 1.0000 |

Only `c_C2H2` changes, by a factor of 2.215. `pd_factor_np(1.3) = 2.2150` and 2.2150^2 / 2.2150 = 2.2150, which is the factor recovered — the second application was squaring it, exactly as audit section 8.3 says.

## Fix 3 — causal weighting in log space, floored, shared schedule

Gate 1 movement: **none, and none is possible.** Causal weighting exists only inside the training loss; evaluation never touches it.

Verified on the failure mode it targets. `cum` is the cumulative chunk residual; the weight is `exp(-eps * cum)`.

| eps * cum | v57 linear-space weight | fixed (log space, floor 1e-8) | v57 underflowed? |
|---|---|---|---|
| 1 | 3.679e-01 | 3.679e-01 | no |
| 10 | 4.540e-05 | 4.540e-05 | no |
| 50 | 1.929e-22 | 1.000e-08 | no |
| 88 | 6.055e-39 | 1.000e-08 | no |
| 104 | 0.000e+00 | 1.000e-08 | **YES** |
| 200 | 0.000e+00 | 1.000e-08 | **YES** |
| 1000 | 0.000e+00 | 1.000e-08 | **YES** |

The v57 weight reaches exactly 0.0 at `eps * cum` around 88, the float32 underflow point of `exp(-x)`. Past that the later collocation chunks contribute **nothing** to the loss and the model trains on the early window only. That is what produced Mono Fair's `wm = 0.000` against COD's `wm = 0.988` and invalidated the comparison (audit B-1). The floored log-space weight bottoms out at 1e-8, which is small but never zero, so the gradient never disappears.

The epsilon schedule is now shared: `EpsilonSchedule(shared=True)` advances on a fixed epoch count rather than on each model's own `wm`, so two models being compared follow the same trajectory. Under v57 they did not — COD's epsilon climbed to the 50.0 cap because its weights stayed high, while Mono Fair's froze near the start because its did not.

## Summary

| fix | moves Gate 1? | measured effect | checkpoint still valid? |
|---|---|---|---|
| 1 unify steady state | **yes** | overall 1.5% -> 7.2%, theta_TO MAE 0.40 -> 1.23 degC | **no — retrain required** |
| 2 double pd_factor | no, by construction | `c_C2H2` RHS_SCALE /2.215; nothing consumes it | yes |
| 3 causal weighting | no, by construction | weight floor 0.0 -> 1e-8; no underflow at any eps*cum; schedule now shared | yes |
| 4-5 | not yet applied | — | — |

