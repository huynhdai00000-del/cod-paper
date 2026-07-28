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

## Fix 4 — randomise the ambient phase

Gate 1 movement: **none.** Only the training profile generator draws an ambient phase; the seed-999 test set builds its own profiles with a hard-coded phase of pi/3.

That is precisely the gap (audit B-5): training saw one phase, the test set another. Verified on 2000 sampled profiles:

| | v57 (phase fixed at 0) | fixed (phase ~ U(0, 2 pi)) |
|---|---|---|
| mean \|Ta(0) - mean(Ta)\| | 0.0000 degC | 2.5012 degC |
| profiles starting at their mean | 100.0% | 0.7% |

Under v57 every single training profile starts at its ambient mean, because sin(0) = 0. The test set's pi/3 phase starts it 0.866 of the amplitude above the mean instead — a systematic offset the model never saw in training. After the fix only 0.7% of training profiles have that special structure.

## Fix 5 — add the missing `.clip()` to the 'step' branch

Gate 1 movement: **none.** Only the training profile generator has a 'step' branch; the test set uses constant or sinusoidal K.

Verified on the full 8000-profile training set, generated through `generate_training_set` with every v57 setting on both arms so that only `clip_step` differs and the v57 column reproduces the stored `transformer_training_v57.npz` exactly:

| | v57 (unclipped) | fixed |
|---|---|---|
| min K over all profiles | 0.257134 | 0.300000 |
| (stored .npz min, for reference) | 0.257134 | — |
| max K over all profiles | 1.500000 | 1.500000 |
| profiles below the 0.3 floor | 12 | 0 |
| profiles above the 1.4 ceiling | 601 | 592 |

The v57 arm reproduces the stored minimum of 0.257134 exactly, which confirms the 'step' branch is the one responsible. After the fix the minimum is exactly the documented floor.

12 of 8000 profiles dipped below the 0.3 floor, and the clip also brings 9 back under the 1.4 ceiling. The 592 profiles still above 1.4 belong to the sinusoidal, `tv_high_amp` and `overload_spike` families, which clip to 1.5 deliberately because they are meant to reach into overload — so 1.4 is the right bound for `step`, matching its piecewise-constant siblings `ramp` and `multi_step`.

A small share of the training mass, so this is a correctness fix rather than a results fix. It is still worth making: a documented sampling range the code does not honour is exactly the kind of discrepancy a reviewer checks.

## Fix 6 — the Arrhenius envelope: bound the temperature, not the rate

Gate 1 movement: **yes.** This one acts on the *evaluation* path — `CODOperator._gas_integral` is the model's entire gas prediction — so it moves Gate 1 without a retrain, like fix 1 and unlike 2-5.

v57 capped the Arrhenius factor at `V_arr.clamp(max=1e4)` in `fast_rhs_torch` and in `_gas_integral`. `fast_rhs_np`, which generates every label through RK45, has no such cap: it bounds the *temperature* at `np.clip(theta_HS + 273.15, 313.15, 573.15)` and is pure Arrhenius inside that envelope. Reference and model were integrating different kinetics (DECISIONS N-1).

Where a single rate cap actually bites, per species — computed from the code's own constants:

| state | Ea kJ/mol | theta_HS at V_arr = 1e4 |
|---|---|---|
| `c_H2` | 112.2 | 245.3 degC |
| `c_C2H2` | 174.6 | 187.2 degC |
| `c_C2H4` | 137.2 | 214.0 degC |
| `c_CO` | 87.3 | 303.6 degC |
| `c_CO2` | 74.8 | 356.7 degC |
| `DP` | 124.7 | 227.6 degC |

One constant, a 170 degC spread in the temperature it enforces. That is the argument against it: no saturation mechanism caps five different reactions at the same dimensionless rate. A physical bound on Arrhenius kinetics is a statement about temperature — above about 300 degC the oil is pyrolysing and this kinetic model no longer describes anything — and the reference already carries exactly that bound.

| quantity | before (v57) | after (fixed) | change |
|---|---|---|---|
| `theta_TO` NMAE % | 1.5 | 1.5 | +0.0 |
| `c_H2` NMAE % | 1.3 | 1.3 | +0.0 |
| `c_C2H2` NMAE % | 2.3 | 1.2 | -1.1 |
| `c_C2H4` NMAE % | 1.6 | 1.4 | -0.3 |
| `c_CO` NMAE % | 1.1 | 1.1 | +0.0 |
| `c_CO2` NMAE % | 1.1 | 1.1 | +0.0 |
| **overall NMAE %** | 1.5 | 1.3 | -0.2 |
| constant K % | 1.2 | 0.9 | -0.3 |
| time-varying % | 1.8 | 1.6 | -0.1 |
| cases < 10% | 99 | 100 | +1 |
| theta_TO MAE degC | 0.399 | 0.399 | +0.000 |

Absolute units, which is where the movement is legible:

| state | before (v57) MAE | after (fixed) MAE | ratio |
|---|---|---|---|
| `theta_TO` | 0.399335 degC | 0.399335 degC | 1.000 |
| `c_H2` | 0.0234039 ppm | 0.0234039 ppm | 1.000 |
| `c_C2H2` | 0.592626 ppm | 0.113757 ppm | 0.192 |
| `c_C2H4` | 0.164648 ppm | 0.0422561 ppm | 0.257 |
| `c_CO` | 0.00600911 ppm | 0.00600911 ppm | 1.000 |
| `c_CO2` | 0.0045081 ppm | 0.0045081 ppm | 1.000 |

`c_C2H2` is the state the cap was distorting and its MAE falls by 5.2x. Eight of the 100 seed-999 cases carry a true hot-spot above the 187.2 degC where the cap first touches acetylene; 6 of them move the model's own prediction, since the quadrature runs on the *predicted* top-oil grid, not the true one. The remaining 94 are bit-identical. The affected cases are the ones audit M-9's initial conditions drive to a 237 degC hot-spot.

Agreement between the two implementations, 4000 random states spanning the whole reachable box in float64 (`audit_port/scripts/14_arrhenius_clamp.py` Q6):

| `fast_rhs_torch` variant | max rel. diff vs `fast_rhs_np` | rows disagreeing |
|---|---|---|
| v57 (rate cap) | 9.998e-01 | 996 / 4000 |
| fixed (temperature envelope) | 5.520e-14 | **0 / 4000** |

A quarter of the box disagreed, by up to 100% of the derivative. After the fix the two agree to float64 round-off everywhere, which is the property a benchmark needs and this one did not have.

**This invalidates the checkpoint again.** `fast_rhs_torch` is the physics residual, so the training objective changed: the stored weights were fitted against a residual whose acetylene channel saturated above 187.2 degC and no longer does. The numbers above are not a claim that either model is better — they measure how far the v57 weights sit from the corrected kinetics.

What the cap was protecting against, and whether the failure mode survives — `audit_port/scripts/14_arrhenius_clamp.py`:

* Not overflow. `exp(B*e*(1/T_ref - 1/T))` is increasing in T with supremum `exp(B*e/T_ref)`, which is 54.83 in the exponent for C2H2, its largest value across the six states. float32 `exp` overflows at 88.7. The factor cannot overflow at any temperature, finite or infinite, so the cap was never an overflow guard.

* It was a magnitude guard on an unbounded network output, and the reference's own envelope is a strictly better one. The physics loss already clamps the predicted state at `STATE_CLAMP_HI[0] = 200` degC top-oil; the worst hot-spot constructible from that corner is 300.6 degC at K = 1.5, which is the 573.15 K envelope to within a degree. The reference bound therefore binds at essentially the same place the old cap was reaching for, while agreeing with ground truth by construction.

* On the operationally realistic sampler the question is moot: the hot-spot reaches 179.8 degC against the 187.2 degC where the cap first touched C2H2, so **0 of 100** cases activate it, against 8 of 100 on the old seed-999 set. The failure mode the cap addressed does not arise on the distribution the benchmark is moving to — but the fix is not conditional on that, because a benchmark whose reference and model solve different equations is invalid whether or not the current sample happens to notice.

The alternative — capping the reference instead — was rejected. It would make the ground truth non-Arrhenius above a species-dependent temperature with no mechanism behind the threshold, require regenerating every label, and leave the benchmark measuring a kinetics no standard describes. Between an arbitrary rate cap and a stated temperature envelope, the envelope is the defensible object, and it is the one already in the reference.

## Summary

| fix | moves Gate 1? | measured effect | checkpoint still valid? |
|---|---|---|---|
| 1 unify steady state | **yes** | overall 1.5% -> 7.2%, theta_TO MAE 0.40 -> 1.23 degC | **no — retrain required** |
| 2 double pd_factor | no, by construction | `c_C2H2` RHS_SCALE /2.215; nothing consumes it | yes |
| 3 causal weighting | no, by construction | weight floor 0.0 -> 1e-8; no underflow at any eps*cum; schedule now shared | yes |
| 4 ambient phase | no | training profiles starting at their ambient mean 100% -> 1% | **no — training distribution changed** |
| 5 step clip | no | min training K 0.2571 -> 0.3000 | **no — training distribution changed** |
| 6 Arrhenius envelope | **yes** | overall 1.5% -> 1.3%, `c_C2H2` MAE 0.5926 -> 0.1138 ppm, 6/100 cases moved | **no — the physics residual changed** |

Fixes 1, 4 and 5 all change the training distribution, so the frozen distribution hash must be re-established and recorded in `DISTRIBUTION_FREEZE.md` before the first retrain. Fixes 2 and 3 leave the distribution untouched. Fix 6 leaves the sampled distribution untouched but changes the equation the labels and the physics residual both refer to, which invalidates the checkpoint for a different reason and does not need a new distribution hash.

