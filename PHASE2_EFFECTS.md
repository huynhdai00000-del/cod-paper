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

## Summary

| fix | moves Gate 1? | measured effect | checkpoint still valid? |
|---|---|---|---|
| 1 unify steady state | **yes** | overall 1.5% -> 7.2%, theta_TO MAE 0.40 -> 1.23 degC | **no — retrain required** |
| 2-5 | not yet applied | — | — |

