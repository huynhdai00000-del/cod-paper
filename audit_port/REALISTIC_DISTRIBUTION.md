# An operationally realistic distribution

Both samplers, N = 100, measured the same way. **Nothing here is frozen** — this is for looking at before anything goes into `DISTRIBUTION_FREEZE.md`.

## 1. Why the old one is not operationally realistic

`profiles.sample_consistent_ic` draws `theta_TO(0)` from `steady_state(K, theta_a) + U(-30, 30)` where `K` and `theta_a` are drawn **independently of the profile that will drive the window**. So the unit starts at a temperature unrelated to its load and spends the window relaxing. Decomposed on the seed-999 test set (`audit_port/scripts/12_swing_decomposition.py`):

| hot-spot swing, degC | mean | median | p90 | max | above 15 |
|---|---|---|---|---|---|
| as drawn | 21.71 | 21.44 | 38.52 | 52.97 | 68% |
| with a profile-consistent IC | 10.85 | 5.11 | 31.11 | 38.69 | 38% |
| the theta_ss forcing alone | 11.23 | 6.14 | 30.74 | 35.86 | 41% |

The IC offset from the profile-consistent value has a median of **28.3 degC** and a maximum of 74.3. Fixing the IC alone removes 76% of the median swing.

But fixing the IC alone is not enough, and the same table says why. Split by case type, a consistent IC gives the constant-K cases a swing of **exactly zero** — constant forcing, constant trajectory — while the time-varying cases stay at 20.3 degC because their own forcing is 21.3 degC. The result is bimodal at 0 and 20, not centred at 10-15. Both the IC **and** the profile amplitudes have to change.

## 2. What the realistic sampler does

**Initial condition.** `theta_TO(0)` is the periodic steady state of the profile itself — the state a unit running this daily pattern would actually be in — plus a recent-history offset `N(0, 3)` clipped to +-8 degC and sensor noise `N(0, 0.5)`. The offset matters: without it every constant-load window would start exactly at its steady state and never move, which is no more realistic than +-30 degC and would make the Jensen gap identically zero on those cases.

**Operating point.** Load is not drawn directly. A fleet is loaded so that temperature stays in band, so the intended mean hot-spot is drawn (`N(86, 11)` clipped to [62, 122] degC, against IEC 60076-7's 98 degC rated and 120 degC normal-cyclic ceiling) and `solve_K_for_hot_spot` inverts for the load factor that achieves it at that site's ambient. **This is what removes the IEC exceedance**: `c_eq = k_gen V_arr / k_dis` is exponential in temperature, so the 37% was the 150 degC initial conditions, not the gas model.

**Gases.** Long-run equilibrium at the unit's own mean hot-spot times a service factor `U(0.45, 1.35)`, with 8% of units carrying an incipient fault (H2, C2H2, C2H4 multiplied by 2-7). A fleet where nothing is ever elevated would make the DGA benchmark trivial.

## 3. Realised hot-spot swing, side by side

Half peak-to-peak of the true hot-spot trajectory, the same quantity C-10's table is indexed by.

| | mean | median | p10 | p90 | max | in 8-18 degC | above 25 degC |
|---|---|---|---|---|---|---|---|
| old sampler | 21.71 | **21.44** | 3.51 | 38.52 | 52.97 | 23% | 40% |
| realistic | 13.67 | **11.20** | 4.51 | 24.74 | 45.31 | 40% | 9% |

Median moves from 21.4 to 11.2 degC, into the 10-15 target band, and the share above 25 degC falls from 40% to 9%.

Underlying temperatures:

| | mean hot-spot | max hot-spot reached | theta_TO(0) mean | theta_TO(0) max |
|---|---|---|---|---|
| old sampler | 102.3 | 236.9 | 81.2 | 141.3 |
| realistic | 94.6 | 179.4 | 75.7 | 122.3 |

The old set reaches a hot-spot of 236.9 degC, past the 187.2 degC where the model's `V_arr` clamp parts company with the reference (DECISIONS N-1). The realistic set peaks at 179.4 degC, so that failure mode does not arise (0% of cases against 6%).

## 4. Gas initial conditions against IEC 60599 attention levels

Both columns are the N = 100 evaluation ICs. Audit M-9's 37.0% is the 8000-IC *training* set; the same old sampler gives a different figure on this smaller draw, so both are reported and a large-sample estimate follows.

| gas | attention ppm | old: above | realistic: above | old median ppm | realistic median ppm |
|---|---|---|---|---|---|
| `c_H2` | 100 | 45% | 15% | 39.6 | 13.5 |
| `c_C2H2` | 35 | 25% | 3% | 0.583 | 0.133 |
| `c_C2H4` | 200 | 25% | 4% | 9.3 | 2.6 |
| `c_CO` | 700 | 21% | 2% | 76.2 | 33.6 |
| `c_CO2` | 2000 | 15% | 2% | 189 | 85.7 |
| **any gas** | | **45%** | **15%** | | |

On a stable sample (realistic sampler, n = 2000):

| gas | above attention |
|---|---|
| `c_H2` | 12.25% |
| `c_C2H2` | 2.95% |
| `c_C2H4` | 3.65% |
| `c_CO` | 2.40% |
| `c_CO2` | 0.85% |
| **any gas** | **12.25%** |

So exceedance falls from **37.0%** on the old 8000-IC training set (audit M-9's figure) to **12.2%**, a factor of 3.0.

**The residue is almost entirely H2, and that is a kinetics problem rather than a sampler problem.** `c_eq = k_gen/k_dis * V_arr` for H2 is 76 ppm at a 110 degC hot-spot against an attention level of 100 ppm, so a transformer in long-run equilibrium at the IEEE reference temperature sits at 76% of the H2 attention level and anything slightly hotter exceeds it. Field practice puts a healthy unit at 5-50 ppm. The sampler could be tuned to hide this by lowering the operating temperature, and deliberately is not: it is further evidence for O-3, that `k_gen` and `k_dis` have no stated source. The other four gases sit at 3.6% or below.

The remainder is the deliberate 8% incipient-fault subpopulation, which is a feature: a benchmark where no unit is ever flagged does not test anything a practitioner cares about.

Worst-case magnitudes tell the same story:

| gas | old max ppm | as x attention | realistic max ppm | as x attention |
|---|---|---|---|---|
| `c_H2` | 4.592e+04 | 459x | 5828 | 58.3x |
| `c_C2H2` | 4.391e+04 | 1255x | 744 | 21.3x |
| `c_C2H4` | 3.919e+04 | 196x | 2866 | 14.3x |
| `c_CO` | 2.443e+04 | 35x | 2832 | 4.0x |
| `c_CO2` | 2.014e+04 | 10x | 2963 | 1.5x |

## 5. The Jensen gap on each distribution

Computed from the true hot-spot trajectory, so no model error is in it.

**Read the medians.** The gap is exponential in swing, so for the high-Ea states a handful of large-swing cases dominate any mean: C2H2's realistic mean is 9.6 against a median of 1.83. The mean describes the tail, not a typical unit.

| state | Ea kJ/mol | old median | realistic median | C-10 analytical at +-15 degC |
|---|---|---|---|---|
| `c_H2` | 112.2 | 2.016 | **1.302** | 1.550 |
| `c_C2H2` | 174.6 | 4.759 | **1.832** | 2.594 |
| `c_C2H4` | 137.2 | 2.718 | **1.477** | 1.877 |
| `c_CO` | 87.3 | 1.570 | **1.168** | 1.313 |
| `c_CO2` | 74.8 | 1.405 | **1.119** | 1.223 |
| `DP` | 124.7 | 2.308 | **1.386** | 1.701 |

Means, and note how far the tail moves them:

| state | old mean | realistic mean |
|---|---|---|
| `c_H2` | 2.580 | 1.957 |
| `c_C2H2` | 9.505 | 9.589 |
| `c_C2H4` | 4.081 | 3.195 |
| `c_CO` | 1.777 | 1.422 |
| `c_CO2` | 1.524 | 1.273 |
| `DP` | 3.211 | 2.442 |

On medians, DP moves from 2.31 to 1.39 and C2H2 from 4.76 to 1.83, against C-10's analytical 1.70 and 2.59 at the +-15 degC reference. The realistic medians now sit just **below** that reference rather than far above it, which is what a median swing of 11.2 degC should give: C-10's reference is +-15 and this distribution is centred a little under it.

That is the honest headline. Quoting the old set's medians instead would have overstated the gap by 67% on DP and 160% on C2H2.

Stratified by swing, to show the mechanism is unchanged and only the distribution moved:

| swing band degC | old n | old DP gap | realistic n | realistic DP gap | analytical DP |
|---|---|---|---|---|---|
| 0-5 | 14 | 1.016 | 16 | 1.045 | 1.018 |
| 5-10 | 5 | 1.078 | 27 | 1.146 | 1.169 |
| 10-15 | 13 | 1.418 | 23 | 1.461 | 1.498 |
| 15-25 | 28 | 2.506 | 25 | 2.506 | 2.460 |
| 25-200 | 40 | 5.321 | 9 | 11.140 | 6.413 |

Same gap at the same swing in both columns. The old distribution was not producing a *different* physics, it was sampling a different place on the same curve.

## 6. What is still open

1. **The load amplitude is calibrated to the target, not measured from a fleet.** Reaching a 10-15 degC hot-spot swing needs a daily load swing of +-12-28%, which is at the upper end of what a real feeder does. That is stated in `RealisticParams` rather than hidden. The honest reading: the Jensen gap matters for cycled units, and a genuinely base-loaded transformer has almost no gap for any method to exploit.
2. **The gas kinetics still have no provenance (O-3).** The IEC exceedance fell because temperatures became realistic, not because `k_gen` and `k_dis` were justified. `c_eq` at a 110 degC hot-spot is 76 ppm of H2 against a 100 ppm attention level, and nothing in the repository says why.
3. **The 8% fault subpopulation is invented.** It is a modelling choice to keep the benchmark non-trivial, not a measured fleet statistic.
4. **Not frozen.** No hash recorded, no test tiers defined. T2 and T3 still need designing on top of this, and the whole point of the freeze protocol is that it happens before the first model is trained against it.

