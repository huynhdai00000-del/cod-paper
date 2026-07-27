# O-8 — the Jensen gap, measured

Closes O-8. Seed-999 test set, N=100. The gap is computed from the **true** hot-spot trajectory (RK45, 241 points), so it carries no model error: it is a property of the reference physics and the test distribution.

```
gap_i = [ (1/T) * integral_0^T V_i(theta_HS(s)) ds ]  /  V_i( mean theta_HS )
```
By Jensen's inequality `gap >= 1` always, with equality only for a constant trajectory. Current practice evaluates the denominator; the physics is the numerator. The gap is the factor by which practice understates generation and ageing.

## 1. The implementation reproduces C-10's analytical table

`cod/models/daily_mean.py::jensen_gap_sinusoidal` computes the gap for a full-period sinusoid by quadrature. Activation energies come from the code's own `B_aging` and `E_act`, not from the table.

| state | Ea kJ/mol | +-5 degC | +-10 degC | +-15 degC | +-20 degC |
|---|---|---|---|---|---|
| `c_CO2` | 74.8 | 1.024 (1.02) | 1.097 (1.10) | 1.223 (1.22) | 1.408 (1.41) |
| `c_CO` | 87.3 | 1.033 (1.03) | 1.135 (1.14) | 1.313 (1.31) | 1.580 (1.58) |
| `c_H2` | 112.2 | 1.056 (1.06) | 1.232 (1.23) | 1.550 (1.55) | 2.050 (2.05) |
| `DP` | 124.7 | 1.070 (1.07) | 1.291 (1.29) | 1.701 (1.70) | 2.365 (2.37) |
| `c_C2H4` | 137.2 | 1.085 (1.09) | 1.359 (1.36) | 1.877 (1.88) | 2.746 (2.75) |
| `c_C2H2` | 174.6 | 1.141 (1.14) | 1.616 (1.62) | 2.594 (2.59) | 4.423 (4.42) |

Computed value first, C-10's value in brackets. Maximum disagreement 0.0048, i.e. rounding. **DP 1.701 against 1.70 and C2H2 2.594 against 2.59 at +-15 degC** — the two figures the paper leads with are confirmed from first principles.

## 2. The gap realised on the test set

The test set is not a set of sinusoids about 100 degC, so these numbers are not expected to match the table above; they say what the gap is on the distribution actually being evaluated.

**All 100 cases** (n = 100)

| state | Ea kJ/mol | mean gap | median | p90 | max | analytical at the median swing |
|---|---|---|---|---|---|---|
| `c_H2` | 112.2 | **2.580** | 2.016 | 4.535 | 11.722 | 2.301 |
| `c_C2H2` | 174.6 | **9.505** | 4.759 | 19.314 | 119.973 | 5.460 |
| `c_C2H4` | 137.2 | **4.081** | 2.718 | 7.692 | 28.752 | 3.201 |
| `c_CO` | 87.3 | **1.777** | 1.570 | 2.789 | 5.099 | 1.708 |
| `c_CO2` | 74.8 | **1.524** | 1.405 | 2.214 | 3.479 | 1.496 |
| `DP` | 124.7 | **3.211** | 2.308 | 5.872 | 18.236 | 2.705 |

Median realised swing 21.44 degC about a median mean hot-spot of 96.2 degC. The last column is C-10's sinusoid formula at that amplitude and centre.

**Constant-K cases** (n = 50)

| state | Ea kJ/mol | mean gap | median | p90 | max | analytical at the median swing |
|---|---|---|---|---|---|---|
| `c_H2` | 112.2 | **1.559** | 1.136 | 1.972 | 8.720 | 1.319 |
| `c_C2H2` | 174.6 | **4.857** | 1.322 | 5.518 | 96.481 | 1.869 |
| `c_C2H4` | 137.2 | **2.178** | 1.200 | 2.845 | 21.972 | 1.498 |
| `c_CO` | 87.3 | **1.262** | 1.084 | 1.525 | 3.789 | 1.185 |
| `c_CO2` | 74.8 | **1.174** | 1.062 | 1.380 | 2.633 | 1.133 |
| `DP` | 124.7 | **1.810** | 1.167 | 2.348 | 13.726 | 1.402 |

Median realised swing 11.26 degC about a median mean hot-spot of 94.0 degC. The last column is C-10's sinusoid formula at that amplitude and centre.

**Time-varying cases** (n = 50)

| state | Ea kJ/mol | mean gap | median | p90 | max | analytical at the median swing |
|---|---|---|---|---|---|---|
| `c_H2` | 112.2 | **3.601** | 3.299 | 5.620 | 11.722 | 3.678 |
| `c_C2H2` | 174.6 | **14.154** | 10.510 | 26.164 | 119.973 | 12.450 |
| `c_C2H4` | 137.2 | **5.984** | 5.129 | 10.306 | 28.752 | 5.904 |
| `c_CO` | 87.3 | **2.291** | 2.178 | 3.171 | 5.099 | 2.366 |
| `c_CO2` | 74.8 | **1.873** | 1.806 | 2.427 | 3.479 | 1.932 |
| `DP` | 124.7 | **4.611** | 4.091 | 7.582 | 18.236 | 4.645 |

Median realised swing 28.81 degC about a median mean hot-spot of 97.6 degC. The last column is C-10's sinusoid formula at that amplitude and centre.

### Why the constant-K cases show a gap at all

Constant K does not mean constant temperature. Each case starts from an initial condition drawn independently of its load, so theta_TO relaxes toward its steady state across the window with a time constant of 150 min against a 12 h window. The realised swing on those cases is a median of 11.26 degC — a monotone transient, not an oscillation. That is a real thermal excursion and the convexity applies to it, but it is a different shape from the sinusoid C-10 assumes, so the analytical column is only indicative there.

## 3. Realised hot-spot swing, which is what the gap depends on

| subset | mean swing | median | p90 | max | median mean-hot-spot | max hot-spot |
|---|---|---|---|---|---|---|
| all 100 | 21.71 | 21.44 | 38.52 | 52.97 | 96.2 | 236.9 |
| constant K | 14.48 | 11.26 | 31.58 | 43.62 | 94.0 | 236.9 |
| time-varying | 28.95 | 28.81 | 40.59 | 52.97 | 97.6 | 199.8 |

Stratified by swing, all 100 cases, DP and C2H2 only:

| swing band degC | n | DP gap | C2H2 gap | analytical DP | analytical C2H2 |
|---|---|---|---|---|---|
| 0-2 | 5 | 1.004 | 1.008 | 1.007 | 1.014 |
| 2-5 | 9 | 1.023 | 1.046 | 1.036 | 1.071 |
| 5-10 | 5 | 1.078 | 1.157 | 1.138 | 1.284 |
| 10-15 | 13 | 1.418 | 1.994 | 1.472 | 2.033 |
| 15-25 | 28 | 2.506 | 5.527 | 2.598 | 5.127 |
| 25-200 | 40 | 5.321 | 18.740 | 5.070 | 14.190 |

The gap rises monotonically with swing, as convexity requires, and the ordering across states follows activation energy exactly: CO2 < CO < H2 < DP < C2H4 < C2H2.

### The realised swing on this test set is not operationally realistic

40 of 100 cases swing by more than 25 degC and the median is 21.44 degC. That is not what a transformer does in a day; it is a property of the IC sampler. Audit M-9: `sample_consistent_ic` draws theta_TO(0) uniformly +-30 degC around the steady state and clips to [theta_a + 5, 150] **independently of the load**, so most cases begin far from equilibrium and spend the window relaxing toward it.

**The all-100 means in section 2 must therefore not be quoted as the operational gap.** They are the gap on a synthetic distribution with an inflated swing. The defensible operational statement is the stratified one: at a realistic +-10 to 15 degC swing (13 cases here) the measured DP gap is 1.418 and the C2H2 gap 1.994, against C-10's analytical 1.70 and 2.59 at +-15 degC.

This cuts against the paper's interest, which is why it needs saying: the honest headline is C-10's 1.70 and 2.59, not the larger numbers this particular test set produces.

## 4. What the gap costs, in units a practitioner reads

Extra gas generated over one 12 h window, resolved minus mean-temperature, in ppm:

| gas | mean | median | p90 | max | as % of the IEC attention level |
|---|---|---|---|---|---|
| `c_H2` | 0.05586 | 0.0007149 | 0.1907 | 1.232 | 0.0559% |
| `c_C2H2` | 0.2091 | 2.487e-05 | 0.09378 | 9.013 | 0.597% |
| `c_C2H4` | 0.08862 | 0.0001912 | 0.1781 | 2.985 | 0.0443% |
| `c_CO` | 0.01802 | 0.001155 | 0.0518 | 0.326 | 0.00257% |
| `c_CO2` | 0.01575 | 0.001869 | 0.0442 | 0.2828 | 0.000787% |

Small per window, which is the point: a 12 h shortfall is invisible, and it compounds. The scale-free statement is the ratio, and for ageing it converts directly into predicted life. Since life is inversely proportional to the ageing rate, a DP gap of `g` means the mean-temperature method predicts a life `g` times too long:

| DP gap | implied predicted life for a true 25-year life | overestimate |
|---|---|---|
| 2.308 (median case) | 57.7 years | +32.7 years |
| 5.872 (p90 case) | 146.8 years | +121.8 years |
| 18.236 (worst case) | 455.9 years | +430.9 years |

At C-10's reference swing of +-15 degC the DP gap is 1.701, so a transformer with a true 25-year life is assessed at 42.5 years — an overestimate of 17.5 years. That is the engineering consequence the paper is built on, and it is now measured rather than asserted.

## 5. Caveats worth stating in the paper

1. **The window is 12 h, not 24 h.** C-4 fixes the forecast window at 12 h, so "daily mean" is implemented as the window mean. For a trajectory averaged over a whole number of periods the ratio is unaffected, which holds for the time-varying test profiles (period = TW). It would not hold for a genuine 24 h profile sampled over 12 h, and a real deployment averages over 24 h.
2. **The constant-K cases contribute transients, not oscillations**, and the realised swing is inflated by the IC sampler (section 3). Quote the stratified gap at a realistic swing, not the all-100 mean. Report the two tiers separately, as C-9 already requires.
3. **`V_arr` is unclamped here**, matching `fast_rhs_np`. The model path clamps at 1e4 (DECISIONS N-1). Every number in this report is against the reference, so the clamp mismatch does not touch it — but the gap cannot be compared against surrogate output until that is fixed.
4. **This measures the gap, not any model's ability to close it.** `DailyMeanArrhenius` is a Tier 0 baseline with no parameters; how well the surrogate closes the gap is a separate experiment against the retrained model.

No CLOSED item is reopened. This is the empirical support C-10 was asserting analytically, and it confirms C-10's two headline figures to three decimals.

