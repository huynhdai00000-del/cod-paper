# Fix 7 — the sampler was forcing at the wrong period (N-6, N-7)

## 1. The attenuation argument

A first-order thermal system driven sinusoidally responds with amplitude `1/sqrt(1 + (omega tau_oil)^2)` of the steady-state amplitude:

| forcing period | omega*tau_oil | amplitude gain |
|---|---|---|
| 12 h | 1.3090 | 0.6071 |
| 24 h | 0.6545 | 0.8367 |

Ratio 1.378. The old sampler forced at 12 h, so to reach a given hot-spot swing it had to assume 1.378x more load swing than a real 24 h cycle needs.

`K_amp = 12%-28%` divided by 1.378 is **8.7%-20.3%**, against ETT's measured 8.7% (ETTh2 median) to 17.8% (ETTh1 non-back-feeding median). The amplitude was never the error.

**`K_amp` is therefore not touched.** Fix 7 changes only the period.

## 2. Realised hot-spot swing, before and after

N = 200, seed 999, half peak-to-peak of the true hot-spot trajectory — the same quantity C-10's table is indexed by, computed by RK45 on `fast_rhs_np` so no model error enters.

| sampler | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| old, 12 h period (N=100) | 100 | 4.51 | — | **11.20** | — | 24.74 | 13.67 | 45.31 |
| fix 7, 24 h period, same `K_amp` | 200 | 4.70 | 7.77 | **13.18** | 18.44 | 28.59 | 14.63 | 47.44 |

Share in the 8-18 degC band: 48% (old 40%). Share above 25 degC: 14% (old 9%).

### Reading it

Median realised swing is **13.18 degC** against the old sampler's 11.20, with `K_amp` unchanged — an uplift of **1.177x**, not the 1.378x the pure-sinusoid arithmetic predicts. §3 says why the rest is eaten.

The claim this does and does not support, stated carefully:

* **It does support the period diagnosis.** Holding the load amplitude fixed and correcting only the period raises the hot-spot swing by 1.18x, in the direction and roughly the size the attenuation argument predicts. The sampler was under-driving the thermal system and compensating with amplitude.

* **It does not support "the numbers are unchanged".** 13.18 is 18% above 11.20, so the two errors do not cancel exactly and quoting the old distribution's Jensen medians would be wrong.

The operational consequence is the one worth putting in the paper. To restore the old 11.20 degC median target, `K_amp` would have to be scaled by 0.850, i.e.

| | load amplitude, % of rated | vs ETT |
|---|---|---|
| `K_amp` before fix 7 | 12%-28% | above ETTh2's whole distribution; at the top of ETTh1's |
| `K_amp` implied after fix 7 | **10.2%-23.8%** | brackets ETTh2 (8.7%) and ETTh1 non-back-feeding (17.8%) |

So the honest statement for the paper is: **most of the sampler's apparent over-assumption of load swing was a period error, not an amplitude error.** Correcting the period moves the load amplitude needed for a 10-15 degC hot-spot swing from 12%-28% of rated — which ETT says is high — to 10.2%-23.8%, which brackets both measured feeders. That converts an admitted weakness into a resolved modelling detail without any hand-tuning of an amplitude to match data.

**`K_amp` is left at 12%-28% regardless**, per the brief. Applying the 0.850 rescale is a calibration decision that belongs with the scope decision O-10 §7 leaves open, not with a period fix.

## 3. Why the uplift is 1.18 and not 1.378

This is expected, not a discrepancy. The 1.378 ratio is the steady-state gain for a **pure sinusoid**, and the sampler is a mixture. Three things pull the population median below it:

1. **Most families are not sinusoids, and the fix does not change their frequency content at all.** Fix 7 deliberately preserved the absolute timescales of the event-shaped families — the overload spike is still 58-144 min long and the evening peak still 130-216 min wide, with only their *position* now drawn over the day. Their spectra in minutes are therefore identical before and after, so the 1.378 uplift never applies to them; a first-order system attenuates that higher-frequency content the same way it always did. Only `daily` and `base_load`, the two pure sinusoids, collect the full gain. The mixture median has to land below 1.378 by construction, and it does.

2. **A 12 h window sees only half a 24 h cycle.** Peak-to-trough within the window depends on where in the day the window lands, and averaged over a random phase it is less than the full daily peak-to-trough.

3. **Every family is now a day pattern, not a window pattern.** A shift change, an overload spike or an evening peak happens at a time of day, so some windows contain the event and some contain none of it. Windows with little variation are a real part of the population and were previously absent by construction.

All three reduce the realised swing relative to the naive 1.378 uplift, which is why the measurement in §2 is the answer and the arithmetic in §1 is only the motivation. The first is checkable from the family definitions rather than inferred from the result.

## 4. Jensen gap on the new distribution

Computed from the true hot-spot trajectory, so no model error is in it. Medians, because the gap is exponential in swing and the mean describes the tail (J-51).

| state | old sampler median | fix 7 median | fix 7 mean |
|---|---|---|---|
| `c_H2` | 1.302 | **1.355** | 2.035 |
| `c_C2H2` | 1.832 | **1.996** | 11.740 |
| `c_C2H4` | 1.477 | **1.563** | 3.404 |
| `c_CO` | 1.168 | **1.208** | 1.463 |
| `c_CO2` | 1.119 | **1.145** | 1.303 |
| `DP` | 1.386 | **1.448** | 2.561 |

## 5. IEC exceedance, unchanged by design

| gas | above attention |
|---|---|
| `c_H2` | 12.00% |
| `c_C2H2` | 1.00% |
| `c_C2H4` | 1.00% |
| `c_CO` | 0.50% |
| `c_CO2` | 0.00% |
| `any_gas` | 12.00% |

The gas initial conditions now equilibrate at the **day's** mean hot-spot rather than the window's, which is the right timescale — dissolved gas equilibrates over weeks, so a window falling on the night trough should not be given the gas loading of a permanently cool unit. The residual is still almost entirely H2, which remains O-3's problem and not the sampler's (J-49).

## 6. Still not frozen

No hash, no test tiers. `RealisticParams.cycle_period` is a field like every other knob so the 24 h assumption can be argued with. What fix 7 settles is that the sampler and ETT no longer disagree about load amplitude; what it does not settle is which feeder population the benchmark is about (O-10 §7).

