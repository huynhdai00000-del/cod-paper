# O-9 — the -3 degC rollout bias, diagnosed

**Result: the bias is an artifact of the diagnostic, not of the model or the physics.** `RolloutResult.theta_bias` subtracts the steady state of the window's *mean* forcing from a top-oil value that is a *lagged* response to that forcing's ripple. A model with zero error reports it too.

Demonstrated three ways that agree:

1. A closed-form first-order-lag calculation predicts a negative, one-signed offset growing with load — `tau_oil = 150` min against a 720 min window gives a 52.6 deg phase lag, so at the window end the response is still below the mean it is returning to (§3).

2. Running the rollout against an exact RK45 integration of `fast_rhs_np` — a model with zero error by construction — measures -2.86 degC at K_base = 0.85 rising monotonically to -3.88 at 1.10, negative in 100% of 540 windows, with no feature at K = 1 (§4). The audit's |bias|_mean of 3.09 degC sits inside that range.

3. Scoring the same trajectories against a reference that already contains the lag — the cyclic endpoint of the window's own forcing — leaves -0.002 degC, i.e. 99.94% of the effect is the lag (§5).

The monotonic growth through K = 1 refutes the manuscript's ETC-staircase explanation a second way: a staircase *at* K = 1 would put a discontinuity there, not a smooth trend through it.

The audit's remaining lead — a ~-3 degC offset between formula A and formula B at high load — survives a size check and is ruled out on shape and on structure. Along the actual rollout trace `A - B` averages -1.15 degC, not -3; it has sd 1.47 against the bias's 0.07; and it changes sign 2 times a year. `formula_A` also never appears on the rollout path at all (§1).

Consequence for the ageing concern in §6: the 10.8 %/K sensitivity arithmetic is correct, but the -3 degC never entered the DP calculation, which is advanced from the predicted trajectory rather than from this reference. The real rollout thermal error remains unmeasured and needs a retrained model (§7).

## 1. The formula A vs B lead: right magnitude, wrong path

The rollout drives `K_base + 0.05 sin(2 pi day/365)` against `Ta = 27 - 12 cos(2 pi day/365)`, so it spans K in [0.80, 1.15] and theta_a in [15, 39]. The formulas over that box:

| K | theta_a | TRUE | A | B | C | A-B | C-B | A-TRUE |
|---|---|---|---|---|---|---|---|---|
| 0.85 | 15 | 55.47 | 57.41 | 55.40 | 56.10 | +2.00 | +0.69 | +1.93 |
| 1.00 | 15 | 68.98 | 70.00 | 69.23 | 69.23 | +0.77 | +0.00 | +1.02 |
| 1.10 | 15 | 80.31 | 79.06 | 80.65 | 79.65 | -1.59 | -1.00 | -1.25 |
| 1.15 | 15 | 86.87 | 83.78 | 87.12 | 85.40 | -3.34 | -1.72 | -3.09 |
| 0.85 | 27 | 69.23 | 69.41 | 68.84 | 69.52 | +0.57 | +0.69 | +0.18 |
| 1.00 | 27 | 83.43 | 82.00 | 83.07 | 83.07 | -1.07 | +0.00 | -1.43 |
| 1.10 | 27 | 95.33 | 91.06 | 94.78 | 93.78 | -3.72 | -1.00 | -4.27 |
| 1.15 | 27 | 102.22 | 95.78 | 101.39 | 99.69 | -5.61 | -1.71 | -6.44 |
| 0.85 | 39 | 82.96 | 81.41 | 82.26 | 82.94 | -0.85 | +0.68 | -1.55 |
| 1.00 | 39 | 97.85 | 94.00 | 96.90 | 96.90 | -2.90 | +0.00 | -3.85 |
| 1.10 | 39 | 110.32 | 103.06 | 108.89 | 107.90 | -5.83 | -0.99 | -7.26 |
| 1.15 | 39 | 117.53 | 107.78 | 115.65 | 113.95 | -7.87 | -1.70 | -9.74 |

**The lead is real in the box.** `A - B` passes through -3 degC inside the rollout's operating range and reaches -7.87 at the corner. Right size, right sign. It is still not the cause, for two independent reasons — and the first only becomes visible once the trace is used instead of the table.

### 1a. Along the actual rollout trace, A - B has the wrong shape

The task is to check the lead against the trace, not against a corner of a table. Evaluating both formulas along the exact `(K_w, Ta_w)` sequence the rollout visits over one year:

| quantity along the trace (K_base = 1.00) | mean | sd | min | max | sign changes |
|---|---|---|---|---|---|
| `A - B` | -1.150 | 1.474 | -3.253 | +0.931 | 2 |
| `C - B` | -0.028 | 0.267 | -0.433 | +0.320 | 3 |
| measured bias (§4, K_base = 1.00) | -3.409 | 0.069 | -3.505 | -2.993 | 0 |

On the trace the lead fails on all three counts:

* **Mean.** `A - B` averages -1.15 degC over the year, not -3. The ~-3 degC figure is its seasonal *extreme* (-3.25), reached only in the hottest weeks, not its typical value.

* **Variability.** `A - B` has sd 1.47 degC and spans 4.2 degC across the year, because it is driven by the seasonal ambient. The measured bias has sd 0.07 and is flat. A quantity that varies by degrees cannot cause one that varies by hundredths.

* **Sign.** `A - B` changes sign 2 times a year and is *positive* for part of it (max +0.93). The bias is negative in every window.

This is the value of checking the trace rather than the table the lead came from. In the table `A - B` reaches -7.87 degC and looks like a candidate; on the trajectory the rollout actually visits it is three times too small on average, twenty times too variable, and not even one-signed.

### 1b. And formula A is not on the rollout path at all

`chi_lifetime_rollout` takes a single `steady_state` argument and uses it for three things — the initial `theta_ss0`, the gas IC through `gas_ic_from_ss`, and `theta_ss_ref`, the reference the bias is scored against. All three are the same function. There is no second formula for the difference to be taken against, in v57 (where it was `formula_B` throughout) or now (where it is `true_fixed_point_np` throughout).

The one real formula mismatch in v57 was between the rollout's reference (B) and the model's own analytic attractor (C), which the model relaxes toward. `C - B` along the trace has mean -0.028 degC and changes sign 3 times a year, so it cannot produce a one-signed 3 degC either. Phase 2 fix 1 removed it entirely, and §4 measures the bias with it gone — unchanged.

So the lead is a coincidence of magnitude. Worth having chased: it was the only quantity in the neighbourhood with both the right size and the right sign. Ruling it out on shape and on structure rather than on size is what leaves §2 as the remaining explanation.

## 2. What `theta_bias` actually measures

```python
@property
def theta_bias(self):
    return self.theta_TO_end - self.theta_ss_ref
```
`theta_TO_end` is the model's top-oil at the **end** of the window. `theta_ss_ref` is `steady_state(K_w, Ta_w)`, the top-oil the unit would settle at if it were driven by the window's *mean* load and ambient forever.

Those are not the same quantity, and the difference is not model error. Within each window the rollout applies a full sine period of ripple:

```python
Ta_s = Ta_w + 2.0 * sin(2 pi tau / T)
K_s  = K_w  + 0.05 * sin(2 pi tau / T)
```
Top-oil follows that ripple through a first-order lag with `tau_oil = 150` min against a window of `T = 720` min. A lagged response to a sinusoid is phase-shifted, so at the instant the forcing returns to its mean — `tau = T`, where `sin(2 pi) = 0` — the *response* has not. It is still on its way back up, from below.

That predicts a bias that is negative, one-signed, present at every window, and **present for a model with no error at all**.

## 3. What the lag predicts, in closed form

For `dtheta/dt = (theta_ss(t) - theta)/tau_oil` driven by a sinusoid of angular frequency `omega`, the periodic response has gain `1/sqrt(1 + (omega tau)^2)` and lag `atan(omega tau)`. At the window end the forcing is at its mean and the response is `-gain * sin(lag)` times the steady-state amplitude.

- `omega tau_oil = 2 pi * 150 / 720 = 1.3090`
- gain = 0.6071, lag = 0.9184 rad = 52.6 deg
- end-of-window factor = `-gain * sin(lag)` = -0.4824

The two ripples' steady-state amplitudes at K = 1:

| source | steady-state amplitude | contribution at window end |
|---|---|---|
| ambient ripple, ±2 degC | 2.000 degC | -0.965 degC |
| load ripple, ±0.05 K | 3.911 degC (`dtheta_ss/dK = 78.2` degC per unit K) | -1.887 degC |
| **total** | | **-2.852 degC** |

That linearisation is taken at K = 1 with the two ripples superposed. Dropping the linearisation — taking the steady-state amplitude as the half range of `true_fixed_point_np` over the ripple itself, which keeps the ETC correction and the K-dependence — gives a prediction per scenario:

| K_base | steady-state amplitude degC | predicted end-of-window bias |
|---|---|---|
| 0.85 | 6.428 | -3.101 degC |
| 0.90 | 6.845 | -3.302 degC |
| 0.95 | 7.308 | -3.525 degC |
| 1.00 | 7.825 | -3.775 degC |
| 1.05 | 8.407 | -4.055 degC |
| 1.10 | 9.065 | -4.373 degC |

Two signatures to check against §4, both of which the manuscript's ETC-staircase story does not have:

* the bias is **one-signed and negative** at every window;

* it **grows monotonically with K_base**, because `dtheta_ss/dK` does. A staircase at K = 1 would put a feature *at* K = 1, not a smooth trend through it.

## 4. The decisive test: run the rollout against ground truth

`ExactModel` integrates `fast_rhs_np` with RK45 at `rtol = 1e-10` and exposes `CODOperator`'s call signature, so `chi_lifetime_rollout` runs against the reference physics itself. Its model error is zero by construction. Any bias it reports belongs to the diagnostic.

| K_base | predicted (§3) | measured mean | median | sd | min | max | windows |
|---|---|---|---|---|---|---|---|
| 0.85 | -3.101 | **-2.863** | -2.865 | 0.052 | -2.937 | -2.547 | 90 |
| 0.90 | -3.302 | **-3.028** | -3.030 | 0.057 | -3.109 | -2.684 | 90 |
| 0.95 | -3.525 | **-3.209** | -3.211 | 0.062 | -3.297 | -2.831 | 90 |
| 1.00 | -3.775 | **-3.409** | -3.411 | 0.069 | -3.505 | -2.993 | 90 |
| 1.05 | -4.055 | **-3.630** | -3.633 | 0.076 | -3.736 | -3.169 | 90 |
| 1.10 | -4.373 | **-3.876** | -3.879 | 0.084 | -3.995 | -3.364 | 90 |

Pooled over all six scenarios: mean **-3.336** degC, sd 0.352, 100.0% of windows negative.

**Both predicted signatures appear.** The bias is negative in every one of the 540 windows, and it grows monotonically from -2.86 degC at K_base = 0.85 to -3.88 at 1.10, with no feature at K = 1.

The prediction overshoots by a consistent 10% (ratio 1.083 to 1.128 across the sweep), and the reason is specific rather than hand-waved: the prediction takes the driving amplitude as **half the peak-to-peak range** of `theta_ss` over the ripple. `theta_ss` is nonlinear in K, so its response to a sinusoidal K is not itself a sinusoid, and half its range exceeds the amplitude of its first harmonic — which is the only component the single-pole gain-and-phase formula applies to. Overshooting is therefore the expected direction. What the prediction gets right is the sign, the shape, and the slope in K_base, which is what identifies the mechanism.

Horizon is 90 windows (45 days) per scenario rather than to end of life. The bias is stationary — the within-scenario sd is 2% of the mean and is the seasonal ambient drift, not spread — so a longer run changes nothing and costs 30x more RK45 solves.

**A model with zero error reports a bias of -3.34 degC, spanning -2.86 to -3.88 across the sweep.** The audit's measured |bias|_mean of 3.09 degC sits inside that range. It is therefore not a model defect and not a physics defect. It is the metric subtracting an unlagged reference from a lagged prediction.

## 5. Separating the phase lag from everything else

If the diagnosis in §2 is right, the bias should vanish against a reference that already contains the ripple's phase lag. `cyclic_endpoint` supplies one: it repeats the window's own forcing until the endpoint stops moving, so it is where top-oil is at the end of a window for a unit already cycling on that window. Nothing about the model enters it.

Same trajectories as §4, three references:

| K_base | vs `steady_state(K_w, Ta_w)` | vs the cyclic endpoint | vs its own endpoint |
|---|---|---|---|
| 0.85 | -2.922 degC | -0.002 degC | 0.000 degC |
| 0.90 | -3.092 degC | -0.002 degC | 0.000 degC |
| 0.95 | -3.279 degC | -0.002 degC | 0.000 degC |
| 1.00 | -3.485 degC | -0.002 degC | 0.000 degC |
| 1.05 | -3.714 degC | -0.003 degC | 0.000 degC |
| 1.10 | -3.970 degC | -0.003 degC | 0.000 degC |

Mean over the six scenarios: **-3.410** degC against the steady state of the mean forcing, **-0.002** degC against the cyclic endpoint of the same forcing. The third column is identically zero because `ExactModel` *is* the true trajectory; it is tabulated so the substitution is explicit.

The ripple's phase lag accounts for 99.94% of the reported bias. The residual is 0.002 degC — three orders of magnitude below the effect and still one-signed, which is what it should be: it is the seasonal ambient drift, `Ta_w` moving between windows so that a unit is never quite at the cyclic state of the window it is currently in. That residual is a real, and negligible, physical lag; the 3.4 degC is not.

Concretely: `RolloutResult.theta_bias` should be scored against a reference integration of `fast_rhs_np` over the same window from the same initial condition. `theta_ss_ref` is worth keeping as its own field — how far the operating point sits from equilibrium is a real diagnostic — but it is not the thing to subtract a prediction from.

## 6. The ageing consequence, which is smaller than feared

O-9 records the concern as: at 10.8 %/K Arrhenius sensitivity a systematic -3 degC understates the ageing rate by roughly 30%, so no end-of-life number can be published until it is understood. The sensitivity arithmetic is right and the conclusion needs revising, because **the -3 degC never entered the DP calculation**.

The DP update reads `theta_for_dp` and `theta_for_dp` is never `theta_ss_ref`:

```python
if dp_source == "reference":
    theta_for_dp = np.full(n_eval, float(steady_state(K_w, Ta_w)))
else:
    theta_for_dp = xp[:, 0].cpu().numpy()      # the predicted trajectory
```
Under `dp_source="model"`, which is the default and the only setting that reflects model quality, DP is advanced from the model's own top-oil trajectory over the whole window — 20 quadrature points, not the endpoint, and not the steady-state reference. The bias field is reported alongside it and consumed by nothing.

Verified: the Arrhenius sensitivity itself, from the code's constants.

| theta_HS | dV/V per +1 degC |
|---|---|
| 80 degC | 12.03% |
| 90 degC | 11.37% |
| 100 degC | 10.77% |
| 110 degC | 10.22% |
| 120 degC | 9.70% |

So 10.8 %/K is the value at about 100 degC, and a *real* systematic -3 degC would indeed cost about 30% of the ageing rate. That remains true and remains the reason to care. What changed is that no such offset has been demonstrated: the 3.09 degC that motivated the concern is an artifact of a diagnostic that does not feed the DP path.

**This does not clear the rollout to publish end-of-life numbers.** It removes one specific reason to distrust them and replaces it with an honest gap: the model's true thermal error over a rollout has not been measured, because the field that was supposed to measure it was measuring something else. §7 says what to run.

## 7. What this leaves open

1. **The real rollout error is still unmeasured.** Fixing `theta_bias` to score against a reference integration gives the number O-9 was actually after. It cannot be produced yet: fix 6 (DECISIONS N-1) invalidated the checkpoint again, so there is no trained model to roll out. This is a post-retrain task and it is the one that decides whether an EOL number is publishable.

2. **Error accumulation across windows is a separate question.** Each window starts from the model's own previous endpoint, so per-window error compounds. That is O-7's subject and this diagnosis says nothing about it either way.

3. **The manuscript's ETC-staircase explanation stays refuted**, and now for a second reason. It was already false at K = 1 (the two formulas coincide there and the Rf clamp is inactive). It is also explaining an effect that has no physical existence.

4. **The fix to `rollout.py` is not applied here.** O-9 asked for a diagnosis, and changing the metric would change reported numbers in the same commit that explains why they were wrong. Separate change, separate before/after.

