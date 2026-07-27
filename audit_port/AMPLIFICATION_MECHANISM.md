# O-1 — where the monolithic baseline's gas outputs come from

Closes O-1. Seed-999 test set, N=100, every arm scored with the same right-edge guard `TW*0.9999` so they are directly comparable.

**Answer: the monolithic baselines' gas concentrations are direct network outputs. They never touch Arrhenius quadrature.** The amplification chain Section 7.1 describes does not exist in the baseline's forward pass, so Section 7.1 and Table 4 need rewriting.

A second finding fell out of the control arm and is arguably more consequential for the paper's own numbers: **COD's reported C2H2 and C2H4 errors are dominated by a reference/model mismatch affecting 6 and 2 of the 100 test cases**, not by model quality. Section 4 below.

## 1. The two forward passes, side by side

`cod/models/cod.py`, `CODOperator.forward` (L334-L355) — the thermal state is the only network output:

```python
L341  delta_TO = (b_th * tr_th).sum(dim=-1, keepdim=True) + self.bias_th
L345  baseline = self._ode_baseline(x0_TO, u_sensors, t)
L346  theta_TO_pred = baseline + t_exp * self.output_scale * delta_TO
L351  theta_TO_grid = self._thermal_predict_grid(
L352      x0_TO, u_sensors, b_th, n_grid=20 if self.training else None).detach()
L354  gases_pred = self._gas_integral(t, u_sensors, x0_gas, theta_TO_grid)
L355  return torch.cat([theta_TO_pred, gases_pred], dim=-1)
```
`_gas_integral` (L289-L332) holds no `nn.Parameter`. It maps the thermal grid to a hot-spot temperature (L315-L319), forms the Arrhenius factor (L321-L323), applies the partial-discharge factor to C2H2 alone (L324-L326) and integrates (L327-L331). The gases are a deterministic function of `(theta_TO_grid, K, x0_gas)` — the thermal prediction is the parent, the gases are its children.

`cod/models/monolithic.py`, `MonolithicFair.forward` (L140-L150) — all six states leave through one linear head:

```python
L146  b  = self.branch(torch.cat([x0n, u], dim=-1))     # (B, p)
L147  tr = self.trunk(tf)                               # (B, p)
L148  raw_out = self.out_proj(dot) + self.bias          # (B, 6) <- Linear(p, 6)
L149  phi = (1 - torch.exp(-t / self.tau)) / (1 - torch.exp(-self.T / self.tau))
L150  return x0 + phi * self.output_scale * raw_out     # all six states
```
`MonolithicMultiHead.forward` (L200-L210) differs only in giving each state its own `p`-dim basis (L208, `raw_out = (b * tr.unsqueeze(1)).sum(-1) + self.bias`); the six outputs are still siblings from one tensor.

So in the monolithic models `theta_TO_pred` and the five gas predictions are **parallel outputs**, not parent and children. The gas channels never read the model's own thermal prediction. Nothing in `monolithic.py` computes `V_arr`, `k_gen`, `k_dis` or a hot-spot temperature; those names do not appear in the file. The only thermal quantity reaching the gas channels is `x0[:, 0:1]`, the *initial* top-oil temperature, passed to `build_trunk_feats` at L147 as a feature.

## 2. Gradient test — the mechanical confirmation

Back-propagate a gas-only loss, `sum(pred[:, 1:]**2)`, and count parameters receiving nonzero gradient. If the gases are analytic, none can.

| model | params | nonzero grad from gas loss | grad norm | nonzero grad from thermal loss |
|---|---|---|---|---|
| COD | 28 | **0** | 0.000e+00 | 28 |
| Mono Fair | 30 | **30** | 1.814e+08 | 30 |
| Mono Multi-head | 28 | **28** | 1.161e+08 | 28 |

COD: **zero of 28** parameter tensors receive gradient from the gas loss, while the thermal loss trains all 28. That is audit M-1 confirmed from the other direction, and it is what "the gases are analytic" means operationally.

Both monolithic baselines: **every** parameter receives gradient from the gas loss. Their gas outputs are learned. There is no quadrature in the path, so there is nothing for Arrhenius to amplify.

## 3. The decisive test — what a 13.4 degC thermal error actually produces

Take each monolithic baseline's predicted theta_TO and push it through COD's cascade with the true gas ICs. This separates what Arrhenius amplification of that thermal error *would* produce from what the architecture *does* produce.

| arm | theta_TO MAE degC |
|---|---|
| COD | 0.3993 |
| Mono Fair | 13.4141 |
| Mono Multi-head | 12.9283 |

Gas MAE in ppm, mean over 100 cases:

| gas | Ea kJ/mol | COD | Mono Fair actual | Mono Fair -> cascade | Mono MH actual | Mono MH -> cascade | true theta -> cascade |
|---|---|---|---|---|---|---|---|
| `c_H2` | 112.2 | 0.0234 | 0.3060 | 0.2082 | 0.2674 | 0.1521 | 0.0001 |
| `c_C2H2` | 174.6 | 0.5926 | 0.7049 | 0.6292 | 0.7043 | 0.6263 | 0.5914 |
| `c_C2H4` | 137.2 | 0.1646 | 0.4292 | 0.2921 | 0.4318 | 0.2673 | 0.1576 |
| `c_CO` | 87.3 | 0.0060 | 0.1219 | 0.0763 | 0.1968 | 0.0582 | 0.0000 |
| `c_CO2` | 74.8 | 0.0045 | 0.1114 | 0.0699 | 0.2447 | 0.0556 | 0.0000 |

Two conclusions, both against Section 7.1.

**(a) The hybrid is better than the baseline, not worse.** Pushing Mono Fair's 13.41 degC thermal error through the Arrhenius cascade gives *lower* gas error than Mono Fair's own head, on every gas:

| gas | Mono Fair actual | through the cascade | ratio |
|---|---|---|---|
| `c_H2` | 0.3060 | 0.2082 | 1.47x better |
| `c_C2H2` | 0.7049 | 0.6292 | 1.12x better |
| `c_C2H4` | 0.4292 | 0.2921 | 1.47x better |
| `c_CO` | 0.1219 | 0.0763 | 1.60x better |
| `c_CO2` | 0.1114 | 0.0699 | 1.59x better |

If the cascade amplified thermal error catastrophically, the hybrid would be far worse than the baseline. It is uniformly better. The monolithic baseline's gas error is not amplified thermal error — it is its own unconstrained head missing the target.

**(b) The amplification is real but sub-ppm.** O-1 predicted that if amplification were operating, 13.41 degC should give gas errors "of order 20 ppm". Measured through the real cascade the largest is 0.63 ppm. That expectation was high by about two orders of magnitude, because generation rates are tiny: `k_gen` spans 9.5e-8 to 2.8e-5 ppm/min, so a 12 h window generates a few ppm at most even with `V_arr` at its ceiling.

### Where Section 7.1's numbers come from

The notebook diagnostic (n15 cell 5) computes the relative change in `V_arr` for a hot-spot perturbation and reports 1,000-35,000%. That arithmetic is right: `V_arr` is exponential in temperature and a 55 degC hot-spot error does change the *instantaneous generation rate* by that factor. What does not follow is the step to concentration error. The rate is multiplied by `k_gen` and integrated over 720 min, which maps a 35,000% rate error to well under 1 ppm of concentration. Section 7.1 quotes a rate ratio as though it were a concentration error.

## 4. The control arm exposed a reference/model mismatch

`true theta -> cascade` is the cascade's error given a *perfect* thermal input. For C2H2 it is 0.5914 ppm against COD's headline 0.5926 ppm — 99.8% of it. For C2H4, 0.1576 of 0.1646 ppm. H2, CO and CO2 have a floor near zero, so their errors are genuinely thermal-driven.

**First hypothesis, refuted.** `_gas_integral` (L332) returns `x0_gas + F_t - k_dis * x0_gas * t`, linearising dissipation at the initial concentration rather than solving `dc/dt = gen - k_dis c`. Solving the same integrand exactly by integrating factor, on the same grid, moves the C2H2 floor by 2.35e-07 ppm — from 0.591367 to 0.591367, i.e. no measurable effect. Recorded because a plausible mechanism that turns out not to be the cause is worth knowing about: the dissipation term really is linearised, and `k_dis * c(0) * T` is large in absolute terms, but the true dissipation over a 12 h window is close enough to linear that it does not show up.

**Actual cause: the ground truth and the model do not use the same Arrhenius factor.**

```python
# cod/data/physics.py  fast_rhs_np  (the ground truth)
T_HS_K = np.clip(theta_HS + 273.15, 313.15, 573.15)
V_arr  = np.exp(B_aging * E_act * (1/T_ref - 1/T_HS_K))          # UNBOUNDED

# cod/data/physics.py  fast_rhs_torch,  and  CODOperator._gas_integral L321
V_arr  = torch.exp(...).clamp(max=1e4)                           # CLAMPED
```
Removing the clamp collapses the floor:

| gas | as shipped | no `V_arr` clamp | reduction | grid points clamped | cases affected |
|---|---|---|---|---|---|
| `c_H2` | 0.000021 | 0.000021 | 1x | 0.00% | 0/100 |
| `c_C2H2` | 0.591367 | 0.000040 | 14,632x | 3.63% | 6/100 |
| `c_C2H4` | 0.157578 | 0.000025 | 6,330x | 1.51% | 2/100 |
| `c_CO` | 0.000008 | 0.000008 | 1x | 0.00% | 0/100 |
| `c_CO2` | 0.000008 | 0.000008 | 1x | 0.00% | 0/100 |

The clamp binds where `V_arr` exceeds 1e4, which for each gas is a temperature threshold set by its activation energy:

| gas | theta_HS at which V_arr = 1e4 |
|---|---|
| `c_H2` | 245.3 degC |
| `c_C2H2` | **187.2 degC** |
| `c_C2H4` | 214.0 degC |
| `c_CO` | 303.6 degC |
| `c_CO2` | 356.7 degC |

The test set reaches a maximum hot-spot temperature of 236.9 degC (mean 123.0, p90 177.2), so C2H2's threshold is crossed and CO's is not. This is a direct consequence of audit M-9: initial conditions run to theta_TO(0) = 150 degC and the hot-spot sits tens of degrees above the top-oil temperature.

The error is concentrated, not spread:

- 6 of 100 cases have the clamp binding for C2H2. On those the C2H2 MAE averages 9.86 ppm; on the other 94 it averages 0.0000 ppm.
- So COD's headline C2H2 figure of 0.5926 ppm is 6 artefact cases diluted across 100, not a property of the surrogate.

Audit section 8.4 lists `V_arr.clamp(max=1e4)` among the clamps that can hide behaviour, but does not note that the reference trajectory lacks the same clamp, and does not quantify it. This is new.

### The comparison restricted to the clean cases

Excluding every case where any clamp bound leaves 94 of 100. On those, the gas errors read very differently:

| gas | COD (all 100) | COD (clean only) | Mono Fair (clean only) | Mono Fair -> cascade (clean only) |
|---|---|---|---|---|
| `c_H2` | 0.0234 | 0.0010 | 0.1234 | 0.0848 |
| `c_C2H2` | 0.5926 | 0.0007 | 0.1009 | 0.0255 |
| `c_C2H4` | 0.1646 | 0.0009 | 0.1713 | 0.0833 |
| `c_CO` | 0.0060 | 0.0005 | 0.0665 | 0.0416 |
| `c_CO2` | 0.0045 | 0.0006 | 0.0759 | 0.0437 |

On the clean subset COD's C2H2 error is 0.0007 ppm against Mono Fair's 0.1009 ppm — a factor of 141, where the all-100 comparison showed only 1.19. The clamp artefact was masking the real gap, in COD's disfavour.

Not fixed here. Aligning the clamps changes the model's forward pass and the reference ODE, so it belongs in its own commit with its own before/after, and it invalidates the checkpoint again. Recorded as a candidate.

## 5. Error distributions, since the means hide the shape

Partly addresses O-2. theta_TO in degC, gases in ppm.

| arm / state | mean | median | p90 | max | argmax case |
|---|---|---|---|---|---|
| COD / `theta_TO` | 0.3993 | 0.3326 | 0.7306 | 2.3991 | 24 (CK) |
| COD / `c_H2` | 0.0234 | 0.0000 | 0.0041 | 1.3881 | 33 (CK) |
| COD / `c_C2H2` | 0.5926 | 0.0000 | 0.0032 | 50.9473 | 33 (CK) |
| COD / `c_C2H4` | 0.1646 | 0.0000 | 0.0036 | 14.7139 | 33 (CK) |
| COD / `c_CO` | 0.0060 | 0.0001 | 0.0022 | 0.2986 | 33 (CK) |
| COD / `c_CO2` | 0.0045 | 0.0001 | 0.0027 | 0.1984 | 33 (CK) |
| Mono Fair / `theta_TO` | 13.4141 | 4.3146 | 39.7990 | 55.1267 | 79 (TV) |
| Mono Fair / `c_H2` | 0.3060 | 0.0612 | 0.3967 | 11.7005 | 33 (CK) |
| Mono Fair / `c_C2H2` | 0.7049 | 0.0954 | 0.1685 | 51.0252 | 33 (CK) |
| Mono Fair / `c_C2H4` | 0.4292 | 0.1796 | 0.3011 | 20.2743 | 33 (CK) |
| Mono Fair / `c_CO` | 0.1219 | 0.0547 | 0.1757 | 2.8825 | 33 (CK) |
| Mono Fair / `c_CO2` | 0.1114 | 0.0592 | 0.1854 | 2.0931 | 33 (CK) |
| Mono Multi-head / `theta_TO` | 12.9283 | 4.9026 | 37.7936 | 54.8563 | 32 (CK) |
| Mono Multi-head / `c_H2` | 0.2674 | 0.0578 | 0.3760 | 11.3871 | 33 (CK) |
| Mono Multi-head / `c_C2H2` | 0.7043 | 0.0677 | 0.2413 | 51.1950 | 33 (CK) |
| Mono Multi-head / `c_C2H4` | 0.4318 | 0.1365 | 0.3825 | 20.5681 | 33 (CK) |
| Mono Multi-head / `c_CO` | 0.1968 | 0.1173 | 0.3348 | 2.8439 | 33 (CK) |
| Mono Multi-head / `c_CO2` | 0.2447 | 0.1807 | 0.3147 | 2.7179 | 8 (CK) |
| Mono Fair -> cascade / `c_H2` | 0.2082 | 0.0006 | 0.4233 | 7.5272 | 33 (CK) |
| Mono Fair -> cascade / `c_C2H2` | 0.6292 | 0.0000 | 0.1756 | 50.9473 | 33 (CK) |
| Mono Fair -> cascade / `c_C2H4` | 0.2921 | 0.0002 | 0.3740 | 14.7139 | 33 (CK) |
| Mono Fair -> cascade / `c_CO` | 0.0763 | 0.0009 | 0.2108 | 1.7533 | 33 (CK) |
| Mono Fair -> cascade / `c_CO2` | 0.0699 | 0.0016 | 0.2219 | 1.1212 | 33 (CK) |
| true theta -> cascade / `c_H2` | 0.0001 | 0.0000 | 0.0002 | 0.0015 | 76 (TV) |
| true theta -> cascade / `c_C2H2` | 0.5914 | 0.0000 | 0.0001 | 50.9473 | 33 (CK) |
| true theta -> cascade / `c_C2H4` | 0.1576 | 0.0000 | 0.0002 | 14.7139 | 33 (CK) |
| true theta -> cascade / `c_CO` | 0.0000 | 0.0000 | 0.0001 | 0.0006 | 33 (CK) |
| true theta -> cascade / `c_CO2` | 0.0000 | 0.0000 | 0.0001 | 0.0006 | 33 (CK) |

The median/mean spread is the story: COD's C2H2 median is 0.000002 ppm against a mean of 0.5926 ppm, four orders of magnitude apart. Reporting the mean alone for this state is misleading in either direction.

## 6. Realised hot-spot swing, handed to O-8

Half peak-to-peak of the true hot-spot trajectory per case. O-8 needs it because the Jensen gap depends on amplitude.

| subset | mean | median | p90 | max |
|---|---|---|---|---|
| all 100 | 21.71 | 21.44 | 38.52 | 52.98 |
| constant K | 14.48 | 11.26 | 31.58 | 43.62 |
| time-varying | 28.95 | 28.81 | 40.59 | 52.98 |

Note for O-8: the constant-K cases still swing, because theta_TO relaxes from its initial condition toward the steady state over the window. The swing is a transient, not a sinusoid, so C-10's sinusoidal table is not directly comparable for those cases.

## 7. What to change in the manuscript

1. **Section 7.1 and Table 4.** Drop the Arrhenius-amplification explanation of the monolithic baseline's gas error: those gases never enter the Arrhenius path. Replace with the measured statement — the monolithic failure is a thermal failure (13.41 degC against 0.399 degC), and the gas channels fail separately because they are unconstrained network outputs.
2. **Never quote a rate ratio as a concentration error.** The 1,000-35,000% figures are relative `V_arr` changes. In ppm the whole effect is under 1 ppm, i.e. under 3% of the tightest IEC 60599 attention level (35 ppm for C2H2).
3. **Fix the clamp mismatch before quoting any C2H2 or C2H4 number.** COD's 0.593 ppm C2H2 is 6 artefact cases out of 100; on the 94 clean cases it is 0.0007 ppm. This one cuts in the paper's favour, which is not a reason to leave it unstated.
4. **The multi-head result reinforces the point.** Mono Multi-head has slightly better theta_TO (12.93 vs 13.41 degC) and comparable gas error, so the bottleneck was never the cause either.

No CLOSED item is reopened. This supports C-10 by removing the competing explanation for the monolithic result, and supports C-9 by showing again that percentage figures on these gases are normalisation artefacts.

