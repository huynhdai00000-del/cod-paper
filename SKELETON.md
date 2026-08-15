# Engineering manuscript skeleton

Updated 2026-08-15. Scope: power-transformer thermal, gas, DP, and CHI trajectories. The empirical basis is the frozen 17-cell matrix with seven seeds per cell.

## Working title

**Cascaded Operator Decomposition for Stable Long-Horizon Transformer State Propagation: A Multi-Backbone Engineering Study**

## Central position

Existing operator-learning studies include coupled, sequential, hierarchical, and decomposed architectures. The open engineering issue is therefore not whether a system can be split. It is whether the one-way thermal-to-chemical topology can be used as a repeatable deployment interface that:

1. remains compatible with different neural backbones;
2. separates short-window thermal approximation from long-horizon chemical state propagation;
3. preserves cyclic thermal information needed by nonlinear ageing kinetics; and
4. produces interpretable DP, gas, and CHI trajectories under repeated rollout.

To address this issue, COD is presented as a modular learned-deterministic topology. A neural operator predicts the upstream thermal trajectory, optionally as a correction to the IEC thermal baseline, while gas and DP states are propagated by their governing updates outside the neural graph. A controlled 2 by 2 matrix separates the effects of cascade topology and analytic baseline across FNO, MIONet, PI-DeepONet, and S-DeepONet.

## Contribution set

1. A backbone-compatible COD interface for transformer thermal-to-chemical propagation.
2. A controlled multi-backbone study that separates cascade and analytic-baseline effects under the same distribution, stopping rule, seven seeds, and one-year endpoint.
3. Evidence that the cascade primarily controls long-horizon gas-state propagation, while the analytic thermal baseline primarily controls cycle-shape attenuation.
4. Two complementary reference implementations: PI-COD for the lowest absolute error in the tested matrix and FNO-COD for the cleanest backbone-transfer comparison.
5. A derived Cobb-Douglas CHI trajectory with a corrected conditional monotonicity theorem.
6. A negative engineering result showing that direct bounded correction is not a reliable refinement under the frozen protocol.

## Paper structure

### 1. Introduction

Start with the operational requirement: loading decisions need transformer thermal and degradation trajectories over many consecutive windows, not only accurate isolated 12-hour predictions.

Establish the gap in three steps:

1. Short-window loss does not test whether recurrent state rollout remains stable.
2. Monolithic neural state updates can turn a small thermal approximation error into persistent gas error through nonlinear temperature-dependent kinetics.
3. Existing decomposition ideas do not by themselves establish which engineering component is responsible for thermal fidelity, state propagation, or cross-backbone repeatability.

Then state the response: COD fixes the thermal trajectory as the learned interface and retains deterministic downstream propagation. Introduce the 2 by 2 study and the CHI output without reporting numerical results in this section.

### 2. Transformer system and engineering task

#### 2.1 State topology

- Upstream state: top-oil or hot-spot thermal trajectory.
- Downstream states: five dissolved gases and DP.
- One-way dependence: thermal history drives gas and DP evolution; downstream states do not enter the thermal equation in the studied mineral-oil model.

#### 2.2 Prediction task

- Input: load and ambient histories with the current transformer state.
- Learned output: upstream thermal trajectory over one window.
- Deterministic output: gas and DP updates over the same window.
- Repeated deployment: the terminal state initializes the next window.

#### 2.3 Primary endpoint

Use absolute gas concentration error in ppm at the end of a one-year free rollout for load factors 0.95 and 1.10. Keep 12-hour MAE as a secondary diagnostic.

### 3. COD engineering topology

#### 3.1 Learned thermal interface

Define the neural thermal operator independently of backbone. For the baseline-equipped form,

\[
\widehat{\theta}(t)=\theta_{\mathrm{IEC}}(t)+\Delta\theta_{\eta}(t).
\]

The same interface accepts FNO, MIONet, PI-DeepONet, or S-DeepONet as \(\Delta\theta_{\eta}\).

#### 3.2 Deterministic downstream propagation

Gas and DP trajectories are computed from the predicted thermal history through the governing quadrature or integration rule. They are not direct neural outputs.

#### 3.3 Two independent design factors

- **Cascade factor:** deterministic downstream propagation versus neural monolithic state updates.
- **Baseline factor:** IEC thermal baseline plus learned correction versus learning the full thermal signal.

This distinction is central to the paper. The cascade and baseline are not treated as one inseparable architecture choice.

### 4. Properties retained from the mathematical formulation

Keep the theorem statements concise in the main text and move proofs to an appendix.

#### 4.1 Cascaded error propagation

State a Lipschitz downstream bound of the form

\[
\|\widehat{x}_{\mathrm{down}}-x_{\mathrm{down}}\|
\le L_{\mathrm{down}}\|\widehat{\theta}-\theta\|.
\]

Present this as a conditional stability statement for the specified operating set, not as a universal guarantee of small error.

#### 4.2 Structural state properties

Retain exact window initialization, DP direction under the stated ageing law, and gas sign or equilibrium conditions only with their explicit assumptions. Do not use these properties to imply numerical accuracy.

#### 4.3 Conditional monotonicity of the adaptive CHI

Let \(x=\chi_{\mathrm{DP}}\), \(y=\chi_{\mathrm{gas}}\), and

\[
\mathrm{CHI}=x^{w(x)}y^{1-w(x)}, \qquad w(x)=1-\frac{x}{2}.
\]

Then

\[
\frac{d\log \mathrm{CHI}}{dt}
=w\frac{\dot{x}}{x}+(1-w)\frac{\dot{y}}{y}
+\dot{w}(\log x-\log y).
\]

A sufficient condition for non-increasing CHI is

\[
\dot{x}\le0,\qquad \dot{y}\le0,\qquad x\le y.
\]

Apply the theorem to seasonal-smoothed sub-indices rather than claiming that smoothing the nonlinear CHI automatically preserves the derivative argument. CHI remains a derived decision-support trajectory; no CHI reference accuracy is claimed.

### 5. Experimental design

#### 5.1 Frozen matrix

- Backbones: FNO, MIONet, PI-DeepONet, S-DeepONet.
- Factors: cascade versus monolithic; analytic baseline versus no baseline.
- Additional design test: bounded-correction PI-COD.
- Seven seeds per declared cell on frozen distribution `fc4cb76c3b32ec17`.

#### 5.2 Reporting rules

- Report convergence rate before accuracy.
- Compute accuracy summaries on converged checkpoints only.
- Report median and full seed range.
- Apply the prespecified two-fold thermal comparability control before attributing a gas difference to cascade topology.
- Use absolute ppm error for gas endpoints.

#### 5.3 Secondary mechanism diagnostics

- Thermal swing ratio on the held-out 100-case set.
- Shape-fidelity gate across tracked swing bands.
- Jensen-gap preservation for gas and DP kinetics.

### 6. Results

#### 6.1 Matrix integrity and convergence

Report 119 production runs, 113 converged checkpoints, complete one-year and swing scoring for every converged checkpoint, and the 1/7 convergence outcome of bounded correction.

#### 6.2 Primary result: long-horizon propagation

Seven of eight matched comparisons pass the thermal control. Across these comparisons, the cascade lowers median one-year gas error in 70/70 gas-by-load endpoints; 57/70 have non-overlapping full seed ranges. Treat PI-DeepONet without the baseline as confounded.

Use `fig2_cascade_rollout_advantage` and the matched-comparison table.

#### 6.3 FNO-COD as the transfer anchor

Show that FNO-COD and matched monolithic FNO have comparable thermal accuracy, while FNO-COD has lower one-year gas error in all ten endpoints with complete seed-range separation.

Use `fig3_fno_rollout_errors`.

#### 6.4 PI-COD as the accuracy anchor

Among baseline-equipped cascade implementations, PI-COD has the lowest median error in all ten gas-by-load endpoints. Do not generalize this ranking beyond the tested matrix.

Use `fig4_cod_absolute_accuracy`.

#### 6.5 Mechanism: baseline and cycle shape

Across in-cascade backbones, baseline-equipped cells pass the shape gate in 28/28 seeds, compared with 8/28 without the baseline. Relate swing attenuation to the nonlinear downstream Jensen gap without making it a substitute for the one-year endpoint.

Use `fig5_baseline_shape_mechanism` and the Jensen-gap table.

#### 6.6 Negative result: bounded correction

Report convergence first. The single converged checkpoint has much larger thermal error than standard PI-COD, so the bounded formulation is a failed design test under this protocol.

Use `fig6_bounded_correction_failure`.

#### 6.7 CHI trajectories

Show representative CHI, DP, and gas trajectories as derived model outputs. Mark intervals where the sufficient monotonicity conditions hold. Do not rank architectures by CHI accuracy because no CHI reference labels exist.

### 7. Discussion

#### 7.1 What the matrix identifies

- Cascade topology controls the propagation pathway.
- The analytic thermal baseline controls cyclic thermal shape for backbones prone to smoothing.
- Absolute accuracy and evidence of backbone transfer are different claims.

#### 7.2 Relation to prior decomposed learning

Position COD by its transformer-specific learned-deterministic interface, deterministic gas and DP rollout, controlled factor separation, and engineering endpoint. Avoid priority language.

#### 7.3 Deployment meaning of CHI

CHI compresses retained state semantics for decision support. Its credibility comes from the definitions of its sub-indices, the stated monotonicity conditions, and transparent trajectories rather than a supervised CHI target.

### 8. Limitations

- Validation uses the frozen simulated distribution and a one-year forcing design.
- Field DGA supports gas-trend plausibility but does not validate full CHI accuracy.
- The present conclusions apply to the one-way mineral-oil transformer model and the four tested backbones.
- The bounded-correction result is one formulation, not a general impossibility result for constrained learning.

### 9. Conclusion

Conclude with the engineering result: replacing neural downstream state updates by a fixed thermal interface and deterministic propagation improves long-horizon reliability across several backbones. Distinguish PI-COD's absolute accuracy from FNO-COD's transfer evidence, and retain CHI as a conditional derived output.

## Main figures and tables

| item | role |
|---|---|
| Figure 1 | COD learned-deterministic engineering topology |
| Figure 2 | One-year cascade advantage across matched backbones |
| Figure 3 | FNO-COD versus matched monolithic FNO |
| Figure 4 | Absolute accuracy across baseline-equipped COD backbones |
| Figure 5 | Analytic baseline and cycle-shape mechanism |
| Figure 6 | Bounded-correction negative result |
| Table 1 | Matched comparisons, thermal control, endpoint counts, and reduction factors |
| Table 2 | Baseline-equipped cascade accuracy and swing summaries |

## Appendix allocation

- Full theorem proofs and assumptions.
- Network and training details.
- Complete per-seed convergence and error tables.
- Full swing-band and Jensen-gap tables.
- CHI axioms and aggregation derivation.
