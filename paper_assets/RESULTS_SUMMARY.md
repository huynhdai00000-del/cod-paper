# Final engineering evidence summary

## Dataset

- 119 production runs on frozen distribution `fc4cb76c3b32ec17`.
- 113 converged checkpoints were scored for one-year rollout and swing fidelity.
- Every declared cell contains seven seeds. Six bounded-correction seeds did not converge; no failed seed enters an accuracy median.

## Claims supported by the completed matrix

1. **COD is a backbone-compatible deployment topology for long-horizon state propagation.** Seven of eight matched cascade-versus-monolithic comparisons pass the thermal comparability control. Across those seven comparisons, the cascade has lower median one-year gas error in 70/70 endpoints; 57/70 also have non-overlapping full seed ranges.
2. **PI-COD gives the best absolute accuracy among the baseline-equipped cascade implementations tested here.** It has the lowest median error in all ten gas-by-load endpoints, with median thermal MAE 0.405 degC and a 7/7 swing-gate pass rate.
3. **FNO-COD provides the clearest backbone-transfer result.** It converges in 7/7 seeds, passes the swing gate in 7/7 seeds, and beats matched monolithic FNO in all ten one-year endpoints with full seed-range separation in all ten.
4. **The analytic thermal baseline and the cascade solve different engineering problems.** Across the four in-cascade backbones, baseline-equipped cells pass the cycle-shape gate in 28/28 seeds, compared with 8/28 without the baseline. The cascade then prevents thermal errors from being learned again as unconstrained gas-state increments.
5. **Directly bounding the learned correction is not a reliable refinement under this protocol.** Only 1/7 bounded-correction seeds converges, and that checkpoint has 14.409 degC thermal MAE versus 0.405 degC for standard PI-COD.

## Claim boundaries

- Do not call COD the first cascade architecture.
- Do not call FNO-COD the most accurate model; PI-COD is more accurate in this matrix. FNO-COD is the strongest transfer and replication example.
- Treat the PI-DeepONet no-baseline cascade-versus-monolithic comparison as confounded because its median thermal MAEs differ by more than the prespecified factor of two.
- Treat CHI as a derived decision-support trajectory. Its role is supported by the retained state semantics and the conditional monotonicity result, not by unavailable CHI reference labels.
