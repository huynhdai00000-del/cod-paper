# Engineering findings from the completed matrix

## Scope and integrity

The final matrix contains 119 production runs on frozen distribution
`fc4cb76c3b32ec17`. Every declared cell has seven seeds. Of the 119 runs, 113
converged and were scored with both the one-year rollout endpoint and the common
100-case swing test. Both aggregators report zero integrity problems.

Six of seven bounded-correction runs did not converge. Their predictions are
retained in the per-seed record but do not enter accuracy medians.

## Primary engineering result

The primary endpoint is absolute gas concentration error at the end of a
one-year free rollout. It is evaluated at load factors 0.95 and 1.10 within each
backbone and baseline level, subject to the prespecified thermal comparability
control.

Seven of eight matched cascade-versus-monolithic comparisons pass the thermal
control. Across their 70 gas-by-load endpoints, the cascade has lower median
error in all 70. Fifty-seven also have non-overlapping full seed ranges; the
remaining 13 are directionally consistent but have overlapping seed ranges.
The PI-DeepONet comparison without the analytic baseline is excluded from this
count because its median thermal MAEs differ by 2.58-fold.

This result identifies a long-horizon propagation effect. Several monolithic
cells achieve acceptable short-window thermal error yet develop large gas-state
offsets under repeated rollout. The in-cascade cells retain deterministic gas
and DP updates and avoid learning those state increments as unconstrained neural
outputs.

## PI-COD accuracy anchor

PI-COD is the most accurate baseline-equipped cascade implementation in the
completed matrix. It has the lowest median absolute gas error in all ten
gas-by-load endpoints. Its median thermal MAE is 0.405 degC with a full seed
range of 0.306 to 0.476 degC, and all seven seeds pass the shape-fidelity gate.

Its median one-year gas errors are:

| load | H2 | C2H2 | C2H4 | CO | CO2 |
|---|---:|---:|---:|---:|---:|
| 0.95 | 0.06199 | 0.001599 | 0.02200 | 0.07723 | 0.1233 |
| 1.10 | 1.108 | 0.1892 | 0.6075 | 0.8025 | 0.9926 |

All values are absolute error in ppm.

## FNO-COD transfer anchor

FNO-COD provides the cleanest evidence that the cascade result is not specific
to PI-DeepONet. It converges in seven of seven seeds, passes the shape-fidelity
gate in seven of seven seeds, and has lower one-year error than matched
monolithic FNO in all ten gas-by-load endpoints. All ten comparisons also have
non-overlapping full seed ranges.

FNO-COD is therefore the transfer and replication anchor. It is not the
absolute-accuracy leader; PI-COD has lower median error in all ten corresponding
endpoints.

## Analytic baseline and cyclic shape

The analytic thermal baseline and the cascade address different parts of the
engineering problem. Across the four in-cascade backbones, baseline-equipped
cells pass the shape-fidelity gate in 28 of 28 seeds, compared with 8 of 28
without the baseline.

FNO gives the clearest mechanism contrast. Its baseline-equipped in-cascade
cell has median swing ratio 1.0356 and passes in seven of seven seeds. Without
the baseline, the corresponding cell has median swing ratio 0.6886 and passes
in zero of seven seeds. Its C2H2 Jensen-gap ratio changes from 0.7108 without
the baseline to 1.0232 with the baseline.

These diagnostics explain cycle-shape attenuation but do not replace the
one-year gas endpoint.

## Negative design result

Directly bounding the learned correction is not a reliable refinement under the
frozen protocol. Only one of seven runs converges. That checkpoint has thermal
MAE 14.409 degC, compared with a PI-COD median of 0.405 degC, and it also has
large downstream gas error. The result is reported by convergence rate before
the single-checkpoint accuracy value.

## CHI role

CHI is retained as a derived decision-support trajectory. It is not an accuracy
target because no CHI reference labels are available. The corrected theorem is
conditional: for smoothed sub-indices `x = chi_DP` and `y = chi_gas`, with
`w(x) = 1 - x/2`, non-increasing `x` and `y` together with `x <= y` are
sufficient for non-increasing adaptive Cobb-Douglas CHI.

Field DGA evidence supports gas-trend plausibility only. It does not validate
the complete CHI trajectory.

## Manuscript position

COD is presented as a learned-deterministic engineering topology rather than a
priority claim about cascade architectures. The evidence base is the controlled
2 by 2 factor separation across four backbones, the one-year gas endpoint, the
swing and Jensen diagnostics, and the bounded-correction failure mode.
