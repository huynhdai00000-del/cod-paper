# Engineering findings from the C-11 matrix

## Scope and provenance

The release contains 117 fresh production runs on frozen distribution
`fc4cb76c3b32ec17`, all from commit `a657d4c7`. Every run has `run.json`,
`model.pt`, and `predictions.npz`. The matrix contains 111 converged checkpoints.
Six of seven bounded-correction COD runs did not converge, so that cell is
reported by failure rate before accuracy.

The primary endpoint was computed for all 111 converged checkpoints at load
factors 0.95 and 1.10 over 730 consecutive 12 h windows. This is a one-year
rollout, consistent with `evaluation.rollout_years: 1` in the frozen configs and
the command in `LAUNCH_LIST.md`. The phrase "2-year rollout" in ANALYSIS_PLAN
Amendment 1 is inconsistent with those executable specifications. It should be
recorded as a post-run documentation discrepancy rather than silently edited.

Two predeclared reusable runs are outside the release: `o5` (`cod`, seed 1) and
`o12` (`cod_no_baseline`, seed 1). The current reports therefore contain six
seeds for those two cells. Restoring the two checkpoint folders completes the
planned seven seeds without new training.

## Primary engineering result

The primary comparison is absolute gas concentration error at the end of the
rollout. It is evaluated within architecture and baseline level, subject to the
predeclared thermal-confound control and the two reporting bars.

Seven of the eight architecture-by-baseline comparisons pass the thermal control.
Across their 70 scenario-by-gas rows, the in-cascade form has lower median error
in every row. Fifty-seven rows pass both the operational floor and seed-separation
bar; the remaining 13 are operationally large but have overlapping seed ranges.
The PI-DeepONet no-baseline comparison is excluded from H1 because its median
thermal MAE differs by 2.63-fold between the two forms.

The effect is not a small improvement to 12 h prediction. Monolithic models can
fit the short-window objective and still develop persistent gas offsets or
unstable free-running trajectories. In-cascade models compute gas evolution from
the governing quadrature and usually keep one-year errors at the scale of a few
ppm. This supports an engineering claim about long-horizon state propagation and
operational stability, not a claim that COD is the first cascade architecture.

For PI-DeepONet with the IEC baseline, standard COD has median rollout errors of
0.0432, 0.00123, 0.0154, 0.0539, and 0.0877 ppm at load factor 0.95 for H2, C2H2,
C2H4, CO, and CO2, respectively. At load factor 1.10, the corresponding medians
are 0.961, 0.204, 0.618, 0.678, and 0.827 ppm. The matched monolithic medians are
larger in all ten comparisons, and every row passes both reporting bars.

## Mechanism result: cycle shape and Jensen gap

All 111 converged checkpoints were scored on the same 100-case held-out set. The
gate tests whether the median predicted-to-reference swing ratio falls below 0.95
in any band whose median thermal MAE is at most 5 degC.

- Standard COD passes all five tracked bands in all six available seeds. Its
  cell median swing ratio is 1.0103, with a seed range from 1.0055 to 1.0176.
- COD without the analytic baseline passes in one of six seeds and fails in five.
  In the 25-200 degC swing band, five of six seeds fail and the cell median ratio
  is 0.8895.
- Both FNO forms with the analytic baseline pass in all seven seeds. Both forms
  without it fail in all seven seeds.
- Both MIONet forms with the analytic baseline pass in all seven seeds. Without
  it, the in-cascade form passes one of seven and the monolithic form three of
  seven.
- The S-DeepONet cells show the same direction but less separation: the baseline
  cells pass in seven of seven and six of seven seeds, while the no-baseline
  cells pass in five of seven and three of seven.

The Jensen-gap tables follow the same pattern. Standard COD preserves the median
gap ratio near one for every reported state. COD without the baseline loses gap
most clearly in C2H2, whose median seed-level ratio is 0.9770 with a range from
0.9351 to 1.0324. FNO without the baseline loses much more, with a C2H2 median
ratio of 0.7108 in cascade and 0.7127 in monolithic form.

These results support a mechanism statement: the analytic thermal baseline
supplies cyclic shape that several neural architectures otherwise smooth. They
do not show that every baseline-equipped seed passes, and the gate is not a
substitute for the rollout endpoint.

## Negative result: bounded correction

Bounded-correction COD is not a successful refinement under the frozen protocol.
Only one of seven seeds converged. That seed has a thermal MAE of 14.409 degC,
compared with 0.389 degC for standard COD, and large downstream gas errors. The
result should be reported as a failed design test. It should not be presented as
a structural guarantee obtained at negligible accuracy cost.

## Required cautions before manuscript revision

1. Do not claim that COD is the first cascade architecture, the only framework
   with the listed properties, or an unexplored idea. Position it as a tested
   process-structured deployment pattern for long-horizon transformer
   thermal-chemical simulation.
2. Do not retain the old v57 headline numbers. The new matrix, convergence rates,
   one-year rollout endpoint, and swing/Jensen results are the evidence base.
3. Do not state that all monolithic models fail to converge. The matrix has
   converged monolithic checkpoints whose short-window objective is acceptable;
   their main failure is long-horizon propagation.
4. Keep PI-DeepONet without baseline out of the primary H1 conclusion because the
   thermal-confound rule excludes it.
5. Report the bounded-correction failure rate before its single converged seed.
6. Separate predeclared results from post-hoc stability descriptions. Early EOL,
   shortened common horizons, and divergence frequency are useful engineering
   diagnostics, but they were not the predeclared H1 metric.
7. Restore `o5` and `o12`, then rerun both postprocessors and both aggregators
   before freezing manuscript tables.
8. Follow the predeclared escalation rule for the eight cells marked with a
   21-seed target if the paper makes seed-separation claims for those cells.

## Manuscript direction

The revised paper should lead with the engineering problem: short-window
surrogate accuracy does not ensure stable multi-window transformer state
propagation. The contribution is the controlled 2 by 2 study across FNO, MIONet,
PI-DeepONet, and S-DeepONet, plus an interpretable transformer implementation in
which the learned thermal state drives deterministic gas quadrature. The theory
can be reduced to the assumptions needed to explain the implementation and moved
to an appendix. The main text should center on convergence, one-year rollout
error in ppm, cycle-shape preservation, Jensen-gap preservation, failure modes,
and operational thresholds.
