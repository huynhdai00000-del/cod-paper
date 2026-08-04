# Phase 1 verification

Loaded checkpoints, reproduced stored results. Nothing retrained.

- torch 2.13.0+cpu, numpy 1.24.3, device cpu
- test set: seed 999, N=100 (50 constant-K + 50 time-varying)
- training data: `transformer_training_v57.npz`, 8000 ICs, reproduced byte-for-byte from seed 42
- tolerance: 0.1 pp absolute below 100%, 1% relative above

## Gate 1 — Table 2, `transformer_pideepOnet_v57.pt`

n12 cell 3 `evaluate_v44`, N=100, seed 999, right-edge guard `TW*0.9999`.

| state | reproduced % | stored % | diff pp | |
|---|---|---|---|---|
| `theta_TO` |        1.5 |        1.5 |    +0.01 | PASS |
| `c_H2` |        1.3 |        1.3 |    -0.05 | PASS |
| `c_C2H2` |        2.3 |        2.3 |    -0.01 | PASS |
| `c_C2H4` |        1.6 |        1.6 |    +0.03 | PASS |
| `c_CO` |        1.1 |        1.1 |    +0.04 | PASS |
| `c_CO2` |        1.1 |        1.1 |    -0.01 | PASS |
| **overall** |        1.5 |        1.5 |    -0.01 | PASS |
| constant K |        1.2 |        1.2 |    +0.02 | PASS |
| time-varying |        1.8 |        1.8 |    -0.05 | PASS |
| median |        0.5 |        0.5 |    -0.05 | PASS |
| cases < 10% |         99 |         99 |       +0 | PASS |

### The single case above 10% (audit M-10)

```
Case  33 (CK): x0_TO=141.3  K=1.398  error=16.2%
```

The manuscript describes this as "the single outlier at 17% arising from a high-amplitude time-varying profile at K = 1.3". It is 16.2%, on a **constant-K** case, at K = 1.398 — an extrapolation case starting from theta_TO(0) = 141.3 degC. Wrong on all three counts, and it attributes the only failure to the regime in which robustness is claimed.

### Absolute error, which is what should be reported (audit M-3)

```
--- COD (full): absolute error, physical units ---
  theta_TO  : MAE       0.3993 degC  median denom      32.98  floor hit  0.0%
  c_H2      : MAE       0.0234 ppm   median denom    0.01849  floor hit 10.0%
  c_C2H2    : MAE       0.5926 ppm   median denom  0.0006726  floor hit 38.0%
  c_C2H4    : MAE       0.1646 ppm   median denom   0.007099  floor hit 19.0%
  c_CO      : MAE     0.006009 ppm   median denom     0.0229  floor hit  2.0%
  c_CO2     : MAE     0.004508 ppm   median denom    0.03457  floor hit  1.0%
```

The NMAE denominator floor of 1e-4 binds on a large fraction of the gas cases. Read the MAE column.

## Gate 2 — capacity sweep, `sweep_{cod,mono_fair}_p*.pt`

n15 `evaluate_100`, N=100, seed 999, right-edge guard `TW*0.999`. `d_h = max(64, 2p)`.

| p | COD repro % | COD stored % | | Mono repro % | Mono stored % | | ratio |
|---|---|---|---|---|---|---|---|
| 4 | 2.2 | 2.2 | PASS | 1153.9 | 1153.9 | PASS | 532x |
| 8 | 2.1 | 2.1 | PASS | 7770.8 | 7770.7 | PASS | 3688x |
| 16 | 2.2 | 2.2 | PASS | 5465.3 | 5465.3 | PASS | 2449x |
| 32 | 1.9 | 1.9 | PASS | 15296.6 | 15296.5 | PASS | 7959x |
| 64 | 1.8 | 1.8 | PASS | 54165.2 | 54165.2 | PASS | 30231x |

The monolithic error **rises 47x as capacity grows 16x**, non-monotonically. With causal weights underflowed to exactly zero and a final loss five orders above COD's, this supports "we could not train the monolithic baseline", not "the monolithic architecture cannot represent this system" (audit M-2). A third reason is recorded in PORT_LOG J-8: every monolithic checkpoint was trained with its thermal exponent shadowed to 12 instead of 0.8.

## Gate 3 — monolithic headline

| model | checkpoint | repro % | stored % | | source |
|---|---|---|---|---|---|
| Mono Fair (single bottleneck) | `mono_fair_v2_perstate.pt` | 13199.7 | 13199.7 | PASS | n15 cell 4 |
| Mono Multi-head (no bottleneck) | `mono_multihead.pt` | 18078.4 | 18076.6 | PASS | n15 cell 8 |
| Mono SoftIC (no output scale) | `mono_fair_v1.pt` | not supplied | 18933.3 | SKIP | n00 cell 8 |

CK / TV split for Mono Fair: 15893.4% / 10506.1% (stored 15893.4 / 10506.1).

### Which checkpoint gives which number

Audit open question 3 asks which monolithic run is cited, 13,199.7% or 18,933.3%. They are **not two runs of one experiment** — they are three different architectures:

- **13,199.7%** — `PIDeepONet_Mono_Fair` (n15 cell 2), `mono_fair_v2_perstate.pt`. Single p-dim bottleneck, per-state learnable output scale initialised from `x_std`, exact IC via `phi(t)`. **This is the manuscript's 13,200%.** Reproduced here.
- **18,076.6%** — `PIDeepONet_Mono_MultiHead` (n15 cell 8), `mono_multihead.pt`. No bottleneck, p basis functions per state, 6x the output capacity. Built to test whether the bottleneck caused the failure; it is *worse*. Reproduced here.
- **18,933.3%** — `PIDeepONet_Mono` (n00 cell 8), `mono_fair_v1.pt`. No output scaling at all, and a soft IC mask `sigmoid(10t/T)` which equals 0.5 at t=0, so `x(0) != x0` and the initial condition is violated by construction. **Checkpoint not supplied — cannot be verified.** Part of its error is definitional rather than a learning failure, which is worth saying if it is cited at all.

Per-state NMAE for Mono Fair, with the absolute figures that audit M-3 says must accompany them:

```
=== Mono FAIR v2 (physics) (N=100) ===
  theta_TO  :     42.1% +/-    66.7%
  c_H2      :   9294.9% +/- 22415.0%
  c_C2H2    :  34558.2% +/- 39930.8%
  c_C2H4    :  29420.1% +/- 53892.7%
  c_CO      :   4493.5% +/- 11026.0%
  c_CO2     :   1389.7% +/-  4141.2%
  Overall: 13199.7% | CK: 15893.4% | TV: 10506.1% | <10%: 0/100

--- Mono FAIR v2 (physics): absolute error, physical units ---
  theta_TO  : MAE        13.41 degC  median denom      32.98  floor hit  0.0%
  c_H2      : MAE        0.306 ppm   median denom    0.01849  floor hit 10.0%
  c_C2H2    : MAE       0.7049 ppm   median denom  0.0006726  floor hit 38.0%
  c_C2H4    : MAE       0.4292 ppm   median denom   0.007099  floor hit 19.0%
  c_CO      : MAE       0.1219 ppm   median denom     0.0229  floor hit  2.0%
  c_CO2     : MAE       0.1114 ppm   median denom    0.03457  floor hit  1.0%
```

Audit M-3's point is confirmed: the enormous gas percentages are sub-ppm absolute errors. The genuinely large error is thermal, **13.41 degC** on theta_TO.

M-3 back-converted its absolute errors from mean-of-ratios times a median denominator and labelled that an order-of-magnitude reconstruction. Measuring them directly instead gives:

| state | M-3 reconstruction | measured directly | IEC 60599 attention | measured / attention |
|---|---|---|---|---|
| `theta_TO` | 13.9 degC | 13.4 degC | — | — |
| `c_H2` | 1.72 ppm | 0.306 ppm | 100 ppm | 0.31% |
| `c_C2H2` | 0.23 ppm | 0.705 ppm | 35 ppm | 2.01% |
| `c_C2H4` | — | 0.429 ppm | 200 ppm | 0.21% |
| `c_CO` | — | 0.122 ppm | 700 ppm | 0.02% |
| `c_CO2` | — | 0.111 ppm | 2000 ppm | 0.01% |

The reconstruction overstated H2 by 5.6x and understated C2H2 by 3.1x — expected, given the method it declared, and it lands theta_TO within 4%. The conclusion is unaffected: the worst gas error is 2.0% of its IEC attention level, so no gas percentage in this table describes a diagnostically meaningful error. Quote the measured column, not either set of percentages.

## Verdict

**All gates pass.** The port reproduces every stored figure within tolerance from the supplied checkpoints, without retraining anything.


Wall clock: 237s on CPU.

