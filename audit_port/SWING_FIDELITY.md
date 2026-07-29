# Does the surrogate flatten the thermal cycle?

**Question.** Neural networks are spectrally biased toward low frequencies, so a thermal surrogate may smooth peaks. If it does it discards part of the Jensen gap the method exists to preserve, and **thermal MAE would not show it** — a flattened trajectory can be close in mean absolute error while systematically under-stating the peak-to-trough range the convex Arrhenius integral is sensitive to.

**The mechanism under test.** COD predicts a correction to an analytic first-order solution, so the cycle shape comes from the IEC baseline and the network's spectral bias has nothing to flatten. Any architecture *without* that baseline has to generate the cycle from the network itself, which is exactly the frequency content spectral bias suppresses. So the prediction is not "neural surrogates flatten cycles" but "neural surrogates flatten cycles **when nothing else supplies the shape**".

## 1. Headline, all three checkpoints

Half peak-to-peak of the hot-spot trajectory, 100 query points across the window, truth by RK45 on `fast_rhs_np` at `rtol = 1e-9`. Live cases are time-varying with a true swing above 1 degC (n = 50).

| model | analytic baseline H | checkpoint | median swing ratio | under-predicting | median thermal MAE degC |
|---|---|---|---|---|---|
| COD v57 | yes | `transformer_pideepOnet_v57.pt` | **1.0121** | 14% | 0.5064 |
| Mono FAIR | no | `mono_fair_v2_perstate.pt` | **0.6802** | 100% | 12.6863 |
| Mono multi-head | no | `mono_multihead.pt` | **0.6493** | 100% | 8.8104 |

**The direction predicted by the mechanism is what appears.** COD, which has the baseline, reproduces the swing to 1.0121 and under-predicts in 14% of cases. Both architectures without the baseline under-predict in **100%** of cases, at a median 0.68 and 0.65 — a third of the swing gone. One-sidedness is the part that is hard to explain any other way: an inaccurate but unbiased model should *over*-predict half peak-to-peak, because independent error inflates the range of a sampled max minus min. Losing swing in every single case is the signature of smoothing, not of error size.

**And the mechanism is still not established, because these checkpoints did not train.** Median thermal MAE is 12.7 and 8.8 degC against COD's 0.51, and in the 25-200 degC band it is 37.8 degC — a model that is not tracking the trajectory at all, whose swing ratio says nothing about spectral bias. The bands where the monolithic models are at least roughly tracking (10-15 and 15-25 degC, MAE 2.2-4.5 degC) lose a more modest 17-28% of the swing, and the ratio there does not worsen with swing — it improves, 0.774 to 0.828 for Mono FAIR. The collapse to 0.61 arrives together with the MAE blowup, which is the reading with the fewest assumptions.

Audit M-2 already established that the monolithic error *rises* 47x as capacity grows 16x, with causal weights underflowed to zero — evidence for "we could not train this baseline", not "this architecture cannot represent the system". Attributing the swing loss to spectral bias would repeat exactly that inference. The repo rule applies unchanged: a model that did not converge is reported as not converged, not converted into a performance number. §5 says what would settle it.

## 2. Full distribution of the swing ratio, per model

| model | population | n | median ratio | Q1 | Q3 | p10 | p90 | median error degC |
|---|---|---|---|---|---|---|---|---|
| COD v57 | all cases | 100 | **1.0038** | 0.9897 | 1.0138 | 0.9703 | 1.0221 | +0.0804 |
| COD v57 | time-varying | 50 | **1.0121** | 1.0078 | 1.0188 | 0.9928 | 1.0221 | +0.3089 |
| COD v57 | time-varying, swing > 1 degC | 50 | **1.0121** | 1.0078 | 1.0188 | 0.9928 | 1.0221 | +0.3089 |
| COD v57 | constant K | 50 | **0.9939** | 0.9732 | 1.0028 | 0.9436 | 1.0208 | -0.0769 |
| Mono FAIR | all cases | 100 | **0.8307** | 0.6274 | 1.0259 | 0.5164 | 1.3414 | -2.5006 |
| Mono FAIR | time-varying | 50 | **0.6802** | 0.6006 | 0.8320 | 0.5164 | 0.9265 | -8.9396 |
| Mono FAIR | time-varying, swing > 1 degC | 50 | **0.6802** | 0.6006 | 0.8320 | 0.5164 | 0.9265 | -8.9396 |
| Mono FAIR | constant K | 50 | **1.0273** | 0.7731 | 1.1574 | 0.5362 | 1.8569 | +0.1277 |
| Mono multi-head | all cases | 100 | **0.7321** | 0.5596 | 0.9773 | 0.3802 | 1.3918 | -3.8400 |
| Mono multi-head | time-varying | 50 | **0.6493** | 0.5582 | 0.7948 | 0.4372 | 0.8894 | -10.2876 |
| Mono multi-head | time-varying, swing > 1 degC | 50 | **0.6493** | 0.5582 | 0.7948 | 0.4372 | 0.8894 | -10.2876 |
| Mono multi-head | constant K | 50 | **0.9785** | 0.5862 | 1.2055 | 0.3386 | 1.8643 | -0.2242 |

## 3. Stratified by true swing — does it worsen where it matters?

A spectral-bias failure shows up as a ratio below 1 that **worsens as the swing grows**. A ratio near 1 that is flat across the bands is the absence of the failure; a ratio that falls with the band is the failure itself.

| model | true swing band | n | median ratio | median error degC | median MAE degC |
|---|---|---|---|---|---|
| COD v57 | 10-15 degC | 3 | 1.0176 | +0.2441 | 0.6340 |
| COD v57 | 15-25 degC | 15 | 1.0174 | +0.3238 | 0.6137 |
| COD v57 | 25-200 degC | 32 | 1.0100 | +0.3288 | 0.4808 |
| Mono FAIR | 10-15 degC | 3 | 0.7741 | -3.1376 | 2.1726 |
| Mono FAIR | 15-25 degC | 15 | 0.8281 | -3.5253 | 3.0241 |
| Mono FAIR | 25-200 degC | 32 | 0.6143 | -11.9437 | 37.7959 |
| Mono multi-head | 10-15 degC | 3 | 0.7198 | -3.9014 | 4.4668 |
| Mono multi-head | 15-25 degC | 15 | 0.7964 | -3.8921 | 4.0488 |
| Mono multi-head | 25-200 degC | 32 | 0.5854 | -13.6162 | 30.6547 |

## 4. The consequence that MAE hides

The reason to measure the swing rather than the error: the Jensen gap carried by the predicted trajectory against the gap carried by the true one. Any flattening shows up here even when MAE is small. The last column is signed so that **negative means gap lost**, i.e. the predicted trajectory carries less Arrhenius acceleration than the true one.

| model | state | median true gap | median predicted gap | median ratio | delta gap, ratio of medians |
|---|---|---|---|---|---|
| COD v57 | `c_H2` | 3.3009 | 3.3514 | 1.0269 | +1.53% |
| COD v57 | `c_C2H2` | 10.5198 | 10.6482 | 1.0424 | +1.22% |
| COD v57 | `c_C2H4` | 5.1323 | 5.2879 | 1.0330 | +3.03% |
| COD v57 | `c_CO` | 2.1784 | 2.1946 | 1.0193 | +0.74% |
| COD v57 | `c_CO2` | 1.8065 | 1.8144 | 1.0152 | +0.44% |
| COD v57 | `DP` | 4.0925 | 4.2002 | 1.0295 | +2.63% |
| Mono FAIR | `c_H2` | 3.3009 | 1.3882 | 0.4642 | -57.94% |
| Mono FAIR | `c_C2H2` | 10.5198 | 2.1330 | 0.2241 | -79.72% |
| Mono FAIR | `c_C2H4` | 5.1323 | 1.6219 | 0.3485 | -68.40% |
| Mono FAIR | `c_CO` | 2.1784 | 1.2199 | 0.6048 | -44.00% |
| Mono FAIR | `c_CO2` | 1.8065 | 1.1562 | 0.6817 | -36.00% |
| Mono FAIR | `DP` | 4.0925 | 1.4960 | 0.4028 | -63.45% |
| Mono multi-head | `c_H2` | 3.3009 | 1.6469 | 0.6535 | -50.11% |
| Mono multi-head | `c_C2H2` | 10.5198 | 3.0734 | 0.4799 | -70.78% |
| Mono multi-head | `c_C2H4` | 5.1323 | 2.0726 | 0.5787 | -59.62% |
| Mono multi-head | `c_CO` | 2.1784 | 1.3551 | 0.7395 | -37.79% |
| Mono multi-head | `c_CO2` | 1.8065 | 1.2486 | 0.7865 | -30.88% |
| Mono multi-head | `DP` | 4.0925 | 1.8404 | 0.6149 | -55.03% |

Thermal MAE over the live cases, for context:

| model | median MAE degC | p90 MAE degC |
|---|---|---|
| COD v57 | 0.5064 | 0.7544 |
| Mono FAIR | 12.6863 | 51.3672 |
| Mono multi-head | 8.8104 | 46.6172 |

MAE and peak-to-trough range are not the same measurement, and only one of them is what the convexity argument depends on. Reading the two tables together is the point: a model can be close in MAE and still lose gap, and the gap column is the one the method's claim rests on.

## 5. What this does not establish

1. **Ablation A is the test this approximates, and its weights do not exist.** Ablation A is COD's architecture with the analytic baseline replaced by the constant `x0` — same network, same pipeline, one variable. Neither the `COD_ablation_study` checkpoint nor the standalone `PI_DeepONet_abl_A` one is among the supplied artifacts, which `cod/models/cod.py` records at `CODNoBaseline`. The monolithic pair used here removes the analytic baseline **and** the cascaded gas integral, and was trained to a far worse optimum, so it is a two-variable substitute. The thermal MAE column is the control that says how much of the swing result is attributable to raw model error rather than to spectral bias.

2. **It is the v57 checkpoint for COD.** Fix 6 invalidated it and the retrain has not happened. The V_arr clamp is not on the thermal path — `theta_TO` comes from the thermal branch and the analytic baseline, neither of which touches Arrhenius — so these thermal predictions are exactly what they always were, but they belong to a model trained on the old distribution.

3. **In distribution only.** Scored on the seed-999 set the models were matched to, so that this measures spectral bias rather than distribution shift. The realistic sampler (fix 7) is a different distribution and the same check has to be rerun there after the retrain.

4. **Every monolithic checkpoint carries the J-8 defect.** Its thermal exponent was shadowed to 12 instead of 0.8 during training. That is a property of the supplied weights, not of the architecture, and it is one more reason the substitute is not Ablation A.

5. **What would settle it**, in increasing order of cost: retrain `CODNoBaseline` on the fix-7 distribution under the same budget as COD and rerun this script, which is the one-variable test and costs one training run; or, if a no-baseline model still will not converge under a fair budget, report that as a convergence result and drop the spectral-bias claim rather than resting it on a broken checkpoint. The delta-learning argument — that predicting a correction to an analytic solution preserves the Jensen gap against the network's low-frequency bias — is worth making only if a converged no-baseline model still loses the gap.

