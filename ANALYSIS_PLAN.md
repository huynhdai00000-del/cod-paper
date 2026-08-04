# Pre-declared analysis plan — C-11 tier-1 matrix and extensions

**Written and committed before any matrix run exists.** The git timestamp on this
file is the evidence for that, and it is the point of the file: with ~220 runs
across ~10 cells and several metrics, deciding *afterwards* which comparison
mattered is how a strength becomes a weakness. The manuscript was desk-rejected
partly for claims that could not be traced to an artifact; a plan written after
the numbers cannot fix that even when every number is honest.

Nothing here may be changed once the first matrix run is committed. If a change is
unavoidable, it is appended as a dated amendment with its reason, never edited in
place, and the analysis reports both versions.

---

## 0. What the matrix is for

C-11 exists to answer one question a reviewer will ask: **when a baseline does
worse, is that the architecture, or is it the cascade it was denied?** Every
tier-1 architecture is therefore run twice — `monolithic` predicting all six
states, `in_cascade` predicting `theta_TO` with the gases by Arrhenius
quadrature — and the pair is the one-variable test.

This plan does **not** test whether decomposition is novel. C-8 settled that it is
not, and no experiment here bears on it.

---

## 1. The primary comparison — one hypothesis, four tests

**H1. Within an architecture, the in-cascade configuration achieves lower
absolute gas error than the monolithic configuration.**

* **Metric:** per-gas MAE in **ppm** (`cod/eval/metrics.py`), on tier
  `T1_in_distribution` (realistic sampler, held-out seed 999). Absolute physical
  units only. **Gas NMAE is not reported anywhere in this analysis** — C-9 as
  tightened 2026-08-02 established that the median 12 h gas variation is
  0.001%–0.046% of each gas's engineering threshold, so a gas percentage is a
  ratio to a physically empty quantity.
* **Unit of test:** one test per architecture — PI-DeepONet, FNO, MIONet,
  S-DeepONet. **Four tests, declared in advance.** Everything else in this
  document is secondary or exploratory and will be labelled as such in the paper.
* **Why gas and not thermal:** the cascade only changes how the *gases* are
  obtained. Both configurations predict `theta_TO` with the same network capacity,
  so thermal error is not where the cascade acts. Making thermal MAE the primary
  metric would test the wrong thing.

**The mandatory control.** The comparison is confounded if the two configurations
do not predict `theta_TO` comparably, because gas error inherits thermal error
through `V_arr`, which is exponential in temperature. So for each architecture:

> If the median thermal MAE of the two configurations differs by more than **2x**,
> the gas comparison for that architecture is reported as **confounded** and does
> not count toward H1 in either direction.

This is declared now precisely because it can go against the claim.

---

## 2. Secondary comparisons — stated now, labelled secondary in the paper

| # | comparison | what it addresses |
|---|---|---|
| S1 | thermal MAE, in-cascade vs monolithic, per architecture | the control above, reported in its own right |
| S2 | COD vs Ablation A (O-12) | what the analytic baseline `H` is worth, one variable |
| S3 | swing ratio and Jensen gap along trajectory, all cells | C-11's honesty protocol; whether a cell keeps the cycle |
| S4 | convergence rate across seeds, per cell | "did the baseline fail, or did you fail to train it" |
| S5 | tier-0 baselines (LSODA, IEC analytic, daily-mean Arrhenius) | the Jensen gap the method exists to capture (C-10) |
| S6 | tier-2 PINN per-profile | the amortisation claim; replaces the §7.2 baseline the audit found missing |
| S7 | tier-3 LSTM/GRU | a pragmatic engineer's first attempt as a reference point |
| S8 | post-hoc cascade (below) | whether the cascade must be in the training loop at all |

**S8 deserves naming as a real risk to the claim.** Take each *monolithic* model's
predicted `theta_TO`, push it through the same Arrhenius quadrature offline, and
score the resulting gases. O-1 already did this for v57 and found it *lowered* gas
error by 1.12x–1.60x. If the post-hoc cascade recovers all of the in-cascade
advantage, then the cascade is a **post-processing step, not an architecture**, and
the paper must say so. That is a materially different and weaker claim than the
one C-11 is set up to support, and it costs no training to check.

---

## 3. What counts as a difference worth reporting

A difference must clear **both** bars. Either alone is insufficient.

**(a) Physical relevance.** The engineering thresholds already in
`cod/eval/metrics.py`, which are what a practitioner needs, not targets tuned
against:

| quantity | threshold | "worth reporting" floor |
|---|---|---|
| `theta_TO` | 2.0 degC (IEC 60076-2 heat-run identification uncertainty) | 0.2 degC |
| `c_H2`, `c_C2H4` | 2.0 ppm (DGA lab repeatability) | 0.2 ppm |
| `c_C2H2` | 1.0 ppm | 0.1 ppm |
| `c_CO` | 5.0 ppm | 0.5 ppm |
| `c_CO2` | 10.0 ppm | 1.0 ppm |
| Jensen gap ratio | — | 0.02, the level at which C-10's analytic curve agrees with measurement (2–5%) |

The floor is 10% of the threshold. A difference below it is reported as "no
operationally meaningful difference" **even if it is statistically clean**.

**(b) Seed separation.** The min–max ranges across seeds must not overlap. With 7
seeds and no distributional assumption, non-overlapping ranges is the honest
statement available; no p-values are computed, and none are needed for a claim of
the form "A is better than B on this benchmark".

**If (a) holds and (b) fails, the finding is "suggestive, seeds overlap".** If (b)
holds and (a) fails, it is "statistically separable, operationally negligible".
Both phrasings are pre-approved; neither is upgraded later.

---

## 4. How seed variation is summarised

* **Median and full min–max range.** Not mean ± sd: audit M-2 and N-9 both
  describe unstable training in the monolithic regime, so the distributions are
  expected to be skewed or bimodal and a mean would describe neither mode.
* **Every seed's `stop_reason` is reported**, in a table, per cell.
* **A non-converged seed is reported as non-converged and is never dropped.**
  Dropping it selects for the seeds that happened to work, which is precisely the
  failure the README's rule 5 exists to prevent. A cell with non-converged seeds
  is summarised by its **convergence rate first** and its error distribution
  second.
* **7 seeds per cell**, escalating to **21 for any cell** with a non-converged
  seed or a max/min thermal MAE ratio above 2 — because for an unstable cell the
  quantity of interest is the failure *rate*, and 7 seeds cannot distinguish a
  15% rate from a 40% one.
* **COD and Ablation A get the same 7 seeds as every baseline.** COD is currently
  n = 1 (O-5) and Ablation A n = 1; a headline method at one seed against
  baselines at seven is an asymmetry that invalidates the comparison.

---

## 5. What would falsify the cascade claim

Declared now, in the form that makes them capable of firing. If any of these
occurs it goes in the paper as stated, not reframed.

**F1 — the cascade does not help.** If for **2 or more of the 4 architectures**
the in-cascade gas MAE is not lower than monolithic by the §3 criteria (excluding
architectures whose comparison is confounded per §1), then "the cascade improves
gas prediction" is not supported and the paper says so.

**F2 — the cascade is only thermal accuracy in disguise.** If in-cascade's gas
advantage vanishes once thermal MAE is matched — i.e. every architecture whose
advantage survives §1's 2x control loses it — then the cascade contributes nothing
beyond making `theta_TO` easier to learn, and must be described that way.

**F3 — the cascade is post-processing, not architecture.** If S8's post-hoc
cascade applied to monolithic outputs recovers the in-cascade gas accuracy within
the §3 floor, then the cascade need not be in the training loop, and the
structural claim reduces to "apply the quadrature afterwards".

**F4 — the gap is not what the cascade preserves.** If monolithic and in-cascade
configurations preserve the Jensen gap equally well (gap ratio difference below
0.02), then the cascade is not the mechanism preserving it, whatever else it does.

**F5 — the baselines were simply untrained.** If a baseline's hyperparameter
search (§6) moves it into the same error band as the in-cascade configuration,
that cell's original result is withdrawn and the tuned one reported instead.

**Precedent that these can fire.** N-8's spectral-bias mechanism predicted that an
architecture without the analytic baseline must smooth the cycle. Ablation A,
converged, one variable, measured the opposite: swing ratio 1.083, Jensen gap
+2.25% to +6.96%, no smoothing at all. That claim was dropped. This plan is
written by someone who has already had to drop one mechanism claim from this
project, and F1–F5 are meant to be equally capable of firing.

---

## 6. The hyperparameter search, and the wall between it and the test set

C-11 requires one search per baseline at the main method's budget, reported.

* Search **only** over hyperparameters PORT_LOG J-90 records as *chosen* rather
  than paper-derived, plus learning rate. Searching a paper's own published values
  would be re-tuning the reference architecture, which is a different and less
  defensible activity.
* **Selection on validation loss only**, from the training distribution.
  README rule 4. This is enforced **structurally**: the search script must not
  construct a test set at all, and that is checked, not trusted.
* **The full search is reported** — every configuration and its validation loss,
  not only the winner. "We tuned it" is only checkable if the search is visible.
* The search runs **before** the 7-seed sweep for that cell, and the sweep uses
  the selected configuration. No cell is tuned after seeing its tier-1 result.

---

## 7. Tiers that are reported but not part of H1

* **Tier 0** (LSODA at matched tolerance, IEC 60076-7 analytic, daily-mean
  Arrhenius) — no training. The daily-mean arm is the direct measurement of the
  Jensen gap that C-10 makes the paper's trunk, and it is the most important
  baseline in the document despite requiring no GPU.
* **Tier 2** — PINN per-profile (amortisation), RBA-PINN (nearest in-domain
  competitor). Both reported against COD, neither part of the cascade test.
* **Tier 3** — LSTM/GRU, supervised. Different paradigm; a reference point, not a
  controlled comparison. Its training needs RK45 labels the physics pipeline never
  generated, so it gets its own wiring and its own round-trip check.

---

## 8. Reporting rules that apply to everything here

1. Absolute physical units are primary; NMAE is secondary for `theta_TO` only and
   absent for gases (C-9).
2. Test tiers are never merged, and every number states its tier and
   `tier_source` (C-9, and the T1 mislabelling that J-83 records).
3. A non-converged model is reported as non-converged with its learning curve,
   never converted into a performance figure (README rule 5).
4. Every quantitative claim in the write-up names the script that produced it and
   that script is committed (CLAUDE.md).
5. Any metric added after this date is checked against the "silent sentinel"
   procedure in PORT_LOG J-89 before it is allowed to produce a number.

---

## 9. Measured costs this plan is scheduled on

All GPU, Tesla T4, all `converged_plateau`, none budget-bound:

| cell | epochs | wall (s) | s/epoch | vs COD | CPU predicted | CPU error |
|---|---|---|---|---|---|---|
| COD (O-5) | 11,900 | 4,911 | 0.4127 | 1.00 | 1.00 | — |
| `cod_no_baseline` | 7,000 | 2,835 | 0.4050 | 0.98 | 1.00 | 1.0x |
| `fno_in_cascade` | 4,700 | 1,161 | 0.2470 | 0.60 | 1.91 | **3.2x** |
| `mionet_in_cascade` | 7,400 | 187 | 0.0253 | 0.06 | 0.16 | **2.6x** |
| `sdeeponet_in_cascade` | 8,100 | 291 | 0.0359 | 0.09 | 1.17 | **13.4x** |

**COD is the most expensive cell**, at 45% of the 10,800 s budget. The local CPU
ranking mispredicted every parallel architecture, by up to 13.4x, and inverted the
order twice. The pattern is consistent: COD is scalar- and sequential-heavy — the
contraction solve for `theta_ss`, the interpolation, small tensors — and gains
least from a GPU, while FFT, GRU and large matmuls collapse. **No schedule in this
project is to be planned from a CPU measurement again.**

`fno_in_cascade`'s 1,161 s is the pre-clamp-fix run; the corrected figure replaces
it when available and is expected similar, since the clamp changes the loss and
not the cost.

Estimated total, ~220 runs dominated by the search: **~60 GPU-hours**, a few hours
of wall clock across 20 accounts.
