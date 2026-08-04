# Revised experimental matrix, given effectively unlimited compute

Proposal, not a decision. C-11 is CLOSED and this does not rewrite it; it asks to
extend it where the binding constraint was cost and that constraint is gone.

**The organising question is not "what can we now afford" but "what was cut for
cost, and what was cut for a reason compute does not touch".** Those are two
different lists and only the first is unlocked.

---

## 0. What unlimited compute does NOT buy

Stated first, because the risk of a compute windfall is spending it where the
limit was never compute.

| limitation | why it stays |
|---|---|
| n = 2 ETT feeders (C-14) | data availability. No number of GPUs produces a third measured machine. The anchor figure stays an operating-regime claim. |
| `k_gen`, `k_dis`, `E_act` assumed (O-3) | closed as a bounded limitation: the Jensen ratio is invariant to them, and calibrating would add more assumptions than it removes. Compute is irrelevant to that argument. |
| `tau_oil`, `DTheta_oil_R`, `n_exp` assumed (O-11) | same, plus the three specific reasons ETT cannot supply them (machines run cold, hot-spot not derivable from OT, OT is a sensor reading). |
| AMORE (C-7) | no public code. The barrier is implementation risk for a single implementer, not GPU time. |
| Battery case study (C-12) | cut for coherence and paper length. Adding it back would reintroduce the asymmetry that motivated cutting it. |
| Goswami, Zanardi, UDE-as-baseline | excluded on overlap and applicability, not cost. |
| Synthetic benchmark | the benchmark is synthetic because no public dataset has hot-spot plus DGA plus load. More runs on synthetic data do not make it less synthetic. |

**Two costs that rise with the number of experiments and are not GPU time.**

1. **Verification burden.** Every new experiment needs its own falsifiable script
   (CLAUDE.md). J-89 records three metrics that returned a plausible number for a
   degenerate input, each caught by a human noticing. Ten more experiments is ten
   more chances for that, and the check does not parallelise across accounts.
2. **Multiple comparisons.** Nine cells x several metrics x many seeds will
   produce some differences that look real by chance. This needs a **pre-declared
   analysis plan** — which comparison decides which question, written before
   looking — or the matrix becomes a fishing expedition with excellent provenance.

---

## 1. Seeds: adaptive, not uniform

**Do not simply raise 7 to 21 everywhere.** Seeds estimate the training variance
of one cell, and the useful number depends on what that cell's distribution looks
like, which is not known in advance.

* **7 seeds for every cell** as the floor. Odd, so the median is a real run;
  enough for a range rather than a mean.
* **Escalate to 21 seeds only for cells that show instability**, defined in
  advance as: any seed not reaching `converged_plateau`, **or** a max/min thermal
  MAE ratio above 2 across the 7.

The reason to escalate is specific, not "we can". If a baseline sometimes trains
and sometimes does not, the quantity a reviewer needs is the **failure rate**, and
a rate is much harder to estimate than a median: 7 seeds cannot distinguish a 15%
failure rate from a 40% one, 21 can. Audit M-2 found exactly this regime —
monolithic error rising 47x with capacity and causal weights underflowing — so
instability in the monolithic cells is the expected case, not a remote one.

**COD itself must get the same 7 seeds.** It is currently n = 1 (O-5). A headline
method reported at one seed against baselines at seven is an asymmetry a reviewer
will find immediately.

**Ablation A to 7 seeds.** Its 6x accuracy result is n = 1 and it is now load
bearing for the paper's only surviving claim about the analytic baseline.

---

## 2. The hyperparameter search — the highest-value addition

C-11 already requires it: "Mỗi baseline được một đợt tìm siêu tham số bằng ngân
sách của phương pháp chính, và đợt tìm đó phải báo cáo." It was cut for cost. It
is the single thing that answers the reviewer question the audit says this paper
will get — *did the baseline fail, or did you fail to train it?* — and without it
every tier-1 number is arguable.

**Design.**

* Search only over hyperparameters **J-90 records as chosen rather than
  paper-derived**, plus learning rate, which dominates. Searching the paper's own
  values would be re-tuning the reference architecture, which is a different and
  less defensible thing.
* Budget per architecture equal to **one main-method training run**, as C-11
  specifies — now trivially affordable, and the point is the protocol, not the
  saving.
* **Selection on validation loss only.** README rule 4: adjustments come from
  validation split from the training distribution, never from a test tier. This
  must be enforced in code, not by discipline — the search script should not have
  the test set available at all.
* Report the whole search, not the winner. A table of every configuration tried
  and its validation loss is what makes "we tuned it" checkable.

Roughly 16 configurations x 6 baseline cells x 1 seed = 96 runs. At the measured
20-50 min, a few hours across accounts.

---

## 3. Tier 2 and tier 3: both in, for different reasons

**Tier 2, PINN per-profile — the highest priority item in this document.** The
audit lists "PINN baseline (§7.2) does not exist" as a *blocking* finding. It is
the evidence for the amortisation argument, which is a headline claim: an
operator trained once against a PINN retrained per profile. 10 profiles x 3 seeds
= 30 runs. This was deferred purely for cost and should not have outlived that
constraint.

**Tier 2, RBA-PINN (Ramirez et al. 2025).** Nearest in-domain competitor, same
problem, same inference chain — temperature learned then ageing derived — and a
reviewer will ask why it is absent. Reimplement on the synthetic benchmark as
C-11 already specifies. **Flag the risk honestly: this is a reimplementation from
a paper, like FNO/MIONet/S-DeepONet, so J-90's adaptation discipline applies and
it carries the same "did we build it right" caveat.** Their real data is not
public and that limitation is stated regardless of compute.

**Tier 3, LSTM and GRU, purely data-driven.** C-11's reasoning stands: it is what
an engineer tries first, and without it the paper lacks a pragmatic reference
point. Cheap now, and the GRU machinery already exists in `sdeeponet.py`.

One thing to get right rather than assume: tier 3 is **supervised**, so it needs
RK45 labels for the training set, which the physics-loss pipeline never generated.
That is about 8,000 solves, a few minutes — but it is a different training path
and needs its own wiring and its own round-trip check, not a config flag.

---

## 4. The 24 h window: scoped, not a second matrix

O-7 asks for both 12 h and 24 h. Note what that actually costs in protocol terms:
`window_minutes` lives **inside the hashed `distribution` block**, so a 24 h
window is a different frozen distribution, a different test set, and a second
benchmark to describe — not a variant run.

**Proposal: train COD only at 24 h, 7 seeds, and compare the Jensen gap and the
rollout thermal error.** That answers the question O-7 is really asking — does the
12 h choice drive the conclusions — for 7 runs instead of 49. Escalate to a full
second matrix only if the answer is yes.

C-4 settled the 12 h window on physical grounds (4.8 x `tau_oil`) and is CLOSED;
this is a robustness check on that decision, not a reopening of it.

---

## 5. EOL at realistic loads: now measurable, and the answer may itself be a finding

Currently censored because low loads need decades of windows. With burn-ins
saturating at 730 the marginal cost is ~0.4 s/window, so:

| K_base | windows to EOL (extrapolated) | approx. years | approx. cost |
|---|---|---|---|
| 1.10 | ~1,024 (measured) | 1.4 | done |
| 1.00 | ~10,000 | 14 | ~1 h |
| 0.95 | ~21,000 | 29 | ~2.5 h |
| 0.85 | ~110,000 | 151 | ~12 h |

All affordable. But **the K = 0.85 figure is worth predicting before measuring**:
a 151-year life at 85% load is longer than any transformer service life, which
would say the assumed kinetic constants produce implausibly slow ageing at low
load. That is a finding about the benchmark's ageing model (O-3's assumed
constants), not about the model, and it should be reported as such rather than
quietly presented as an EOL prediction.

Recommend measuring K = 1.00 and 0.95, and reporting 0.85 as "beyond any service
life, which bounds the practical relevance of the EOL claim at low load".

---

## 6. Proposed matrix

| tier | cells | seeds | runs |
|---|---|---|---|
| 0 | LSODA at matched rtol, IEC analytic, daily-mean Arrhenius | n/a | no training |
| 1 | COD, MonolithicFair, FNO x2, MIONet x2, S-DeepONet x2, Ablation A | 7 (21 if unstable) | 63 |
| 1-hp | hyperparameter search, 6 baseline cells | 1 | ~96 |
| 2 | PINN per-profile (10 profiles) | 3 | 30 |
| 2 | RBA-PINN | 7 | 7 |
| 3 | LSTM, GRU | 7 | 14 |
| aux | COD at 24 h window | 7 | 7 |
| aux | EOL rollouts K = 1.00, 0.95 | n/a | ~4 h |

About 220 training runs, dominated by the search. At the measured 20-50 min and
20 accounts, this is well under a day of wall clock.

**What I would still leave out**, beyond §0: a third and fourth architecture
family added merely because there is room. Each new architecture costs a paper
read, an adaptation decision recorded in J-90, a fairness argument, and a
verification script — none of which parallelises. Four architectures is already
enough to separate "the cascade helps" from "this one architecture is odd", which
is what the matrix is for.

---

## 7. What this needs before it starts

1. **A pre-declared analysis plan.** Which comparison answers which question,
   written before any of it is read. Without it the multiple-comparisons problem
   above turns a strength into a weakness.
2. **The search must not be able to see the test set.** Enforced structurally.
3. **The tier-3 supervised path needs building** — label generation, a trainer,
   and a checkpoint round-trip like every other cell.
4. **Real GPU costs for MIONet and S-DeepONet**, in flight. The last CPU-based
   estimate was wrong in direction for FNO, so nothing here is scheduled on a
   local measurement.
