# Theorem 2(iii), Corollary 1 and Remark 3 — drop-in replacement text

**The manuscript is not in this repository**, so this is not an edit of the
existing wording — it is replacement text written against the measured numbers,
to be substituted for whatever Theorem 2(iii), Corollary 1 and Remark 3 currently
say. Where the current text is quoted below it is quoted as the user described it
("handles this case qualitatively"), not from the source.

**The measurement:** `audit_port/scripts/32_cod_figure_and_monotonicity.py`,
n = 300 cases from the frozen realistic sampler (`fc4cb76c3b32ec17`), gas initial
conditions compared against the window-mean equilibrium the trajectory relaxes
toward.

| gas | IC above `c_eq` | actually decreased over the window |
|---|---|---|
| `c_H2` | 32.7% | 33.0% |
| `c_C2H2` | 26.3% | 25.0% |
| `c_C2H4` | 29.3% | 29.7% |
| `c_CO` | 27.3% | 27.7% |
| `c_CO2` | 31.0% | 31.7% |
| **any gas** | **47.3%** | **47.7%** |

---

## 1. Why this is a refutation, not a caveat

The current text treats `c_i,0 > c_i,eq` as an edge case to be acknowledged. On
this benchmark it is **not an edge case: it is 47% of the population**, and it is
so *by construction* — `sample_realistic_ic` draws
`gases = c_eq * U(0.45, 1.35)`, so 39% of draws exceed `c_eq` before the 8%
fault-injected minority multiplies three gases by a further 2-7x.

A qualitative acknowledgement of a condition that fails on nearly half the
benchmark is the same class of problem the audit found elsewhere in this
manuscript: a statement whose scope is not the scope the experiments occupy. It
must become a hypothesis of the theorem, and the fraction must be reported.

---

## 2. What survives, and what does not — keep these separate

The current text states monotonicity and non-negativity together. **Only one of
them survives, and they have different proofs**, so combining them hides the
failure.

**Non-negativity survives, unconditionally.** The gas dynamics are
`dc_i/dt = k_gen,i V_arr,i(theta_HS) - k_dis,i c_i`, with
`k_gen,i V_arr,i > 0` everywhere on the admissible temperature envelope. At
`c_i = 0` the derivative is `k_gen,i V_arr,i(theta_HS) > 0`, so the boundary is
repelling and `c_i(0) > 0` implies `c_i(t) > 0` for all `t`. This holds whether
the initial condition is above or below equilibrium, and it is what the cascade's
structural guarantee actually rests on.

**Monotonicity does not survive.** Writing `c_i,eq(t) = k_gen,i V_arr,i / k_dis,i`
for the instantaneous equilibrium,

    dc_i/dt = k_dis,i ( c_i,eq(t) - c_i )

so `c_i` increases exactly when it is **below** the instantaneous equilibrium and
decreases when it is above. `c_i,0 <= c_i,eq` is therefore not a technical
convenience; it is the *entire* content of the monotonicity claim, and on this
distribution it fails 47% of the time.

---

## 3. Replacement text

### Theorem 2(iii) — restated

> **(iii) Sign-definite relaxation.** Let `c_i,eq(t) = k_gen,i V_arr,i(theta_HS(t)) / k_dis,i`
> denote the instantaneous equilibrium. Then for every `i` and every `t`,
>
>     sign( dc_i/dt ) = sign( c_i,eq(t) - c_i(t) ),
>
> so `c_i` moves monotonically toward `c_i,eq` and cannot cross it. In
> particular, **if `c_i,0 <= inf_t c_i,eq(t)` then `c_i` is non-decreasing on the
> window**, and if `c_i,0 >= sup_t c_i,eq(t)` it is non-increasing.

This is strictly stronger than the original where the original held, and true
where it did not: it states the invariant (`c_i` is attracted to `c_i,eq` and
never overshoots) rather than one of its two corollaries.

### Theorem 2 — new part (iv), replacing what non-negativity was bundled into

> **(iv) Positive invariance.** The positive orthant is forward invariant:
> `c_i,0 > 0` implies `c_i(t) > 0` for all `t > 0`, for every `i`, with no
> condition on the initial concentration relative to equilibrium, since
> `dc_i/dt |_{c_i = 0} = k_gen,i V_arr,i(theta_HS) > 0`.

### Corollary 1 — restated

> **Corollary 1.** A cascade predictor that obtains `c_i` by evaluating the
> quadrature of Theorem 2 from any predicted temperature trajectory inherits
> (iii) and (iv) exactly, for any temperature trajectory whatsoever — including a
> wrong one. The guarantees are properties of the quadrature, not of the accuracy
> of the network upstream of it.
>
> A predictor that outputs `c_i` directly inherits neither, and in particular has
> no mechanism forcing `c_i` toward `c_i,eq`, so a steady-state offset in `c_i`
> is not corrected by longer integration.

**This is the corollary the paper should lead with**, because it is exactly the
mechanism the revised primary hypothesis tests (ANALYSIS_PLAN Amendment 1) and it
is the one that survives the monotonicity failure untouched.

### Remark 3 — restated, with the number

> **Remark 3.** The hypothesis `c_i,0 <= inf_t c_i,eq(t)` of Theorem 2(iii) is not
> generic. On the benchmark distribution used here, initial concentrations are
> drawn as a service-history factor times the equilibrium at the unit's own mean
> hot-spot, `c_eq * U(0.45, 1.35)`, with a minority carrying an incipient fault.
> Measured over n = 300 cases, **47.3% of cases begin with at least one gas above
> its window-mean equilibrium, and 47.7% show at least one gas decreasing over the
> window** (per gas: 25-33%). The monotone case is therefore slightly less than
> half the population, and figures showing gases rising throughout are not
> representative of it.
>
> This is a deliberate property of the sampler, not a defect: a fleet contains
> units whose dissolved-gas levels are falling because they are cooler than they
> have been, and a benchmark in which every gas only rises would not be a DGA
> benchmark. The sign-definite form of (iii) covers both cases, and (iv) is
> unconditional.

---

## 4. Consequences elsewhere in the manuscript

1. **Any figure captioned as showing monotone gas growth** must be checked against
   this. `audit_port/figures/state_predictions.pdf`, the median case under the
   current sampler, is the honest replacement for Fig 4 and should be captioned
   with the reason the ground truth changed.
2. **Claims of the form "the cascade guarantees monotone gas evolution" must be
   withdrawn** and replaced with the sign-definite statement plus positive
   invariance. The second is the one with operational content — a diagnostic that
   can output a negative concentration is unusable, and this one cannot.
3. **DECISIONS N-4 is unaffected.** It concerns the Jensen gap vanishing for
   constant load, which is a separate statement about swing amplitude.
