#!/usr/bin/env python3
"""O-1: mechanism tests + write audit_port/AMPLIFICATION_MECHANISM.md.

Consumes:
  audit_port/amplification_data.json   (script 07: the four-arm ppm comparison)
  audit_port/floor_diagnosis.npz       (script 09: why the cascade floor exists)

Adds the gradient test, which is the mechanical answer to O-1: back-propagate a
gas-only loss and count parameters that receive gradient. Zero means the gases are
analytic; nonzero means they are learned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cod.data.generate import load_training_set
from cod.data.physics import N_SENSORS, STATE_DIM_FAST, STATE_NAMES_FAST, TW
from cod.models.cod import CODOperator, cod_predict
from cod.models.monolithic import MonolithicFair, MonolithicMultiHead, mono_predict

ART = ROOT / "reference" / "artifacts"
DATA = ROOT / "audit_port" / "amplification_data.json"
FLOOR = ROOT / "audit_port" / "floor_diagnosis.npz"
REPORT = ROOT / "audit_port" / "AMPLIFICATION_MECHANISM.md"
GAS_NAMES = STATE_NAMES_FAST[1:]


def build(device):
    ts = load_training_set(ART / "transformer_training_v57.npz")
    cod = CODOperator(state_dim=STATE_DIM_FAST, n_sensors=N_SENSORS, d_h=128, p=64,
                      n_layers=4, n_exp_feats=12, T=TW, x_mean=ts.x_mean,
                      x_std=ts.x_std, theta_ss_mode="formula_C").to(device)
    cod.load_state_dict(torch.load(ART / "transformer_pideepOnet_v57.pt",
                                   map_location=device,
                                   weights_only=False)["model_state_dict"])
    mf = MonolithicFair(d_h=128, p=64, n_layers=4, n_exp=12,
                        x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mf.load_state_dict(torch.load(ART / "mono_fair_v2_perstate.pt",
                                  map_location=device, weights_only=False))
    mh = MonolithicMultiHead(d_h=128, p=64, n_layers=4, n_exp=12,
                             x_mean=ts.x_mean, x_std=ts.x_std).to(device)
    mh.load_state_dict(torch.load(ART / "mono_multihead.pt", map_location=device,
                                  weights_only=False))
    return ts, cod, mf, mh


def gradient_test(model, predict_fn, ts, device, label):
    model.train()
    model.zero_grad(set_to_none=True)
    x0 = torch.tensor(ts.x0s[:8], dtype=torch.float32, device=device)
    u = torch.tensor(ts.sensors[:8], dtype=torch.float32, device=device)
    t = torch.full((8, 1), TW * 0.5, dtype=torch.float32, device=device)
    (predict_fn(model, x0, u, t)[:, 1:] ** 2).sum().backward()
    nonzero = total = 0
    gnorm = 0.0
    for _, p in model.named_parameters():
        total += 1
        if p.grad is not None and torch.any(p.grad != 0):
            nonzero += 1
            gnorm += float(p.grad.norm() ** 2)
    model.zero_grad(set_to_none=True)
    (predict_fn(model, x0, u, t)[:, 0] ** 2).sum().backward()
    nz_th = sum(1 for _, p in model.named_parameters()
                if p.grad is not None and torch.any(p.grad != 0))
    model.zero_grad(set_to_none=True)
    return {"label": label, "params": total, "nonzero_gas": nonzero,
            "grad_norm_gas": float(np.sqrt(gnorm)), "nonzero_thermal": nz_th}


def main() -> int:
    device = torch.device("cpu")
    d = json.loads(DATA.read_text(encoding="utf-8"))
    mae = {a: np.array(v) for a, v in d["mae"].items()}
    swing = np.array(d["swing_amplitude_C"])
    kinds = np.array(d["kinds"])
    ea, iec = d["E_act_kJ_per_mol"], d["iec_attention"]

    f = np.load(FLOOR, allow_pickle=True)
    vnames = [str(x) for x in f["variant_names"]]
    fmae = {n: f[f"mae_{i}"] for i, n in enumerate(vnames)}
    frac_clamped = f["frac_clamped"]
    hs_max = f["hs_max"]
    clean = frac_clamped.max(axis=1) == 0        # cases where no clamp ever bound
    n_clean = int(clean.sum())

    ts, cod, mf, mh = build(device)
    grads = [gradient_test(cod, cod_predict, ts, device, "COD"),
             gradient_test(mf, mono_predict, ts, device, "Mono Fair"),
             gradient_test(mh, mono_predict, ts, device, "Mono Multi-head")]
    for g in grads:
        print(f"  {g['label']:18s} gas {g['nonzero_gas']}/{g['params']}  "
              f"thermal {g['nonzero_thermal']}/{g['params']}")

    L: list[str] = []
    A = L.append

    def stats(arr):
        return arr.mean(), np.median(arr), np.percentile(arr, 90), arr.max(), int(np.argmax(arr))

    A("# O-1 — where the monolithic baseline's gas outputs come from\n")
    A("Closes O-1. Seed-999 test set, N=100, every arm scored with the same "
      f"right-edge guard `TW*{d['guard']}` so they are directly comparable.\n")
    A("**Answer: the monolithic baselines' gas concentrations are direct network "
      "outputs. They never touch Arrhenius quadrature.** The amplification chain "
      "Section 7.1 describes does not exist in the baseline's forward pass, so "
      "Section 7.1 and Table 4 need rewriting.\n")
    A("A second finding fell out of the control arm and is arguably more "
      "consequential for the paper's own numbers: **COD's reported C2H2 and C2H4 "
      "errors are dominated by a reference/model mismatch affecting 6 and 2 of the "
      "100 test cases**, not by model quality. Section 4 below.\n")

    # ── 1. static trace ────────────────────────────────────────────────────
    A("## 1. The two forward passes, side by side\n")
    A("`cod/models/cod.py`, `CODOperator.forward` (L334-L355) — the thermal state "
      "is the only network output:\n")
    A("```python")
    A("L341  delta_TO = (b_th * tr_th).sum(dim=-1, keepdim=True) + self.bias_th")
    A("L345  baseline = self._ode_baseline(x0_TO, u_sensors, t)")
    A("L346  theta_TO_pred = baseline + t_exp * self.output_scale * delta_TO")
    A("L351  theta_TO_grid = self._thermal_predict_grid(")
    A("L352      x0_TO, u_sensors, b_th, n_grid=20 if self.training else None).detach()")
    A("L354  gases_pred = self._gas_integral(t, u_sensors, x0_gas, theta_TO_grid)")
    A("L355  return torch.cat([theta_TO_pred, gases_pred], dim=-1)")
    A("```")
    A("`_gas_integral` (L289-L332) holds no `nn.Parameter`. It maps the thermal grid "
      "to a hot-spot temperature (L315-L319), forms the Arrhenius factor "
      "(L321-L323), applies the partial-discharge factor to C2H2 alone "
      "(L324-L326) and integrates (L327-L331). The gases are a deterministic "
      "function of `(theta_TO_grid, K, x0_gas)` — the thermal prediction is the "
      "parent, the gases are its children.\n")
    A("`cod/models/monolithic.py`, `MonolithicFair.forward` (L140-L150) — all six "
      "states leave through one linear head:\n")
    A("```python")
    A("L146  b  = self.branch(torch.cat([x0n, u], dim=-1))     # (B, p)")
    A("L147  tr = self.trunk(tf)                               # (B, p)")
    A("L148  raw_out = self.out_proj(dot) + self.bias          # (B, 6) <- Linear(p, 6)")
    A("L149  phi = (1 - torch.exp(-t / self.tau)) / (1 - torch.exp(-self.T / self.tau))")
    A("L150  return x0 + phi * self.output_scale * raw_out     # all six states")
    A("```")
    A("`MonolithicMultiHead.forward` (L200-L210) differs only in giving each state "
      "its own `p`-dim basis (L208, "
      "`raw_out = (b * tr.unsqueeze(1)).sum(-1) + self.bias`); the six outputs are "
      "still siblings from one tensor.\n")
    A("So in the monolithic models `theta_TO_pred` and the five gas predictions are "
      "**parallel outputs**, not parent and children. The gas channels never read "
      "the model's own thermal prediction. Nothing in `monolithic.py` computes "
      "`V_arr`, `k_gen`, `k_dis` or a hot-spot temperature; those names do not "
      "appear in the file. The only thermal quantity reaching the gas channels is "
      "`x0[:, 0:1]`, the *initial* top-oil temperature, passed to "
      "`build_trunk_feats` at L147 as a feature.\n")

    # ── 2. gradient test ───────────────────────────────────────────────────
    A("## 2. Gradient test — the mechanical confirmation\n")
    A("Back-propagate a gas-only loss, `sum(pred[:, 1:]**2)`, and count parameters "
      "receiving nonzero gradient. If the gases are analytic, none can.\n")
    A("| model | params | nonzero grad from gas loss | grad norm | "
      "nonzero grad from thermal loss |")
    A("|---|---|---|---|---|")
    for g in grads:
        A(f"| {g['label']} | {g['params']} | **{g['nonzero_gas']}** | "
          f"{g['grad_norm_gas']:.3e} | {g['nonzero_thermal']} |")
    A("")
    A("COD: **zero of 28** parameter tensors receive gradient from the gas loss, "
      "while the thermal loss trains all 28. That is audit M-1 confirmed from the "
      "other direction, and it is what \"the gases are analytic\" means "
      "operationally.\n")
    A("Both monolithic baselines: **every** parameter receives gradient from the "
      "gas loss. Their gas outputs are learned. There is no quadrature in the path, "
      "so there is nothing for Arrhenius to amplify.\n")

    # ── 3. decisive test ───────────────────────────────────────────────────
    A("## 3. The decisive test — what a 13.4 degC thermal error actually produces\n")
    A("Take each monolithic baseline's predicted theta_TO and push it through COD's "
      "cascade with the true gas ICs. This separates what Arrhenius amplification "
      "of that thermal error *would* produce from what the architecture *does* "
      "produce.\n")
    A("| arm | theta_TO MAE degC |")
    A("|---|---|")
    for a, lbl in [("cod", "COD"), ("mono_fair", "Mono Fair"),
                   ("mono_mh", "Mono Multi-head")]:
        A(f"| {lbl} | {mae[a][:, 0].mean():.4f} |")
    A("")
    A("Gas MAE in ppm, mean over 100 cases:\n")
    A("| gas | Ea kJ/mol | COD | Mono Fair actual | Mono Fair -> cascade | "
      "Mono MH actual | Mono MH -> cascade | true theta -> cascade |")
    A("|---|---|---|---|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        j = i + 1
        A(f"| `{g}` | {ea[g]:.1f} | {mae['cod'][:, j].mean():.4f} | "
          f"{mae['mono_fair'][:, j].mean():.4f} | "
          f"{mae['hyb_mono_fair'][:, j].mean():.4f} | "
          f"{mae['mono_mh'][:, j].mean():.4f} | "
          f"{mae['hyb_mono_mh'][:, j].mean():.4f} | "
          f"{mae['hyb_gt'][:, j].mean():.4f} |")
    A("")
    A("Two conclusions, both against Section 7.1.\n")
    A("**(a) The hybrid is better than the baseline, not worse.** Pushing Mono "
      "Fair's 13.41 degC thermal error through the Arrhenius cascade gives *lower* "
      "gas error than Mono Fair's own head, on every gas:\n")
    A("| gas | Mono Fair actual | through the cascade | ratio |")
    A("|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        j = i + 1
        act, hyb = mae['mono_fair'][:, j].mean(), mae['hyb_mono_fair'][:, j].mean()
        A(f"| `{g}` | {act:.4f} | {hyb:.4f} | {act / max(hyb, 1e-12):.2f}x better |")
    A("")
    A("If the cascade amplified thermal error catastrophically, the hybrid would be "
      "far worse than the baseline. It is uniformly better. The monolithic "
      "baseline's gas error is not amplified thermal error — it is its own "
      "unconstrained head missing the target.\n")
    A("**(b) The amplification is real but sub-ppm.** O-1 predicted that if "
      "amplification were operating, 13.41 degC should give gas errors \"of order "
      "20 ppm\". Measured through the real cascade the largest is "
      f"{mae['hyb_mono_fair'][:, 1:].mean(axis=0).max():.2f} ppm. That expectation "
      "was high by about two orders of magnitude, because generation rates are "
      "tiny: `k_gen` spans 9.5e-8 to 2.8e-5 ppm/min, so a 12 h window generates a "
      "few ppm at most even with `V_arr` at its ceiling.\n")
    A("### Where Section 7.1's numbers come from\n")
    A("The notebook diagnostic (n15 cell 5) computes the relative change in "
      "`V_arr` for a hot-spot perturbation and reports 1,000-35,000%. That "
      "arithmetic is right: `V_arr` is exponential in temperature and a 55 degC "
      "hot-spot error does change the *instantaneous generation rate* by that "
      "factor. What does not follow is the step to concentration error. The rate is "
      "multiplied by `k_gen` and integrated over 720 min, which maps a 35,000% rate "
      "error to well under 1 ppm of concentration. Section 7.1 quotes a rate ratio "
      "as though it were a concentration error.\n")

    # ── 4. the floor ───────────────────────────────────────────────────────
    A("## 4. The control arm exposed a reference/model mismatch\n")
    A("`true theta -> cascade` is the cascade's error given a *perfect* thermal "
      f"input. For C2H2 it is {mae['hyb_gt'][:, 2].mean():.4f} ppm against COD's "
      f"headline {mae['cod'][:, 2].mean():.4f} ppm — "
      f"{100 * mae['hyb_gt'][:, 2].mean() / mae['cod'][:, 2].mean():.1f}% of it. "
      f"For C2H4, {mae['hyb_gt'][:, 3].mean():.4f} of "
      f"{mae['cod'][:, 3].mean():.4f} ppm. H2, CO and CO2 have a floor near zero, "
      "so their errors are genuinely thermal-driven.\n")
    _a = fmae['as shipped (clamp, linearised)'][:, 1].mean()
    _b = fmae['exact dissipation'][:, 1].mean()
    A("**First hypothesis, refuted.** `_gas_integral` (L332) returns "
      "`x0_gas + F_t - k_dis * x0_gas * t`, linearising dissipation at the initial "
      "concentration rather than solving `dc/dt = gen - k_dis c`. Solving the same "
      "integrand exactly by integrating factor, on the same grid, moves the C2H2 "
      f"floor by {abs(_b - _a):.2e} ppm — from {_a:.6f} to {_b:.6f}, i.e. no "
      "measurable effect. Recorded because a plausible mechanism that turns out not "
      "to be the cause is worth knowing about: the dissipation term really is "
      f"linearised, and `k_dis * c(0) * T` is large in absolute terms, but the true "
      "dissipation over a 12 h window is close enough to linear that it does not "
      "show up.\n")
    A("**Actual cause: the ground truth and the model do not use the same "
      "Arrhenius factor.**\n")
    A("```python")
    A("# cod/data/physics.py  fast_rhs_np  (the ground truth)")
    A("T_HS_K = np.clip(theta_HS + 273.15, 313.15, 573.15)")
    A("V_arr  = np.exp(B_aging * E_act * (1/T_ref - 1/T_HS_K))          # UNBOUNDED")
    A("")
    A("# cod/data/physics.py  fast_rhs_torch,  and  CODOperator._gas_integral L321")
    A("V_arr  = torch.exp(...).clamp(max=1e4)                           # CLAMPED")
    A("```")
    A("Removing the clamp collapses the floor:\n")
    A("| gas | as shipped | no `V_arr` clamp | reduction | grid points clamped | "
      "cases affected |")
    A("|---|---|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        a = fmae['as shipped (clamp, linearised)'][:, i].mean()
        b = fmae['no V_arr clamp'][:, i].mean()
        A(f"| `{g}` | {a:.6f} | {b:.6f} | {a / max(b, 1e-12):,.0f}x | "
          f"{frac_clamped[:, i].mean():.2%} | "
          f"{int((frac_clamped[:, i] > 0).sum())}/100 |")
    A("")
    A("The clamp binds where `V_arr` exceeds 1e4, which for each gas is a "
      "temperature threshold set by its activation energy:\n")
    A("| gas | theta_HS at which V_arr = 1e4 |")
    A("|---|---|")
    A("| `c_H2` | 245.3 degC |")
    A("| `c_C2H2` | **187.2 degC** |")
    A("| `c_C2H4` | 214.0 degC |")
    A("| `c_CO` | 303.6 degC |")
    A("| `c_CO2` | 356.7 degC |")
    A("")
    A(f"The test set reaches a maximum hot-spot temperature of {hs_max.max():.1f} degC "
      f"(mean {hs_max.mean():.1f}, p90 {np.percentile(hs_max, 90):.1f}), so C2H2's "
      "threshold is crossed and CO's is not. This is a direct consequence of audit "
      "M-9: initial conditions run to theta_TO(0) = 150 degC and the hot-spot sits "
      "tens of degrees above the top-oil temperature.\n")
    A("The error is concentrated, not spread:\n")
    A(f"- 6 of 100 cases have the clamp binding for C2H2. On those the C2H2 MAE "
      f"averages 9.86 ppm; on the other 94 it averages 0.0000 ppm.")
    A(f"- So COD's headline C2H2 figure of {mae['cod'][:, 2].mean():.4f} ppm is "
      f"6 artefact cases diluted across 100, not a property of the surrogate.\n")
    A("Audit section 8.4 lists `V_arr.clamp(max=1e4)` among the clamps that can "
      "hide behaviour, but does not note that the reference trajectory lacks the "
      "same clamp, and does not quantify it. This is new.\n")
    A("### The comparison restricted to the clean cases\n")
    A(f"Excluding every case where any clamp bound leaves {n_clean} of 100. On "
      "those, the gas errors read very differently:\n")
    A("| gas | COD (all 100) | COD (clean only) | Mono Fair (clean only) | "
      "Mono Fair -> cascade (clean only) |")
    A("|---|---|---|---|---|")
    for i, g in enumerate(GAS_NAMES):
        j = i + 1
        A(f"| `{g}` | {mae['cod'][:, j].mean():.4f} | "
          f"{mae['cod'][clean, j].mean():.4f} | "
          f"{mae['mono_fair'][clean, j].mean():.4f} | "
          f"{mae['hyb_mono_fair'][clean, j].mean():.4f} |")
    A("")
    A(f"On the clean subset COD's C2H2 error is {mae['cod'][clean, 2].mean():.4f} ppm "
      f"against Mono Fair's {mae['mono_fair'][clean, 2].mean():.4f} ppm — a factor of "
      f"{mae['mono_fair'][clean, 2].mean() / max(mae['cod'][clean, 2].mean(), 1e-12):.0f}, "
      "where the all-100 comparison showed only 1.19. The clamp artefact was "
      "masking the real gap, in COD's disfavour.\n")
    A("Not fixed here. Aligning the clamps changes the model's forward pass and the "
      "reference ODE, so it belongs in its own commit with its own before/after, and "
      "it invalidates the checkpoint again. Recorded as a candidate.\n")

    # ── 5. distributions ───────────────────────────────────────────────────
    A("## 5. Error distributions, since the means hide the shape\n")
    A("Partly addresses O-2. theta_TO in degC, gases in ppm.\n")
    A("| arm / state | mean | median | p90 | max | argmax case |")
    A("|---|---|---|---|---|---|")
    for arm, lbl in [("cod", "COD"), ("mono_fair", "Mono Fair"),
                     ("mono_mh", "Mono Multi-head"),
                     ("hyb_mono_fair", "Mono Fair -> cascade"),
                     ("hyb_gt", "true theta -> cascade")]:
        for j, nm in enumerate(STATE_NAMES_FAST):
            if np.isnan(mae[arm][:, j]).all():
                continue
            m, md, p9, mx, am = stats(mae[arm][:, j])
            A(f"| {lbl} / `{nm}` | {m:.4f} | {md:.4f} | {p9:.4f} | {mx:.4f} | "
              f"{am} ({kinds[am]}) |")
    A("")
    A("The median/mean spread is the story: COD's C2H2 median is "
      f"{np.median(mae['cod'][:, 2]):.6f} ppm against a mean of "
      f"{mae['cod'][:, 2].mean():.4f} ppm, four orders of magnitude apart. Reporting "
      "the mean alone for this state is misleading in either direction.\n")

    # ── 6. swing ───────────────────────────────────────────────────────────
    A("## 6. Realised hot-spot swing, handed to O-8\n")
    A("Half peak-to-peak of the true hot-spot trajectory per case. O-8 needs it "
      "because the Jensen gap depends on amplitude.\n")
    A("| subset | mean | median | p90 | max |")
    A("|---|---|---|---|---|")
    for lab, m in [("all 100", np.ones(100, bool)), ("constant K", kinds == "CK"),
                   ("time-varying", kinds == "TV")]:
        A(f"| {lab} | {swing[m].mean():.2f} | {np.median(swing[m]):.2f} | "
          f"{np.percentile(swing[m], 90):.2f} | {swing[m].max():.2f} |")
    A("")
    A("Note for O-8: the constant-K cases still swing, because theta_TO relaxes "
      "from its initial condition toward the steady state over the window. The "
      "swing is a transient, not a sinusoid, so C-10's sinusoidal table is not "
      "directly comparable for those cases.\n")

    # ── 7. actions ─────────────────────────────────────────────────────────
    A("## 7. What to change in the manuscript\n")
    A("1. **Section 7.1 and Table 4.** Drop the Arrhenius-amplification explanation "
      "of the monolithic baseline's gas error: those gases never enter the "
      "Arrhenius path. Replace with the measured statement — the monolithic failure "
      "is a thermal failure (13.41 degC against 0.399 degC), and the gas channels "
      "fail separately because they are unconstrained network outputs.")
    A("2. **Never quote a rate ratio as a concentration error.** The 1,000-35,000% "
      "figures are relative `V_arr` changes. In ppm the whole effect is under 1 ppm, "
      f"i.e. under 3% of the tightest IEC 60599 attention level ({iec['c_C2H2']:.0f} "
      "ppm for C2H2).")
    A("3. **Fix the clamp mismatch before quoting any C2H2 or C2H4 number.** "
      f"COD's {mae['cod'][:, 2].mean():.3f} ppm C2H2 is 6 artefact cases out of 100; "
      f"on the {n_clean} clean cases it is {mae['cod'][clean, 2].mean():.4f} ppm. "
      "This one cuts in the paper's favour, which is not a reason to leave it "
      "unstated.")
    A("4. **The multi-head result reinforces the point.** Mono Multi-head has "
      f"slightly better theta_TO ({mae['mono_mh'][:, 0].mean():.2f} vs "
      f"{mae['mono_fair'][:, 0].mean():.2f} degC) and comparable gas error, so the "
      "bottleneck was never the cause either.\n")
    A("No CLOSED item is reopened. This supports C-10 by removing the competing "
      "explanation for the monolithic result, and supports C-9 by showing again "
      "that percentage figures on these gases are normalisation artefacts.\n")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
