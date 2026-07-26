"""Building blocks shared by the COD model and the monolithic baselines.

PHASE 1 — FAITHFUL PORT.

`ModifiedMLP` and the 32-dimensional trunk feature builder each appear in three
notebooks under different names but with identical bodies:

    ModifiedMLP        n12 cell 0 L389 | n15 cell 2 L165 | n00 cell 4 L252
                       (n00 calls it `ModMLP` with abbreviated attribute names
                        `eU/eV/f/hl/o/a`, and `ModifiedMLP_orig` with the same
                        names as n12; only the n12/n15 spelling is a valid
                        state_dict key for the stored checkpoints, so that is
                        the one ported)
    build_trunk_feats  n12 cell 0 L502 (`_thermal_trunk_feat`, a method)
                       n15 cell 2 L180 | n00 cell 4 L136 (a free function)

The two trunk-feature spellings are equivalent. n12 computes the query index
twice — `clamp(t/T*(ns-1), 0, ns-2+1e-6)` inside `_interp_at_t` for (K_t, Ta_t),
and `clamp(t/T*(ns-1), 0, ns-1-1e-6)` for the K-history block — while n15/n00
compute it once with the first bound and reuse it. The results are identical for
every t in [0, T]: where tn <= ns-2 both floor to floor(tn), and where
ns-2 < tn <= ns-1 both floor to ns-2. That equivalence is why n15's COD scores
the same 1.5% as n12's from the same checkpoint. Ported once, as the free
function, so there is one definition rather than three.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModifiedMLP(nn.Module):
    """Gated MLP with U/V encoding (Wang et al. 2022).

    Attribute names are load-bearing: `enc_U`, `enc_V`, `first`, `hidden.N`,
    `out` are the state_dict keys in every stored checkpoint.
    """

    def __init__(self, d_in: int, d_h: int, d_out: int, n_layers: int = 4):
        super().__init__()
        self.enc_U = nn.Linear(d_in, d_h)
        self.enc_V = nn.Linear(d_in, d_h)
        self.first = nn.Linear(d_in, d_h)
        self.hidden = nn.ModuleList([nn.Linear(d_h, d_h) for _ in range(n_layers - 1)])
        self.out = nn.Linear(d_h, d_out)
        self.act = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        U = self.act(self.enc_U(x))
        V = self.act(self.enc_V(x))
        h = self.act(self.first(x))
        for layer in self.hidden:
            z = self.act(layer(h))
            h = z * U + (1.0 - z) * V
        return self.out(h)


def build_trunk_feats(t, u_sensors, x0_TO, T, ns, exp_rates, tau_buf,
                      R_buf, ne_buf, me_buf, Do_buf, Dhs_buf, ac_buf, Tr_buf):
    """The 32-dimensional trunk input: 28 features plus 4 K-history features.

    Layout, in order:
        exps        (n_exp_feats)  exp(-t * rate_k / tau)
        rising      (n_exp_feats)  1 - exps
        tn          (1)            t / T
        driving     (1)            (theta_ss(t) - theta_TO(0)) / 80
        K_n         (1)            (K(t) - 0.3) / 1.3
        Ta_n        (1)            (Ta(t) - 15) / 35
        Km_n        (1)            (mean K over [0,t] - 0.3) / 1.3
        Kt_n        (1)            (K(t) - K(0)) / 0.8
        dm_n        (1)            (mean theta_ss over [0,t] - theta_TO(0)) / 80
        dr_n        (1)            (range of theta_ss over the window) / 80

    `ne_buf` and `me_buf` are passed in rather than read from module globals
    because the monolithic baselines register `ne = 12.0` here — the thermal
    exponent shadowed by the constructor's `n_exp=12` argument (PORT_LOG J-8).
    That defect must reach this function, since the monolithic checkpoints were
    trained with it.
    """
    t_sq = t.squeeze(-1)
    exps = torch.exp(-t * exp_rates / tau_buf)
    rising = 1.0 - exps
    tn = t / T

    # Interpolate (K, Ta) at the query time
    tn_idx = torch.clamp(t_sq / T * (ns - 1), 0, ns - 2 + 1e-6)
    idx = torch.clamp(tn_idx.long(), 0, ns - 2)
    frac = tn_idx - idx.float()
    K_t = (torch.gather(u_sensors[:, :ns], 1, idx.unsqueeze(1)).squeeze(1) * (1 - frac)
           + torch.gather(u_sensors[:, :ns], 1, (idx + 1).unsqueeze(1)).squeeze(1) * frac)
    Ta_t = (torch.gather(u_sensors[:, ns:], 1, idx.unsqueeze(1)).squeeze(1) * (1 - frac)
            + torch.gather(u_sensors[:, ns:], 1, (idx + 1).unsqueeze(1)).squeeze(1) * frac)

    # theta_ss at the query time == steady_state.formula_C (n_exp for the oil
    # rise, m_exp for the hot-spot gradient inside the Rf estimate)
    fm = ((1 + K_t ** 2 * R_buf) / (1 + R_buf)) ** me_buf
    fn = ((1 + K_t ** 2 * R_buf) / (1 + R_buf)) ** ne_buf
    tHS0 = Ta_t + Do_buf * fn + Dhs_buf * fm
    Rf = (1 + ac_buf * (tHS0 - Tr_buf)).clamp(0.8, 1.5)
    tss = Ta_t + Do_buf * ((1 + K_t ** 2 * R_buf * Rf) / (1 + R_buf)) ** ne_buf

    drv = ((tss - x0_TO.squeeze(-1)) / 80).unsqueeze(-1)
    Kn = ((K_t - 0.3) / 1.3).unsqueeze(-1)
    Tan = ((Ta_t - 15) / 35).unsqueeze(-1)

    # K history over the whole sensor grid
    Ks = u_sensors[:, :ns]
    tss_s_fm = ((1 + Ks ** 2 * R_buf) / (1 + R_buf)) ** me_buf
    tss_s_fn = ((1 + Ks ** 2 * R_buf) / (1 + R_buf)) ** ne_buf
    tHS0_s = u_sensors[:, ns:] + Do_buf * tss_s_fn + Dhs_buf * tss_s_fm
    Rf_s = (1 + ac_buf * (tHS0_s - Tr_buf)).clamp(0.8, 1.5)
    tss_s = u_sensors[:, ns:] + Do_buf * ((1 + Ks ** 2 * R_buf * Rf_s) / (1 + R_buf)) ** ne_buf

    idx1 = torch.clamp(tn_idx.long(), 0, ns - 2) + 1
    Kcs = torch.cumsum(Ks, 1)
    Km0t = torch.gather(Kcs, 1, (idx1 - 1).unsqueeze(1)).squeeze(1) / idx1.float()
    Ktr = K_t - Ks[:, 0]
    sscs = torch.cumsum(tss_s, 1)
    dm = torch.gather(sscs, 1, (idx1 - 1).unsqueeze(1)).squeeze(1) / idx1.float()
    dr = tss_s.max(dim=-1).values - tss_s.min(dim=-1).values

    return torch.cat([
        exps, rising, tn, drv, Kn, Tan,
        ((Km0t - 0.3) / 1.3).unsqueeze(-1),
        (Ktr / 0.8).unsqueeze(-1),
        ((dm - x0_TO.squeeze(-1)) / 80).unsqueeze(-1),
        (dr / 80).unsqueeze(-1),
    ], dim=-1)


def interp_sensors(sensors, t_v, T, ns):
    """Interpolate (K, Ta) at collocation times, expanding sensors over queries.

    `sensors` is (B, 2*ns); `t_v` is (B*Q, 1). Returns (B*Q, 2) = [K, Ta].
    Identical in n12 cell 0 L705, cell 2 L1105 and cell 2 L1315 (audit §8.1
    marks all three as identical), so ported once.
    """
    BQ = t_v.shape[0]
    B = sensors.shape[0]
    Q = BQ // B
    tn = torch.clamp(t_v.detach().squeeze(-1) / T * (ns - 1), 0, ns - 2 + 1e-6)
    idx = torch.clamp(tn.long(), 0, ns - 2)
    frac = tn - idx.float()
    s_exp = sensors.unsqueeze(1).expand(B, Q, 2 * ns).reshape(BQ, 2 * ns)
    K_lo = torch.gather(s_exp[:, :ns], 1, idx.unsqueeze(1)).squeeze(1)
    K_hi = torch.gather(s_exp[:, :ns], 1, (idx + 1).unsqueeze(1)).squeeze(1)
    K_t = K_lo * (1 - frac) + K_hi * frac
    Ta_lo = torch.gather(s_exp[:, ns:], 1, idx.unsqueeze(1)).squeeze(1)
    Ta_hi = torch.gather(s_exp[:, ns:], 1, (idx + 1).unsqueeze(1)).squeeze(1)
    Ta_t = Ta_lo * (1 - frac) + Ta_hi * frac
    return torch.stack([K_t, Ta_t], dim=-1)


__all__ = ["ModifiedMLP", "build_trunk_feats", "interp_sensors"]
