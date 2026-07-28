#!/usr/bin/env python3
"""O-10: what a real transformer's daily load swing actually is.

`RealisticParams.K_amp = (0.12, 0.28)` is calibrated to a target — it is the load
swing that produces a 10-15 degC hot-spot swing — and `realistic.py` says so
plainly, calling it "at the upper end of what a real feeder does". That is an
assertion with nothing behind it. ETT has two years of hourly load from two
operating transformers, so it can be checked.

Data: github.com/zhouhaoyi/ETDataset, ETT-small/ETTh{1,2}.csv, 17420 hourly rows
each, 2016-07-01 to 2018-06-26. Two separate units. Columns are High/Middle/Low
UseFul and UseLess load (active and reactive power by customer class) plus OT,
the oil temperature. Downloaded to `data/ett/` (gitignored; the report carries the
fetch command).

Run:  python audit_port/scripts/15_ett_load_calibration.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "ett"
OUT = ROOT / "audit_port" / "ETT_LOAD_CALIBRATION.md"

P_IDX = [0, 2, 4]      # HUFL, MUFL, LUFL — active power
Q_IDX = [1, 3, 5]      # HULL, MULL, LULL — reactive power
OT_IDX = 6

SERIES = ("ETTh1", "ETTh2")


def load(name: str):
    rows = list(csv.reader(open(DATA / f"{name}.csv")))
    ts = np.array([datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows[1:]])
    v = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    P = v[:, P_IDX].sum(axis=1)
    Q = v[:, Q_IDX].sum(axis=1)
    S = np.sqrt(P ** 2 + Q ** 2)
    dead = np.abs(v[:, :6]).sum(axis=1) == 0.0     # meter outage: every channel 0
    return {"ts": ts, "P": P, "Q": Q, "S": S, "OT": v[:, OT_IDX], "dead": dead}


def blocks(d, hours: int):
    """Non-overlapping `hours`-long blocks from midnight. Complete, clean blocks only.

    A block is dropped if any hour has every load channel at exactly zero, which
    is a meter outage rather than an unloaded transformer.
    """
    h0 = d["ts"][0].replace(hour=0, minute=0, second=0)
    idx = np.array([int((t - h0).total_seconds() // 3600) for t in d["ts"]])
    blk = idx // hours
    keep, dropped = [], 0
    for b in np.unique(blk):
        m = blk == b
        if m.sum() != hours:
            continue
        if d["dead"][m].any():
            dropped += 1
            continue
        keep.append(m)
    return keep, dropped


def qtable(x: np.ndarray) -> dict:
    return {"n": int(len(x)),
            "p10": float(np.percentile(x, 10)), "q1": float(np.percentile(x, 25)),
            "median": float(np.median(x)), "q3": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)), "mean": float(x.mean()),
            "max": float(x.max())}


def row(label: str, d: dict, pct: bool = True) -> str:
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.2f}")
    return (f"| {label} | {d['n']} | {f(d['p10'])} | {f(d['q1'])} | "
            f"**{f(d['median'])}** | {f(d['q3'])} | {f(d['p90'])} | "
            f"{f(d['mean'])} | {f(d['max'])} |")


HDR = ("| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |\n"
       "|---|---|---|---|---|---|---|---|---|")


def realistic_K_base(n: int = 2000) -> np.ndarray:
    from cod.data.physics import N_SENSORS
    from cod.data.realistic import DEFAULTS, make_realistic_profile
    rng = np.random.RandomState(11)
    return np.array([make_realistic_profile(rng, DEFAULTS)[:N_SENSORS].mean()
                     for _ in range(n)])


def _summary_rows(D):
    """Daily swing / rated (p99/0.85) for the four populations, for the summary."""
    out = []
    for n in SERIES:
        S = D[n]["S"]
        base = np.percentile(S, 99) / 0.85
        k, _ = blocks(D[n], 24)
        x = np.array([(S[m].max() - S[m].min()) / (2.0 * base) for m in k])
        out.append((n + ", all days", x))
        if n == "ETTh1":
            P = D[n]["P"]
            bf = np.array([bool((P[m] < 0).any()) for m in k])
            out.append((n + ", non-back-feeding days", x[~bf]))
            out.append((n + ", back-feeding days", x[bf]))
    return sorted(out, key=lambda r: np.median(r[1]))


def main() -> int:
    md: list[str] = []
    A = md.append
    D = {n: load(n) for n in SERIES}

    A("# O-10 — the daily load swing, measured against ETT\n")
    A("`RealisticParams.K_amp = (0.12, 0.28)` was set to hit a target hot-spot "
      "swing of 10-15 degC, not measured from a fleet. `cod/data/realistic.py` "
      "says so and calls it \"at the upper end of what a real feeder does\". This "
      "checks that sentence against two years of hourly load from two operating "
      "transformers.\n")
    A("**Report only. `RealisticParams` is not touched** — see §7 for what would "
      "have to be decided first.\n")

    A("## Summary\n")
    A("The two ETT units disagree with each other by as much as either disagrees "
      "with `RealisticParams`, and that is the result.\n")
    A("Median daily swing as a fraction of rated (`p99/0.85` proxy, §3b), against "
      "`K_amp`'s 12-28%:\n")
    A("| | n days | median | below the band | inside it | above it |")
    A("|---|---|---|---|---|---|")
    for lbl, x in _summary_rows(D):
        A(f"| {lbl} | {len(x)} | **{np.median(x):.1%}** | {(x < 0.12).mean():.1%} "
          f"| {((x >= 0.12) & (x <= 0.28)).mean():.1%} | {(x > 0.28).mean():.1%} |")
    A("")
    A("* **ETTh2 swings roughly half what `RealisticParams` assumes.** 85% of its "
      "days fall below even the bottom of the band, and not one of its 725 days "
      "reaches the top.\n")
    A("* **ETTh1 sits inside the band, and above it whenever the sun is out.** Its "
      "net load reverses at midday in spring — 51% of noon hours have negative "
      "active power, concentrated in March-June. That is photovoltaic back-feed, "
      "i.e. exactly the renewable-driven cycling the manuscript opens on.\n")
    A("* **The PV explains part of the gap between the units, not all of it.** "
      "Removing back-feeding days takes ETTh1 from 24.8% to 17.8%, so roughly 7 of "
      "the 16 points separating it from ETTh2. The remaining 9 points are simply "
      "two different feeders. Any single calibrated `K_amp` is a claim about which "
      "feeder the benchmark is about.\n")
    A("* **Both units' measured top-oil swings are far below the 10-15 degC the "
      "sampler targets at the hot spot** — medians of 2.39 and 5.60 degC in "
      "amplitude. §5 gives three reasons that is suggestive rather than decisive, "
      "and it is the number most likely to be flattering the sampler.\n")
    A("So `K_amp = 0.12-0.28` is not uniformly too high. It is about right for a "
      "PV-back-fed feeder on a quiet day, low for one on a sunny day, and about "
      "double for a conventionally loaded unit. The consequence for the Jensen "
      "headline is in §6: it cannot be a single number, which is what C-10 already "
      "concluded on other grounds.\n")

    # ── data ──────────────────────────────────────────────────────────────
    A("## 1. Data and the three choices that had to be made\n")
    A("`github.com/zhouhaoyi/ETDataset`, `ETT-small/ETTh1.csv` and `ETTh2.csv`. "
      "Two separate units, 17420 hourly rows each, 2016-07-01 to 2018-06-26. "
      "Columns are High/Middle/Low UseFul and UseLess load — active and reactive "
      "power by customer class — plus `OT`, the oil temperature. Fetch:\n")
    A("```")
    A("python -c \"import urllib.request as u;[u.urlretrieve("
      "'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'+f,"
      "'data/ett/'+f) for f in ('ETTh1.csv','ETTh2.csv')]\"")
    A("```\n")
    A("**Never pooled.** ETTh1 and ETTh2 are different transformers in different "
      "counties and §2 shows they do not behave alike; a pooled number would hide "
      "the only interesting thing in this dataset.\n")
    A("**Apparent power, not active power.** IEC 60076-7's load factor K is a "
      "current ratio and losses go as I^2, so what a thermal model responds to is "
      "`|S| = sqrt(P^2 + Q^2)` summed over the three classes. It is also the only "
      "combination that stays non-negative, which matters because ETTh1's active "
      "power is negative in 13.3% of hours.\n")
    A("**Peak-to-trough is twice the amplitude.** `K = K_base + K_amp*sin(...)`, so "
      "`K_amp` is compared against **half** the peak-to-trough range throughout. "
      "Getting this backwards doubles the answer.\n")
    A("**Outage screening.** Blocks containing an hour with every load channel at "
      "exactly zero are dropped as meter outages rather than unloaded "
      "transformers:\n")
    A("| series | 24 h blocks kept | dropped | 12 h blocks kept | dropped |")
    A("|---|---|---|---|---|")
    for n in SERIES:
        k24, d24 = blocks(D[n], 24)
        k12, d12 = blocks(D[n], 12)
        A(f"| {n} | {len(k24)} | {d24} | {len(k12)} | {d12} |")
    A("")

    # ── reverse flow ──────────────────────────────────────────────────────
    A("## 2. ETTh1 back-feeds at midday; ETTh2 never does\n")
    A("This is not a data defect and it changes how §3 reads, so it comes first.\n")
    A("Fraction of hours with negative total active power:\n")
    A("| series | overall | by hour of day (00, 03, 06, 09, 12, 15, 18, 21) |")
    A("|---|---|---|")
    for n in SERIES:
        P, ts = D[n]["P"], D[n]["ts"]
        hr = np.array([t.hour for t in ts])
        by = [f"{(P[hr == h] < 0).mean():.0%}" for h in range(0, 24, 3)]
        A(f"| {n} | {(P < 0).mean():.1%} | {', '.join(by)} |")
    A("")
    A("| series | by month (Jan…Dec) |")
    A("|---|---|")
    for n in SERIES:
        P, ts = D[n]["P"], D[n]["ts"]
        mo = np.array([t.month for t in ts])
        A(f"| {n} | {', '.join(f'{(P[mo == m] < 0).mean():.0%}' for m in range(1, 13))} |")
    A("")
    A("ETTh1: zero at night, 51% at noon, peaking March-June. A midday-only, "
      "spring-peaking reversal is photovoltaic back-feed and nothing else. ETTh2 "
      "has no negative hour in two years (min total active power 11.3).\n")
    A("The consequence for the numbers below: ETTh1's net load passes through zero "
      "most spring middays, so its *relative* range is mechanically large. That is "
      "a real thermal duty — the unit genuinely unloads and reloads — but it is a "
      "different regime from a feeder whose load merely rises and falls, and the "
      "two should not be averaged into one calibration.\n")

    # ── section 3: swing ──────────────────────────────────────────────────
    A("## 3. Daily load swing\n")
    A("Per 24 h block, `(max - min) / (2 * denominator)` of `|S|`. Two "
      "denominators, because the two answer different questions and `K_amp` is "
      "written in the second.\n")

    A("### 3a. As a fraction of the day's own mean load\n")
    A("Needs no assumption about the unit's rating.\n")
    A(HDR)
    rel_mean = {}
    for n in SERIES:
        k, _ = blocks(D[n], 24)
        S = D[n]["S"]
        x = np.array([(S[m].max() - S[m].min()) / (2.0 * S[m].mean()) for m in k])
        rel_mean[n] = x
        A(row(n, qtable(x)))
    A("")
    A(f"**This normalisation breaks down on ETTh1** and the {np.median(rel_mean['ETTh1']):.0%} "
      "median is not a usable number. Its net load spends midday near zero (§2), "
      "which depresses the daily mean at the same time as it widens the range, so "
      "the ratio is inflated at both ends. That is why §3b exists and why the "
      "rated normalisation is the one carried into §6. On ETTh2, which never "
      "back-feeds, the two normalisations agree in ordering and 3a is the more "
      "trustworthy of them.\n")

    A("### 3b. As a fraction of rated load\n")
    A("`K` is per-unit of rated, so `K_amp = 0.12` means twelve percent **of "
      "rated**, not of the day's mean. ETT publishes no nameplate rating, so this "
      "needs a proxy and the answer moves with it. All three are shown rather than "
      "one being chosen silently.\n")
    A("`rated = p99` asserts the unit reaches nameplate in 1% of hours, which is "
      "aggressive; real units peak nearer 0.7-0.9 pu, and assuming a lower peak "
      "loading makes rated *larger* and the normalised swing *smaller*. The middle "
      "row is the most defensible and is the one quoted elsewhere in this file.\n")
    rel_rated = {}
    for proxy, desc in [("p99", "rated = p99(|S|) — the peak hour is nameplate"),
                        ("p99/0.85", "rated = p99(|S|)/0.85 — peak loading is 0.85 pu"),
                        ("max", "rated = max(|S|) over the full record")]:
        A(f"\n**{desc}**\n")
        A(HDR)
        for n in SERIES:
            S = D[n]["S"]
            base = {"p99": np.percentile(S, 99),
                    "p99/0.85": np.percentile(S, 99) / 0.85,
                    "max": S.max()}[proxy]
            k, _ = blocks(D[n], 24)
            x = np.array([(S[m].max() - S[m].min()) / (2.0 * base) for m in k])
            rel_rated[(n, proxy)] = x
            A(row(n, qtable(x)))
    A("")

    A("### 3c. ETTh1 split by whether that day back-fed\n")
    A("How much of ETTh1's swing is the PV, rather than ordinary load following.\n")
    A(HDR)
    S1, P1 = D["ETTh1"]["S"], D["ETTh1"]["P"]
    base1 = np.percentile(S1, 99) / 0.85
    k1, _ = blocks(D["ETTh1"], 24)
    split = {"back-feeding days": [], "non-back-feeding days": []}
    for m in k1:
        x = (S1[m].max() - S1[m].min()) / (2.0 * base1)
        split["back-feeding days" if (P1[m] < 0).any()
              else "non-back-feeding days"].append(x)
    for lbl, xs in split.items():
        A(row(f"ETTh1, {lbl}", qtable(np.array(xs))))
    A("")
    _h1 = np.median(rel_rated[("ETTh1", "p99/0.85")])
    _h1n = np.median(split["non-back-feeding days"])
    _h2 = np.median(rel_rated[("ETTh2", "p99/0.85")])
    A(f"Strip the back-feed and ETTh1's median falls from {_h1:.1%} to {_h1n:.1%}. "
      f"That is {100 * (_h1 - _h1n):.1f} of the {100 * (_h1 - _h2):.1f} points "
      f"separating the two units — **the photovoltaic reversal explains under half "
      f"the difference.** ETTh1 on a day with no back-feed at all still swings "
      f"{_h1n / _h2:.1f}x ETTh2. The rest is not distributed generation, it is "
      "simply that these are two different feeders serving two different mixes of "
      "customer.\n")
    A("This is the part that resists a single calibrated number. Had the PV "
      "accounted for the whole gap, `K_amp` could have been set from the "
      "conventional baseline with a documented uplift for renewable duty. It does "
      "not, so the between-feeder spread is irreducible at this sample size, and "
      "two units cannot estimate it.\n")

    # ── section 4: 12 h ───────────────────────────────────────────────────
    A("## 4. The benchmark window is 12 hours, and that compounds it\n")
    A("`TW = 720` min. The `daily` family in `make_realistic_profile` puts a "
      "**full** sine period inside that window, so a case labelled \"daily\" "
      "completes an entire load cycle in twelve hours. A real daily cycle takes "
      "twenty-four, so a twelve-hour slice of real data contains about half of "
      "one.\n")
    A("Same statistic on non-overlapping 12 h blocks, relative to the block's own "
      "mean, against 3a:\n")
    A(HDR)
    rel12 = {}
    for n in SERIES:
        k, _ = blocks(D[n], 12)
        S = D[n]["S"]
        x = np.array([(S[m].max() - S[m].min()) / (2.0 * S[m].mean()) for m in k])
        rel12[n] = x
        A(row(n + ", 12 h", qtable(x)))
    A("")
    for n in SERIES:
        r = np.median(rel12[n]) / np.median(rel_mean[n])
        A(f"- {n}: median 12 h swing is {r:.2f}x the median 24 h swing "
          f"({np.median(rel12[n]):.1%} against {np.median(rel_mean[n]):.1%}).")
    A("\nThe mismatch therefore compounds. The sampler asks for an amplitude at or "
      "above the top of the real range **and** completes a full cycle of it in "
      "half the time a real cycle takes. A 12 h window of real data sees appreciably "
      "less than a full swing; a 12 h window of sampled data sees a whole period.\n")

    # ── section 5: OT ─────────────────────────────────────────────────────
    A("## 5. The oil temperature ETT actually recorded\n")
    A("Beyond what O-10 asked for, and worth having: `OT` is a real thermal "
      "response to this real load, so it measures the quantity `K_amp` exists to "
      "produce rather than its input.\n")
    A("Daily top-oil peak-to-trough, and half of it, in degC:\n")
    A("| series | n | p10 | Q1 | median | Q3 | p90 | max |")
    A("|---|---|---|---|---|---|---|---|")
    ot_amp = {}
    for n in SERIES:
        k, _ = blocks(D[n], 24)
        OT = D[n]["OT"]
        pt = np.array([OT[m].max() - OT[m].min() for m in k])
        ot_amp[n] = pt / 2.0
        A(f"| {n} peak-to-trough | {len(pt)} | {np.percentile(pt, 10):.2f} | "
          f"{np.percentile(pt, 25):.2f} | **{np.median(pt):.2f}** | "
          f"{np.percentile(pt, 75):.2f} | {np.percentile(pt, 90):.2f} | "
          f"{pt.max():.2f} |")
        a = pt / 2.0
        A(f"| {n} amplitude | {len(a)} | {np.percentile(a, 10):.2f} | "
          f"{np.percentile(a, 25):.2f} | **{np.median(a):.2f}** | "
          f"{np.percentile(a, 75):.2f} | {np.percentile(a, 90):.2f} | "
          f"{a.max():.2f} |")
    A("")
    A("Against a sampler targeting a **hot-spot** amplitude of 10-15 degC, the "
      "measured **top-oil** amplitudes are 2.39 and 5.60 degC at the median, and "
      "4.17 and 8.90 at p90.\n")
    A("**Three reasons this is suggestive and not decisive, stated because the "
      "comparison is not apples to apples and the gap is large enough to be worth "
      "attacking properly:**\n")
    A("1. Hot-spot swings more than top-oil. The gradient "
      "`DTheta_HS_R * ((1 + K^2 R)/(1 + R))^m_exp` moves with the load and in "
      "phase with it, so hot-spot amplitude exceeds top-oil amplitude — plausibly "
      "by a factor near two at these load levels, not by the factor of four the "
      "comparison above would need.\n")
    A("2. **These units run cold.** Median `OT` is 11.4 degC on ETTh1 and 26.6 on "
      "ETTh2, against `hot_spot_mean = 86` degC in `RealisticParams`. Temperature "
      "rise scales roughly as `K^(2n)`, so a lightly loaded unit shows a small "
      "absolute swing even under a large relative load swing. The load swing in §3 "
      "transfers to a hotter unit; this temperature swing does not.\n")
    A("3. `OT` is top-oil at a sensor, subject to its own filtering and placement. "
      "It is not `theta_TO` as the model defines it.\n")
    A("The honest statement is therefore: the load-swing measurement in §3 is the "
      "one that transfers, and §5 is a flag that the thermal chain deserves the "
      "same treatment — see §7.\n")

    # ── section 6: the comparison ─────────────────────────────────────────
    A("## 6. Against `RealisticParams`, and what it does to the Jensen headline\n")
    kb = realistic_K_base()
    kbm = float(np.median(kb))
    A(f"`K_amp ~ U(0.12, 0.28)` is per-unit of rated. The sampler's realised "
      f"`K_base` has median {kbm:.3f} (Q1 {np.percentile(kb, 25):.3f}, Q3 "
      f"{np.percentile(kb, 75):.3f}, n = {len(kb)}), so the same amplitudes as a "
      f"fraction of the unit's own mean load are {0.12 / kbm:.0%} to "
      f"{0.28 / kbm:.0%}.\n")
    A("Both conventions, side by side. ETT rows are median [Q1, Q3], 24 h blocks, "
      "rated proxy `p99/0.85`.\n")
    A("| | fraction of own mean | fraction of rated |")
    A("|---|---|---|")
    A(f"| **`RealisticParams.K_amp`** | {0.12 / kbm:.0%} – {0.28 / kbm:.0%} | "
      f"**12% – 28%** |")
    for n in SERIES:
        a, b = rel_mean[n], rel_rated[(n, "p99/0.85")]
        A(f"| {n} | {np.median(a):.1%} [{np.percentile(a, 25):.1%}, "
          f"{np.percentile(a, 75):.1%}] | {np.median(b):.1%} "
          f"[{np.percentile(b, 25):.1%}, {np.percentile(b, 75):.1%}] |")
    nb = np.array(split["non-back-feeding days"])
    A(f"| ETTh1, no back-feed | — | {np.median(nb):.1%} "
      f"[{np.percentile(nb, 25):.1%}, {np.percentile(nb, 75):.1%}] |")
    A("")

    A("### Where the sampler's band sits in the real distribution\n")
    A("\"`K_amp = 0.12` is at the Nth percentile of real days.\"\n")
    A("| series | normalisation | pct at 0.12 | pct at 0.28 | real days in band |")
    A("|---|---|---|---|---|")
    for n in SERIES:
        for lbl, x, lo, hi in [
            ("fraction of rated (p99/0.85)", rel_rated[(n, "p99/0.85")], 0.12, 0.28),
            ("fraction of own mean", rel_mean[n], 0.12 / kbm, 0.28 / kbm),
        ]:
            p_lo, p_hi = 100.0 * (x < lo).mean(), 100.0 * (x < hi).mean()
            A(f"| {n} | {lbl} | {p_lo:.1f} | {p_hi:.1f} | {p_hi - p_lo:.1f}% |")
    x = nb
    A(f"| ETTh1 no back-feed | fraction of rated (p99/0.85) | "
      f"{100.0 * (x < 0.12).mean():.1f} | {100.0 * (x < 0.28).mean():.1f} | "
      f"{100.0 * ((x >= 0.12) & (x < 0.28)).mean():.1f}% |")
    A("")

    A("### Consequence for the Jensen gap\n")
    A("The gap is a function of hot-spot swing and hot-spot swing is driven by "
      "load swing, so an inflated `K_amp` inflates the headline. C-10's analytical "
      "curve:\n")
    A("| hot-spot swing | DP gap | C2H2 gap |")
    A("|---|---|---|")
    for amp, dp, c2 in [(5, 1.07, 1.14), (10, 1.29, 1.62), (15, 1.70, 2.59),
                        (20, 2.37, 4.42)]:
        A(f"| ±{amp} degC | {dp:.2f} | {c2:.2f} |")
    A("")
    A("The realistic sampler currently produces a median hot-spot swing of 11.20 "
      "degC and measured medians of DP 1.386 and C2H2 1.832 "
      "(`REALISTIC_DISTRIBUTION.md` §5). Those numbers rest entirely on `K_amp`. "
      "Halving the load amplitude, which is what matching ETTh2 or a "
      "non-back-feeding ETTh1 day would mean, moves the swing toward 5-6 degC, "
      "where the curve gives DP near 1.08 and C2H2 near 1.17 — a gap small enough "
      "that the whole argument changes character.\n")
    A("**Said plainly, since the brief asked for it: on the one conventionally "
      "loaded unit in this dataset, a real transformer swings about half what "
      "`RealisticParams` assumes, and a Jensen headline computed on the current "
      "sampler is correspondingly optimistic.** On the one unit with midday "
      "photovoltaic reversal it is not — that unit sits inside the band on quiet "
      "days and above it on sunny ones — and that is the population the manuscript "
      "claims to address.\n")
    A("Two units cannot settle which of those the benchmark should be. What they "
      "do settle is that the current `K_amp` is defensible only with a scope "
      "statement the paper has not made, and that quoting a Jensen gap without one "
      "is quoting the flattering half of a two-point sample. §7 is about making "
      "the statement rather than picking the half.\n")

    # ── section 7 ─────────────────────────────────────────────────────────
    A("## 7. What is not settled, and what this does not establish\n")
    A("1. **No parameter has been changed.** The brief said report first. Setting "
      "`K_amp` from this would mean choosing which population the benchmark is "
      "about, and that is a scope decision, not a calibration one.\n")
    A("2. **No nameplate rating.** §3b rests on a proxy; §3a does not, which is "
      "why it is primary. The `p99` and `max` rows are there so the sensitivity is "
      "visible.\n")
    A("3. **Two units, not a fleet.** Two transformers in two Chinese counties over "
      "one two-year span bound nothing. The spread *between* two units is already "
      "as large as the effect being measured, which is itself the finding.\n")
    A("4. **These are distribution transformers.** The manuscript's framing is "
      "cycling driven by renewables. ETTh1 turns out to fit that framing well and "
      "ETTh2 does not, but neither is a transmission-connected unit.\n")
    A("5. **The thermal chain is untested against data and now clearly should be.** "
      "ETT gives `OT` alongside the load that produced it, which is enough to test "
      "`tau_oil`, `DTheta_oil_R` and `n_exp` directly instead of assuming them from "
      "IEC defaults — the same class of gap as O-3 on the gas kinetics. §5 is the "
      "reason to do it: measured top-oil swing is well below what the sampler's "
      "assumed thermal parameters imply, and that discrepancy is currently "
      "unexplained. Larger than O-10 and deliberately not attempted here.\n")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    for n in SERIES:
        print(f"{n}: 24h/own-mean median {np.median(rel_mean[n]):.3f}  "
              f"24h/rated median {np.median(rel_rated[(n, 'p99/0.85')]):.3f}  "
              f"12h/own-mean median {np.median(rel12[n]):.3f}  "
              f"OT amplitude median {np.median(ot_amp[n]):.2f} degC")
    print(f"ETTh1 non-back-feed days /rated median {np.median(nb):.3f}")
    print(f"K_base median {kbm:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
