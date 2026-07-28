# O-10 — the daily load swing, measured against ETT

`RealisticParams.K_amp = (0.12, 0.28)` was set to hit a target hot-spot swing of 10-15 degC, not measured from a fleet. `cod/data/realistic.py` says so and calls it "at the upper end of what a real feeder does". This checks that sentence against two years of hourly load from two operating transformers.

**Report only. `RealisticParams` is not touched** — see §7 for what would have to be decided first.

## Summary

The two ETT units disagree with each other by as much as either disagrees with `RealisticParams`, and that is the result.

Median daily swing as a fraction of rated (`p99/0.85` proxy, §3b), against `K_amp`'s 12-28%:

| | n days | median | below the band | inside it | above it |
|---|---|---|---|---|---|
| ETTh2, all days | 725 | **8.7%** | 85.2% | 14.8% | 0.0% |
| ETTh1, non-back-feeding days | 320 | **17.8%** | 15.6% | 77.5% | 6.9% |
| ETTh1, all days | 722 | **24.8%** | 6.9% | 56.1% | 37.0% |
| ETTh1, back-feeding days | 402 | **29.7%** | 0.0% | 39.1% | 60.9% |

* **ETTh2 swings roughly half what `RealisticParams` assumes.** 85% of its days fall below even the bottom of the band, and not one of its 725 days reaches the top.

* **ETTh1 sits inside the band, and above it whenever the sun is out.** Its net load reverses at midday in spring — 51% of noon hours have negative active power, concentrated in March-June. That is photovoltaic back-feed, i.e. exactly the renewable-driven cycling the manuscript opens on.

* **The PV explains part of the gap between the units, not all of it.** Removing back-feeding days takes ETTh1 from 24.8% to 17.8%, so roughly 7 of the 16 points separating it from ETTh2. The remaining 9 points are simply two different feeders. Any single calibrated `K_amp` is a claim about which feeder the benchmark is about.

* **Both units' measured top-oil swings are far below the 10-15 degC the sampler targets at the hot spot** — medians of 2.39 and 5.60 degC in amplitude. §5 gives three reasons that is suggestive rather than decisive, and it is the number most likely to be flattering the sampler.

So `K_amp = 0.12-0.28` is not uniformly too high. It is about right for a PV-back-fed feeder on a quiet day, low for one on a sunny day, and about double for a conventionally loaded unit. The consequence for the Jensen headline is in §6: it cannot be a single number, which is what C-10 already concluded on other grounds.

## 1. Data and the three choices that had to be made

`github.com/zhouhaoyi/ETDataset`, `ETT-small/ETTh1.csv` and `ETTh2.csv`. Two separate units, 17420 hourly rows each, 2016-07-01 to 2018-06-26. Columns are High/Middle/Low UseFul and UseLess load — active and reactive power by customer class — plus `OT`, the oil temperature. Fetch:

```
python -c "import urllib.request as u;[u.urlretrieve('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'+f,'data/ett/'+f) for f in ('ETTh1.csv','ETTh2.csv')]"
```

**Never pooled.** ETTh1 and ETTh2 are different transformers in different counties and §2 shows they do not behave alike; a pooled number would hide the only interesting thing in this dataset.

**Apparent power, not active power.** IEC 60076-7's load factor K is a current ratio and losses go as I^2, so what a thermal model responds to is `|S| = sqrt(P^2 + Q^2)` summed over the three classes. It is also the only combination that stays non-negative, which matters because ETTh1's active power is negative in 13.3% of hours.

**Peak-to-trough is twice the amplitude.** `K = K_base + K_amp*sin(...)`, so `K_amp` is compared against **half** the peak-to-trough range throughout. Getting this backwards doubles the answer.

**Outage screening.** Blocks containing an hour with every load channel at exactly zero are dropped as meter outages rather than unloaded transformers:

| series | 24 h blocks kept | dropped | 12 h blocks kept | dropped |
|---|---|---|---|---|
| ETTh1 | 722 | 3 | 1445 | 6 |
| ETTh2 | 725 | 0 | 1451 | 0 |

## 2. ETTh1 back-feeds at midday; ETTh2 never does

This is not a data defect and it changes how §3 reads, so it comes first.

Fraction of hours with negative total active power:

| series | overall | by hour of day (00, 03, 06, 09, 12, 15, 18, 21) |
|---|---|---|
| ETTh1 | 13.3% | 0%, 0%, 0%, 28%, 51%, 38%, 0%, 0% |
| ETTh2 | 0.0% | 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0% |

| series | by month (Jan…Dec) |
|---|---|
| ETTh1 | 7%, 15%, 18%, 24%, 22%, 23%, 10%, 9%, 8%, 9%, 9%, 8% |
| ETTh2 | 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0%, 0% |

ETTh1: zero at night, 51% at noon, peaking March-June. A midday-only, spring-peaking reversal is photovoltaic back-feed and nothing else. ETTh2 has no negative hour in two years (min total active power 11.3).

The consequence for the numbers below: ETTh1's net load passes through zero most spring middays, so its *relative* range is mechanically large. That is a real thermal duty — the unit genuinely unloads and reloads — but it is a different regime from a feeder whose load merely rises and falls, and the two should not be averaged into one calibration.

## 3. Daily load swing

Per 24 h block, `(max - min) / (2 * denominator)` of `|S|`. Two denominators, because the two answer different questions and `K_amp` is written in the second.

### 3a. As a fraction of the day's own mean load

Needs no assumption about the unit's rating.

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 722 | 28.7% | 42.6% | **62.2%** | 75.0% | 83.5% | 59.0% | 123.0% |
| ETTh2 | 725 | 11.1% | 14.0% | **17.4%** | 22.5% | 28.7% | 18.8% | 70.7% |

**This normalisation breaks down on ETTh1** and the 62% median is not a usable number. Its net load spends midday near zero (§2), which depresses the daily mean at the same time as it widens the range, so the ratio is inflated at both ends. That is why §3b exists and why the rated normalisation is the one carried into §6. On ETTh2, which never back-feeds, the two normalisations agree in ordering and 3a is the more trustworthy of them.

### 3b. As a fraction of rated load

`K` is per-unit of rated, so `K_amp = 0.12` means twelve percent **of rated**, not of the day's mean. ETT publishes no nameplate rating, so this needs a proxy and the answer moves with it. All three are shown rather than one being chosen silently.

`rated = p99` asserts the unit reaches nameplate in 1% of hours, which is aggressive; real units peak nearer 0.7-0.9 pu, and assuming a lower peak loading makes rated *larger* and the normalised swing *smaller*. The middle row is the most defensible and is the one quoted elsewhere in this file.


**rated = p99(|S|) — the peak hour is nameplate**

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 722 | 16.0% | 21.8% | **29.1%** | 36.6% | 41.4% | 28.7% | 57.0% |
| ETTh2 | 725 | 6.4% | 8.3% | **10.2%** | 12.4% | 15.2% | 10.5% | 31.7% |

**rated = p99(|S|)/0.85 — peak loading is 0.85 pu**

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 722 | 13.6% | 18.5% | **24.8%** | 31.2% | 35.2% | 24.4% | 48.5% |
| ETTh2 | 725 | 5.5% | 7.0% | **8.7%** | 10.6% | 13.0% | 8.9% | 27.0% |

**rated = max(|S|) over the full record**

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 722 | 12.8% | 17.4% | **23.3%** | 29.4% | 33.2% | 23.0% | 45.7% |
| ETTh2 | 725 | 4.0% | 5.2% | **6.4%** | 7.8% | 9.6% | 6.6% | 20.0% |

### 3c. ETTh1 split by whether that day back-fed

How much of ETTh1's swing is the PV, rather than ordinary load following.

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1, back-feeding days | 402 | 21.8% | 25.3% | **29.7%** | 33.6% | 36.8% | 29.6% | 48.5% |
| ETTh1, non-back-feeding days | 320 | 10.1% | 14.0% | **17.8%** | 22.2% | 26.6% | 17.8% | 38.3% |

Strip the back-feed and ETTh1's median falls from 24.8% to 17.8%. That is 7.0 of the 16.1 points separating the two units — **the photovoltaic reversal explains under half the difference.** ETTh1 on a day with no back-feed at all still swings 2.0x ETTh2. The rest is not distributed generation, it is simply that these are two different feeders serving two different mixes of customer.

This is the part that resists a single calibrated number. Had the PV accounted for the whole gap, `K_amp` could have been set from the conventional baseline with a documented uplift for renewable duty. It does not, so the between-feeder spread is irreducible at this sample size, and two units cannot estimate it.

## 4. The benchmark window is 12 hours, and that compounds it

`TW = 720` min. The `daily` family in `make_realistic_profile` puts a **full** sine period inside that window, so a case labelled "daily" completes an entire load cycle in twelve hours. A real daily cycle takes twenty-four, so a twelve-hour slice of real data contains about half of one.

Same statistic on non-overlapping 12 h blocks, relative to the block's own mean, against 3a:

| series | n | p10 | Q1 | median | Q3 | p90 | mean | max |
|---|---|---|---|---|---|---|---|---|
| ETTh1, 12 h | 1445 | 19.2% | 27.5% | **48.8%** | 65.4% | 77.3% | 48.2% | 131.0% |
| ETTh2, 12 h | 1451 | 6.0% | 8.8% | **11.5%** | 15.6% | 22.0% | 12.9% | 56.6% |

- ETTh1: median 12 h swing is 0.79x the median 24 h swing (48.8% against 62.2%).
- ETTh2: median 12 h swing is 0.66x the median 24 h swing (11.5% against 17.4%).

The mismatch therefore compounds. The sampler asks for an amplitude at or above the top of the real range **and** completes a full cycle of it in half the time a real cycle takes. A 12 h window of real data sees appreciably less than a full swing; a 12 h window of sampled data sees a whole period.

## 5. The oil temperature ETT actually recorded

Beyond what O-10 asked for, and worth having: `OT` is a real thermal response to this real load, so it measures the quantity `K_amp` exists to produce rather than its input.

Daily top-oil peak-to-trough, and half of it, in degC:

| series | n | p10 | Q1 | median | Q3 | p90 | max |
|---|---|---|---|---|---|---|---|
| ETTh1 peak-to-trough | 722 | 2.39 | 3.59 | **4.78** | 6.54 | 8.30 | 18.50 |
| ETTh1 amplitude | 722 | 1.20 | 1.79 | **2.39** | 3.27 | 4.15 | 9.25 |
| ETTh2 peak-to-trough | 725 | 3.60 | 6.59 | **11.21** | 15.16 | 17.80 | 26.81 |
| ETTh2 amplitude | 725 | 1.80 | 3.30 | **5.60** | 7.58 | 8.90 | 13.40 |

Against a sampler targeting a **hot-spot** amplitude of 10-15 degC, the measured **top-oil** amplitudes are 2.39 and 5.60 degC at the median, and 4.17 and 8.90 at p90.

**Three reasons this is suggestive and not decisive, stated because the comparison is not apples to apples and the gap is large enough to be worth attacking properly:**

1. Hot-spot swings more than top-oil. The gradient `DTheta_HS_R * ((1 + K^2 R)/(1 + R))^m_exp` moves with the load and in phase with it, so hot-spot amplitude exceeds top-oil amplitude — plausibly by a factor near two at these load levels, not by the factor of four the comparison above would need.

2. **These units run cold.** Median `OT` is 11.4 degC on ETTh1 and 26.6 on ETTh2, against `hot_spot_mean = 86` degC in `RealisticParams`. Temperature rise scales roughly as `K^(2n)`, so a lightly loaded unit shows a small absolute swing even under a large relative load swing. The load swing in §3 transfers to a hotter unit; this temperature swing does not.

3. `OT` is top-oil at a sensor, subject to its own filtering and placement. It is not `theta_TO` as the model defines it.

The honest statement is therefore: the load-swing measurement in §3 is the one that transfers, and §5 is a flag that the thermal chain deserves the same treatment — see §7.

## 6. Against `RealisticParams`, and what it does to the Jensen headline

`K_amp ~ U(0.12, 0.28)` is per-unit of rated. The sampler's realised `K_base` has median 0.879 (Q1 0.786, Q3 0.966, n = 2000), so the same amplitudes as a fraction of the unit's own mean load are 14% to 32%.

Both conventions, side by side. ETT rows are median [Q1, Q3], 24 h blocks, rated proxy `p99/0.85`.

| | fraction of own mean | fraction of rated |
|---|---|---|
| **`RealisticParams.K_amp`** | 14% – 32% | **12% – 28%** |
| ETTh1 | 62.2% [42.6%, 75.0%] | 24.8% [18.5%, 31.2%] |
| ETTh2 | 17.4% [14.0%, 22.5%] | 8.7% [7.0%, 10.6%] |
| ETTh1, no back-feed | — | 17.8% [14.0%, 22.2%] |

### Where the sampler's band sits in the real distribution

"`K_amp = 0.12` is at the Nth percentile of real days."

| series | normalisation | pct at 0.12 | pct at 0.28 | real days in band |
|---|---|---|---|---|
| ETTh1 | fraction of rated (p99/0.85) | 6.9 | 63.0 | 56.1% |
| ETTh1 | fraction of own mean | 2.1 | 12.9 | 10.8% |
| ETTh2 | fraction of rated (p99/0.85) | 85.2 | 100.0 | 14.8% |
| ETTh2 | fraction of own mean | 23.3 | 93.4 | 70.1% |
| ETTh1 no back-feed | fraction of rated (p99/0.85) | 15.6 | 93.1 | 77.5% |

### Consequence for the Jensen gap

The gap is a function of hot-spot swing and hot-spot swing is driven by load swing, so an inflated `K_amp` inflates the headline. C-10's analytical curve:

| hot-spot swing | DP gap | C2H2 gap |
|---|---|---|
| ±5 degC | 1.07 | 1.14 |
| ±10 degC | 1.29 | 1.62 |
| ±15 degC | 1.70 | 2.59 |
| ±20 degC | 2.37 | 4.42 |

The realistic sampler currently produces a median hot-spot swing of 11.20 degC and measured medians of DP 1.386 and C2H2 1.832 (`REALISTIC_DISTRIBUTION.md` §5). Those numbers rest entirely on `K_amp`. Halving the load amplitude, which is what matching ETTh2 or a non-back-feeding ETTh1 day would mean, moves the swing toward 5-6 degC, where the curve gives DP near 1.08 and C2H2 near 1.17 — a gap small enough that the whole argument changes character.

**Said plainly, since the brief asked for it: on the one conventionally loaded unit in this dataset, a real transformer swings about half what `RealisticParams` assumes, and a Jensen headline computed on the current sampler is correspondingly optimistic.** On the one unit with midday photovoltaic reversal it is not — that unit sits inside the band on quiet days and above it on sunny ones — and that is the population the manuscript claims to address.

Two units cannot settle which of those the benchmark should be. What they do settle is that the current `K_amp` is defensible only with a scope statement the paper has not made, and that quoting a Jensen gap without one is quoting the flattering half of a two-point sample. §7 is about making the statement rather than picking the half.

## 7. What is not settled, and what this does not establish

1. **No parameter has been changed.** The brief said report first. Setting `K_amp` from this would mean choosing which population the benchmark is about, and that is a scope decision, not a calibration one.

2. **No nameplate rating.** §3b rests on a proxy; §3a does not, which is why it is primary. The `p99` and `max` rows are there so the sensitivity is visible.

3. **Two units, not a fleet.** Two transformers in two Chinese counties over one two-year span bound nothing. The spread *between* two units is already as large as the effect being measured, which is itself the finding.

4. **These are distribution transformers.** The manuscript's framing is cycling driven by renewables. ETTh1 turns out to fit that framing well and ETTh2 does not, but neither is a transmission-connected unit.

5. **The thermal chain is untested against data and now clearly should be.** ETT gives `OT` alongside the load that produced it, which is enough to test `tau_oil`, `DTheta_oil_R` and `n_exp` directly instead of assuming them from IEC defaults — the same class of gap as O-3 on the gas kinetics. §5 is the reason to do it: measured top-oil swing is well below what the sampler's assumed thermal parameters imply, and that discrepancy is currently unexplained. Larger than O-10 and deliberately not attempted here.

