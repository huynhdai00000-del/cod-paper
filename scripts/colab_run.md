# Running one config on Colab

Training happens on Colab, not on the development machine (DECISIONS C-5). The
local machine has 4 CPU cores and no GPU: one 25,000-epoch run there is 12-22
hours, against roughly an hour on a T4. Locally you write code and smoke-test the
wiring; Colab does the real runs.

The pattern below is designed so **several accounts can run different configs in
parallel against one shared Drive folder**. Each run writes into its own
subdirectory named after the config hash and seed, so nothing collides.

---

## One cell, start to finish

Runtime → Change runtime type → **T4 GPU** first. Then:

```python
# 1. Mount the shared Drive folder
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/COD_project/results'      # shared across accounts

# 2. Get the code. Pin a commit so the run is reproducible.
!git clone -q https://github.com/<user>/cod-paper.git /content/cod-paper
%cd /content/cod-paper
!git checkout -q <COMMIT_SHA>        # omit for tip of main, but then record the sha
!pip install -q -e .

# 3. Confirm the GPU is actually there before spending an hour
import torch
assert torch.cuda.is_available(), 'No GPU: change the runtime type first'
print(torch.cuda.get_device_name(0), '| torch', torch.__version__)

# 4. Run one config
!python -u scripts/run.py \
    --config configs/example_cod_seed1.yaml \
    --out "{OUT}" \
    --device cuda
```

`run.py` writes `run.json` and `loss_history.json` into

```
<OUT>/<experiment.name>_<experiment.variant>_s<seed>_<config_hash>/
```

so two accounts running different configs, or the same config at different seeds,
never overwrite each other. Two accounts running the *same* config at the *same*
seed would — see "Parallel runs" below.

---

## Splitting work across accounts

Give each account a different config file, or the same file with a different seed.
The output directory carries both, so the shared folder self-organises.

| account | command |
|---|---|
| A | `--config configs/example_cod_seed1.yaml` |
| B | `--config configs/example_cod_seed2.yaml` |
| C | `--config configs/example_cod_seed3.yaml` |

To vary only the seed without writing three files, edit `training.seed` in a copy.
Do **not** pass a seed on the command line — the seed belongs in the config so that
the config hash changes with it and the run record stays self-describing.

For the C-11 baseline matrix, one config per (architecture, configuration, seed):
six architecture/configuration pairs × 3 seeds = 18 configs. At roughly an hour
each that is about 18 GPU-hours, inside Kaggle's 30 h/week, and it parallelises
across accounts with no coordination beyond distinct config files.

---

## Before a run whose numbers will be reported

1. **Commit and push first.** `run.py` calls `provenance.warn_if_dirty()` and
   records `git_dirty` in `run.json`. A result from a dirty tree cannot be traced
   back to code, which is exactly how the audit lost track of which run produced
   which number.
2. **Pin the commit** in the clone step and note the sha, so the run can be
   reproduced later even after main moves.
3. **Freeze the distribution.** Once `DISTRIBUTION_FREEZE.md` exists, pass
   `--freeze-hash <hash>`; the run aborts if the `distribution` block has changed.
   Phase 2 fixes 1, 4 and 5 all changed the training distribution, so the frozen
   hash has to be re-established before the first real retrain.

---

## Cost, and why it is what it is

Per-epoch cost on a T4 was about 0.11 s for the v57 model. Phase 2 fix 1 replaced
a closed-form steady state with a contraction solve, which cost 2.6x per epoch
until the solve was cached at dataset generation; it is now about 1.5x
(`audit_port/scripts/11_check_ss_cache.py` measures this, and asserts the cached
forward pass is bit-identical to the uncached one).

So budget roughly **1.5x the old wall clock** for a full run. The residual overhead
is the one place `theta_ss` has to stay differentiable in `t` — the query-time value
feeding the `driving` trunk feature — and removing it would change the model rather
than just cache it.

If a run is going to exceed the session limit, do not silently shorten it. Pass
`--max-wall-seconds` so the harness stops cleanly and records
`stop_reason='wall_clock_budget'` with `converged=False`. A truncated run reported
as a performance number is the failure mode this whole harness exists to prevent.

---

## Smoke test, on any machine

Proves the wiring without training anything. Takes a couple of minutes on CPU:

```bash
python scripts/run.py --config configs/example_cod_seed1.yaml \
    --max-epochs 100 --n-ic 50 --device cpu
```

The output directory gains a `_smoke` suffix and `run.json` records
`status: smoke_test`, so a smoke run can never be mistaken for a real one.

---

## Retrieving results

Everything needed to interpret a run is in its `run.json`: commit sha, config hash,
distribution hash, seed, device, library versions, `converged`, `stop_reason`,
`fair_comparison_candidate`, the pathology report (clamp fractions, causal-weight
minimum), NMAE per state, and absolute MAE in physical units with per-state
floor-hit rates. `loss_history.json` holds the full learning curve.

Read them locally with:

```python
import json, glob
for p in sorted(glob.glob('results/*/run.json')):
    d = json.load(open(p, encoding='utf-8'))
    print(d['config']['config_hash'], d['seed'],
          d['outcome']['stop_reason'],
          f"converged={d['outcome']['converged']}",
          f"theta_TO MAE={d['evaluation']['mae_physical_units']['theta_TO']['mae']:.3f} degC")
```
