# Distribution changelog

Every change to a hashed `distribution` block, with the date, the reason, and the
hashes either side of it. Referenced by `cod/config.py`'s
`assert_distribution_unchanged`, whose error message tells you to come here.

A change recorded here must also be disclosed in the paper. The point of the
protocol is that widening a sampling range to chase a metric cannot be done
quietly; it can still be done, but it leaves this trail.

---

## 2026-07-29 — put the realistic sampler on the config path

| config | distribution hash before | after |
|---|---|---|
| `configs/example_cod_seed1.yaml` | `9bf8b092546cfa30` | `fc4cb76c3b32ec17` |
| `configs/v57_faithful.yaml` | `7a98d381e402e7c8` | `3ad5f68876934c75` |

**Reason.** The first freeze, recorded earlier the same day, hashed a block the
training path largely did not read. `run.py` called `generate_training_set`,
which takes `seed`, `steady_state_formula`, a phase boolean and one flag derived
from `profile_families`; the block's `K_base`, `ambient_base`,
`ambient_amplitude` and nine family definitions reached no sampler at all, while
the ranges that actually applied were hardcoded in `cod/data/profiles.py`.
Editing `K_base` in the YAML moved the hash and changed no data.

Worse, `cod/data/realistic.py` — the fix-7 sampler that every Jensen-gap number
in `audit_port/` is computed on — was not on the config path in any form. Training
would have used one distribution and the paper would have reported from another,
which is the mismatch the audit found in the original manuscript.

**What changed.** `distribution.sampler` now names the sampler inside the hashed
block. For `kind: realistic`, `params` must list every field of
`RealisticParams`; `from_config` rejects a config with a field missing and
equally one with a key that is not a field. `configs/v57_faithful.yaml` declares
`kind: v57` and takes no params, because its ranges must stay immutable for the
Phase 1 reproduction gates.

**Nothing had been trained on the superseded hashes.** Both smoke runs against
them were budget-capped wiring tests that reported NOT CONVERGED, so no result
depends on them. `9bf8b092546cfa30` and `7a98d381e402e7c8` are recorded here for
the audit trail, not because anything was scored under them.

**Verified.** `audit_port/scripts/20_verify_config_binding.py`: editing
`cycle_period` from 1440 to 720 changes `sensors` by up to 15.17 and `x0s` by up
to 8.44, moves the median load swing from 0.1045 to 0.1261, and moves the
distribution hash to `cefe1e0e2f9251dd`. `scripts/verify_phase1.py --gate 1`
still reproduces Table 2 on the v57 path.
