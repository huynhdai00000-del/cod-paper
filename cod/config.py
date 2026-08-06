"""Config loading with a freeze mechanism.

The audit found that the training distribution was widened repeatedly in
response to test metrics, and that three different numbers existed for the
same experiment with no way to tell which config produced which. This module
makes that impossible to do accidentally: every config is hashed, and the hash
travels with every result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def canonical_hash(obj: Any) -> str:
    """Stable hash of a nested dict, independent of key order.

    Uses sorted keys and a fixed separator so that reformatting the YAML file
    does not change the hash, but changing any value does.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Config:
    """An immutable experiment configuration.

    Attributes
    ----------
    raw
        The parsed YAML contents.
    path
        Where the config was loaded from.
    hash
        Canonical hash of the whole config.
    distribution_hash
        Canonical hash of the ``distribution`` block alone. Reported separately
        so that a claim of the form "the training distribution was frozen
        before any model was trained" can be checked against the record.
    """

    raw: dict
    path: Path
    hash: str
    distribution_hash: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def summary(self) -> dict:
        #: `experiment.variant` is the cell identity and it has to travel in the
        #: record, not only in the directory name. `model.kind` cannot stand in
        #: for it: the factorial's with-baseline cells share a `kind` with their
        #: without-baseline partners (`fno_in_cascade` is both
        #: `fno_in_cascade` and `fno_baseline_in_cascade`), and
        #: `cod_bounded_correction` shares `kind: cod` with COD itself. Without
        #: the variant an aggregator can only recover the cell by re-hashing the
        #: local config tree, which fails the moment a result is read on a
        #: machine where that file has moved (PORT_LOG J-92, J-94).
        return {
            "config_path": str(self.path),
            "config_hash": self.hash,
            "distribution_hash": self.distribution_hash,
            "experiment_name": self.raw["experiment"].get("name"),
            "variant": self.raw["experiment"].get("variant"),
        }


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    required = {"experiment", "distribution", "model", "training", "evaluation"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{path} is missing required blocks: {sorted(missing)}")

    return Config(
        raw=raw,
        path=path,
        hash=canonical_hash(raw),
        distribution_hash=canonical_hash(raw["distribution"]),
    )


def assert_distribution_unchanged(cfg: Config, expected_hash: str) -> None:
    """Fail loudly if the training distribution has been edited.

    Call this in any script that is meant to run against a frozen
    distribution. Pass the hash that was recorded when the distribution was
    first frozen. If somebody widens a sampling range to chase a test metric,
    this raises instead of silently producing a better number.
    """
    if cfg.distribution_hash != expected_hash:
        raise RuntimeError(
            "Training distribution has changed since it was frozen.\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {cfg.distribution_hash}\n"
            "If the change is intentional, record it in CHANGELOG_DISTRIBUTION.md "
            "with the date and reason, then update the frozen hash. Every such "
            "change must be disclosed in the paper."
        )
