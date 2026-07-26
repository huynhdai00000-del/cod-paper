"""Provenance capture for every run.

Answers the question the audit could not: which run produced this number.
Every result file carries the git commit, the config hash, the seed, and the
environment it ran in.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Provenance:
    timestamp_utc: str
    git_commit: str | None
    git_dirty: bool | None
    git_branch: str | None
    hostname: str
    platform: str
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    gpu_name: str | None
    numpy_version: str | None
    scipy_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def capture() -> Provenance:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    torch_version = cuda_version = gpu_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    numpy_version = scipy_version = None
    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        pass
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        pass

    return Provenance(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=(bool(status) if status is not None else None),
        git_branch=branch,
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        numpy_version=numpy_version,
        scipy_version=scipy_version,
    )


def write_run_record(
    out_dir: str | Path,
    config_summary: dict,
    seed: int,
    extra: dict | None = None,
) -> Path:
    """Write ``run.json`` next to the results of a run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "provenance": capture().to_dict(),
        "config": config_summary,
        "seed": seed,
    }
    if extra:
        record.update(extra)

    path = out_dir / "run.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    return path


def warn_if_dirty() -> None:
    """Print a warning if the working tree has uncommitted changes.

    A result produced from a dirty tree cannot be reproduced from the commit
    hash alone, which is exactly how the audit lost track of which code
    produced which number.
    """
    prov = capture()
    if prov.git_commit is None:
        print("[provenance] WARNING: not a git repository, results are not traceable")
    elif prov.git_dirty:
        print(
            "[provenance] WARNING: uncommitted changes present. "
            "Commit before a run whose numbers will be reported."
        )
