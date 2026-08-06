"""What experimental cell a config is, derived from the config itself.

The C-11 matrix is a 2x2 factorial per architecture — cascade x analytic
baseline — plus two COD-family variants that sit outside it. Which cell a run
belongs to decides which table it lands in and which comparison it takes part
in, so it must be derived from the same fields `run.py` reads to build the
model, not from a directory name and not from a lookup table maintained
separately.

Two ways of guessing the cell that this module exists to make unnecessary:

  * **`model.kind` is not the cell.** `fno_in_cascade` and
    `fno_baseline_in_cascade` share it, and so do `cod` and
    `cod_bounded_correction`. Grouping runs by `model_kind` silently pools two
    factorial cells into one.
  * **The directory name is not the cell.** `run.py` builds it as
    `{name}_{variant}_s{seed}_{hash}[_smoke][_tag]`, and both `name` and
    `variant` may contain underscores, so it does not parse back unambiguously.

PORT_LOG J-92 requires the with-baseline cells to be labelled as hybrids —
"FNO + IEC delta-learning", never "FNO" — because reporting a hybrid under the
source paper's name attributes our modification to their published method.
`CellFactors.label` is that name, computed once here so no table can lose it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

#: model kind -> the published architecture it instantiates. PI-DeepONet covers
#: COD, Ablation A and the MonolithicFair family: all four are the same branch /
#: trunk Modified-MLP operator, differing in the two factors below.
_ARCH = {
    "fno_monolithic": "FNO",
    "fno_in_cascade": "FNO",
    "mionet_monolithic": "MIONet",
    "mionet_in_cascade": "MIONet",
    "sdeeponet_monolithic": "S-DeepONet",
    "sdeeponet_in_cascade": "S-DeepONet",
    "cod": "PI-DeepONet",
    "cod_no_baseline": "PI-DeepONet",
    "monolithic_fair": "PI-DeepONet",
    "monolithic_multihead": "PI-DeepONet",
    "monolithic_softic": "PI-DeepONet",
}

#: Whether the gases come from the Arrhenius quadrature (`in_cascade`) or from
#: the network directly (`monolithic`). COD and Ablation A are cascade models.
_CASCADE = {
    "fno_monolithic": False, "fno_in_cascade": True,
    "mionet_monolithic": False, "mionet_in_cascade": True,
    "sdeeponet_monolithic": False, "sdeeponet_in_cascade": True,
    "cod": True, "cod_no_baseline": True,
    "monolithic_fair": False, "monolithic_multihead": False,
    "monolithic_softic": False,
}

#: The four factorial cells, in the numbering PORT_LOG J-92 uses.
_CELL_NUMBER = {(False, True): 1,    # monolithic, with baseline
                (False, False): 2,   # monolithic, without
                (True, True): 3,     # in-cascade, with baseline
                (True, False): 4}    # in-cascade, without


@dataclass(frozen=True)
class CellFactors:
    """Where one config sits in the factorial."""
    variant: str            #: `experiment.variant`; the cell's identity
    model_kind: str
    architecture: str
    cascade: bool           #: factor A: gases by quadrature
    baseline: bool          #: factor B: the IEC analytic baseline H(t)
    bounded_correction: bool  #: ANALYSIS_PLAN Amendment 2; NOT a factor
    cell_number: int        #: 1-4, per J-92
    label: str              #: what a table is allowed to call it
    in_factorial: bool      #: False for the Amendment-2 variant

    def to_dict(self) -> dict:
        return asdict(self)


def cell_factors(raw: dict) -> CellFactors:
    """Read the factor levels out of a parsed config.

    `raw` is the whole config mapping, i.e. `Config.raw`.
    """
    model = raw.get("model", {})
    kind = model.get("kind")
    if kind not in _ARCH:
        raise ValueError(
            f"cell_factors: unknown model kind {kind!r}. Add it to cod/cells.py "
            "rather than letting a run fall into no cell — a run whose cell is "
            "unknown must be reported as unresolved, never pooled by guesswork.")

    variant = raw.get("experiment", {}).get("variant")
    if not variant:
        raise ValueError("cell_factors: experiment.variant is missing; it is "
                         "the cell identity")

    cascade = _CASCADE[kind]
    # `cod_no_baseline` IS the no-baseline level by construction — it replaces
    # H(t) with the constant x0 — and carries no `use_baseline` key. Everything
    # else declares the factor explicitly, defaulting to the published form of
    # the architecture, which for every tier-1 baseline is "no analytic
    # baseline" and for COD is "has one".
    if kind == "cod_no_baseline":
        baseline = False
    elif kind == "cod":
        baseline = True
    else:
        baseline = bool(model.get("use_baseline", False))
    bounded = bool(model.get("bounded_correction", False))

    arch = _ARCH[kind]
    # J-92: the hybrid is not the source architecture, and the name says so.
    if baseline and arch != "PI-DeepONet":
        label = f"{arch} + IEC delta-learning"
    elif kind == "cod":
        label = "COD (bounded)" if bounded else "COD"
    elif kind == "cod_no_baseline":
        label = "Ablation A (COD, no baseline H)"
    elif arch == "PI-DeepONet":
        label = "PI-DeepONet + IEC delta-learning" if baseline else "PI-DeepONet"
    else:
        label = arch

    return CellFactors(
        variant=variant, model_kind=kind, architecture=arch, cascade=cascade,
        baseline=baseline, bounded_correction=bounded,
        cell_number=_CELL_NUMBER[(cascade, baseline)], label=label,
        # Amendment 2 is reported against COD on the same seeds, not as a fifth
        # factorial variable; folding it in would double the matrix to test a
        # design refinement rather than the hypothesis.
        in_factorial=not bounded,
    )
