#!/usr/bin/env python3
"""Single entry point for every experiment.

Usage:
    python scripts/run.py --config configs/example_cod_seed1.yaml --out results/
    python scripts/run.py --config configs/example_cod_seed1.yaml \
        --max-epochs 100 --n-ic 50 --device cpu       # wiring smoke test

Reads a config, builds the model and the dataset from it, trains through the
shared harness, evaluates, and writes `run.json` carrying the commit hash, the
config hash, the distribution hash, the seed and the environment.

Two things this deliberately does NOT do:

  * report a performance number for a model that did not converge. The harness
    returns `converged=False` with a `stop_reason`, and that is written into
    `run.json` and printed. A non-converged run is reported as non-converged.
  * treat NMAE as the headline. `cod/eval/metrics.py` reports absolute error in
    physical units first, with the fraction of cases where the normalisation
    denominator hit its floor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cod.config import load_config
from cod.data.generate import (build_realistic_test_set, build_test_set,
                               generate_realistic_training_set,
                               generate_training_set, load_training_set)
from cod.data.physics import STATE_DIM_FAST, STATE_NAMES_FAST
from cod.data.realistic import RealisticParams
from cod.data.steady_state import formula_A, true_fixed_point_np
from cod.eval.benchmark import evaluate
from cod.eval.metrics import TRANSFORMER_STATES, evaluate_state, report
from cod.models.cod import CODOperator, cod_predict
from cod.models.monolithic import (
    MonolithicFair,
    MonolithicMultiHead,
    MonolithicSoftIC,
    mono_predict,
)
from cod.provenance import warn_if_dirty, write_run_record
from cod.training.harness import ConvergenceCriterion, train
from cod.training.train import CODTrainer, SharedPhysicsTrainer

MODEL_BUILDERS = {
    "cod": CODOperator,
    "monolithic_fair": MonolithicFair,
    "monolithic_multihead": MonolithicMultiHead,
    "monolithic_softic": MonolithicSoftIC,
}


def build_model(cfg, x_mean, x_std, device):
    """Instantiate the model named by `cfg['model']['kind']`."""
    m = cfg["model"]
    kind = m["kind"]
    if kind not in MODEL_BUILDERS:
        raise ValueError(f"unknown model kind {kind!r}; "
                         f"expected one of {sorted(MODEL_BUILDERS)}")

    d_h = int(m.get("branch", {}).get("width", 128))
    p = int(m.get("basis_dim", 64))
    n_layers = int(m.get("branch", {}).get("layers", 4))
    n_exp_feats = int(m.get("n_exp_feats", 12))
    n_sensors = int(cfg["distribution"].get("n_sensors", 100))
    T = float(cfg["distribution"]["window_minutes"])

    if kind == "cod":
        model = CODOperator(
            state_dim=STATE_DIM_FAST, n_sensors=n_sensors, d_h=d_h, p=p,
            n_layers=n_layers, n_exp_feats=n_exp_feats, T=T,
            x_mean=x_mean, x_std=x_std,
            theta_ss_mode=m.get("steady_state", "true_fixed_point"),
        )
        return model, cod_predict

    model = MODEL_BUILDERS[kind](d_h=d_h, p=p, n_layers=n_layers,
                                n_exp=n_exp_feats, x_mean=x_mean, x_std=x_std)
    return model, mono_predict


def build_trainer(cfg, model, predict_fn, ts, device, max_epochs):
    """Pick the training loop the config asks for.

    `loop: train_v34` is what trained the headline COD model; `train_physics` is
    what trained every baseline and every capacity-sweep checkpoint. They are
    different loops, not two names for one (see cod/training/train.py), so the
    config has to say which.
    """
    t = cfg["training"]
    loop = t.get("loop", "train_v34" if cfg["model"]["kind"] == "cod"
                 else "train_physics")
    cw = t.get("causal_weighting", {})
    kwargs = dict(
        x0s=ts.x0s, sensors=ts.sensors, device=device,
        lr=float(t.get("lr", 1e-3)),
        n_fb=int(t.get("batch_size", 128)),
        n_col=int(t.get("n_collocation", 80 if loop == "train_v34" else 60)),
        n_chunks=int(t.get("n_chunks", 5)),
        max_epochs=max_epochs,
        seed=int(t.get("seed", 0)),
        causal_log_space=bool(cw.get("log_space", True)),
        causal_floor=float(cw.get("weight_floor", 1e-8)),
        causal_schedule_shared=bool(cw.get("schedule_shared", True)),
    )
    if loop == "train_v34":
        # Cached theta_ss on the sensor grid, so fix 1's contraction solve runs once
        # per sample at dataset time instead of every forward pass.
        return CODTrainer(model, theta_ss=ts.ensure_theta_ss(), **kwargs), loop
    return SharedPhysicsTrainer(model, predict_fn, **kwargs), loop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--freeze-hash", default=None,
                    help="Expected distribution hash; fails if it has changed.")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override the config's epoch ceiling (smoke tests).")
    ap.add_argument("--max-wall-seconds", type=float, default=None,
                    help="Override the config's wall-clock ceiling. Equal wall "
                         "clock, not equal epochs, is the fair budget when two "
                         "models differ in cost per step (audit B-1).")
    ap.add_argument("--n-ic", type=int, default=None,
                    help="Override the config's n_ic (smoke tests).")
    ap.add_argument("--n-test", type=int, default=None,
                    help="Override the number of evaluation cases.")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output directory, to keep runs apart.")
    ap.add_argument("--theta-ss", default=None,
                    choices=["true_fixed_point", "formula_C"],
                    help="Override the model's analytic attractor. Used to run the "
                         "v57-physics control against the corrected physics.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing run in the output directory. "
                         "Without it a collision is an error, not a silent "
                         "overwrite of the earlier result.")
    ap.add_argument("--train-data", default=None,
                    help="Path to a stored .npz. Required to reproduce a "
                         "checkpoint; omit to generate from the config.")
    args = ap.parse_args()

    warn_if_dirty()
    cfg = load_config(args.config)

    if args.freeze_hash:
        from cod.config import assert_distribution_unchanged
        assert_distribution_unchanged(cfg, args.freeze_hash)

    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dist = cfg["distribution"]
    n_ic = args.n_ic if args.n_ic is not None else int(dist["n_ic"])
    max_epochs = (args.max_epochs if args.max_epochs is not None
                  else int(cfg["training"]["convergence"]["max_epochs"]))
    T = float(dist["window_minutes"])

    smoke = args.max_epochs is not None or args.n_ic is not None
    out_dir = (Path(args.out) / f"{cfg['experiment']['name']}_"
               f"{cfg['experiment']['variant']}_s{seed}_{cfg.hash}"
               f"{'_smoke' if smoke else ''}"
               f"{('_' + args.tag) if args.tag else ''}")

    # A second run of the same config and seed lands on the same directory name,
    # so without this the previous run.json, model.pt and loss_history.json are
    # overwritten in place and the earlier result is gone with no record that it
    # existed. Two runs that differ only in something outside the config hash —
    # a code change, a GPU, a library version — are exactly the pair worth
    # keeping, and provenance is what run.json is for.
    if (out_dir / "run.json").exists() and not args.overwrite:
        raise SystemExit(
            f"[run] {out_dir} already holds a run.json.\n"
            "  Refusing to overwrite it. Use --tag to name this run separately, "
            "or --overwrite if replacing the old one is what you mean.")

    print(f"[run] config          {args.config}")
    print(f"[run] config hash     {cfg.hash}")
    print(f"[run] distribution    {cfg.distribution_hash}")
    print(f"[run] seed            {seed}")
    print(f"[run] device          {device}")
    print(f"[run] n_ic            {n_ic}"
          f"{'  (OVERRIDDEN)' if args.n_ic is not None else ''}")
    print(f"[run] max_epochs      {max_epochs}"
          f"{'  (OVERRIDDEN)' if args.max_epochs is not None else ''}")
    print(f"[run] out             {out_dir}")
    if smoke:
        print("[run] SMOKE TEST: budgets overridden on the command line. "
              "The numbers below prove the wiring runs; they are not results.")

    # ── Data ───────────────────────────────────────────────────────────────
    # Which sampler is a config decision inside the hashed block, so the frozen
    # hash records which one produced the data. `realistic` is the default for
    # new work; `v57` exists to reproduce transformer_training_v57.npz and the
    # Phase 1 gates, and its ranges are hardcoded on purpose (see
    # `generate_training_set`).
    # The test set is the seed-999 benchmark and is built the same way whichever
    # sampler drew the training data, so its IC formula is read here rather than
    # inside a sampler branch.
    v57_ic = dist.get("steady_state_formula") == "A"
    ic_formula = formula_A if v57_ic else true_fixed_point_np

    sampler = dist.get("sampler")
    if sampler is None:
        raise ValueError(
            f"{args.config}: distribution.sampler is missing.\n"
            "  Declare it explicitly — which sampler drew the data is part of the\n"
            "  distribution, so it belongs inside the hashed block:\n"
            "    sampler: {kind: realistic, params: {...}}   # new work\n"
            "    sampler: {kind: v57}                        # reproduce v57")
    kind = sampler.get("kind")

    data_provenance = {"source": args.train_data or "generated",
                       "sampler": kind}

    if args.train_data:
        ts = load_training_set(args.train_data)
        print(f"[data] loaded {len(ts)} ICs from {args.train_data}")
    elif kind == "realistic":
        params = RealisticParams.from_config(sampler.get("params", {}))
        ts = generate_realistic_training_set(n_ic=n_ic,
                                             seed=int(dist.get("seed", 42)),
                                             params=params)
        data_provenance["realistic_params"] = params.to_config_dict()
        print(f"[data] generated {len(ts)} ICs  realistic sampler  "
              f"(cycle_period={params.cycle_period:g} min, "
              f"K_amp={params.K_amp}, hot_spot_mean={params.hot_spot_mean:g})")
    elif kind == "v57":
        if sampler.get("params"):
            raise ValueError(
                f"{args.config}: distribution.sampler.kind is 'v57' but params "
                "were given.\n  The v57 sampler's ranges are hardcoded in "
                "cod/data/profiles.py so that 'reproduces v57' cannot be changed "
                "by editing a YAML file. Remove the params block, or switch to "
                "kind: realistic.")
        phase_range = dist.get("ambient_phase", [0.0, 6.283185])
        randomise_phase = float(phase_range[1]) > float(phase_range[0])
        clip_step = True
        for fam in dist.get("profile_families", []):
            if fam.get("kind") == "step" and fam.get("clipped") is False:
                clip_step = False
        ts = generate_training_set(n_ic=n_ic, seed=int(dist.get("seed", 42)),
                                   randomise_ambient_phase=randomise_phase,
                                   steady_state=ic_formula, clip_step=clip_step)
        data_provenance.update({
            "steady_state_formula": "A" if v57_ic else "true_fixed_point",
            "randomise_ambient_phase": randomise_phase,
            "clip_step": clip_step,
        })
        print(f"[data] generated {len(ts)} ICs  v57 sampler (DEPRECATED, "
              f"reproduction only)  "
              f"(steady_state={'A' if v57_ic else 'true_fixed_point'}, "
              f"randomise_phase={randomise_phase}, clip_step={clip_step})")
    else:
        raise ValueError(
            f"{args.config}: distribution.sampler.kind is {kind!r}; "
            "expected 'realistic' or 'v57'.")

    # ── Model ──────────────────────────────────────────────────────────────
    if args.theta_ss is not None:
        cfg.raw["model"]["steady_state"] = args.theta_ss
        print(f"[model] attractor OVERRIDDEN to {args.theta_ss}")
    model, predict_fn = build_model(cfg, ts.x_mean, ts.x_std, device)
    model = model.to(device)
    print(f"[model] {cfg['model']['kind']}  {model.n_parameters():,} parameters")

    # ── Train ──────────────────────────────────────────────────────────────
    trainer, loop = build_trainer(cfg, model, predict_fn, ts, device, max_epochs)
    crit_cfg = dict(cfg["training"]["convergence"])
    crit_cfg["max_epochs"] = max_epochs
    if args.max_wall_seconds is not None:
        crit_cfg["max_wall_seconds"] = args.max_wall_seconds
    crit = ConvergenceCriterion(**crit_cfg)
    print(f"[train] loop={loop}  criterion={crit}")
    outcome = train(trainer, crit, log_every=max(1, max_epochs // 5))

    # ── Persist the weights ────────────────────────────────────────────────
    # Written before evaluation, so a run whose evaluation crashes does not throw
    # away the training. Until this existed `run.py` wrote run.json and
    # loss_history.json and dropped the model on exit, which made every downstream
    # measurement that needs `model(x0, u, t)` — swing fidelity, the rollout
    # thermal error, the whole C-11 matrix — impossible to run without retraining.
    #
    # The buffers `x_mean_TO` / `x_std_TO` are part of `state_dict()`, so a
    # checkpoint carries its own normalisation and does not depend on regenerating
    # the training set to be loadable. The hashes travel with it because a
    # checkpoint whose distribution cannot be identified is not evidence.
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kind": cfg["model"]["kind"],
        "theta_ss_mode": cfg["model"].get("steady_state", "true_fixed_point"),
        "x_mean": np.asarray(ts.x_mean).tolist(),
        "x_std": np.asarray(ts.x_std).tolist(),
        "config_hash": cfg.hash,
        "distribution_hash": cfg.distribution_hash,
        "seed": seed,
        "converged": bool(outcome.converged),
        "stop_reason": outcome.stop_reason,
        "config": cfg.raw,
    }, ckpt_path)
    print(f"[ckpt] wrote {ckpt_path}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    tier_cfg = cfg["evaluation"]["tiers"]
    tier_name = next(iter(tier_cfg))
    tier = tier_cfg[tier_name]
    n_test = args.n_test if args.n_test is not None else int(tier["n_cases"])
    guard = float(cfg["evaluation"].get("ground_truth", {})
                  .get("right_edge_guard", 0.9999))

    # Which distribution the tier is drawn from is a declared property of the
    # tier, not something inferred from its name. Before this, every tier got
    # `build_test_set` — the v57-era benchmark — whatever the sampler had been,
    # so a realistic-sampler run reported an out-of-family score under the label
    # `T1_in_distribution` (audit_port/TEST_SET_PROVENANCE.md: seven of nine
    # distributional axes outside training support). A tier now says what it is.
    tier_source = tier.get("source")
    if tier_source is None:
        raise ValueError(
            f"{args.config}: evaluation.tiers.{tier_name} has no `source`.\n"
            "  Which distribution a tier draws from decides whether its label is\n"
            "  true, so it has to be declared:\n"
            "    source: realistic_sampler   # the frozen sampler, held-out seed\n"
            "    source: v57_benchmark       # the seed-999 legacy benchmark")

    if tier_source == "realistic_sampler":
        if kind != "realistic":
            raise ValueError(
                f"{args.config}: tier {tier_name} draws from the realistic "
                f"sampler but training used sampler kind {kind!r}. That is the "
                "train/test mismatch this check exists to catch.")
        if int(tier["seed"]) == int(dist.get("seed", 42)):
            raise ValueError(
                f"{args.config}: tier {tier_name} uses seed {tier['seed']}, the "
                "same seed that draws the training set. That is a subset of "
                "training, not a test set.")
        cases = build_realistic_test_set(
            n_test=n_test, seed=int(tier["seed"]),
            params=RealisticParams.from_config(sampler.get("params", {})), T=T)
    elif tier_source == "v57_benchmark":
        cases = build_test_set(n_test=n_test, seed=int(tier["seed"]), T=T,
                               steady_state=ic_formula)
    else:
        raise ValueError(
            f"{args.config}: evaluation.tiers.{tier_name}.source is "
            f"{tier_source!r}; expected 'realistic_sampler' or 'v57_benchmark'.")
    print(f"[eval] tier {tier_name}  source={tier_source}  "
          f"seed={tier['seed']}  n={n_test}")
    bench = evaluate(model, predict_fn, cases, label=cfg["experiment"]["variant"],
                     T=T, t_clip_frac=guard, device=device)
    print("\n" + bench.summary())
    print("\n" + bench.physical_summary())

    # The metric the paper should report: absolute error in physical units,
    # with the floor-hit rate, and the test tier stated explicitly.
    gt = np.stack([
        __import__("cod.data.generate", fromlist=["rk45_ground_truth"])
        .rk45_ground_truth(c.x0, c.K_sensors, c.Ta_sensors,
                           np.linspace(0, T, 50), T=T, t_clip_frac=guard)
        for c in cases
    ])
    with torch.no_grad():
        preds = []
        t_q = torch.tensor(np.linspace(0, T, 50), dtype=torch.float32,
                           device=device).unsqueeze(-1)
        for c in cases:
            s_k = torch.tensor(np.concatenate([c.K_sensors, c.Ta_sensors]),
                               dtype=torch.float32, device=device).unsqueeze(0)
            x0_t = torch.tensor(c.x0, dtype=torch.float32,
                                device=device).unsqueeze(0)
            preds.append(predict_fn(model, x0_t.expand(50, -1).contiguous(),
                                    s_k.expand(50, -1).contiguous(),
                                    t_q).cpu().numpy())
    pred = np.stack(preds)

    phys = {}
    for i, spec in enumerate(TRANSFORMER_STATES[:STATE_DIM_FAST]):
        phys[spec.name] = evaluate_state(pred[:, :, i], gt[:, :, i], spec)
    print("\n" + report(phys, tier=tier_name))

    # ── Record ─────────────────────────────────────────────────────────────
    extra = {
        "status": "smoke_test" if smoke else "run",
        "training_loop": loop,
        "model_kind": cfg["model"]["kind"],
        "model_parameters": model.n_parameters(),
        "n_ic": len(ts),
        "device": str(device),
        "theta_ss_mode": cfg["model"].get("steady_state", "true_fixed_point"),
        "wall_clock_overridden": args.max_wall_seconds is not None,
        "convergence_criterion": crit.__dict__,
        "outcome": outcome.to_dict(),
        "fair_comparison_candidate": outcome.is_fair_comparison_candidate(),
        "evaluation": {
            "tier": tier_name,
            # Without this a run.json cannot say which distribution produced its
            # numbers, which is how O-5's out-of-family score came to be filed
            # under `T1_in_distribution`.
            "tier_source": tier_source,
            "tier_seed": int(tier["seed"]),
            "n_cases": n_test,
            "right_edge_guard": guard,
            "nmae_per_state_pct": dict(zip(STATE_NAMES_FAST,
                                           bench.per_state_pct.round(4).tolist())),
            "nmae_overall_pct": bench.overall_pct,
            "nmae_constant_K_pct": bench.ck_pct,
            "nmae_time_varying_pct": bench.tv_pct,
            "nmae_denominator_floor_hit_frac": dict(
                zip(STATE_NAMES_FAST, bench.floor_hit_frac().round(4).tolist())),
            "mae_physical_units": {k: v.to_dict() for k, v in phys.items()},
        },
        # Sampler-specific, so that a run.json never records flags belonging to a
        # sampler it did not use. The realistic branch writes the resolved
        # parameters rather than the flags, which is the record that matters:
        # it is what the distribution hash is a hash of.
        "data_provenance": data_provenance,
    }
    # loss_history can be 25,000 floats; keep the file readable.
    # The clamp and causal-weight series are the same size and go to their own
    # file for the same reason. They are not optional detail: the peak alone
    # cannot say whether a clamp fired only while the weights were random or was
    # still firing at the midpoint, and only the second invalidates the run.
    clamp_hist = extra["outcome"].pop("clamp_history", {})
    cw_hist = extra["outcome"].pop("causal_weight_history", [])
    extra["outcome"]["clamp_history_len"] = {k: len(v) for k, v in clamp_hist.items()}
    hist = extra["outcome"].pop("loss_history", [])
    extra["outcome"]["loss_history_len"] = len(hist)
    extra["outcome"]["loss_history_head"] = [round(float(x), 8) for x in hist[:20]]
    extra["outcome"]["loss_history_tail"] = [round(float(x), 8) for x in hist[-20:]]

    path = write_run_record(out_dir, cfg.summary(), seed, extra=extra)
    (out_dir / "loss_history.json").write_text(
        json.dumps([float(x) for x in hist]), encoding="utf-8")
    (out_dir / "clamp_history.json").write_text(
        json.dumps({"clamp": clamp_hist, "causal_weight_min": cw_hist}),
        encoding="utf-8")
    print("\n--- when each clamp was active ---")
    print(outcome.pathology.clamp_onset_table(outcome.clamp_history))

    # The per-case predictions and ground truth, so every table in the paper can
    # be rebuilt offline. run.json holds aggregates only, and an aggregate cannot
    # be un-aggregated: O-2 wants median / p90 / max with the case index of the
    # max, the C-11 honesty protocol wants a swing table and a Jensen-gap table,
    # and none of that is recoverable from a mean. Regenerating it costs a full
    # RK45 pass over the test set plus a forward pass; storing it costs ~250 kB.
    # float64, not float32. The gas errors are differences of order 1e-4 ppm
    # between values of order 1e-2, so float32 storage loses ~3e-4 relative on
    # exactly the quantity the absolute-ppm argument rests on — measured, in
    # `25_checkpoint_roundtrip.py` check 4, which is what caught it. The file
    # goes from ~250 kB to ~500 kB.
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred.astype(np.float64), gt=gt.astype(np.float64),
        t_eval=np.linspace(0, T, 50).astype(np.float64),
        x0=np.stack([c.x0 for c in cases]).astype(np.float32),
        K_sensors=np.stack([c.K_sensors for c in cases]).astype(np.float32),
        Ta_sensors=np.stack([c.Ta_sensors for c in cases]).astype(np.float32),
        kind=np.array([c.kind for c in cases]),
        family=np.array([c.family or "" for c in cases]),
        state_names=np.array(STATE_NAMES_FAST),
    )
    print(f"\n[run] wrote {path}")
    print(f"[run] wrote {out_dir / 'predictions.npz'} "
          f"({pred.shape[0]} cases x {pred.shape[1]} times x {pred.shape[2]} states)")

    if not outcome.converged:
        print("[run] This run did NOT converge. Report it as non-converged, "
              f"with stop_reason={outcome.stop_reason!r} and its learning curve. "
              "Do not quote the metrics above as a performance figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
