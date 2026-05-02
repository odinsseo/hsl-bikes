"""Post-hoc statistical tests for RQ1–RQ3 using paired station-level WMAPE vectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
)
from .data import build_station_series, load_graph_bundle, load_split, load_communities
from .train_eval import (
    bootstrap_mean_ci,
    cluster_bootstrap_mean_ci,
    cluster_paired_sign_permutation_pvalue,
    build_station_cohort_indices,
    load_station_city_lookup,
    paired_sign_permutation_pvalue,
)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (valid under arbitrary test dependence)."""
    p = np.asarray(list(p_values), dtype=float)
    m = int(p.size)
    if m == 0:
        return []
    if m == 1:
        return [float(min(1.0, p[0]))]
    order = np.argsort(p)
    sp = p[order]
    w = (m - np.arange(m, dtype=float)) * sp
    adj_sorted = np.zeros(m, dtype=float)
    running = 0.0
    for i in range(m):
        running = max(running, float(w[i]))
        adj_sorted[i] = float(min(1.0, running))
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return [float(x) for x in out.tolist()]


def _load_wmape_vector(scores_dir: Path, experiment_id: str) -> np.ndarray:
    path = scores_dir / f"{experiment_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing station score file: {path}")
    data = np.load(path)
    if "wmape_by_station" not in data:
        raise ValueError(f"Missing wmape_by_station in {path}")
    return np.asarray(data["wmape_by_station"], dtype=float)


@dataclass(frozen=True)
class ContrastSpec:
    rq: str
    contrast_id: str
    h0: str
    experiment_a: str
    experiment_b: str
    label_a: str
    label_b: str


def _graph_propagation_rows(results: pd.DataFrame) -> pd.DataFrame:
    g = results[results["model"] == "graph_propagation"].copy()
    return g


def build_contrast_specs(results: pd.DataFrame, *, rqs: set[str]) -> list[ContrastSpec]:
    """Define planned contrasts for selected RQs; verifies experiment_ids exist."""
    g = _graph_propagation_rows(results)
    ids = set(g["experiment_id"].astype(str))

    specs: list[ContrastSpec] = []

    def need(eid: str) -> None:
        if eid not in ids:
            raise ValueError(
                f"Required experiment_id {eid!r} not found in results.csv graph_propagation rows"
            )

    if "RQ1" in rqs:
        for cand, label in (
            ("RQ1_DE_STATION", "DE"),
            ("RQ1_DC_STATION", "DC"),
            ("RQ1_DE_DC_STATION", "DE+DC"),
        ):
            need("RQ1_SD_STATION")
            need(cand)
            specs.append(
                ContrastSpec(
                    rq="RQ1",
                    contrast_id=f"SD_vs_{label}",
                    h0=f"Mean paired station test WMAPE difference ({label} − SD) equals 0",
                    experiment_a=cand,
                    experiment_b="RQ1_SD_STATION",
                    label_a=label,
                    label_b="SD",
                )
            )

    if "RQ2" in rqs:
        need("RQ2_ALL_STATION")
        for pair_id, label in (
            ("RQ2_SD_DE_STATION", "SD+DE"),
            ("RQ2_SD_DC_STATION", "SD+DC"),
            ("RQ2_DE_DC_STATION", "DE+DC"),
        ):
            need(pair_id)
            specs.append(
                ContrastSpec(
                    rq="RQ2",
                    contrast_id=f"all_view_vs_{label.replace('+', '_')}",
                    h0=f"Mean paired station test WMAPE difference (all-view − {label}) equals 0",
                    experiment_a="RQ2_ALL_STATION",
                    experiment_b=pair_id,
                    label_a="SD+DE+DC+ATD",
                    label_b=label,
                )
            )

    if "RQ3" in rqs:
        pairs = (
            ("RQ3_ALL_STATION", "RQ3_ALL_COMMUNITY", "SD+DE+DC+ATD"),
            ("RQ3_FUNCTIONAL_STATION", "RQ3_FUNCTIONAL_COMMUNITY", "DE+DC"),
        )
        for sid, cid, gset in pairs:
            need(sid)
            need(cid)
            specs.append(
                ContrastSpec(
                    rq="RQ3",
                    contrast_id=f"station_vs_community_{gset.replace('+', '_')}",
                    h0=(
                        "Mean paired station test WMAPE difference "
                        "(station native − broadcast community to stations) equals 0"
                    ),
                    experiment_a=sid,
                    experiment_b=cid,
                    label_a=f"station/{gset}",
                    label_b=f"community→station/{gset}",
                )
            )

    return specs


def run_one_contrast(
    *,
    spec: ContrastSpec,
    cohort: str,
    cohort_idx: np.ndarray,
    scores_dir: Path,
    rng: np.random.Generator,
    n_permutations: int,
    n_bootstrap: int,
    ci_level: float,
    alpha: float,
    two_sided: bool,
    cluster_aware: bool = False,
    cluster_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    va = _load_wmape_vector(scores_dir, spec.experiment_a)
    vb = _load_wmape_vector(scores_dir, spec.experiment_b)
    if va.shape != vb.shape:
        raise ValueError(
            f"Vector length mismatch {spec.contrast_id}: {va.shape} vs {vb.shape}"
        )

    idx = cohort_idx
    if idx.size == 0:
        return {
            "rq": spec.rq,
            "contrast": spec.contrast_id,
            "cohort": cohort,
            "H0": spec.h0,
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "p_holm": np.nan,
            "alpha": alpha,
            "reject_H0": False,
            "n_stations_used": 0,
            "two_sided": two_sided,
            "label_a": spec.label_a,
            "label_b": spec.label_b,
            "experiment_a": spec.experiment_a,
            "experiment_b": spec.experiment_b,
            "note": "empty_cohort",
        }

    a_sub = va[idx]
    b_sub = vb[idx]
    mask = np.isfinite(a_sub) & np.isfinite(b_sub)
    diff = a_sub[mask] - b_sub[mask]
    # If cluster labels were supplied for all stations, restrict them to the
    # cohort indices so boolean masks align with the subset arrays below.
    cluster_labels_sub: np.ndarray | None = None
    if cluster_labels is not None:
        try:
            cluster_labels_sub = np.asarray(cluster_labels)[idx]
        except Exception:
            # If subsetting fails, leave as None and proceed with station-level methods
            cluster_labels_sub = None
    if diff.size == 0:
        mean_delta = np.nan
        ci_lo, ci_hi = (np.nan, np.nan)
        p_raw = np.nan
        p_cluster = np.nan
    else:
        mean_delta = float(np.nanmean(diff))
        if cluster_aware and cluster_labels_sub is not None:
            # cluster-aware bootstrap CI
            ci_lo, ci_hi = cluster_bootstrap_mean_ci(
                diff,
                cluster_labels=cluster_labels_sub[mask],
                rng=rng,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
            )
            # cluster-aware permutation p-value
            p_cluster = cluster_paired_sign_permutation_pvalue(
                a_sub[mask],
                b_sub[mask],
                cluster_labels=cluster_labels_sub[mask],
                rng=rng,
                n_permutations=n_permutations,
            )
            # also compute the raw station-level permutation for reference
            p_raw = paired_sign_permutation_pvalue(
                a_sub[mask],
                b_sub[mask],
                rng=rng,
                n_permutations=n_permutations,
            )
        else:
            ci_lo, ci_hi = bootstrap_mean_ci(
                diff,
                rng=rng,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
            )
            # sample = A, reference = B => diff A-B matches our vector definition
            p_raw = paired_sign_permutation_pvalue(
                a_sub[mask],
                b_sub[mask],
                rng=rng,
                n_permutations=n_permutations,
            )
            p_cluster = np.nan

    return {
        "rq": spec.rq,
        "contrast": spec.contrast_id,
        "cohort": cohort,
        "H0": spec.h0,
        "mean_delta": mean_delta,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p_raw,
        "p_holm": np.nan,
        "p_cluster": p_cluster,
        "alpha": alpha,
        "reject_H0": False,
        "n_stations_used": int(diff.size) if np.isfinite(mean_delta) else 0,
        "two_sided": two_sided,
        "cluster_aware": bool(cluster_aware),
        "label_a": spec.label_a,
        "label_b": spec.label_b,
        "experiment_a": spec.experiment_a,
        "experiment_b": spec.experiment_b,
    }


def run(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.random_state)
    output_dir: Path = args.output_dir
    scores_dir: Path = (
        args.scores_dir
        if args.scores_dir is not None
        else output_dir / "station_scores"
    )
    results_path = output_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}")

    results = pd.read_csv(results_path)
    rqs_set = set(str(x).upper() for x in args.rqs)
    contrast_specs = build_contrast_specs(results, rqs=rqs_set)

    station_index, _ = load_graph_bundle(args.graph_dir)
    train_df = load_split(args.train)
    train_station = build_station_series(train_df, station_index)
    city_lookup = load_station_city_lookup(args.stations_dir)
    cohorts = build_station_cohort_indices(
        train_station,
        station_index,
        city_lookup,
        float(args.sparse_quantile),
    )

    # Optional cluster mapping used for cluster-aware resampling
    cluster_labels: np.ndarray | None = None
    if bool(getattr(args, "cluster_aware", False)):
        station_to_group = load_communities(args.communities, station_index)
        cluster_labels = np.array(
            [station_to_group[s] for s in station_index], dtype=object
        )

    rows: list[dict[str, Any]] = []
    for cohort_name, cohort_idx in cohorts.items():
        for rq in sorted(rqs_set):
            batch: list[dict[str, Any]] = []
            for spec in contrast_specs:
                if spec.rq != rq:
                    continue
                row = run_one_contrast(
                    spec=spec,
                    cohort=cohort_name,
                    cohort_idx=cohort_idx,
                    scores_dir=scores_dir,
                    rng=rng,
                    n_permutations=args.permutation_resamples,
                    n_bootstrap=args.bootstrap_resamples,
                    ci_level=float(args.ci_level),
                    alpha=float(args.alpha),
                    two_sided=bool(args.two_sided),
                    cluster_aware=bool(getattr(args, "cluster_aware", False)),
                    cluster_labels=cluster_labels,
                )
                batch.append(row)
            # Select which p-values to use for multiple-comparison adjustment.
            if bool(getattr(args, "cluster_aware", False)):
                pvals = [
                    (
                        float(
                            r.get("p_cluster")
                            if r.get("p_cluster") is not None
                            else np.nan
                        )
                        if np.isfinite(
                            r.get("p_cluster")
                            if r.get("p_cluster") is not None
                            else np.nan
                        )
                        else np.nan
                    )
                    for r in batch
                ]
            else:
                pvals = [
                    float(r["p_value"]) if np.isfinite(r["p_value"]) else np.nan
                    for r in batch
                ]
            finite_mask = [np.isfinite(p) for p in pvals]
            finite_ps = [p for p, ok in zip(pvals, finite_mask) if ok]
            adj_finite = holm_bonferroni(finite_ps) if finite_ps else []
            fi = 0
            for r, ok in zip(batch, finite_mask):
                if ok:
                    r["p_holm"] = adj_finite[fi]
                    fi += 1
                    r["reject_H0"] = bool(r["p_holm"] < float(args.alpha))
                else:
                    r["p_holm"] = np.nan
                    r["reject_H0"] = False
                rows.append(r)

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "rq_hypothesis_tests.csv"
    out_df.to_csv(out_path, index=False)

    summary = {
        "alpha": float(args.alpha),
        "ci_level": float(args.ci_level),
        "two_sided": bool(args.two_sided),
        "cluster_aware": bool(getattr(args, "cluster_aware", False)),
        "permutation_resamples": int(args.permutation_resamples),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "random_state": int(args.random_state),
        "sparse_quantile": float(args.sparse_quantile),
        "rqs": sorted(args.rqs),
        "rows_written": int(len(rows)),
        "paths": {
            "rq_hypothesis_tests": str(out_path),
            "station_scores": str(scores_dir),
            "results": str(results_path),
        },
    }
    summary_path = output_dir / "rq_hypothesis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ hypothesis tests from rq_runner station_scores and results.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=None,
        help="Directory with {experiment_id}.npz (default: <output-dir>/station_scores)",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument(
        "--stations-dir",
        type=Path,
        default=None,
        help="Directory with station CSV for city cohorts (default: train_eval default)",
    )
    parser.add_argument("--sparse-quantile", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--two-sided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report two-sided sign permutation (current train_eval helper is two-sided on mean|diff|).",
    )
    parser.add_argument(
        "--cluster-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable cluster-aware resampling (use community clusters for permutation/bootstrap).",
    )
    parser.add_argument("--permutation-resamples", type=int, default=9999)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--rqs",
        nargs="+",
        default=["RQ1", "RQ2", "RQ3"],
        choices=["RQ1", "RQ2", "RQ3"],
        help="Which RQs to emit rows for",
    )
    ns = parser.parse_args(argv)
    if ns.stations_dir is None:
        from .train_eval import DEFAULT_STATIONS_DIR

        ns.stations_dir = DEFAULT_STATIONS_DIR
    return ns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
