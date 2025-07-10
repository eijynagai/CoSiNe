#!/usr/bin/env python
import os
import sys
import argparse
import json
import logging
import multiprocessing
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import networkx as nx
import pandas as pd
import psutil
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, wait

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.label_propagation import run_label_propagation
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Runtime benchmark for community detection methods.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Path to CSV file containing benchmark scenarios with columns: n, avg_deg, mu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save benchmark results and metadata",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, multiprocessing.cpu_count()),
        help="Maximum number of parallel worker processes",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each batch of tasks",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 5, 13, 47, 82],
        help="List of random seeds to use for graph generation",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance parameter for LFR graph generation",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5000,
        help="Maximum iterations for LFR graph generation",
    )
    return parser.parse_args()


def log_environment(output_dir: Path) -> None:
    """Save environment metadata to JSON in output directory."""
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory_total_mb": psutil.virtual_memory().total / 1e6,
        "packages": {
            "networkx": nx.__version__,
            "pandas": pd.__version__,
        },
    }
    timestamp = time.strftime("%Y%m%d_%H%M")
    md_path = output_dir / f"runtime_benchmark_metadata_{timestamp}.json"
    with md_path.open("w") as mf:
        json.dump(meta, mf, indent=2)
    logger.info("Saved metadata to %s", md_path)


def load_methods(config_path: Path) -> Dict[str, Tuple[str, Dict]]:
    """Load community detection methods configuration from YAML file."""
    with config_path.open("r") as cf:
        cfg = yaml.safe_load(cf)
    # Define methods with parameters
    methods = {
        "LouvainSigned": ("signed", {"alpha": 0.6, "resolution": 1.0}),
        "Louvain": ("pos", {"resolution": 1.0}),
        "Leiden": ("pos", {"resolution": 1.0}),
        "Greedy": ("pos", {}),
        "LPA": ("pos", {}),
    }
    return methods

# -----------------------------------------------------------------------------
# Load color mapping for methods from config/colors.yml
# -----------------------------------------------------------------------------
colors_path = project_root / "config" / "colors.yml"
try:
    with colors_path.open() as cf:
        _colors_cfg = yaml.safe_load(cf)
    METHOD_COLORS = _colors_cfg.get("method_colors", {})
    logger.info("Loaded method colors from %s", colors_path)
except Exception as e:
    logger.warning("Could not load colors from %s: %s", colors_path, e)
    # Fallback to default cycling colors
    METHOD_COLORS = {}


def run_benchmark_task(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    P_minus: float,
    P_plus: float,
    k: int,
    min_community: int,
    seed: int,
    name: str,
    gtype: str,
    params: Dict,
    tol: float,
    max_iters: int,
) -> Dict:
    """Run a single benchmark task: generate graph and run detection method."""
    build_time = None
    res = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            res = generate_signed_LFR_benchmark_graph(
                n,
                tau1,
                tau2,
                mu,
                P_minus,
                P_plus,
                k,
                min_community,
                seed + attempt,
            )
        except nx.exception.ExceededMaxIterations:
            logger.error("Attempt %d/3: LFR max iterations exceeded", attempt + 1)
            continue
        build_time = time.perf_counter() - t0
        if res is not None:
            break
        logger.error("Attempt %d/3: LFR returned None", attempt + 1)
    if res is None:
        return {
            "n": n,
            "tau1": tau1,
            "tau2": tau2,
            "mu": mu,
            "P_minus": P_minus,
            "P_plus": P_plus,
            "avg_deg": k,
            "min_community": min_community,
            "Seed": seed,
            "Method": name,
            "error": "LFR_failed_after_retries",
        }
    Gs, Gp, Gn = res

    try:
        t0_detect = time.perf_counter()
        if gtype == "signed":
            comm = run_louvain_signed(Gp, Gn, **params)
        else:
            if name == "Louvain":
                comm = run_louvain(Gp, **params)
            elif name == "Leiden":
                comm = run_leiden(Gp, **params)
            elif name == "Greedy":
                comm = run_greedy_modularity(Gp)
            elif name == "LPA":
                comm = run_label_propagation(Gp)
        detect_time = time.perf_counter() - t0_detect
    except Exception as e:
        return {
            "n": n,
            "tau1": tau1,
            "tau2": tau2,
            "mu": mu,
            "P_minus": P_minus,
            "P_plus": P_plus,
            "avg_deg": k,
            "min_community": min_community,
            "Seed": seed,
            "Method": name,
            "error": f"DetectionError: {e}",
        }

    # compute number of communities detected
    if isinstance(comm, dict):
        num_comms = len(set(comm.values()))
    else:
        num_comms = len(set(comm))

    return {
        "n": n,
        "tau1": tau1,
        "tau2": tau2,
        "mu": mu,
        "P_minus": P_minus,
        "P_plus": P_plus,
        "avg_deg": k,
        "min_community": min_community,
        "Seed": seed,
        "Method": name,
        "build_time_s": round(build_time, 4),
        "detect_time_s": round(detect_time, 4),
        "num_comms": num_comms,
    }


def run_batch(
    scenario: Dict,
    seeds: List[int],
    methods: Dict[str, Tuple[str, Dict]],
    timeout: int,
    tol: float,
    max_iters: int,
    max_workers: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Run a batch of benchmark tasks for a single scenario.

    Args:
        scenario: Dict with keys 'n', 'avg_deg', 'mu' describing the scenario.
        seeds: List of random seeds.
        methods: Dict of methods to run.
        timeout: Timeout in seconds for the batch.
        tol: Tolerance parameter for LFR generation.
        max_iters: Max iterations for LFR generation.
        max_workers: Max parallel workers.

    Returns:
        Tuple of (records, failures) lists of dicts.
    """
    # Ensure required scenario fields exist or confirm defaults
    required = ["n","tau1","tau2","mu","P_minus","P_plus","avg_deg","min_community"]
    defaults = {"tau1":2.8,"tau2":1.5,"P_minus":0.3,"P_plus":0.15,"avg_deg":30,"min_community":30}
    for field in required:
        if field not in scenario or pd.isna(scenario[field]):
            resp = input(f"Scenario '{scenario.get('scenario','')}': missing '{field}'. Use default {defaults.get(field)}? (y/N) ")
            if resp.lower() != 'y':
                sys.exit(f"Missing scenario field '{field}'. Aborting.")
            scenario[field] = defaults[field]
    n = int(scenario["n"])
    tau1 = float(scenario["tau1"])
    tau2 = float(scenario["tau2"])
    mu = float(scenario["mu"])
    P_minus = float(scenario["P_minus"])
    P_plus = float(scenario["P_plus"])
    k = int(scenario["avg_deg"])
    min_community = int(scenario["min_community"])

    batch = [
        (
          n, tau1, tau2, mu,
          P_minus, P_plus,
          k, min_community,
          seed,
          name, gtype, params,
          tol, max_iters
        )
        for seed in seeds
        for name, (gtype, params) in methods.items()
    ]
    records = []
    failure_log = []

    logger.info("Starting batch n=%d, avg_deg=%d, mu=%.3f with %d tasks", n, k, mu, len(batch))

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_benchmark_task, *t): t for t in batch}

        done, not_done = wait(futures.keys(), timeout=timeout)
        for f in not_done:
            f.cancel()

        logger.info("Batch done: %d succeeded, %d timed out", len(done), len(not_done))

        for f in done:
            res = f.result()
            if res is None:
                continue
            # add scenario label
            res["scenario"] = scenario.get("scenario", "")
            if "error" in res:
                failure_log.append(res)
            else:
                records.append(res)

        exe.shutdown(wait=False, cancel_futures=True)

    return records, failure_log


def save_results(
    output_dir: Path,
    df: pd.DataFrame,
    failures: pd.DataFrame,
    summary: pd.DataFrame,
    timestamp: str,
) -> None:
    """Save benchmark results, failures, and summary CSV files.

    Args:
        output_dir: Directory to save files.
        df: DataFrame with successful results.
        failures: DataFrame with failure records.
        summary: DataFrame with aggregated summary statistics.
        timestamp: Timestamp string for filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / f"runtime_benchmark_{timestamp}.csv"
    df.to_csv(run_csv, index=False)
    logger.info("Saved runtime results to %s", run_csv)

    if not failures.empty:
        failure_csv = output_dir / f"runtime_benchmark_failures_{timestamp}.csv"
        failures.to_csv(failure_csv, index=False)
        logger.warning("Saved %d failures to %s", len(failures), failure_csv)

    summary_csv = output_dir / f"runtime_benchmark_summary_{timestamp}.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info("Saved summary to %s", summary_csv)


def main(args: argparse.Namespace) -> None:
    """Main entry point for the runtime benchmark script."""
    methods = load_methods(project_root / "config" / "colors.yml")

    scenarios_df = pd.read_csv(args.scenarios)
    all_records = []
    all_failures = []

    for _, scenario in scenarios_df.iterrows():
        records, failures = run_batch(
            scenario=scenario.to_dict(),
            seeds=args.seeds,
            methods=methods,
            timeout=args.timeout,
            tol=args.tol,
            max_iters=args.max_iters,
            max_workers=args.workers,
        )
        all_records.extend(records)
        all_failures.extend(failures)

    df = pd.DataFrame(all_records)
    failure_log = pd.DataFrame(all_failures)

    if not df.empty:
        summary = df.groupby(["Method", "n", "tau1", "tau2", "mu"]).agg(
            build_time_mean=("build_time_s", "mean"),
            build_time_std=("build_time_s", "std"),
            detect_time_mean=("detect_time_s", "mean"),
            detect_time_std=("detect_time_s", "std"),
        ).reset_index()
    else:
        summary = pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_environment(args.output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M")
    save_results(args.output_dir, df, failure_log, summary, timestamp)

    total = len(all_records) + len(all_failures)
    if total > 0:
        logger.info(
            "LFR generation failed %d/%d times (%.1f%%)",
            len(all_failures),
            total,
            100 * len(all_failures) / total,
        )
    else:
        logger.info("No benchmark tasks were run.")


    # ---- Per-scenario plotting ----
    # Ensure 'scenario' column exists
    if "scenario" not in df.columns:
        logger.warning("No 'scenario' column; skipping per-scenario plots.")
        return

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(df["scenario"].unique())
    for scen in scenarios:
        df_s = df[df["scenario"] == scen]

        # 1) Detection time vs avg_deg (boxplot)
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(
            data=df_s,
            x="avg_deg",
            y="detect_time_s",
            hue="Method",
            palette=METHOD_COLORS
        )
        ax.set_yscale("log")
        ax.set_title(f"Detection Time vs Average Degree ({scen})")
        ax.set_xlabel("Average degree (⟨k⟩)")
        ax.set_ylabel("Detection time (s) [log scale]")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out1 = plot_dir / f"{scen}_runtime_vs_degree.png"
        plt.savefig(out1, dpi=300)
        plt.close()
        logger.info("Saved plot: %s", out1)

        # 2) Detection time vs mu (lineplot)
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=df_s,
            x="mu",
            y="detect_time_s",
            hue="Method",
            style="Method",
            markers=True,
            dashes=True,
            estimator="mean",
            errorbar="sd",
            palette=METHOD_COLORS
        )
        ax.set_yscale("log")
        ax.set_title(f"Detection Time vs Mixing μ ({scen})")
        ax.set_xlabel("Mixing parameter μ")
        ax.set_ylabel("Detection time (s) [log scale]")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out2 = plot_dir / f"{scen}_runtime_vs_mu.png"
        plt.savefig(out2, dpi=300)
        plt.close()
        logger.info("Saved plot: %s", out2)

        # 3) Detection time vs P_plus (lineplot)
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=df_s,
            x="P_plus",
            y="detect_time_s",
            hue="Method",
            style="Method",
            markers=True,
            dashes=True,
            estimator="mean",
            errorbar="sd",
            palette=METHOD_COLORS
        )
        ax.set_yscale("log")
        ax.set_title(f"Detection Time vs P_plus ({scen})")
        ax.set_xlabel("Positive edge fraction P_plus")
        ax.set_ylabel("Detection time (s) [log scale]")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out3 = plot_dir / f"{scen}_runtime_vs_pplus.png"
        plt.savefig(out3, dpi=300)
        plt.close()
        logger.info("Saved plot: %s", out3)

        # 4) Detection time vs number of communities (scatter + regression)
        plt.figure(figsize=(8, 5))
        # scatter of raw points
        sns.scatterplot(
            data=df_s,
            x="num_comms",
            y="detect_time_s",
            hue="Method",
            palette=METHOD_COLORS,
            alpha=0.6,
            s=50
        )
        # overlay linear regression per method
        for method in df_s["Method"].unique():
            sns.regplot(
                data=df_s[df_s["Method"] == method],
                x="num_comms",
                y="detect_time_s",
                scatter=False,
                color=METHOD_COLORS.get(method),
                line_kws={"linewidth": 2},
            )
        ax = plt.gca()
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_title(f"Detection Time vs Number of Communities ({scen})")
        ax.set_xlabel("Number of detected communities")
        ax.set_ylabel("Detection time (s) [log scale]")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out4 = plot_dir / f"{scen}_runtime_vs_num_comms.png"
        plt.savefig(out4, dpi=300)
        plt.close()
        logger.info("Saved plot: %s", out4)

    # ---- Cross-scenario global plots ----
    # 5) Detection time vs number of nodes (scatter + regression)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="n",
        y="detect_time_s",
        hue="Method",
        palette=METHOD_COLORS,
        alpha=0.6,
        s=50
    )
    for method in df["Method"].unique():
        sns.regplot(
            data=df[df["Method"] == method],
            x="n",
            y="detect_time_s",
            scatter=False,
            color=METHOD_COLORS.get(method),
            line_kws={"linewidth": 2},
        )
    ax = plt.gca()
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_title("Detection Time vs Number of Nodes")
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("Detection time (s) [log scale]")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out5 = plot_dir / "runtime_vs_n.png"
    plt.savefig(out5, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", out5)

    # 6) Number of communities vs number of nodes (scatter + regression)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="n",
        y="num_comms",
        hue="Method",
        palette=METHOD_COLORS,
        alpha=0.6,
        s=50
    )
    for method in df["Method"].unique():
        sns.regplot(
            data=df[df["Method"] == method],
            x="n",
            y="num_comms",
            scatter=False,
            color=METHOD_COLORS.get(method),
            line_kws={"linewidth": 2},
        )
    ax = plt.gca()
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_title("Number of Communities vs Number of Nodes")
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("Number of detected communities")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out6 = plot_dir / "num_comms_vs_n.png"
    plt.savefig(out6, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", out6)


if __name__ == "__main__":
    args = parse_args()
    main(args)