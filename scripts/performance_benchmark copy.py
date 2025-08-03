#!/usr/bin/env python
import os
import sys
import time
import logging
import json
import platform
import argparse
import multiprocessing
import psutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, TimeoutError

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import yaml
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from skimage.metrics import variation_of_information
except ImportError:
    def variation_of_information(gt, pred):
        # fallback: return nan
        return np.nan

from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.label_propagation import run_label_propagation

# -----------------------------------------------------------------------------
# argparse and utility functions
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performance benchmark for community detection methods.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Path to CSV file with scenario parameters (columns: mu,P_minus,P_plus, plus optional n,avg_deg,min_community)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save benchmark results and plots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, multiprocessing.cpu_count()),
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds per batch",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1,11)),
        help="Random seeds for each scenario",
    )
    return parser.parse_args()

def log_environment(output_dir: Path) -> None:
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory_total_mb": psutil.virtual_memory().total / 1e6,
        "packages": {
            "networkx": nx.__version__,
            "pandas": pd.__version__,
            "seaborn": sns.__version__,
            "scikit-learn": f"{f1_score.__module__.split('.')[0]}",
            "sklearn": f"{f1_score.__module__.split('.')[0]}",
            "psutil": psutil.__version__ if hasattr(psutil, "__version__") else "",
        },
    }
    timestamp = time.strftime("%Y%m%d_%H%M")
    md_path = output_dir / f"performance_benchmark_metadata_{timestamp}.json"
    with md_path.open("w") as mf:
        json.dump(meta, mf, indent=2)
    logging.info("Saved metadata to %s", md_path)

# -----------------------------------------------------------------------------
# Load color mapping for methods from config/colors.yml
# -----------------------------------------------------------------------------
colors_path = project_root / "config" / "colors.yml"
try:
    with colors_path.open() as cf:
        _colors_cfg = yaml.safe_load(cf)
    METHOD_COLORS = _colors_cfg.get("method_colors", {})
    logging.info("Loaded method colors from %s", colors_path)
except Exception as e:
    logging.warning("Could not load colors from %s: %s", colors_path, e)
    METHOD_COLORS = {}

# -----------------------------------------------------------------------------
# Define methods (mirroring runtime_benchmark.py)
# -----------------------------------------------------------------------------
METHODS = {
    "LouvainSigned": ("signed", {"alpha": 0.6, "resolution": 1.0}),
    "Louvain": ("pos", {"resolution": 1.0}),
    "Leiden": ("pos", {"resolution": 1.0}),
    "Greedy": ("pos", {}),
    "LPA": ("pos", {}),
}

# -----------------------------------------------------------------------------
# Per-task execution: run_performance_task
# -----------------------------------------------------------------------------
def run_performance_task(
    scenario: dict,
    seed: int,
    method_name: str,
    method_info: tuple,
) -> dict:
    """Run a single performance benchmark task: scenario dict, seed, method."""
    # Get parameters with defaults
    n = int(scenario.get("n", 2000))
    tau1 = float(scenario.get("tau1", 3.0))
    tau2 = float(scenario.get("tau2", 1.5))
    mu = float(scenario["mu"])
    P_minus = float(scenario["P_minus"])
    P_plus = float(scenario["P_plus"])
    avg_deg = int(scenario.get("avg_deg", 10))
    min_community = int(scenario.get("min_community", 50))
    gtype, params = method_info
    # 1) generate
    try:
        t0_build = time.perf_counter()
        Gs, Gp, Gn = generate_signed_LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu,
            P_minus=P_minus, P_plus=P_plus,
            average_degree=avg_deg, min_community=min_community,
            seed=seed,
        )
        build_time = time.perf_counter() - t0_build
    except Exception as e:
        return {
            "n": n, "tau1": tau1, "tau2": tau2, "mu": mu, "P_minus": P_minus, "P_plus": P_plus,
            "avg_deg": avg_deg, "min_community": min_community, "Seed": seed, "Method": method_name,
            "error": f"gen:{e}"
        }

    nodes = sorted(Gs.nodes())
    # 2) run
    try:
        t0 = time.perf_counter()
        if gtype == "signed":
            comm = run_louvain_signed(Gp, Gn, **params)
        else:
            if method_name == "Louvain":
                comm = run_louvain(Gp, **params)
            elif method_name=="Leiden":
                comm = run_leiden(Gp, **params)
            elif method_name=="Greedy":
                comm = run_greedy_modularity(Gp)
            elif method_name=="LPA":
                comm = run_label_propagation(Gp)
        detect_time = time.perf_counter() - t0
    except Exception as e:
        return {
            "n": n, "tau1": tau1, "tau2": tau2, "mu": mu, "P_minus": P_minus, "P_plus": P_plus,
            "avg_deg": avg_deg, "min_community": min_community, "Seed": seed, "Method": method_name,
            "error": f"det:{e}"
        }

    # 3) ground truth & flat lists
    gt = [Gs.nodes[n]["community"] for n in nodes]
    pred = [comm.get(n, -1) for n in nodes]  

    # 4) pairwise AUPRC using index-based comparison
    y_true, y_score = [], []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            y_true.append(int(gt[i] == gt[j]))
            y_score.append(int(pred[i] == pred[j]))
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = np.nan

    # 5) frustration index = frac of neg-edge inside communities
    neg_inside = 0
    total_neg = 0
    for u, v, d in Gs.edges(data=True):
        if d['weight'] < 0:
            total_neg += 1
            if comm.get(u) == comm.get(v):
                neg_inside += 1
    frustration = neg_inside / total_neg if total_neg > 0 else 0.0

    # 6) variation of information (if available)
    try:
        vi = variation_of_information(gt, pred)
    except Exception:
        vi = np.nan

    return dict(
        n=n, tau1=tau1, tau2=tau2, mu=mu, P_minus=P_minus, P_plus=P_plus,
        avg_deg=avg_deg, min_community=min_community, Seed=seed, Method=method_name,
        NMI=normalized_mutual_info_score(gt, pred),
        ARI=adjusted_rand_score(gt, pred),
        F1=f1_score(gt, pred, average="macro"),
        AUPRC=auprc,
        VI=vi,
        frustration=frustration,
        detect_time_s=round(detect_time, 4),
        build_time_s=round(build_time, 4),
    )

# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------
def run_batch(
    scenario: dict,
    seeds: list,
    methods: dict,
    timeout: int,
    max_workers: int,
) -> tuple:
    """Run a batch of performance tasks for a single scenario."""
    batch = [
        (scenario, seed, name, info)
        for seed in seeds
        for name, info in methods.items()
    ]
    records = []
    failures = []
    logging.info(
        f"Starting batch mu={scenario.get('mu')}, P_minus={scenario.get('P_minus')}, P_plus={scenario.get('P_plus')} with {len(batch)} tasks"
    )
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_performance_task, *t): t for t in batch}
        done, not_done = wait(futures.keys(), timeout=timeout)
        for f in not_done:
            f.cancel()
        for f in done:
            try:
                res = f.result()
            except Exception as e:
                t = futures[f]
                failures.append({
                    "mu": scenario.get("mu"),
                    "P_minus": scenario.get("P_minus"),
                    "P_plus": scenario.get("P_plus"),
                    "Seed": t[1],
                    "Method": t[2],
                    "error": f"exception:{e}",
                })
                continue
            if res is None:
                continue
            if "error" in res:
                failures.append(res)
            else:
                # add scenario label if present
                res["scenario"] = scenario.get("scenario", "")
                records.append(res)
        exe.shutdown(wait=False, cancel_futures=True)
    return records, failures

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scenarios_df = pd.read_csv(args.scenarios)
    all_records = []
    all_failures = []

    for _, scenario in scenarios_df.iterrows():
        recs, fails = run_batch(
            scenario=scenario.to_dict(),
            seeds=args.seeds,
            methods=METHODS,
            timeout=args.timeout,
            max_workers=args.workers,
        )
        all_records.extend(recs)
        all_failures.extend(fails)

    df = pd.DataFrame(all_records)
    failure_log = pd.DataFrame(all_failures)

    # --- summary stats per method / parameter combo ---
    if not df.empty:
        summary = df.groupby(["Method", "mu", "P_minus", "P_plus"]).agg(
            NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
            ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
            F1_mean=("F1", "mean"), F1_std=("F1", "std"),
            AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"),
            VI_mean=("VI", "mean"), VI_std=("VI", "std"),
            frustration_mean=("frustration", "mean"), frustration_std=("frustration", "std"),
            build_time_mean=("build_time_s", "mean"), build_time_std=("build_time_s", "std"),
            detect_time_mean=("detect_time_s", "mean"), detect_time_std=("detect_time_s", "std"),
        ).reset_index()
    else:
        summary = pd.DataFrame()

    timestamp = time.strftime("%Y%m%d_%H%M")
    outdir = args.output_dir
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / f"performance_benchmark_{timestamp}.csv", index=False)
    failure_log.to_csv(outdir / f"performance_benchmark_failures_{timestamp}.csv", index=False)
    summary.to_csv(outdir / f"performance_benchmark_summary_{timestamp}.csv", index=False)
    logging.info("Saved %d records, %d failures", len(df), len(failure_log))
    log_environment(outdir)

    # -----------------------------------------------------------------------------
    # plotting: boxplots per metric by method & scenario
    # -----------------------------------------------------------------------------
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    metrics = ["NMI", "ARI", "F1", "AUPRC", "VI", "frustration"]
    for m in metrics:
        plt.figure(figsize=(8, 5))
        # draw one box per method using hue, then remove duplicate legend
        ax = sns.boxplot(
            data=df,
            x="Method",
            y=m,
            hue="Method",
            palette=METHOD_COLORS,
            dodge=False
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(f"{m} by method")
        ax.set_xlabel("")
        ax.set_ylabel(m)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_dir / f"performance_{m}.png", dpi=300)
        plt.close()

    # ------------------------------------------------------------------------------
    # paired delta‐plots: signed vs leiden
    # -----------------------------------------------------------------------------
    ref = df[df.Method == "Leiden"]
    sig = df[df.Method == "LouvainSigned"]
    for m in ["NMI", "ARI", "F1"]:
        merged = pd.merge(sig, ref, on=["mu", "P_minus", "P_plus", "Seed"], suffixes=("_sig", "_ref"))
        merged["Δ" + m] = merged[f"{m}_sig"] - merged[f"{m}_ref"]
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x="mu", y="Δ" + m, data=merged, palette="vlag")
        ax.set_title(f"Δ{m}: SignedLouvain – Leiden")
        ax.set_xlabel("Mixing μ")
        ax.set_ylabel(f"Δ{m}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"delta_{m}.png", dpi=300)
        plt.close()

    logging.info("Performance benchmark complete.")


if __name__ == "__main__":
    main()