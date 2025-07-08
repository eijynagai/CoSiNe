#!/usr/bin/env python
import os
import sys
from pathlib import Path

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, os.path.join(project_root, "src"))

import yaml
import networkx as nx
import multiprocessing
import platform
import json
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, wait

# Timeout per worker task (seconds)
TASK_TIMEOUT = 30

# Load method colors from config/colors.yml
color_config_path = project_root / "config" / "colors.yml"
with open(color_config_path, "r") as cf:
    cfg = yaml.safe_load(cf)
METHOD_COLORS = cfg["method_colors"]

import logging
import time

import pandas as pd
# import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.label_propagation import run_label_propagation
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

logging.basicConfig(level=logging.INFO)

methods = {
    "LouvainSigned": ("signed", {"alpha": 1.0, "resolution": 1.0}),
    "Louvain":      ("pos",    {"resolution": 1.0}),
    "Leiden":       ("pos",    {"resolution": 1.0}),
    "Greedy":       ("pos",    {}),
    "LPA":          ("pos",    {}),
}

sizes   = [1000, 3000] #,3000
degrees = [5, 15] #, 15
mixings = [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seeds   = [1, 5, 13, 35, 82] #1, 5, 9, 13, 17, 23, 35, 41, 47, 82
logging.info(f"Running benchmarks with seeds: {seeds}")

# --- Parallelized benchmark logic ---
def run_benchmark_task(n, k, mu, seed, name, gtype, params):
    # Graph generation with increased iterations, tolerance, and retry
    build_time = None
    res = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            res = generate_signed_LFR_benchmark_graph(
                n=n, tau1=2.8, tau2=1.5, mu=mu,
                P_minus=0.3, P_plus=0.15,
                average_degree=k, min_community=30,
                seed=seed + attempt,    # vary seed on retry
                tol=1e-5,               # relax tolerance
                max_iters=5000,         # increase iteration cap
            )
        except nx.exception.ExceededMaxIterations:
            logging.error("Attempt %d/%d: LFR max iterations exceeded", attempt+1, 3)
            continue
        build_time = time.perf_counter() - t0
        if res is not None:
            break
        logging.error("Attempt %d/%d: LFR returned None", attempt+1, 3)
    if res is None:
        failure_log.append({
            "n": n, "avg_deg": k, "mu": mu,
            "seed": seed, "method": name,
            "error": "LFR_failed_after_retries"
        })
        return None
    Gs, Gp, Gn = res

    # detection with error capture
    try:
        t0_detect = time.perf_counter()
        if gtype == "signed":
            run_louvain_signed(Gp, Gn, **params)
        else:
            if name == "Louvain":
                run_louvain(Gp, **params)
            elif name == "Leiden":
                run_leiden(Gp, **params)
            elif name == "Greedy":
                run_greedy_modularity(Gp)
            elif name == "LPA":
                run_label_propagation(Gp)
        detect_time = time.perf_counter() - t0_detect
    except Exception as e:
        logging.error("Detection failed [%s, seed %d]: %s", name, seed, e)
        failure_log.append({
            "n": n, "avg_deg": k, "mu": mu,
            "seed": seed, "method": name,
            "error": f"DetectionError: {e}"
        })
        return None

    return {
        "n": n,
        "avg_deg": k,
        "mu": mu,
        "Seed": seed,
        "Method": name,
        "build_time_s": round(build_time, 4),
        "detect_time_s": round(detect_time, 4),
    }

##
# Batch by (n, avg_deg) to limit concurrency and avoid stalling
##
records = []
failure_log = []

batch_iter = list((n, k) for n in sizes for k in degrees)
for n, k in tqdm(batch_iter, desc="Batches (n,deg)", total=len(batch_iter)):
    batch = [
        (n, k, mu, seed, name, gtype, params)
        for mu in mixings
        for seed in seeds
        for name, (gtype, params) in methods.items()
    ]
    logging.info("Starting batch n=%d, avg_deg=%d with %d tasks", n, k, len(batch))

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_benchmark_task, *t): t for t in batch}

        # now wrap the as_completed in its own tqdm
        for f in tqdm(as_completed(futures), desc=f"Tasks n={n} deg={k}", total=len(batch)):
            try:
                res = f.result(timeout=TASK_TIMEOUT)
            except Exception as e:
                task = futures[f]
                logging.error("Task %r failed: %s", task, e)
                failure_log.append({**_task_to_dict(task), "error": str(e)})
                continue
            if res:
                records.append(res)

        # cancel any stragglers
        for f in futures:
            if not f.done():
                f.cancel()
        exe.shutdown(wait=False, cancel_futures=True)

        logging.info("Batch done: %d succeeded, %d timed out", len(done), len(not_done))

        for f in done:
            try:
                res = f.result()
            except Exception as e:
                task = futures[f]
                logging.error("Task %r failed: %s", task, e)
                failure_log.append({**_task_to_dict(task), "error": str(e)})
            else:
                if res:
                    records.append(res)

        # Cancel any remaining futures before shutdown
        exe.shutdown(wait=False, cancel_futures=True)

df = pd.DataFrame(records)
os.makedirs("results", exist_ok=True)
df.to_csv("results/runtime_benchmark.csv", index=False)

# Save failures and report failure rate
if failure_log:
    df_fail = pd.DataFrame(failure_log)
    failure_path = project_root / "results" / "runtime_benchmark_failures.csv"
    df_fail.to_csv(failure_path, index=False)
    logging.warning(f"Saved {len(failure_log)} failures to {failure_path}")
total = len(records) + len(failure_log)
logging.info(f"LFR generation failed {len(failure_log)}/{total} times "
             f"({len(failure_log)/total:.1%})")

# --- Save environment metadata ---
meta = {
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": multiprocessing.cpu_count(),
    "memory_total_mb": psutil.virtual_memory().total / 1e6,
    "packages": {
        "networkx": nx.__version__,
        "pandas": pd.__version__,
        "seaborn": sns.__version__,
    },
}
md_path = project_root / "results" / "runtime_benchmark_metadata.json"
with open(md_path, "w") as mf:
    json.dump(meta, mf, indent=2)
logging.info("Saved metadata to %s", md_path)

# --- Compute and save summary statistics ---
summary = df.groupby(["Method", "n", "avg_deg", "mu"]).agg(
    build_time_mean=("build_time_s", "mean"),
    build_time_std=("build_time_s", "std"),
    detect_time_mean=("detect_time_s", "mean"),
    detect_time_std=("detect_time_s", "std"),
).reset_index()
summary_path = project_root / "results" / "runtime_benchmark_summary.csv"
summary.to_csv(summary_path, index=False)
logging.info("Saved summary to %s", summary_path)

# ---- Plotting ----
PLOT_DIR = os.path.join("results")
import pathlib
PLOT_DIR = pathlib.Path(PLOT_DIR)
PLOT_DIR.mkdir(exist_ok=True)

SIZES = sizes
DEGREES = degrees

# 6) Plot: runtime vs degree per graph size (boxplot)
for n_val in SIZES:
    df_n = df[df["n"] == n_val]
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(
        data=df_n,
        x="avg_deg",
        y="detect_time_s",
        hue="Method",
        palette=METHOD_COLORS
    )
    ax.set_title(f"Runtime vs. Degree (n={n_val})")
    ax.set_xlabel("Average degree")
    ax.set_ylabel("Runtime (s)")
    plt.tight_layout()
    out2 = PLOT_DIR / f"runtime_vs_degree_n{n_val}.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    logging.info("Saved plot: %s", out2)

# 7) Plot: runtime vs mixing parameter mu per graph size
for n_val in SIZES:
    df_n = df[df["n"] == n_val]
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(
        data=df_n,
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
    ax.set_title(f"Runtime vs. Mixing (n={n_val})")
    ax.set_xlabel("Mixing parameter μ")
    ax.set_ylabel("Runtime (s)")
    plt.tight_layout()
    out3 = PLOT_DIR / f"runtime_vs_mu_n{n_val}.png"
    plt.savefig(out3, dpi=300)
    plt.close()
    logging.info("Saved plot: %s", out3)
