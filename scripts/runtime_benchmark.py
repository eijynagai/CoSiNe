#!/usr/bin/env python
import os
import sys
from pathlib import Path

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, os.path.join(project_root, "src"))

import yaml
import networkx as nx
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load method colors from config/colors.yml
color_config_path = project_root / "config" / "colors.yml"
with open(color_config_path, "r") as cf:
    cfg = yaml.safe_load(cf)
METHOD_COLORS = cfg["method_colors"]

import logging
import time

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.label_propagation import run_label_propagation
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

logging.basicConfig(level=logging.INFO)

methods = {
    "LouvainSigned": ("signed", {"alpha": 1.0, "resolution": 1.0}),
    "Leiden": ("pos", {"resolution": 1.0}),
}

sizes = [1000, 2000, 5000]
degrees = [5, 10, 20]
mixings = [0.05, 0.2, 0.5]
seeds = list(range(1, 11))  # seeds 1 through 10 for robust estimates
seeds = list(range(1, 11))  # seeds 1 through 10 for robust estimates
logging.info(f"Running benchmarks with seeds: {seeds}")

total_runs = len(sizes) * len(degrees) * len(mixings) * len(seeds) * len(methods)
pbar = tqdm(total=total_runs, desc="Benchmark runs", unit="run")

failure_log = []

# --- Parallelized benchmark logic ---
def run_benchmark_task(n, k, mu, seed, name, gtype, params):
    # Graph generation with increased iterations, tolerance, and retry
    build_time = None
    res = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            res = generate_signed_LFR_benchmark_graph(
                n=n, tau1=3.0, tau2=1.5, mu=mu,
                P_minus=0.2, P_plus=0.05,
                average_degree=k, min_community=50,
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

    # detection
    t0 = time.perf_counter()
    if gtype == "signed":
        run_louvain_signed(Gp, Gn, **params)
    else:
        if name == "Louvain":
            run_louvain(Gp, **params)
        elif name == "Leiden":
            run_leiden(Gp, **params)
        elif name == "Greedy":
            run_greedy_modularity(Gp)
        elif name == "Infomap":
            run_infomap(Gp)
        elif name == "LPA":
            run_label_propagation(Gp)
    detect_time = time.perf_counter() - t0

    return {
        "n": n,
        "avg_deg": k,
        "mu": mu,
        "seed": seed,
        "method": name,
        "build_time_s": round(build_time, 4),
        "detect_time_s": round(detect_time, 4),
    }

# Build all tasks
tasks = [
    (n, k, mu, seed, name, gtype, params)
    for n in sizes for k in degrees for mu in mixings
    for seed in seeds for name, (gtype, params) in methods.items()
]
total_runs = len(tasks)
records = []

# Run in parallel
max_workers = min(len(tasks), multiprocessing.cpu_count())
with ProcessPoolExecutor(max_workers=max_workers) as exe:
    futures = [
        exe.submit(run_benchmark_task, *t) for t in tasks
    ]
    for f in tqdm(as_completed(futures), total=total_runs, desc="Benchmark runs"):
        res = f.result()
        if res is not None:
            records.append(res)

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
        hue="method",
        palette=METHOD_COLORS
    )
    ax.set_title(f"Runtime vs. Degree (n={n_val})")
    ax.set_xlabel("Average degree")
    ax.set_ylabel("Runtime (s)")
    plt.tight_layout()
    out2 = PLOT_DIR / f"runtime_vs_degree_n{n_val}.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved plot: {out2}")

# 7) Plot: runtime vs mixing parameter mu per graph size
for n_val in SIZES:
    df_n = df[df["n"] == n_val]
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(
        data=df_n,
        x="mu",
        y="detect_time_s",
        hue="method",
        style="method",
        markers=True, dashes=True,
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
    print(f"Saved plot: {out3}")
