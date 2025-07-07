#!/usr/bin/env python
import os, sys, time, logging, json, platform
from pathlib import Path
import multiprocessing
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

import yaml
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.metrics import f1_score, average_precision_score
from skimage.metrics import variation_of_information  # if installed, or implement VI manually

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.label_propagation import run_label_propagation

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# load colors
with open(project_root / "config" / "colors.yml") as cf:
    METHOD_COLORS = yaml.safe_load(cf)["method_colors"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TASK_TIMEOUT = 60  # sec per graph+detect

# methods to compare
METHODS = {
    "LouvainSigned": ("signed", {"alpha":1.0, "resolution":1.0}),
    "Louvain":       ("pos",    {"resolution":1.0}),
    "Leiden":        ("pos",    {"resolution":1.0}),
    "Greedy":        ("pos",    {}),
    "LPA":           ("pos",    {}),
}

# sweep parameters
MUS     = [0.05, 0.20, 0.50]
P_MINUS = [0.1, 0.3, 0.5]
P_PLUS  = [0.1, 0.3, 0.5]
SEEDS   = list(range(1, 11))

# fixed LFR exponents
TAU1, TAU2 = 3.0, 1.5
N, AVG_DEG, MIN_COMM = 2000, 10, 50

# ------------------------------------------------------------------------------
# helper to run one method on one graph
# ------------------------------------------------------------------------------
failure_log = []

def worker(task):
    mu, p_minus, p_plus, seed, mname, (gtype, params) = task
    # 1) generate
    try:
        t0_build = time.perf_counter()
        Gs, Gp, Gn = generate_signed_LFR_benchmark_graph(
            n=N, tau1=TAU1, tau2=TAU2, mu=mu,
            P_minus=p_minus, P_plus=p_plus,
            average_degree=AVG_DEG, min_community=MIN_COMM,
            seed=seed,
        )
        build_time = time.perf_counter() - t0_build
    except Exception as e:
        failure_log.append(dict(mu=mu, P_minus=p_minus, P_plus=p_plus,
                                seed=seed, method=mname, error=f"gen:{e}"))
        return None

    nodes = sorted(Gs.nodes())
    # 2) run
    try:
        t0 = time.perf_counter()
        if gtype=="signed":
            comm = run_louvain_signed(Gp, Gn, **params)
        else:
            if mname=="Louvain":
                comm = run_louvain(Gp, **params)
            elif mname=="Leiden":
                comm = run_leiden(Gp, **params)
            elif mname=="Greedy":
                comm = run_greedy_modularity(Gp)
            elif mname=="LPA":
                comm = run_label_propagation(Gp)
        detect_time = time.perf_counter()-t0
    except Exception as e:
        failure_log.append(dict(mu=mu, P_minus=p_minus, P_plus=p_plus,
                                seed=seed, method=mname, error=f"det:{e}"))
        return None

    # 3) ground truth & flat lists
    gt = [Gs.nodes[n]["community"] for n in nodes]
    pred = [comm.get(n,-1) for n in nodes]  # fill missing

    # 4) pairwise AUPRC using index-based comparison
    y_true, y_score = [], []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            y_true.append(int(gt[i] == gt[j]))
            y_score.append(int(pred[i] == pred[j]))
    auprc = average_precision_score(y_true, y_score)

    # 5) frustration index = frac of neg-edge inside communities
    neg_inside = 0; total_neg=0
    for u,v,d in Gs.edges(data=True):
        if d['weight']<0:
            total_neg+=1
            if comm.get(u)==comm.get(v): neg_inside+=1
    frustration = neg_inside/total_neg if total_neg>0 else 0.0

    # 6) variation of information (if available)
    try:
        vi = variation_of_information(gt,pred)
    except:
        vi = np.nan

    return dict(
        mu=mu, P_minus=p_minus, P_plus=p_plus, Seed=seed, Method=mname,
        NMI=normalized_mutual_info_score(gt,pred),
        ARI=adjusted_rand_score(gt,pred),
        F1=f1_score(gt,pred,average="macro"),
        AUPRC=auprc,
        VI=vi,
        frustration=frustration,
        detect_time_s=round(detect_time,4),
        BuildTime=round(build_time, 4),
    )

# ------------------------------------------------------------------------------
# build task list
# ------------------------------------------------------------------------------
def main():
    tasks = []
    for mu in MUS:
        for pm in P_MINUS:
            for pp in P_PLUS:
                for seed in SEEDS:
                    for name,info in METHODS.items():
                        tasks.append((mu,pm,pp,seed,name,info))

    logging.info(f"Total tasks: {len(tasks)}")

    # ------------------------------------------------------------------------------
    # execute in parallel
    # ------------------------------------------------------------------------------
    results = []
    with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 30)) as exe:
        futures = {exe.submit(worker,t):t for t in tasks}
        for f in as_completed(futures):
            try:
                out = f.result(timeout=TASK_TIMEOUT)
            except TimeoutError:
                mu, pm, pp, seed, mname, _ = futures[f]
                failure_log.append({
                    "mu": mu,
                    "P_minus": pm,
                    "P_plus": pp,
                    "Seed": seed,
                    "Method": mname,
                    "error": "timeout"
                })
                continue
            except Exception as e:
                failure_log.append(dict(error=str(e)))
                continue
            if out: results.append(out)

    # ------------------------------------------------------------------------------
    # save raw & failures
    # ------------------------------------------------------------------------------
    os.makedirs(project_root/"results",exist_ok=True)
    pd.DataFrame(results).to_csv(project_root/"results"/"performance_benchmark.csv",index=False)
    pd.DataFrame(failure_log).to_csv(project_root/"results"/"performance_benchmark_failures.csv",index=False)
    logging.info("Saved %d records, %d failures", len(results), len(failure_log))

    # --- summary stats per method / parameter combo ---
    df = pd.DataFrame(results)
    summary = df.groupby(["Method","mu","P_minus","P_plus"]).agg(
        NMI_mean=("NMI","mean"), NMI_std=("NMI","std"),
        ARI_mean=("ARI","mean"), ARI_std=("ARI","std"),
        F1_mean=("F1","mean"), F1_std=("F1","std"),
        AUPRC_mean=("AUPRC","mean"), AUPRC_std=("AUPRC","std"),
        VI_mean=("VI","mean"), VI_std=("VI","std"),
        frustration_mean=("frustration","mean"), frustration_std=("frustration","std"),
        BuildTime_mean=("BuildTime","mean"), BuildTime_std=("BuildTime","std"),
        detect_time_mean=("detect_time_s","mean"), detect_time_std=("detect_time_s","std"),
    ).reset_index()
    summary.to_csv(project_root/"results"/"performance_benchmark_summary.csv", index=False)
    logging.info("Saved summary stats to results/performance_benchmark_summary.csv")

    # metadata
    meta = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        cpus=multiprocessing.cpu_count(),
        mem_mb=psutil.virtual_memory().total/1e6,
    )
    meta["packages"] = {
        "networkx": nx.__version__,
        "pandas": pd.__version__,
        "seaborn": sns.__version__,
        "scikit-learn": sklearn.__version__,
    }
    with open(project_root/"results"/"performance_benchmark_metadata.json","w") as f:
        json.dump(meta,f,indent=2)

    # ------------------------------------------------------------------------------
    # plotting: boxplots per metric
    # ------------------------------------------------------------------------------
    plot_dir = project_root/"results"/"plots"
    plot_dir.mkdir(exist_ok=True)

    metrics = ["NMI","ARI","F1","AUPRC","VI","frustration"]
    for m in metrics:
        plt.figure(figsize=(8,5))
        ax = sns.boxplot(data=df, x="Method", y=m, palette=METHOD_COLORS)
        ax.set_title(f"{m} by method")
        ax.set_xlabel("")
        ax.set_ylabel(m)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_dir/f"performance_{m}.png",dpi=300)
        plt.close()

    # ------------------------------------------------------------------------------
    # paired delta‐plots: signed vs leiden
    # ------------------------------------------------------------------------------
    ref = df[df.Method=="Leiden"]
    sig = df[df.Method=="LouvainSigned"]
    for m in ["NMI","ARI","F1"]:
        merged = pd.merge(sig, ref, on=["mu","P_minus","P_plus","Seed"], suffixes=("_sig","_ref"))
        merged["Δ"+m] = merged[f"{m}_sig"]-merged[f"{m}_ref"]
        plt.figure(figsize=(6,4))
        ax = sns.barplot(x="mu", y="Δ"+m, data=merged, palette="vlag")
        ax.set_title(f"Δ{m}: SignedLouvain – Leiden")
        ax.set_xlabel("Mixing μ")
        ax.set_ylabel(f"Δ{m}")
        plt.tight_layout()
        plt.savefig(plot_dir/f"delta_{m}.png",dpi=300)
        plt.close()

    logging.info("Performance benchmark complete.")


if __name__ == "__main__":
    main()