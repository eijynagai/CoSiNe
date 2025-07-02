#!/usr/bin/env python
import logging
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.label_propagation import run_label_propagation
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

sys.path.append(os.path.abspath("src"))

logging.basicConfig(level=logging.INFO)

methods = {
    "LouvainSigned": ("signed", {"alpha": 1.0, "resolution": 1.0}),
    "Leiden": ("pos", {"resolution": 1.0}),
}

sizes = [1000, 2000, 5000, 10000]
degrees = [5, 10, 20]
mixings = [0.05, 0.2, 0.5]
seeds = [42, 43, 44]

total_runs = len(sizes) * len(degrees) * len(mixings) * len(seeds) * len(methods)
pbar = tqdm(total=total_runs, desc="Benchmark runs", unit="run")

records = []
for n in sizes:
    for k in degrees:
        for mu in mixings:
            for seed in seeds:
                Gs, Gp, Gn = generate_signed_LFR_benchmark_graph(
                    n=n,
                    tau1=3.0,
                    tau2=1.5,
                    mu=mu,
                    P_minus=0.2,
                    P_plus=0.05,
                    average_degree=k,
                    min_community=50,
                    seed=seed,
                )
                for name, (gtype, params) in methods.items():
                    start = time.perf_counter()
                    if gtype == "signed":
                        _ = run_louvain_signed(Gp, Gn, **params)
                    else:
                        if name == "Louvain":
                            _ = run_louvain(Gp, **params)
                        elif name == "Leiden":
                            _ = run_leiden(Gp, **params)
                        elif name == "Greedy":
                            _ = run_greedy_modularity(Gp)
                        elif name == "Infomap":
                            _ = run_infomap(Gp)
                        elif name == "LPA":
                            _ = run_label_propagation(Gp)
                    elapsed = time.perf_counter() - start

                    records.append(
                        {
                            "n": n,
                            "avg_deg": k,
                            "mu": mu,
                            "seed": seed,
                            "method": name,
                            "runtime_s": round(elapsed, 4),
                        }
                    )
                    pbar.update(1)

pbar.close()

df = pd.DataFrame(records)
os.makedirs("results", exist_ok=True)
df.to_csv("results/runtime_benchmark.csv", index=False)
