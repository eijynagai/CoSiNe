#!/usr/bin/env python3
"""
Generate scenarios_param.csv with the recommended 5x3x2x3 grid:
  μ in [0.4,0.5,0.6,0.7,0.8]
  P_minus in [0.1,0.2,0.3]
  average_degree in [10,20]
  n in [2000,5000,10000]
Fixed parameters: tau1=3.0, tau2=1.5, P_plus=0.9, min_community=20.
"""
import itertools
import pandas as pd

# Define grid
mus = [0.4, 0.5, 0.6, 0.7, 0.8]
p_minuses = [0.1, 0.2, 0.3]
avg_degs = [10, 20]
ns = [2000, 5000, 10000]

#Generate scenarios_param.csv with a smaller “quick-and-dirty” subset grid:
#mus = [0.4, 0.6, 0.8]
#p_minuses = [0.1, 0.3]
#avg_degs = [10]
#ns = [2000, 10000]

rows = []
for n, mu, p_minus, avg_deg in itertools.product(ns, mus, p_minuses, avg_degs):
    scenario = f"n{n}_mu{int(mu*100)}_pm{int(p_minus*100)}_k{avg_deg}"
    rows.append({
        "scenario": scenario,
        "n": n,
        "tau1": 3.0,
        "tau2": 1.5,
        "mu": mu,
        "P_minus": p_minus,
        "P_plus": 0.9,
        "avg_deg": avg_deg,
        "min_community": 20,
    })

df = pd.DataFrame(rows)
csv_path = "scenarios_param_fig2.csv"
df.to_csv(csv_path, index=False)
print(f"Generated {len(df)} scenarios in {csv_path}")