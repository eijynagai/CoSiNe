import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import optuna.visualization as vis
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

# ensure the top-level 'src' directory (containing the CoSiNe package) is on PYTHONPATH
script_path = Path(__file__).resolve()
src_dir = next(p for p in script_path.parents if (p / "CoSiNe").is_dir())
sys.path.insert(0, str(src_dir))

# Default seeds, can be overridden via CLI
DEFAULT_SEEDS = [42, 43, 44, 47, 51]

# placeholder for scenario definitions; will be loaded from CSV if provided
SCENARIOS = [
    {
        "n": 10000,
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.12,
        "P_minus": 0.1,
        "P_plus": 0.01,
        "min_community": 20,
        "average_degree": 25,
    },
]

###############################################################################
# 1) CONFIGURE LOGGING (if you want logs in a file or console)
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


###############################################################################
# 2) DEFINE A HELPER FUNCTION TO RUN COMMUNITY DETECTION & COMPUTE NMI
###############################################################################
def run_detection_and_get_nmi(alpha, gamma, seeds):
    """
    Runs community detection with (alpha, gamma) and returns.

    the average NMI over all scenarios x seeds.
    """
    nmis = []
    for cfg in SCENARIOS:
        for seed in seeds:
            try:
                g_signed, g_pos, g_neg = generate_signed_LFR_benchmark_graph(
                    n=cfg["n"],
                    tau1=cfg["tau1"],
                    tau2=cfg["tau2"],
                    mu=cfg["mu"],
                    P_minus=cfg["P_minus"],
                    P_plus=cfg["P_plus"],
                    min_community=cfg["min_community"],
                    average_degree=cfg["average_degree"],
                    seed=seed,
                )
            except (ValueError, RuntimeError) as e:
                logger.warning("Scenario %s seed %s: LFR failed: %s", cfg, seed, e)
                continue
            communities = run_louvain_signed(
                g_pos, g_neg, alpha=alpha, resolution=gamma
            )
            nodes = sorted(g_signed.nodes())
            gt = [g_signed.nodes[n]["community"] for n in nodes]
            pred = [communities[n] for n in nodes]
            nmis.append(normalized_mutual_info_score(gt, pred))
    return sum(nmis) / len(nmis) if nmis else 0.0


###############################################################################
# 3) DEFINE THE OPTUNA OBJECTIVE FUNCTION
###############################################################################
def objective(trial, seeds):
    """
    Optuna calls this function multiple times with different alpha, gamma values.

    Then we return NMI.
    """
    alpha = trial.suggest_float("alpha", 0.1, 1.0)
    gamma = trial.suggest_float("gamma", 0.1, 3.0)

    # 1) Run detection and get average NMI
    mean_nmi = run_detection_and_get_nmi(alpha, gamma, seeds)

    # 2) We want to maximize NMI
    return mean_nmi


###############################################################################
# 4) MAIN FUNCTION: SETUP THE STUDY, RUN OPTIMIZATION
###############################################################################
def main(args):
    logger.info("Starting Bayesian Optimization for alpha,gamma to maximize NMI...")
    seeds = args.seeds
    n_trials = args.n_trials

    # Create a study that "maximizes" the returned objective
    study = optuna.create_study(direction="maximize")

    start_t = time.time()
    study.optimize(
        lambda trial: objective(trial, seeds), n_trials=n_trials, n_jobs=args.n_jobs
    )
    elapsed = time.time() - start_t

    # Extract results
    best_alpha = study.best_params["alpha"]
    best_gamma = study.best_params["gamma"]
    best_value = study.best_value

    logger.info(
        "Done. Best trial took alpha=%.3f, gamma=%.3f",
        best_alpha,
        best_gamma,
    )
    logger.info("Max NMI found: %.4f", best_value)
    logger.info("Elapsed time: %.2f sec", elapsed)

    best_params = {
        "alpha": study.best_params["alpha"],
        "gamma": study.best_params["gamma"],
        "nmi": study.best_value,
    }

    # write results into the package directory, not the CWD
    output_dir = (
        Path(src_dir) / "CoSiNe" / "benchmarks" / "hyperparam_tuning" / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "best_params_nmi.json").open("w") as f:
        json.dump(best_params, f, indent=4)

    trials_dataframe = study.trials_dataframe()
    trials_dataframe.to_csv(str(output_dir / "optuna_trials_nmi.csv"), index=False)

    # Generate and save plots
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html(str(output_dir / "optimization_history.html"))

    fig2 = vis.plot_param_importances(study)
    fig2.write_html(str(output_dir / "param_importances.html"))


###############################################################################
# 5) ENTRY POINT
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for LouvainSigned",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials (reduced for quick runs)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of random seeds for averaging",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Path to CSV of LFR scenario parameters",
    )
    args = parser.parse_args()
    # load scenarios CSV if provided
    if args.scenarios:
        scenarios_df = pd.read_csv(args.scenarios)
        SCENARIOS.clear()
        # Determine column names for avg degree and min community
        avg_deg_col = (
            "average_degree" if "average_degree" in scenarios_df.columns else "avg_deg"
        )
        min_comm_col = (
            "min_community" if "min_community" in scenarios_df.columns else "min_comm"
        )
        required_columns = [
            "n",
            "tau1",
            "tau2",
            "mu",
            "P_minus",
            "P_plus",
            min_comm_col,
            avg_deg_col,
        ]
        missing_cols = [
            col for col in required_columns if col not in scenarios_df.columns
        ]
        if missing_cols:
            logger.exception("Missing columns %s in scenarios CSV", missing_cols)
            sys.exit(1)
        for _, row in scenarios_df.iterrows():
            SCENARIOS.append(
                {
                    "n": int(row["n"]),
                    "tau1": float(row["tau1"]),
                    "tau2": float(row["tau2"]),
                    "mu": float(row["mu"]),
                    "P_minus": float(row["P_minus"]),
                    "P_plus": float(row["P_plus"]),
                    "min_community": int(row[min_comm_col]),
                    "average_degree": float(row[avg_deg_col]),
                }
            )
    main(args)
