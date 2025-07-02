import json
import logging
import time
from pathlib import Path

import optuna
import optuna.visualization as vis
from sklearn.metrics import normalized_mutual_info_score

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

###############################################################################
# 1) CONFIGURE LOGGING (if you want logs in a file or console)
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


###############################################################################
# 2) DEFINE A HELPER FUNCTION TO RUN COMMUNITY DETECTION & COMPUTE NMI
###############################################################################
def run_detection_and_get_nmi(alpha, gamma):
    """
    Runs community detection with (alpha, gamma) and returns
    the average NMI over 1 or more synthetic LFR graphs.

    Steps include:
      1. Generate or load LFR graph(s).
      2. Run detection method(s) with alpha, gamma (run_louvain_signed).
      3. Compare predicted communities to ground-truth.
      4. Return the mean NMI as a float.
    """
    # Averaging Over Multiple Seeds
    seeds = [42, 43, 44]
    nmis = []
    for seed in seeds:
        G_signed, G_pos, G_neg = generate_signed_LFR_benchmark_graph(
            n=10000,
            tau1=3,
            tau2=1.5,
            mu=0.12,
            P_minus=0.1,
            P_plus=0.01,
            min_community=20,
            average_degree=25,
            seed=seed,
        )
        communities = run_louvain_signed(G_pos, G_neg, alpha=alpha, resolution=gamma)
        nodes = sorted(G_signed.nodes())
        ground_truth = [G_signed.nodes[n]["community"] for n in nodes]
        predicted = [communities[n] for n in nodes]
        nmis.append(normalized_mutual_info_score(ground_truth, predicted))
    return sum(nmis) / len(nmis)


###############################################################################
# 3) DEFINE THE OPTUNA OBJECTIVE FUNCTION
###############################################################################
def objective(trial):
    """
    Optuna calls this function multiple times with different alpha, gamma values,
    then we return NMI
    """
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    gamma = trial.suggest_float("gamma", 0.1, 3.0)

    # 1) Run detection and get average NMI
    mean_nmi = run_detection_and_get_nmi(alpha, gamma)

    # 2) We want to maximize NMI
    return mean_nmi


###############################################################################
# 4) MAIN FUNCTION: SETUP THE STUDY, RUN OPTIMIZATION
###############################################################################
def main():
    logging.info("Starting Bayesian Optimization for alpha,gamma to maximize NMI...")

    # Create a study that "maximizes" the returned objective
    study = optuna.create_study(direction="maximize")

    # Let's do 30 trials for demonstration
    # (increase if each trial is fast or you need more thorough search)
    n_trials = 1000

    start_t = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start_t

    # Extract results
    best_alpha = study.best_params["alpha"]
    best_gamma = study.best_params["gamma"]
    best_value = study.best_value

    logging.info(
        f"Done. Best trial took alpha={best_alpha:.3f}, gamma={best_gamma:.3f}"
    )
    logging.info(f"Max NMI found: {best_value:.4f}")
    logging.info(f"Elapsed time: {elapsed:.2f} sec")

    best_params = {
        "alpha": study.best_params["alpha"],
        "gamma": study.best_params["gamma"],
        "nmi": study.best_value,
    }

    output_dir = Path("src/CoSiNe/benchmarks/hyperparam_tuning/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(output_dir / "best_params_nmi.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    df = study.trials_dataframe()
    df.to_csv(str(output_dir / "optuna_trials_nmi.csv"), index=False)

    # Generate and save plots
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html(str(output_dir / "optimization_history.html"))

    fig2 = vis.plot_param_importances(study)
    fig2.write_html(str(output_dir / "param_importances.html"))


###############################################################################
# 5) ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
