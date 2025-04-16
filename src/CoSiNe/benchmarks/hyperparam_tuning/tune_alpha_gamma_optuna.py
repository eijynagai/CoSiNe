import logging
import os
import time

import optuna
from sklearn.metrics import normalized_mutual_info_score

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
    get_ground_truth_communities,
)

# If your script references community detection methods directly,
# you might also import them from:
# from CoSiNe.community_detection.louvain_signed import run_louvain_signed
# ... etc.

###############################################################################
# 1) CONFIGURE LOGGING (if you want logs in a file or console)
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


###############################################################################
# 2) DEFINE A HELPER FUNCTION TO RUN YOUR DETECTION & COMPUTE NMI
###############################################################################
def run_detection_and_get_nmi(alpha, gamma):
    """
    Runs your community detection with (alpha, gamma) and returns
    the average NMI over 1 or more synthetic LFR graphs.

    Steps might include:
      1. Generate or load your LFR graph(s).
      2. Run your detection method(s) with alpha, gamma (e.g., run_louvain_signed).
      3. Compare predicted communities to ground-truth.
      4. Return the mean NMI as a float.
    """
    # -------------------------------------------------------------------------
    # EXAMPLE SKELETON (pseudocode)
    # -------------------------------------------------------------------------

    # Possibly generate an LFR graph (or multiple):
    # G_signed, G_pos, G_neg = generate_signed_LFR_benchmark_graph(
    #     n=250, tau1=3.0, tau2=1.5, mu=0.1,
    #     P_minus=0.5, P_plus=0.8, min_community=20,
    #     average_degree=5, seed=10
    # )

    # # Then run detection with alpha, gamma (e.g. "louvain_signed")
    # # This is just an example if your code looks like run_louvain_signed(pos, neg, alpha=..., resolution=...)
    # # or if you prefer a separate function that returns predicted communities:
    # communities_dict = run_louvain_signed(G_pos, G_neg, alpha=alpha, resolution=gamma)

    # # Convert that dictionary into a list of predicted labels
    # node_list = sorted(G_signed.nodes())
    # predicted = [communities_dict[node] for node in node_list]

    # # Compare predicted to ground truth
    # ground_truth = get_ground_truth_communities(G_signed)
    # nmi = normalized_mutual_info_score(ground_truth, predicted)

    # # If you want multiple runs or multiple graphs, average them:
    # mean_nmi = nmi

    # # Return the final float
    return nmi


###############################################################################
# 3) DEFINE THE OPTUNA OBJECTIVE FUNCTION
###############################################################################
def objective(trial):
    """
    Optuna calls this function multiple times with different alpha, gamma values,
    then we return -NMI (because Optuna by default "minimizes";
    or we can set direction="maximize" in create_study).
    """
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    gamma = trial.suggest_float("gamma", 0.5, 1.25)

    # 1) Run detection and get average NMI
    mean_nmi = run_detection_and_get_nmi(alpha, gamma)

    # 2) We want to maximize NMI => minimize negative NMI
    return -mean_nmi


###############################################################################
# 4) MAIN FUNCTION: SETUP THE STUDY, RUN OPTIMIZATION
###############################################################################
def main():
    logging.info("Starting Bayesian Optimization for alpha,gamma to maximize NMI...")

    # Create a study that "minimizes" the returned objective
    study = optuna.create_study(direction="minimize")
    # If you prefer to avoid negative, you could do:
    # study = optuna.create_study(direction="maximize")
    # and then just return mean_nmi in the objective

    # Let's do 30 trials for demonstration
    # (increase if each trial is fast or you need more thorough search)
    n_trials = 30

    start_t = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start_t

    # Extract results
    best_alpha = study.best_params["alpha"]
    best_gamma = study.best_params["gamma"]
    best_value = study.best_value  # This is negative NMI if direction="minimize"
    best_nmi = -best_value  # Convert back to positive NMI

    logging.info(
        f"Done. Best trial took alpha={best_alpha:.3f}, gamma={best_gamma:.3f}"
    )
    logging.info(f"Max NMI found: {best_nmi:.4f}")
    logging.info(f"Elapsed time: {elapsed:.2f} sec")


###############################################################################
# 5) ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
