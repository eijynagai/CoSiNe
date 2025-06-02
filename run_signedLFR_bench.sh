# Run test 1: simple parameters
#poetry run python -m CoSiNe.benchmarks.benchmark_community_detection_signed_graph
#tmux new-session -d -s "signedLFR" poetry run python -m  CoSiNe.benchmarks.run_pipeline

# Run test 2: using list of params.csv 
#time poetry run python -m CoSiNe.benchmarks.run_pipeline


# Run test 3: hyperparameter tunning
time poetry run python -m CoSiNe.benchmarks.hyperparam_tuning.tune_alpha_gamma_optuna