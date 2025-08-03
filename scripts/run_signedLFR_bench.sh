#!/usr/bin/env bash
# issues in dev laptop with poetry and envs
#conda deactivate 2>/dev/null || true

#unning on servers
# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosine

# Clean results
#bash run_clean.sh

# Run test 1: simple parameters
#poetry run python -m CoSiNe.benchmarks.benchmark_community_detection_signed_graph
#tmux new-session -d -s "signedLFR" poetry run python -m  CoSiNe.benchmarks.run_pipeline

# Run test 2: using list of params.csv 
#time poetry run python -m CoSiNe.benchmarks.run_pipeline

# Run test 3: hyperparameter tunning --> working
#time poetry run python -m CoSiNe.benchmarks.hyperparam_tuning.tune_alpha_gamma_optuna

# Run method comparison
#time poetry run python compare_methods.py
#time poetry run python correlation_analysis.py
#time python compare_methods.py
python ../src/CoSiNe/benchmarks/hyperparam_tuning/tune_alpha_gamma_optuna.py \
  --scenarios ../config/batch/scenarios_param.csv \
  --n-trials 10 \
  --seeds 3 5 13 47 82 \
  --n-jobs 30

# Run runtime comparison #2
#time poetry run python runtime_benchmark.py
#time poetry run python plot_runtime.py
python runtime_benchmark.py \
  --scenarios /work3/Users/nagai/projects/CoSiNe/config/batch/scenarios_param.csv \
  --output-dir  /work3/Users/nagai/projects/CoSiNe/results/ \
  --timeout  60 \
  --seeds  3 5 13 47 82

# Run performance_benchmark
#time poetry run python performance_benchmark.py
python performance_benchmark.py \
  --scenarios /work3/Users/nagai/projects/CoSiNe/config/batch/scenarios_param.csv \
  --output-dir /work3/Users/nagai/projects/CoSiNe/results/ \
  --timeout 60 \
  --seeds 3 5 13 47 82