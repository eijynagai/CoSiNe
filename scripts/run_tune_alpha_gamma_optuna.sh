source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosine

python ../src/CoSiNe/benchmarks/hyperparam_tuning/tune_alpha_gamma_optuna.py \
  --scenarios ../config/batch/scenarios_param.csv \
  --n-trials 10 \
  --seeds 42 43 44 47 51 \
  --n-jobs 4