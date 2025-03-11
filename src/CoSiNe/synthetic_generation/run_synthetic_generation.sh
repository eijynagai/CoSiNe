# Preprocess, estimate parameters and generate synthetic data for the PBMC dataset

# Preprocess the PBMC dataset
python3 network_cleaner.py --input ../../../data/PBMC/pbmc_cdi.csv --output ../../../data/PBMC/pbmc_cdi_clean.csv --drop 0 --sep ","

# Convert to tab delimited file
tr ',' '\t' < ../../../data/PBMC/pbmc_cdi_clean.csv > ../../../data/PBMC/pbmc_cdi_clean.tsv

# Create partition for the PBMC dataset
python3 calc_leiden_partition.py --input ../../../data/PBMC/pbmc_cdi_clean.csv --output ../../../data/PBMC/pbmc_cdi_partition.csv

# Convert to tab delimited file
tr ',' '\t' < ../../../data/PBMC/pbmc_cdi_partition.csv > ../../../data/PBMC/pbmc_cdi_partition.tsv

# Estimate parameters for the PBMC dataset
python3 est_net_prop.py -n ../../../data/PBMC/pbmc_cdi_clean.tsv -c ../../../data/PBMC/pbmc_cdi_partition.tsv

# Generate synthetic data for the PBMC dataset
python3 gen_lfr.py -n ../../../data/PBMC/pbmc_cdi_partition.json -lp binary_networks/lfr_mac -cm 20 > data_synthetic/pbmc_cdi_leiden.log
