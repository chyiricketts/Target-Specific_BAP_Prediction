# Target-Specific BAP Predictions
## Chyi Ricketts
### In collaboration with the Ballester Lab at Imperial College London
### MRes in Cancer Informatics

## Creating Environment
Three Enviornments within configs: conda env create --file __.yml
1. ts_BAP.yml : main environment with typical bioinformatics tools
2. ts_aev-plig.yml : environment for aev-plig with torch packages
3. plec.yml : environment for creating PLEC features. Specific openbabel dependency

## Downloading Raw Data
1. PDBbindv2020:
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_other_PL.tar.gz (potentially have to log in through their website)
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_refined.tar.gz (potentially have to log in through their website)
   - Ensure it is placed within data/raw/pdbbind/general-set/ and data/raw/pdbbind/refined-set, respectively

2. BindingNet:
   - wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
   - Ensure it is placed within data/raw/bindingnet
  
3. BindingDB-DCS:
   - wget https://www.bindingdb.org/rwd/data/surflex/surflex.tar
   - wget https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202606_tsv.zip
   - tar -xvzf data/raw/surflex.tar -C data/raw/bindingdb/
   - Ensure it is placed within data/raw/bindingdb

## Preparation
run scripts/run_preparation.sh to create 2 files: 
- data/processed/all_proteins.csv
- data/processed/fep_prediction_info.csv

## Creating Target-Specific Training Sets and Data Splitting Methods
Run scripts/run_training-sets.sh to create training sets within data/training_sets. 
This script is computationally heavy to run as it goes through several steps to create a training set for each protein for its target specific set and at each 0.1 interval for structural similarity thresholds (using input data/structural_similarity/tm*)
1. Filters for complexes at the specified threshold or within the target-specific dataset
2. Adds ECFP6 Tanimoto Fingerprints between the train set and Ross FEP Test Set (using only the subset containing the protein of interest)
3. Creates 5-fold Data Splits for train set and validation set
   - Cluster Split using Butina 0.8
   - Random Split
   - Scaffold Split using Bemis-Murcko scaffolds
   - Stratified split for regression by binning pK values
4. Creates a report for each data split to analyze the distribution and similarity of the train/valid sets

e.g. 
../data/training_sets/MCL1
├── cluster-split
│   ├── MCL1_0.1_cluster_fold-1_train.csv
├── scaffold-split
├── random-split
├── stratified-split
└── ts-MCL1.csv


## Target-Specific AEV-PLIG


## Other Models: Tree-based and Regression Learning Algorithms

Using ECIF and PLEC Features
- Run scripts/run_plec-ecif_features.sh to create plec_features.csv and ecif_features.csv in raw_data
