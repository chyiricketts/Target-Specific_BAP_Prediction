# Target-Specific BAP Predictions
### Chyi Ricketts, MRes in Cancer Informatics
In collaboration with the Ballester Lab at Imperial College London

## Creating Environment
Three Enviornments within configs: conda env create --file __.yml
1. ts_BAP.yml : main environment with typical bioinformatics tools
2. ts_aev-plig.yml : environment for aev-plig with torch packages
3. plec.yml : environment for creating PLEC features. Specific openbabel dependency

## Downloading Raw Data
1. PDBbindv2020:
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_other_PL.tar.gz 
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_refined.tar.gz
      - Due to data security, one must log in manually through their website: http://pdbbind-plus.org.cn
   - Ensure it is placed within data/raw/pdbbind/general-set/ and data/raw/pdbbind/refined-set, respectively

2. BindingNet:
   - wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
   - Ensure it is placed within data/raw/bindingnet
  
3. BindingDB-DCS:
   - wget https://www.bindingdb.org/rwd/data/surflex/surflex.tar
   - wget https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202606_tsv.zip
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

Preparation
- Run scripts/run_fep_graphs.sh for FEP Test set graph generation required for evaluation: 
   - src/models/AEV-PLIG-models/fep_graphs.py
- Outputs: 
   - data/processed/fep_graphs.pickle

AEV-PLIG and GATv2
- See scripts/run_AEV-PLIG_template.sh for more detailed information
- For an individual model, run scripts/AEV-PLIG_example/run_aev-plig.sh
   - src/models/AEV-PLIG-models/generate_graphs.py
   - src/models/AEV-PLIG-models/create_pytorch_data.py
   - src/models/AEV-PLIG-models/train.py
   - src/models/AEV-PLIG-models/fep_predictions.py
- Outputs within AEV-PLIG_example/
   - 


## Tree-based and Regression Learning Algorithms using Fingerprint Features

Preparation
- Run scripts/run_plec-ecif_features.sh for ECIF and PLEC feature generation
   - src/models/tree-regression-models/plec_features.py
   - src/models/tree-regression-models/ecif_features.py
- Outputs: 
   - data/processed/raw_all_ecif_features.csv
   - data/processed/fep_ecif_features.csv   
   - data/processed/raw_all_plec_features.csv
   - data/processed/fep_plec_features.csv

Tree-based and Regression Learning Algorithms
- See scripts/run_tree-regression_template.sh for more detailed information
- For an individual model, run scripts/tree-regression_example/run_tree-regression.sh
   - src/models/tree-regression-models/merge_dataset.py
   - src/models/tree-regression-models/train_models.py
   - src/models/tree-regression-models/eval_script.py
- Outputs within tree-regression_example/: 
   - models_cv/ containing 5 models trained on each fold
   - models_seeded/ containing 10 models trained on different random seeds
   - json/ containing data-config.json, train_best_params.json, xgb_cv_summary.json
   - figures/ containing 3 plots for optuna optimization visualization
   - tree-regression_example_cv_predictions.csv
   - tree-regression_example_seeded_predictions.csv


## Final Models
IN PROGRESS. The best target-specific model will be placed here for reproducibility. This model still needs to be determined as this project is ongoing. 