# Target-Specific BAP Predictions
### Chyi Ricketts, MRes in Cancer Informatics, under the supervision of Dr. Pedro Ballester, Imperial College London

Target-specific binding affinity prediction-based scoring functions are trained and evaluated in the accompanying [thesis](https://chyiricketts.onrender.com/papers/cr725_Ricketts_2026_Ballester_MResPaper.pdf). 

In this repo:

Methods for target-specific training set curation and model training for each model (ECIF::RF, ECIF::XGB, ECIF::SVR, PLEC::RF, PLEC::XGB, PLEC::SVR, and AEV-PLIG::GATv2) are shared
The best performing models for each of the five protein targets: MCL1, SYK, HIF2α, PFKFB3, and MAPK14 are shared along with their hyperparameters and computational specifications"


## Creating Environments

Three environments are available within `configs/`:

```bash
conda env create --file ts_BAP.yml
conda env create --file ts_aev-plig.yml
conda env create --file plec.yml
```

1. **ts_BAP.yml**
   - Main environment with typical bioinformatics tools.

2. **ts_aev-plig.yml**
   - Environment for AEV-PLIG with PyTorch packages.

3. **plec.yml**
   - Environment for creating PLEC features.
   - Requires a specific OpenBabel dependency.


## Downloading Raw Data

### 1. PDBbind v2020

```bash
wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_other_PL.tar.gz
wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_refined.tar.gz
```

> Due to data access restrictions, you must manually log in at:
>
> http://pdbbind-plus.org.cn

Place the files in:

```text
data/raw/pdbbind/general-set/
data/raw/pdbbind/refined-set/
```

### 2. BindingNet

```bash
wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
```

Place the file in:

```text
data/raw/bindingnet/
```

### 3. BindingDB-DCS

```bash
wget https://www.bindingdb.org/rwd/data/surflex/surflex.tar
wget https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202606_tsv.zip
```

Place the files in:

```text
data/raw/bindingdb/
```


## Preparation

Run:

```bash
bash scripts/run_preparation.sh
```

This generates:

```text
data/processed/all_proteins.csv
data/processed/fep_prediction_info.csv
```


## Creating Target-Specific Training Sets and Data Splitting Methods

Run:

```bash
bash scripts/run_training-sets.sh
```

Outputs are created in:

```text
data/training_sets/
```

The script:

1. Filters complexes at the specified structural similarity threshold.
2. Adds ECFP6 Tanimoto fingerprint similarity between the training set and Ross FEP test set.
3. Creates 5-fold train/validation splits:
   - Cluster split (Butina 0.8)
   - Random split
   - Scaffold split (Bemis–Murcko)
   - Stratified regression split
4. Generates reports for split analysis.

Example output structure:

```text
data/training_sets/MCL1/
├── cluster-split/
│   ├── MCL1_0.1_cluster_fold-1_train.csv
│   └── ...
├── scaffold-split/
├── random-split/
├── stratified-split/
└── ts-MCL1.csv
```



## Target-Specific AEV-PLIG

### Preparation

Run:

```bash
bash scripts/run_fep_graphs.sh
```

Underlying script:

```text
src/models/AEV-PLIG-models/fep_graphs.py
```

Output:

```text
data/processed/fep_graphs.pickle
```

### AEV-PLIG and GATv2

For an example run:

```bash
bash scripts/AEV-PLIG_example/run_aev-plig.sh
```

Pipeline:

```text
src/models/AEV-PLIG-models/generate_graphs.py
src/models/AEV-PLIG-models/create_pytorch_data.py
src/models/AEV-PLIG-models/train.py
src/models/AEV-PLIG-models/fep_predictions.py
```

Outputs:

```text
AEV-PLIG_example/
├── graphs.pkl
├── processed/
├── json/
├── models/
├── logs/
└── AEV-PLIG_example_predictions.csv
```


## Tree-Based and Regression Models

### Feature Generation

Run:

```bash
bash scripts/run_plec-ecif_features.sh
```

Scripts:

```text
src/models/tree-regression-models/plec_features.py
src/models/tree-regression-models/ecif_features.py
```

Outputs:

```text
data/processed/raw_all_ecif_features.csv
data/processed/fep_ecif_features.csv
data/processed/raw_all_plec_features.csv
data/processed/fep_plec_features.csv
```

### Training and Evaluation

Example:

```bash
bash scripts/tree-regression_example/run_tree-regression.sh
```

Pipeline:

```text
src/models/tree-regression-models/merge_dataset.py
src/models/tree-regression-models/train_models.py
src/models/tree-regression-models/eval_script.py
```

Outputs:

```text
tree-regression_example/
├── models_cv/
├── models_seeded/
├── json/
├── figures/
├── tree-regression_example_cv_predictions.csv
└── tree-regression_example_seeded_predictions.csv
```

## Final Hyperparameters

Hyperparameter files from Optuna optimisation are provided in `hyperparameters/`.

## Final Predictions

Test set predictions on the Ross FEP Benchmark are provided for the best famiily-specific ECIF::RF model 
configurations in `predictions/`. The best family-specific ECIF::RF model is
selected by Overlap@5 with Kendall's τ used as a tiebreaker.

| Protein | Best Threshold |
|---------|---------------|
| MCL1    | 0.9           |
| SYK     | 0.3           |
| HIF2α   | 0.5           |
| PFKFB3  | 0.3           |
| MAPK14  | 0.4           |


