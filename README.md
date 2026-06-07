# Target-Specific BAP Predictions
## Chyi Ricketts
### In collaboration with the Ballester Lab at Imperial College London
### MRes in Cancer Informatics

## Creating Environment
conda env create --file aev-plig-linux.yml <- fix this

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

## Filtering Target-Specific Sets using Metadata

## Target-Specific AEV-PLIG

## Other Models

## Logic for my own reference

project/
├── src/                     # ALL Python logic (reusable code)
│   ├── data/
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/
│   │   ├── model.py
│   │   └── train.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils.py
│
├── scripts/                # ENTRY POINTS (what you actually run)
│   ├── preprocess.sh
│   ├── split.sh
│   ├── train.sh
│   └── run_all.sh
│
├── configs/               # experiment settings (VERY important for reproducibility)
│   ├── preprocess.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── outputs/
│   ├── logs/
│   ├── models/
│   └── results/
│
└── README.md