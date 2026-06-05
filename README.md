# Target-Specific BAP Predictions
## Chyi Ricketts
### In collaboration with the Ballester Lab at Imperial College London
### MRes in Cancer Informatics

## Creating Environment
conda env create --file aev-plig-linux.yml

# Downloading Raw Data
1. PDBbindv2020:
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_other_PL.tar.gz (potentially have to log in through their website)
   - wget http://pdbbind-plus.org.cn/download/PDBbind_v2020_refined.tar.gz (potentially have to log in through their website)
   - tar -xvzf data/raw/PDBbind_v2020_other_PL.tar.gz -C data/raw/pdbbind/general-set/
   - tar -xvzf data/raw/PDBbind_v2020_other_PL.tar.gz -C data/raw/pdbbind/refined-set/
   - Ensure it is placed within data/raw/pdbbind

2. BindingNet:
   - wget to http://bindingnet.huanglab.org.cn/api/api/download/binding_database
   - tar -xvzf data/raw/binding_database.tar.gz -C data/raw/bindingnet/
   - Ensure it is placed within data/raw/bindingnet
  
3. BindingDB-DCS:
   - wget https://www.bindingdb.org/rwd/data/surflex/surflex.tar
   - tar -xvzf data/raw/surflex.tar -C data/raw/bindingdb/
   - Ensure it is placed within data/raw/bindingdb
   
  
## Filtering Target-Specific Sets using Metadata

## Target-Specific AEV-PLIG

## Other Models
