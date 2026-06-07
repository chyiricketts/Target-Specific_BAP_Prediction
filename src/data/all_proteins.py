# creates all_proteins.csv
# a csv with the path to all pdb files

from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

all_proteins = []
main_folder = Path.cwd().parents[0]

# pdbbind
pdbbind_df = pd.read_csv(os.path.join(main_folder, "data/files/pdbbind_processed.csv"), index_col=0)
for i, pdb in tqdm(enumerate(pdbbind_df["PDB_code"])):
    if pdbbind_df["refined"][i]:
        folder = "data/raw/pdbbind/refined-set/"
    else:
        folder = "data/raw/pdbbind/general-set/"
    protein_path = os.path.join(main_folder, folder, pdb, f'{pdb}_protein.pdb')
    if not os.path.exists(protein_path): print("Path does not exist:", protein_path)
    unique_id = f"pdbbind_{pdb}"
    all_proteins.append({
        "unique_id": unique_id,
        "path": protein_path
    })

# bindingnet
bindingnet_df = pd.read_csv(os.path.join(main_folder, "data/files/bindingnet_processed.csv"), index_col=0)
folder = "data/raw/bindingnet/from_chembl_client/"
for index, row in tqdm(bindingnet_df.iterrows()):
    unique_identify = row['unique_identify']
    target = row['target']
    pdb = row['pdb']
    compnd = row['compnd']
    protein_path = os.path.join(main_folder, folder + f"{pdb}/rec_h_opt.pdb")
    if not os.path.exists(protein_path): print("Path does not exist:", protein_path)
    unique_id = f"bindingnet_{unique_identify}"
    all_proteins.append({
        "unique_id": unique_id,
        "path": protein_path
    })

# bindingdb
bindingdb_df = pd.read_csv(os.path.join(main_folder, "data/files//bindingdb_processed.csv"), index_col=0)
folder = "data/raw/bindingdb/surflex/"
for index, row in tqdm(bindingdb_df.iterrows()):
    protein_path = os.path.join(main_folder, folder, row["folder"], row["pdb_file"])
    if not os.path.exists(protein_path): print("Path does not exist:", protein_path)
    uid = row["unique_id"]
    unique_id = f"bindingdb_{uid}"
    all_proteins.append({
        "unique_id": unique_id,
        "path": protein_path
    })

print("Length of protein df: ", len(all_proteins))

df = pd.DataFrame(all_proteins)
df.to_csv(os.path.join(main_folder, "data/processed/all_proteins.csv"), index=False)