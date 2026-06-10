# For generating ALL types of PLEC Features

import os
from tqdm import tqdm
from os import listdir
from rdkit import Chem
from openbabel import pybel
from oddt.fingerprints import PLEC
import glob 
import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path.cwd().parents[0]

def compute_plec(ligand_path, receptor_path):
    ligand = next(pybel.readfile("sdf", str(ligand_path)))
    receptor = next(pybel.readfile("pdb", str(receptor_path)))

    return PLEC(
        ligand=ligand,
        protein=receptor,
        depth_protein=4,
        depth_ligand=2,
        distance_cutoff=4.5,
        sparse=False
    )


# Start of database spplecic generation

def build_plec_dataframe(unique_ids, plecs, extra_cols=None):
    if len(plecs) == 0:
        raise ValueError("No PLEC features generated.")

    feature_matrix = np.vstack(plecs)

    plec_cols = [f"PLEC_{i}" for i in range(feature_matrix.shape[1])]

    plec_df = pd.DataFrame(feature_matrix, columns=plec_cols)
    plec_df.insert(0, "unique_id", unique_ids)

    if extra_cols:
        for idx, (col_name, values) in enumerate(extra_cols.items(), start=1):
            plec_df.insert(idx, col_name, values)

    return plec_df


def generate_PDBbind_PLEC():
    print("Generating PLEC features for pdbbind")

    data = pd.read_csv(os.path.join(BASE_DIR, "data", "files", "pdbbind_processed.csv"), index_col=0)

    unique_ids = []
    plecs = []
    failed_list = []

    for row in data.itertuples():
        
        if row.refined:
            folder = os.path.join(BASE_DIR, "data/raw/pdbbind/refined-set")
        else:
            folder = os.path.join(BASE_DIR, "data/raw/pdbbind/general-set")

        pdb = row.PDB_code

        ligand = os.path.join(folder, pdb, f'{pdb}_ligand.sdf')
        receptor = os.path.join(folder, pdb, f'{pdb}_protein.pdb')

        try: 
            plec = compute_plec(ligand, receptor)
            unique_ids.append(pdb)
            plecs.append(plec)
        except Exception as e: 
            failed_list.append(pdb)

    print(f"Length of failed: {len(failed_list)}")

    return build_plec_dataframe(unique_ids, plecs)

def generate_Bindingnet_PLEC():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "files", "bindingnet_processed.csv"), index_col=0)
    folder = os.path.join(BASE_DIR, "data/raw/bindingnet/from_chembl_client/")

    unique_ids = []
    plecs = []
    failed_list = []

    for row in df.itertuples():
        unique_identify = row.unique_identify
        target = row.target
        pdb = row.pdb
        compnd = row.compnd

        sdf_file = folder + f"{pdb}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf"
        protein_path = folder + f"{pdb}/rec_h_opt.pdb"

        receptor = protein_path
        ligand = sdf_file

        try: 
            plec = compute_plec(ligand, receptor)
            unique_ids.append(unique_identify)
            plecs.append(plec)
        except Exception as e: 
            failed_list.append(pdb)
        
    print(f"Length of failed: {len(failed_list)}")

    return build_plec_dataframe(unique_ids, plecs)


def generate_BindingDB_PLEC():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/files/bindingdb_processed.csv"), index_col=0)
    folder = os.path.join(BASE_DIR, "data/raw/bindingdb/surflex/")

    unique_ids = []
    plecs = []
    failed_list = []

    for row in df.itertuples():
        mol2_file = folder + row.folder+ "/" + row.mol2_file
        sdf_path_true = Path(mol2_file.replace(".mol2", ".sdf"))
        sdf_path = mol2_file.replace(".mol2", ".sdf")
        unique_id = row.unique_id

        # Only create SDF if it doesn't exist
        if not sdf_path_true.exists():
            lig = Chem.MolFromMol2File(str(mol2_file))
            lig = Chem.AddHs(lig, addCoords=True)
            Chem.MolToMolFile(lig, str(sdf_path))
            print("creating sdf file")

        protein_path = folder + row.folder + "/" + row.pdb_file

        receptor = protein_path
        ligand = sdf_path

        try: 
            plec = compute_plec(ligand, receptor)
            unique_ids.append(unique_id)
            plecs.append(plec)
        except Exception as e: 
            failed_list.append(row.folder)  
              
    print(f"Length of failed: {len(failed_list)}")

    return build_plec_dataframe(unique_ids, plecs)


"""
# Tried using FEP data for training, but is not valid as it is augmented data, similar to the test set
def generate_FEP_PLEC_fortraining():
    fep_prediction_info = pd.read_csv(os.path.join(BASE_DIR, "fep/analysis/only_in_1343.csv"))
    print(f"Number of complexes in FEP for training/valid: {len(fep_prediction_info)}")

    unique_id_list = []
    plec_list = []
    dg_list = []
    failed_list = []

    for _, row in fep_prediction_info.iterrows():
        pdb_path = row.pdb_file
        sdf_path = row.sdf_file

        try: 
            plec = compute_plec(sdf_path, pdb_path)
            plec_list.append(plec)
            unique_id_list.append(row.get("unique_id", np.nan))
        except Exception as e: 
            print(f"FAILED: {row.get('unique_id', np.nan)} | {e}")
            failed_list.append(row.get("unique_id", np.nan))
 
    return build_plec_dataframe(
        unique_id_list,
        plec_list
    )
"""


# Generates raw_fep_plec_features.csv
def generate_FEP_PLEC():
    fep_prediction_info = pd.read_csv(os.path.join(BASE_DIR, "data/processed/fep_prediction_info.csv"))
    print(f"Number of complexes in FEP test set: {len(fep_prediction_info)}")

    unique_id_list = []
    plec_list = []
    dg_list = []
    failed_list = []

    for _, row in fep_prediction_info.iterrows():
        pdb_path = row.pdb_file
        sdf_path = row.sdf_file

        try: 
            plec = compute_plec(sdf_path, pdb_path)
            plec_list.append(plec)
            unique_id_list.append(row.get("unique_id", np.nan))
            dg_list.append(row.get("dg_exp", np.nan))
        except Exception as e: 
            print(f"FAILED: {row.get('unique_id', np.nan)} | {e}")
            failed_list.append(row.get("unique_id", np.nan))
 
    return build_plec_dataframe(
        unique_id_list,
        plec_list,
        extra_cols={"dg_exp": dg_list}
    )


if __name__ == "__main__": 

    # GENERATING RAW FEATURES FOR DATABASES -- TAKES 4 HOURS
    merged_out = os.path.join(BASE_DIR, "data/processed/raw_all_plec_features.csv")

    # Generate directly into memory
    pdbbind_plec_df = generate_PDBbind_PLEC()
    bindingnet_plec_df = generate_Bindingnet_PLEC()
    bindingdb_plec_df = generate_BindingDB_PLEC()
    #fep_plec_df_training = generate_FEP_PLEC_fortraining()

    # Merge all dataframes
    all_plec_df = pd.concat(
        [pdbbind_plec_df, bindingnet_plec_df, bindingdb_plec_df],
        ignore_index=True
    )

    # Save only merged dataframe
    all_plec_df.to_csv(merged_out, index=False)

    print(
        f"Shapes:\n"
        f"  PDBbind:    {pdbbind_plec_df.shape}\n"
        f"  BindingNet: {bindingnet_plec_df.shape}\n"
        f"  BindingDB:  {bindingdb_plec_df.shape}\n"
        #f"  FEP train:  {fep_plec_df_training.shape}\n"
        f"  Merged:     {all_plec_df.shape}"
    )

    # GENERATE FEP FEATURES
    fep_df = generate_FEP_PLEC()
    fep_df.to_csv(os.path.join(BASE_DIR, "data/processed/fep_plec_features.csv"), index=False)
