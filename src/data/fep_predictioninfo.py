# makes both 1184 and 1343 prediction infos and then merges them

import os
import glob
import pandas as pd
import re
from pathlib import Path

all_proteins = []
main_folder = Path.cwd().parents[0]

# defining file paths in terms of cwd
input_path = os.path.join(main_folder, "data/fep/fep_benchmark_inputs/structure_inputs")
dg_path = os.path.join(main_folder, "data/fep/21_4_results/ligand_predictions")
metadata = pd.read_csv(os.path.join(main_folder, "data/fep/21_4_results/benchmark_output_metadata.csv"))

info = []

# loop over protein
for _, row in metadata.iterrows():
    group_name = row["Group name"]
    group_abbrev = row["Group abbreviation"]
    protein_input_name = row["Input file naming scheme"]
    protein_output_name = row["Output file naming scheme"]
    #print("info", group_abbrev, protein_input_name)

    dg_file_path = os.path.join(dg_path, group_abbrev, protein_output_name + ".csv")

    if not os.path.exists(dg_file_path):
        print(f"Missing: {dg_file_path}")
        continue

    dg_file = pd.read_csv(dg_file_path)

    #  unique_id creation
    dg_file["Cleaned Ligand name"] = dg_file["Ligand name"].apply(
        lambda x: re.sub(r'[^\w\-.]', '_', str(x))
    )

    dg_file["unique_id"] = (
        protein_input_name + "_molecule_" + dg_file["Cleaned Ligand name"]
    )

    # loop over ligands
    for _, lig_row in dg_file.iterrows():
        ligand_name = lig_row["Ligand name"]
        cleaned_name = lig_row["Cleaned Ligand name"]
        exp_dg = lig_row["Exp. dG (kcal/mol)"]
        fep_pred_dg = lig_row["Pred. dG (kcal/mol)"]
        unique_id = lig_row["unique_id"]
        #print(unique_id)

        pdb_file_path = os.path.join(
            input_path,
            group_abbrev,
            protein_input_name + "_protein.pdb"
        )

        sdf_file_paths = os.path.join(
            input_path,
            group_abbrev,
            protein_input_name + "_ligands_split",
            f"{unique_id}.sdf"
        )

        sdf_files = glob.glob(sdf_file_paths)
        if len(sdf_files) == 1:
            sdf_file_path = sdf_files[0]
        else:
            continue

        info.append({
            "unique_id": unique_id,
            "pdb_file": pdb_file_path,
            "sdf_file": sdf_file_path,
            "protein": protein_input_name,
            "dg_exp_1184": exp_dg,
            "fep_dg_pred": fep_pred_dg
        })

# save properly
df_1184 = pd.DataFrame(info)
print("Made prediction info 1184 of length:", len(df_1184))
df_1184.to_csv(os.path.join(main_folder, "data/processed/fep_prediction_info.csv"))