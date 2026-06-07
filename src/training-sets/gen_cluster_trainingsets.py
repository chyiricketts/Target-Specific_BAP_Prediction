# to make the full database for train/validation splits
# combines the 4 sources of database
# does tanimoto similarity filter to the test set.
# returns to output

import os
import glob as glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

BASE_DIR = "/rds/general/user/cr725/home/aev-plig_research"



def build_dataset(protein, pdbbind_path, bindingnet_path, bindingdb_path):
    print("********************************")
    print(f"Building Dataset for {protein}")

    # csvs
    pdbbind_data = pd.read_csv(pdbbind_path, index_col=0)
    bindingnet_data = pd.read_csv(bindingnet_path, index_col=0)
    bindingdb_data = pd.read_csv(bindingdb_path, index_col=0)
    fep_data = pd.read_csv(os.path.join(BASE_DIR, f"training_sets/fep_train/fep_train_{protein}.csv"))

    data = []

    # pdbbind
    print(f"PDBbind data contains {len(pdbbind_data)} rows")
    for _, row in tqdm(pdbbind_data.iterrows()):
        if row["refined"]:
            folder = os.path.join(BASE_DIR, "databases/pdbbind/refined-set")
        else:
            folder = os.path.join(BASE_DIR, "databases/pdbbind/general-set")
        pdb = row["PDB_code"]
        ligand_path = os.path.join(folder, pdb, f'{pdb}_ligand.mol2')
        protein_path = os.path.join(folder, pdb, f'{pdb}_protein.pdb')
        info = {
            "unique_id": pdb, 
            "database": "pdbbind",
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pK": row["-logKd/Ki"]
        }
        data.append(info)

    # bindingnet
    print(f"BindingNet data contains {len(bindingnet_data)} rows")
    folder = os.path.join(BASE_DIR, "databases/bindingnet/from_chembl_client/")
    for index, row in tqdm(bindingnet_data.iterrows()):
        unique_identify = row['unique_identify']
        target = row['target']
        pdb = row['pdb']
        compnd = row['compnd']
        ligand_path = folder + f"{pdb}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf"
        protein_path = folder + f"{pdb}/rec_h_opt.pdb"
        info = {
            "unique_id": unique_identify, 
            "database": "bindingnet",
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pK": row["-logAffi"]
        }
        data.append(info)


    # bindingdb
    print(f"BindingDB data contains {len(bindingdb_data)} rows")
    folder = os.path.join(BASE_DIR, "databases/bindingdb/surflex/")
    for index, row in tqdm(bindingdb_data.iterrows()):
        ligand_path = folder + row["folder"] + "/" + row["mol2_file"]
        protein_path = folder + row["folder"] + "/" + row["pdb_file"]

        info = {
            "unique_id": row["unique_id"],
            "database": "bindingdb",
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pK": row["pK"]
        }
        data.append(info)

    # fep
    print(f"FEP data contains {len(fep_data)} rows")
    for _, row in tqdm(fep_data.iterrows(), total=len(fep_data)):
        protein_path = row["pdb_file"]
        ligand_path = row["sdf_file"]
        info = {
            "unique_id": row["unique_id"],
            "database": "fep",
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pK": row["dg_exp_1343"]/-1.36
        }
        data.append(info)

    # finalizing
    df = pd.DataFrame(data)
    df["pK"] = pd.to_numeric(df["pK"], errors="coerce")
    df["protein_exists"] = df["protein_path"].apply(os.path.exists)
    df["ligand_exists"] = df["ligand_path"].apply(os.path.exists)
    print("Protein files found:", df["protein_exists"].sum())
    print("Ligand files found:", df["ligand_exists"].sum())
    print("Total rows:", len(df))

    return df


# helper function for tanimoto fingerprints
def standardize_mol(sdf_file):
    mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
    if mol is None:
        print("mol is none")
        return None
    try:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return None
    return mol


# helper function for tanimoto fingerprints
def gen_ts_fep(protein_fullname, morgan_gen):
    # Fixed Morgan Fingerprints for 1184 FEP Data
    fep_csv = pd.read_csv(os.path.join(BASE_DIR, "fep/analysis/prediction_info_1184.csv"))
    
    fep_csv = fep_csv[fep_csv["protein"] == protein_fullname]
    
    fep_data = []
    for _, row in fep_csv.iterrows():
        unique_id = row["unique_id"]
        sdf_file = row["sdf_file"]

        try: 
            lig = standardize_mol(sdf_file)
        except: 
            print(f"[WARN] Failed to standardize ligand for {sdf_file}")
            continue
        if lig is None:
            print(f"[WARN] Failed to process ligand for {sdf_file}")
            continue

        fp = morgan_gen.GetFingerprint(lig)

        fep_data.append({
            "fp": fp,
            "mol": lig,
            "fep_lig_match": os.path.basename(sdf_file)
        })

    print("Length of FEP data:", len(fep_data))
    return fep_data

# helper function for tanimoto fingerprints
def tanimoto_process(data, fep_data, morgan_gen):
     # database storage variable
    db_data = []
    fep_fps = [x["fp"] for x in fep_data]

    for _, row in data.iterrows():

        sdf_file = row["ligand_path"].replace(".mol2", ".sdf")
        pdb_file = row["protein_path"]

        try: 
            lig = standardize_mol(sdf_file)
        except: 
            print(f"[WARN] Failed to standardize ligand for ({sdf_file})")
            continue 
        if lig is None:
            print(f"[WARN] Failed to process ligand for ({sdf_file})")
            continue

        
        fp = morgan_gen.GetFingerprint(lig)
        entry = row.to_dict()

        entry["fp"] = fp
        db_data.append(entry)

    results = []

    for db_idx, db_item in enumerate(db_data):

        sims = BulkTanimotoSimilarity(db_item["fp"], fep_fps)

        best_fep_idx = max(range(len(sims)), key=lambda i: sims[i])
        best_sim = sims[best_fep_idx]

        # Start from ALL existing columns
        result = db_item.copy()
        result.pop("fp", None) # remove fp

        # Add new similarity info
        result["fep_lig_match"] = fep_data[best_fep_idx]["fep_lig_match"]
        result["ts-max_tanimoto_fep_benchmark"] = best_sim

        results.append(result)

    results = pd.DataFrame(results)
    return results


# main functino for tanimoto fingerprints
def add_fep_tanimoto(df, protein_fullname):
    print("*****************************")
    print(f"Adding Tanimoto fingerprints for {protein_fullname} on dataset of length {len(df)}")
    morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)
    fep_data = gen_ts_fep(protein_fullname, morgan_gen)
    df = tanimoto_process(df, fep_data, morgan_gen)

    # remove the ones above 0.9
    before = len(df)
    df = df[df["ts-max_tanimoto_fep_benchmark"] < 0.9].reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} compounds with tanimoto >= 0.9")
    print(f"Remaining compounds: {after}")
    
    return df


def valid_tanimoto(train_df, val_df, morgan_gen): 
    print("Building train fingerprints...")
    train_fps = []
    train_ids = []

    for _, row in train_df.iterrows():
        sdf_file = row["ligand_path"].replace(".mol2", ".sdf")

        lig = standardize_mol(sdf_file)
        if lig is None:
            continue

        fp = morgan_gen.GetFingerprint(lig)

        train_fps.append(fp)
        train_ids.append(row["unique_id"])

    print(f"Train molecules: {len(train_fps)}")

    # -----------------------------
    # 2. Compute VALID → TRAIN similarity
    # -----------------------------
    results = []

    print("Computing valid → train similarities...")

    for _, row in val_df.iterrows():
        sdf_file = row["ligand_path"].replace(".mol2", ".sdf")

        lig = standardize_mol(sdf_file)
        if lig is None:
            continue

        fp = morgan_gen.GetFingerprint(lig)

        sims = BulkTanimotoSimilarity(fp, train_fps)

        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        best_sim = sims[best_idx]

        # optional: top-k mean similarity
        top_k = sorted(sims, reverse=True)[:5]
        mean_top5 = sum(top_k) / len(top_k)

        results.append({
            "unique_id": row["unique_id"],
            "best_train_match": train_ids[best_idx],
            "max_tanimoto_to_train": best_sim,
            "mean_top5_tanimoto_to_train": mean_top5,
            "pK": row.get("pK", None)
        })

    results_df = pd.DataFrame(results)

    print("\n=== VALID → TRAIN SIMILARITY CUTOFF COUNTS ===")

    for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9]:

        n = (
            results_df["max_tanimoto_to_train"] >= cutoff
        ).sum()

        pct = 100 * n / len(results_df)

        print(
            f"Tanimoto >= {cutoff:.1f}: "
            f"{n:4d} complexes "
            f"({pct:.1f}%)"
        )
    
    return results_df


# to generate fps for all of them
def build_fps(df, radius=3, fpSize=2048):
    """
    Precompute Morgan fingerprints for entire dataset.
    Returns aligned fps list + filtered dataframe.
    """
    print("Building Morgan fingerprints...")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=fpSize)

    fps = []
    keep_rows = []

    for idx, row in df.iterrows():
        sdf_file = row["ligand_path"].replace(".mol2", ".sdf")
        mol = standardize_mol(sdf_file)
        if mol is None:
            continue
        fps.append(morgan_gen.GetFingerprint(mol))
        keep_rows.append(idx)
    df = df.iloc[keep_rows].reset_index(drop=True)
    print(f"Kept {len(df)} molecules with valid fingerprints")
    return df, fps



def cluster_kfold(name, df, n_splits=5, similarity_cutoff=0.8, random_state=42):
    """
    Cluster-aware K-fold CV split using
    Butina clustering on Morgan fingerprints.
    Entire clusters are assigned to folds together.
    """
    print("**************************")
    print(f"Cluster K-Fold (K={n_splits}) on dataset of length {len(df)}")

    random.seed(random_state)
    np.random.seed(random_state)

    df = df.copy().reset_index(drop=True)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048,)

    fps = []
    keep_rows = []

    for idx, row in df.iterrows():
        sdf_file = row["ligand_path"].replace(".mol2", ".sdf")
        try:
            mol = standardize_mol(sdf_file)
        except:
            continue
        if mol is None:
            continue
        fp = morgan_gen.GetFingerprint(mol)
        fps.append(fp)
        keep_rows.append(idx)

    df = df.iloc[keep_rows].reset_index(drop=True)
    print(f"Successfully fingerprinted {len(df)} molecules")

    # Pairwise distances — computed ONCE for all folds
    dists = []
    for i in range(1, len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # Cluster ONCE
    clusters = Butina.ClusterData(
        dists,
        len(fps),
        1 - similarity_cutoff,
        isDistData=True
    )
    clusters = [list(c) for c in clusters]
    print(f"Generated {len(clusters)} clusters")

    # Sort by size descending for better balance
    clusters = sorted(clusters, key=len, reverse=True)

    # Assign clusters to folds greedily
    # Always assign to the smallest fold (by current size)
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for cluster in clusters:
        smallest = int(np.argmin(fold_sizes))
        folds[smallest].extend(cluster)
        fold_sizes[smallest] += len(cluster)

    print(f"Fold sizes: {fold_sizes}")

    # Build (train_df, val_df) pairs
    splits = []
    folds_indices = []
    for val_fold_idx in range(n_splits):
        val_indices   = folds[val_fold_idx]
        train_indices = [
            idx for f in range(n_splits)
            if f != val_fold_idx
            for idx in folds[f]
        ]
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df   = df.iloc[val_indices].reset_index(drop=True)
        splits.append((train_df, val_df))
        folds_indices.append((train_indices, val_indices))

        print(f"  Fold {val_fold_idx+1}: "
              f"train={len(train_df)}, val={len(val_df)}")

        # Save folds
        output_dir = os.path.join(BASE_DIR, "training_sets", "final", name)
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir,f"{name}_cluster_fold-{val_fold_idx+1}_train2.csv")
        val_path = os.path.join(output_dir,f"{name}_cluster_fold-{val_fold_idx+1}_valid2.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

    return splits, folds_indices, df, fps



def generate_split_report(
    name,
    splits,
    folds_indices,
    df,
    fps,
    title="Split Report"
):

    print("\n" + "=" * 60)
    print(f"{title}: {name}")
    print("=" * 60)

    summary = []

    for i, (train_df, val_df) in enumerate(splits):

        print(f"\n--- Fold {i+1} ---")

        # --------------------------
        # 1. pK distribution
        # --------------------------
        print("pK stats:")
        print(val_df["pK"].describe()[["mean", "std", "min", "max"]])

        # --------------------------
        # 2. similarity leakage
        # --------------------------
        train_idx, val_idx = folds_indices[i]

        train_fps = [fps[j] for j in train_idx]
        val_fps   = [fps[j] for j in val_idx]

        max_sims = []

        for vfp in val_fps:
            sims = BulkTanimotoSimilarity(vfp, train_fps)
            max_sims.append(max(sims))

        max_sims = np.array(max_sims)

        print("\nTanimoto leakage:")
        print(f"  mean max: {max_sims.mean():.3f}")
        print(f"  median:   {np.median(max_sims):.3f}")
        print(f"  max:      {max_sims.max():.3f}")
        print(f"  >0.7:     {(max_sims > 0.7).sum()}")
        print(f"  >0.8:     {(max_sims > 0.8).sum()}")
        print(f"  >0.9:     {(max_sims > 0.9).sum()}")

        summary.append({
            "fold": i+1,
            "pK_mean": val_df["pK"].mean(),
            "pK_std": val_df["pK"].std(),
            "sim_mean_max": max_sims.mean(),
            "sim_median": np.median(max_sims),
            "sim_max": max_sims.max(),
            "gt0.7": (max_sims > 0.7).sum(),
            "gt0.8": (max_sims > 0.8).sum(),
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS FOLDS")
    print(summary_df)

    return summary_df


if __name__ == "__main__":

    morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)

    # first generate fingerprints for random generation

    proteins = {
        "MCL1": "mcl1_extra_flips",
        "SYK": "syk_4puz_fullmap",
        "PFKFB3": "pfkfb3_automap",
        "HIF2A": "hif2a_automap",
        "MAPK14": "p38"
    }
    
    proteins = {
        "PFKFB3": "pfkfb3_automap"
    }

    for protein, protein_fullname in proteins.items():

        pdbbind_data_path = os.path.join(BASE_DIR, f"training_sets/target-specific/pdbbind_processed_{protein}.csv")
        bindingnet_data_path = os.path.join(BASE_DIR, f"training_sets/target-specific/bindingnet_processed_{protein}.csv")
        bindingdb_data_path = os.path.join(BASE_DIR, f"training_sets/target-specific/bindingdb_processed_{protein}.csv")

        df = build_dataset(protein, pdbbind_data_path, bindingnet_data_path, bindingdb_data_path)
        df = add_fep_tanimoto(df, protein_fullname)

        cluster_splits, cluster_fold_indices, filtered_df, fps = cluster_kfold(protein, df)
        report_cluster = generate_split_report("cluster", cluster_splits, cluster_fold_indices, filtered_df, fps)
        
        for threshold in np.arange(0.9, -0.1, -0.1):
            threshold_str = f"{threshold:.1f}"
            name = protein + "_" + threshold_str
            
            print("\n\n*********************************")
            print(name)
            print("*********************************")

            # tm thresholds -- finish later, make sure it doesn't overwrite the others
            pdbbind_data_path = os.path.join(BASE_DIR, f"training_sets/tm-align_filter/{protein}/tm-{protein}-{threshold_str}_pdbbind.csv")
            bindingnet_data_path = os.path.join(BASE_DIR, f"training_sets/tm-align_filter/{protein}/tm-{protein}-{threshold_str}_bindingnet.csv")
            bindingdb_data_path = os.path.join(BASE_DIR, f"training_sets/tm-align_filter/{protein}/tm-{protein}-{threshold_str}_bindingdb.csv")

            df = build_dataset(protein, pdbbind_data_path, bindingnet_data_path, bindingdb_data_path)
            df = add_fep_tanimoto(df, protein_fullname)

            cluster_splits, cluster_fold_indices, filtered_df, fps = cluster_kfold(name, df)
            report_cluster = generate_split_report("cluster", cluster_splits, cluster_fold_indices, filtered_df, fps)
