# For generating ALL types of ECIF Features
# Top is the ECIF Generation code

from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import os
import glob
from pathlib import Path

BASE_DIR = Path.cwd().parents[0]

"""
This is from the ECIF Calculation code. all below is necessary
"""
# Possible predefined protein atoms
ECIF_ProteinAtoms = ['C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1',
                     'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1',
                     'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0',
                     'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0']

# Possible ligand atoms according to the PDBbind 2016 "refined set"
ECIF_LigandAtoms = ['Br;1;1;0;0;0', 'C;3;3;0;1;1', 'C;4;1;1;0;0', 'C;4;1;2;0;0',
                     'C;4;1;3;0;0', 'C;4;2;0;0;0', 'C;4;2;1;0;0', 'C;4;2;1;0;1',
                     'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1', 'C;4;3;0;0;0',
                     'C;4;3;0;0;1', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;4;4;0;0;0', 'C;4;4;0;0;1', 'C;5;3;0;0;0', 'C;5;3;0;1;1',
                     'C;6;3;0;0;0', 'Cl;1;1;0;0;0', 'F;1;1;0;0;0', 'I;1;1;0;0;0',
                     'N;3;1;0;0;0', 'N;3;1;1;0;0', 'N;3;1;2;0;0', 'N;3;2;0;0;0',
                     'N;3;2;0;0;1', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;0;1',
                     'N;3;2;1;1;1', 'N;3;3;0;0;0', 'N;3;3;0;0;1', 'N;3;3;0;1;1',
                     'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;1;0;0', 'N;4;2;2;0;0',
                     'N;4;2;2;0;1', 'N;4;3;0;0;0', 'N;4;3;0;0;1', 'N;4;3;1;0;0',
                     'N;4;3;1;0;1', 'N;4;4;0;0;0', 'N;4;4;0;0;1', 'N;5;2;0;0;0',
                     'N;5;3;0;0;0', 'N;5;3;0;1;1', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'O;2;2;0;0;0', 'O;2;2;0;0;1', 'O;2;2;0;1;1', 'P;5;4;0;0;0',
                     'P;6;4;0;0;0', 'P;6;4;0;0;1', 'P;7;4;0;0;0', 'S;2;1;0;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0', 'S;2;2;0;0;1', 'S;2;2;0;1;1',
                     'S;3;3;0;0;0', 'S;3;3;0;0;1', 'S;4;3;0;0;0', 'S;6;4;0;0;0',
                     'S;6;4;0;0;1', 'S;7;4;0;0;0']

PossibleECIF = [i[0]+"-"+i[1] for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]

ELEMENTS_ProteinAtoms = ["C","N","O", "S"]
ELEMENTS_LigandAtoms = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
PossibleELEMENTS = [i[0]+"-"+i[1] for i in product(ELEMENTS_ProteinAtoms, ELEMENTS_LigandAtoms)]

LigandDescriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
                      'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
                      'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ',
                      'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
                      'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1',
                      'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA10',
                      'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
                      'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
                      'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
                      'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10',
                      'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
                      'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
                      'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
                      'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
                      'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
                      'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
                      'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
                      'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                      'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
                      'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_N',
                      'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
                      'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
                      'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                      'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide',
                      'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azo', 'fr_barbitur',
                      'fr_benzene', 'fr_bicyclic', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                      'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                      'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                      'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                      'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitroso', 'fr_oxazole',
                      'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                      'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN',
                      'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                      'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_urea']

DescCalc = MolecularDescriptorCalculator(LigandDescriptors)


def GetAtomType(atom):
# This function takes an atom in a molecule and returns its type as defined for ECIF
    
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())), 
               ]

    return(";".join(AtomType))


def LoadSDFasDF(SDF):
# This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF
    
    m = Chem.MolFromMolFile(SDF, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    
    ECIF_atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(GetAtomType(atom))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            ECIF_atoms.append(entry)

    df = pd.DataFrame(ECIF_atoms)
    df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE","X","Y","Z"]
    if len(set(df["ECIF_ATOM_TYPE"]) - set(ECIF_LigandAtoms)) > 0:
        pass
        #print("WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)

Atom_Keys=pd.read_csv("/rds/general/user/cr725/home/aev-plig_research/other_models/ECIF/data/PDB_Atom_Keys.csv", sep=",")

def LoadPDBasDF(PDB):
# This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF

    ECIF_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                ECIF_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
        pass
        #print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)


def GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=6.0):
# This function returns the protein-ligand atom-type pairs for a given distance cutoff
    
    # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)
    
    # Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    # Get all possible pairs
    Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    Pairs = pd.concat([Pairs,Distances], axis=1)
    Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    # Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
    Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
    return Pairs


def GetECIF(PDB_protein, SDF_ligand, distance_cutoff=6.0):
# Main function for the calculation of ECIF
    Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
    ECIF = [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]
    return ECIF


def GetELEMENTS(PDB_protein, SDF_ligand, distance_cutoff=6.0):
# Function for the calculation of ELEMENTS
    Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
    ELEMENTS = [list(Pairs["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS]
    return ELEMENTS


def GetRDKitDescriptors(SDF):
# Function for the calculation of ligand descriptors
    mol = Chem.MolFromMolFile(SDF, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return DescCalc.CalcDescriptors(mol)



# End of ECIF Functions



# Start of database specific generation


def build_ecif_dataframe(unique_ids, ecifs, extra_cols=None):
    if len(ecifs) == 0:
        raise ValueError("No ECIF features generated.")

    feature_matrix = np.vstack(ecifs)

    ecif_cols = [f"ECIF_{i}" for i in range(feature_matrix.shape[1])]

    ecif_df = pd.DataFrame(feature_matrix, columns=ecif_cols)
    ecif_df.insert(0, "unique_id", unique_ids)

    if extra_cols:
        for idx, (col_name, values) in enumerate(extra_cols.items(), start=1):
            ecif_df.insert(idx, col_name, values)

    return ecif_df


def generate_PDBbind_ECIF():
    print("Generating ECIF features for pdbbind")

    data = pd.read_csv(os.path.join(BASE_DIR, "data", "files", "pdbbind_processed.csv"), index_col=0)

    unique_ids = []
    ecifs = []
    failed_list = []

    for _, row in tqdm(data.iterrows()):
        
        if row["refined"]:
            folder = os.path.join(BASE_DIR, "data/raw/pdbbind/refined-set")
        else:
            folder = os.path.join(BASE_DIR, "data/raw/pdbbind/general-set")

        pdb = row["PDB_code"]

        ligand = os.path.join(folder, pdb, f'{pdb}_ligand.sdf')
        receptor = os.path.join(folder, pdb, f'{pdb}_protein.pdb')

        try: 
            ecif = GetECIF(receptor, ligand, distance_cutoff=6.0)
            unique_ids.append(pdb)
            ecifs.append(ecif)
        except Exception as e: 
            failed_list.append(pdb)

    print(f"Length of failed: {len(failed_list)}")

    return build_ecif_dataframe(unique_ids, ecifs)

def generate_Bindingnet_ECIF():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "files", "bindingnet_processed.csv"), index_col=0)
    folder = os.path.join(BASE_DIR, "data/raw/bindingnet/from_chembl_client/")

    unique_ids = []
    ecifs = []
    failed_list = []

    for index, row in tqdm(df.iterrows()):
        unique_identify = row['unique_identify']
        target = row['target']
        pdb = row['pdb']
        compnd = row['compnd']

        sdf_file = folder + f"{pdb}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf"
        protein_path = folder + f"{pdb}/rec_h_opt.pdb"

        receptor = protein_path
        ligand = sdf_file

        try: 
            ecif = GetECIF(receptor, ligand, distance_cutoff=6.0)
            unique_ids.append(unique_identify)
            ecifs.append(ecif)
        except Exception as e: 
            failed_list.append(pdb)
        
    print(f"Length of failed: {len(failed_list)}")

    return build_ecif_dataframe(unique_ids, ecifs)


def generate_BindingDB_ECIF():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/files/bindingdb_processed.csv"), index_col=0)
    folder = os.path.join(BASE_DIR, "data/raw/bindingdb/surflex/")

    unique_ids = []
    ecifs = []
    failed_list = []

    for index, row in tqdm(df.iterrows()):
        mol2_file = folder + row["folder"] + "/" + row["mol2_file"]
        sdf_path_true = Path(mol2_file.replace(".mol2", ".sdf"))
        sdf_path = mol2_file.replace(".mol2", ".sdf")
        unique_id = row["unique_id"]

        # Only create SDF if it doesn't exist
        if not sdf_path_true.exists():
            lig = Chem.MolFromMol2File(str(mol2_file))
            lig = Chem.AddHs(lig, addCoords=True)
            Chem.MolToMolFile(lig, str(sdf_path))
            print("creating sdf file")

        protein_path = folder + row["folder"] + "/" + row["pdb_file"]

        receptor = protein_path
        ligand = sdf_path

        try: 
            ecif = GetECIF(receptor, ligand, distance_cutoff=6.0)
            unique_ids.append(unique_id)
            ecifs.append(ecif)
        except Exception as e: 
            failed_list.append(row["folder"])  
              
    print(f"Length of failed: {len(failed_list)}")

    return build_ecif_dataframe(unique_ids, ecifs)


"""
# Tried using FEP data for training, but is not valid as it is augmented data, similar to the test set
def generate_FEP_ECIF_fortraining():
    fep_prediction_info = pd.read_csv(os.path.join(BASE_DIR, "fep/analysis/only_in_1343.csv"))
    print(f"Number of complexes in FEP for training/valid: {len(fep_prediction_info)}")

    unique_id_list = []
    ecif_list = []
    dg_list = []
    failed_list = []

    for _, row in fep_prediction_info.iterrows():
        pdb_path = row["pdb_file"]
        sdf_path = row["sdf_file"]

        try: 
            ecif = GetECIF(pdb_path, sdf_path, distance_cutoff=6.0)
            ecif_list.append(ecif)
            unique_id_list.append(row.get("unique_id", np.nan))
        except Exception as e: 
            print(f"FAILED: {row.get('unique_id', np.nan)} | {e}")
            failed_list.append(row.get("unique_id", np.nan))
 
    return build_ecif_dataframe(
        unique_id_list,
        ecif_list
    )
"""


# Generates raw_fep_ecif_features.csv
def generate_FEP_ECIF():
    fep_prediction_info = pd.read_csv(os.path.join(BASE_DIR, "data/processed/fep_prediction_info.csv"))
    print(f"Number of complexes in FEP test set: {len(fep_prediction_info)}")

    unique_id_list = []
    ecif_list = []
    dg_list = []
    failed_list = []

    for _, row in fep_prediction_info.iterrows():
        pdb_path = row["pdb_file"]
        sdf_path = row["sdf_file"]

        try: 
            ecif = GetECIF(pdb_path, sdf_path, distance_cutoff=6.0)
            ecif_list.append(ecif)
            unique_id_list.append(row.get("unique_id", np.nan))
            dg_list.append(row.get("dg_exp", np.nan))
        except Exception as e: 
            print(f"FAILED: {row.get('unique_id', np.nan)} | {e}")
            failed_list.append(row.get("unique_id", np.nan))
 
    return build_ecif_dataframe(
        unique_id_list,
        ecif_list,
        extra_cols={"dg_exp": dg_list}
    )


if __name__ == "__main__": 

    # GENERATING RAW FEATURES FOR DATABASES -- TAKES 4 HOURS
    merged_out = os.path.join(BASE_DIR, "data/processed/raw_all_plec_features.csv")

    # Generate directly into memory
    pdbbind_ecif_df = generate_PDBbind_ECIF()
    bindingnet_ecif_df = generate_Bindingnet_ECIF()
    bindingdb_ecif_df = generate_BindingDB_ECIF()
    #fep_ecif_df_training = generate_FEP_ECIF_fortraining()

    # Merge all dataframes
    all_ecif_df = pd.concat(
        [pdbbind_ecif_df, bindingnet_ecif_df, bindingdb_ecif_df, fep_ecif_df_training],
        ignore_index=True
    )

    # Save only merged dataframe
    all_ecif_df.to_csv(merged_out, index=False)

    print(
        f"Shapes:\n"
        f"  PDBbind:    {pdbbind_ecif_df.shape}\n"
        f"  BindingNet: {bindingnet_ecif_df.shape}\n"
        f"  BindingDB:  {bindingdb_ecif_df.shape}\n"
        #f"  FEP train:  {fep_ecif_df_training.shape}\n"
        f"  Merged:     {all_ecif_df.shape}"
    )

    # GENERATE FEP FEATURES
    fep_df = generate_FEP_ECIF()
    fep_df.to_csv(os.path.join(BASE_DIR, "data/processed/fep_plec_features.csv"), index=False)
