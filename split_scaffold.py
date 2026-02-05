import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
import random
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
import umap
from sklearn.preprocessing import StandardScaler

def generate_scaffold(smiles, include_chirality=False):
    """
    Generate a Murcko scaffold from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)

def scaffold_split(df, smiles_col='canonical_smiles', test_size=0.1, random_state=42):
    """
    Perform a scaffold split on a DataFrame.
    """
    random.seed(random_state)
    
    scaffolds = defaultdict(list)
    for idx, smiles in df[smiles_col].items():
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)
        else:
            # Handle cases where scaffold generation fails, maybe put them in a random split pool
            # For now, we can assign them to a 'no_scaffold' group
            scaffolds['no_scaffold'].append(idx)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    test_indices = []
    train_indices = []
    
    target_test_size = int(len(df) * test_size)

    for scaffold_set in scaffold_sets:
        if len(test_indices) < target_test_size:
            test_indices.extend(scaffold_set)
        else:
            train_indices.extend(scaffold_set)
            
    # In case the last added scaffold group made the test set too large,
    # move excess molecules from the test set to the training set.
    while len(test_indices) > target_test_size:
        train_indices.append(test_indices.pop())

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    return train_df, test_df

def calculate_tanimoto_similarity(smiles, smiles_pool):
    """
    Calculate Tanimoto similarity between a SMILES string and a pool of SMILES strings using ECFP fingerprints.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 10, nBits=4096)
    
    pool_mols = [Chem.MolFromSmiles(s) for s in smiles_pool]
    pool_mols = [m for m in pool_mols if m is not None]
    
    pool_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 10, nBits=4096) for m in pool_mols]
    
    similarities = BulkTanimotoSimilarity(fp, pool_fps)
    
    return similarities

def get_ecfp_fingerprints(smiles_list):
    """
    Convert a list of SMILES to ECFP fingerprints.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 10, nBits=4096) for m in mols]
    return [np.array(fp) for fp in fps]

def split_train_test(df, train_csv, test_csv, test_size=0.2, random_state=42):

    # แยกข้อมูลเป็นชุดฝึกสอนและทดสอบ
    train_df, test_df = scaffold_split(df, test_size=test_size, random_state=random_state)

    # บันทึกข้อมูลที่แยกแล้วลงในไฟล์ CSV ใหม่
    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)

    print(f"✅ แยกข้อมูลเสร็จสิ้น: {len(train_df)} ตัวอย่างในชุดฝึกสอน, {len(test_df)} ตัวอย่างในชุดทดสอบ")
    return train_df, test_df


def umap_chemical_space_plot(labels_train, train_df,test_df=None, labels_test=None, title="UMAP Chemical Space", output_dir="results", output_file="umap_chemical_space.svg"):
    class_train = labels_train['Class']
    if labels_test is not None:
        class_test = labels_test['Class']

    # 1) Standardize using train only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = None
    if test_df is not None:
        test_scaled = scaler.transform(test_df)
    # 2) Fit UMAP on train only
    reducer = umap.UMAP(random_state=42)
    train_emb = reducer.fit_transform(train_scaled)

    # 3) Transform test (if provided)
    test_emb = None
    if test_scaled is not None:
        # umap-learn supports transform if the reducer was fit with transformable=True (default True).
        # If using older umap versions, transform may still be available.
        try:
            test_emb = reducer.transform(test_scaled)
        except AttributeError:
            # ถ้าไม่มี transform method (เก่า) ให้แจ้งผู้ใช้
            raise RuntimeError(
                "UMAP reducer ไม่มี .transform() — ตรวจสอบเวอร์ชันของ umap-learn (แนะนำ >=0.5).\n"
                "ทางเลือก: รวม test ในการ plot โดยไม่ transform หรือใช้ fit_transform แต่จะเกิด data leakage"
            )

    # 4) Plot
    plt.figure(figsize=(3,3))
    plt.scatter(train_emb[:, 0], train_emb[:, 1], label="Train", edgecolors='black', c = class_train.map({0: 'red', 1: 'blue'}))
    if test_emb is not None:
        plt.scatter(test_emb[:, 0], test_emb[:, 1], label="Test", edgecolors='black', marker='p', c = class_test.map({0: 'orange', 1: 'green'}))
    plt.title(title, fontsize=12, weight='bold', style='italic')
    plt.xlabel("UMAP 1", fontsize=12, weight='bold', style='italic')
    plt.ylabel("UMAP 2", fontsize=12, weight='bold', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=500)
    plt.close()
    return train_emb, test_emb

def main(random_state=0):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    # โหลดข้อมูลที่ไม่มี duplicate
    df = pd.read_csv("data_no_duplicate.csv", index_col=0)
    # แยกข้อมูลเป็นชุดฝึกสอนและทดสอบ
    train_df, test_df = split_train_test(df, f"train_{random_state}.csv", f"test_{random_state}.csv", test_size=0.1, random_state=random_state)

    # Calculate Tanimoto similarity between test set and training set
    train_smiles = train_df['canonical_smiles'].tolist()
    test_smiles = test_df['canonical_smiles'].tolist()

    max_similarities = []
    for test_s in test_smiles:
        similarities = calculate_tanimoto_similarity(test_s, train_smiles)
        if similarities:
            max_similarities.append(max(similarities))

    if max_similarities:
        mean_sim = statistics.mean(max_similarities)
        std_sim = statistics.stdev(max_similarities) if len(max_similarities) > 1 else 0.0
        stats_text = f"Tc score: {mean_sim:.2f} ± {std_sim:.2f}"

        plt.figure(figsize=(3, 3))
        sns.histplot(max_similarities, bins=30, kde=True)
        plt.title('Similarity of Test to Pool', fontsize=12, weight='bold', style='italic')
        plt.xlabel('Max $T_c$ Scores', fontsize=12, weight='bold', style='italic')
        plt.ylabel('Molecules', fontsize=12, weight='bold', style='italic')
        plt.text(0.3, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top')
        plot_path = os.path.join(output_dir, f'test_vs_train_similarity_{random_state}.svg')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=500)
        plt.close()
        print(f"✅ Similarity plot saved to {plot_path}")

    # UMAP Plot based on ECFP
    train_fps = get_ecfp_fingerprints(train_smiles)
    test_fps = get_ecfp_fingerprints(test_smiles)

    if train_fps and test_fps:
        # Combine train and test fingerprints for UMAP fitting
        combined_fps = np.array(train_fps + test_fps)
        
        # Create labels for plotting
        num_train = len(train_fps)
        num_test = len(test_fps)
        labels = ['Pool'] * num_train + ['Test'] * num_test

        # Fit UMAP on the combined dataset
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard', random_state=random_state)
        embedding = reducer.fit_transform(combined_fps)
        
        # Separate embeddings for train and test
        train_emb = embedding[:num_train]
        test_emb = embedding[num_train:]

        # save UMAP csv
        umap_df_train = pd.DataFrame(train_emb, columns=['UMAP1', 'UMAP2'], index=train_df.index)
        umap_df_train['Set'] = 'Train'
        umap_df_test = pd.DataFrame(test_emb, columns=['UMAP1', 'UMAP2'], index=test_df.index)
        umap_df_test['Set'] = 'Test'
        umap_df = pd.concat([umap_df_train, umap_df_test])
        umap_df.to_csv(os.path.join(output_dir, f'umap_ecfp_coordinates_{random_state}.csv'))
        print(f"✅ UMAP coordinates saved to {os.path.join(output_dir, f'umap_ecfp_coordinates_{random_state}.csv')}")

        # Plotting
        plt.figure(figsize=(4, 3))
        plt.grid(True, alpha=0.5, linestyle='--')
        
        # Plot train data
        plt.scatter(train_emb[:, 0], train_emb[:, 1], label="Pool", c='gray', alpha=0.5)
        
        # Plot test data with color based on max similarity
        if max_similarities:
            sc = plt.scatter(test_emb[:, 0], test_emb[:, 1], label="Test", c=max_similarities, cmap='inferno', vmin=0, vmax=1)
            cbar = plt.colorbar(sc)
            cbar.set_label('Max $T_c$ to Pool')
        else:
            plt.scatter(test_emb[:, 0], test_emb[:, 1], label="Test", c='blue', alpha=0.7)

        if 'stats_text' in locals():
            plt.text(0.3, 0.15, stats_text, transform=plt.gca().transAxes, fontsize=9,
                     verticalalignment='top')
        plt.title('Similarity of Test to Pool', fontsize=12, weight='bold', style='italic')
        plt.xlabel("UMAP 1", fontsize=12, weight='bold', style='italic')
        plt.ylabel("UMAP 2", fontsize=12, weight='bold', style='italic')
        
        plt.tight_layout()
        umap_plot_path = os.path.join(output_dir, f'umap_ecfp_chemical_space_{random_state}.svg')
        plt.savefig(umap_plot_path, dpi=500)
        plt.close()
        print(f"✅ UMAP plot saved to {umap_plot_path}")


if __name__ == "__main__":
    for seed in [0, 10, 20]:
        main(random_state=seed)