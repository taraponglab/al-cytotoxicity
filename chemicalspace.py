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


def get_ecfp_fingerprints(smiles_list):
    """
    Convert a list of SMILES to ECFP fingerprints.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if not mols:
        return []
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 10, nBits=4096) for m in mols]
    return [np.array(fp) for fp in fps]

def plot_umap_space(fingerprints_dict, smiles_dict, fitted_reducer, output_path, title):
    """
    A helper function to TRANSFORM, plot the chemical space, and save hit coordinates.
    """
    all_fps = []
    all_smiles = []
    labels = []
    
    plot_config = {
        'Pool':         {'color': 'gray',   'alpha': 0.5, 'zorder': 1},
        'Initial Hit':  {'color': 'orange', 'alpha': 0.5, 'zorder': 2},
        'Baseline Hit': {'color': 'blue',   'alpha': 0.5, 'zorder': 3},
        'Acquired Hit': {'color': 'blue',   'alpha': 0.5, 'zorder': 4}
    }
    
    plot_order = ['Pool', 'Initial Hit', 'Baseline Hit', 'Acquired Hit']

    for label_name in plot_order:
        if label_name in fingerprints_dict and fingerprints_dict[label_name]:
            fps_list = fingerprints_dict[label_name]
            smiles_list = smiles_dict[label_name]
            all_fps.extend(fps_list)
            all_smiles.extend(smiles_list)
            labels.extend([label_name] * len(fps_list))

    if not all_fps:
        print(f"Warning: No fingerprints to plot for '{title}'. Skipping.")
        return

    # Use the pre-fitted reducer to TRANSFORM the data for this plot
    embedding = fitted_reducer.transform(np.array(all_fps))
    labels_array = np.array(labels)

    # --- Save Hit Coordinates to CSV ---
    coord_df = pd.DataFrame({
        'SMILES': all_smiles,
        'Label': labels_array,
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1]
    })
    # Filter for hits only
    hit_coord_df = coord_df[coord_df['Label'] != 'Pool'].copy()
    csv_output_path = output_path.replace('.svg', '_hits_coordinates.csv')
    hit_coord_df.to_csv(csv_output_path, index=False)
    print(f"✅ Hit coordinates saved to {csv_output_path}")


    # --- Plotting ---
    plt.figure(figsize=(5, 3))

    for label_name in plot_order:
        if label_name in fingerprints_dict and fingerprints_dict[label_name]:
            mask = labels_array == label_name
            if np.any(mask):
                config = plot_config[label_name]
                plt.scatter(embedding[mask, 0], embedding[mask, 1], label=label_name, **config)

    plt.xlabel("UMAP1", fontsize=12, weight='bold', style='italic')
    plt.ylabel("UMAP2", fontsize=12, weight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"✅ UMAP plot saved to {output_path}")


def main(random_seed = 0, model=''):
    output_dir = f"chemical_space_{random_seed}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    # --- 1. Load Data ---
    print("Loading datasets...")
    train = pd.read_csv(f'train_{random_seed}.csv', index_col=0)
    test = pd.read_csv(f'test_{random_seed}.csv', index_col=0)
    result  = pd.read_csv(os.path.join(f"active_learning{model}_{random_seed}", "active_learning_acquired_samples.csv"))
    baseline_result = pd.read_csv(os.path.join(f"active_learning_{random_seed}", "baseline_hit_100_samples.csv"))

    # --- 2. Define Molecule Sets ---
    initial_mol = result[result["Round"] == 0]
    pool = train[~train['canonical_smiles'].isin(initial_mol['SMILES'])]
    initial_hit = initial_mol[initial_mol['True_Label'] == 1]
    hit_baseline = baseline_result[(baseline_result["True_Label"] == 1) & (baseline_result["Source"] != "Initial_Random")]

    # --- 3. Generate All Fingerprints and Collect SMILES ---
    print("Generating fingerprints and collecting SMILES for all molecule sets...")
    pool_fps = get_ecfp_fingerprints(pool['canonical_smiles'].tolist())
    pool_smiles = pool['canonical_smiles'].tolist()

    test_fps = get_ecfp_fingerprints(test['canonical_smiles'].tolist())
    test_smiles = test['canonical_smiles'].tolist()

    hit_baseline_fps = get_ecfp_fingerprints(hit_baseline['SMILES'].tolist())
    hit_baseline_smiles = hit_baseline['SMILES'].tolist()

    initial_hit_fps = get_ecfp_fingerprints(initial_hit['SMILES'].tolist())
    initial_hit_smiles = initial_hit['SMILES'].tolist()

    strategies = ["random", "uncertainty", "margin", "entropy", "novelty", "diversity", "confidence_toxic"]
    acquired_fps_by_strategy = {}
    acquired_smiles_by_strategy = {}
    for strategy in strategies:
        acquired = result[(result["Strategy"] == strategy) & (result["Round"] != 0)]
        hit_in_acquired = acquired[acquired["True_Label"] == 1]
        if not hit_in_acquired.empty:
            acquired_fps_by_strategy[strategy] = get_ecfp_fingerprints(hit_in_acquired['SMILES'].tolist())
            acquired_smiles_by_strategy[strategy] = hit_in_acquired['SMILES'].tolist()

    # --- 4. FIT UMAP Reducer ONCE on ALL data ---
    print("\nCombining all fingerprints to create a master UMAP model...")
    master_fps_list = []
    master_fps_list.extend(pool_fps)
    master_fps_list.extend(test_fps)
    master_fps_list.extend(hit_baseline_fps)
    master_fps_list.extend(initial_hit_fps)
    for strategy in strategies:
        if strategy in acquired_fps_by_strategy:
            master_fps_list.extend(acquired_fps_by_strategy[strategy])
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard', random_state=random_seed)
    print(f"Fitting UMAP on {len(master_fps_list)} total fingerprints...")
    reducer.fit(np.array(master_fps_list))
    print("Master UMAP model has been fitted.")

    # --- 5. Create Baseline UMAP Plot (using the fitted reducer) ---
    print("\n--- Generating Baseline Chemical Space Plot ---")
    baseline_fps_dict = {'Pool': pool_fps, 'Test': test_fps, 'Initial Hit': initial_hit_fps, 'Baseline Hit': hit_baseline_fps}
    baseline_smiles_dict = {'Pool': pool_smiles, 'Test': test_smiles, 'Initial Hit': initial_hit_smiles, 'Baseline Hit': hit_baseline_smiles}
    baseline_plot_path = os.path.join(output_dir, 'umap_space_baseline.svg')
    plot_umap_space(baseline_fps_dict, baseline_smiles_dict, reducer, baseline_plot_path, 'Chemical Space: Baseline Hits')

    # --- 6. Loop Through Strategies for Acquired Hits Plots (using the same fitted reducer) ---
    for strategy in strategies:
        if strategy not in acquired_fps_by_strategy or not acquired_fps_by_strategy[strategy]:
            print(f"\n--- No acquired hits for strategy: {strategy}. Skipping plot. ---")
            continue

        print(f"\n--- Generating plot for strategy: {strategy} ---")
        strategy_fps_dict = {'Pool': pool_fps, 'Test': test_fps, 'Initial Hit': initial_hit_fps, 'Acquired Hit': acquired_fps_by_strategy[strategy]}
        strategy_smiles_dict = {'Pool': pool_smiles, 'Test': test_smiles, 'Initial Hit': initial_hit_smiles, 'Acquired Hit': acquired_smiles_by_strategy[strategy]}
        strategy_plot_path = os.path.join(output_dir, f'umap_space_acquired_{strategy}.svg')
        plot_umap_space(strategy_fps_dict, strategy_smiles_dict, reducer, strategy_plot_path, f'Chemical Space: Acquired Hits ({strategy.capitalize()})')

if __name__ == "__main__":
    for model in ['', '_CNN', '_GCN']:
        for seed in [0, 10, 20]:
            print(f"\n================ Running Chemical Space Analysis for Random Seed: {seed} ================\n")
            main(random_seed=seed, model=model)
        print("\nAll chemical space analyses completed.")
