import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, pairwise_distances, confusion_matrix, precision_score, recall_score, auc
from scipy.stats import entropy
from tqdm import tqdm
import umap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")


# === 0.5. Fingerprint and Descriptor Calculation Functions ===
def calculate_ecfp(df, smiles_col, radius=10, nBits=4096):
    '''
    Compute ECFP fingerprints, radius = 10, nBits = 4096
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_ecfp(smiles):
       try:
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               print(f"SMILES conversion failed for: {smiles}")
               return [None] * nBits
           fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
           return [int(bit) for bit in fingerprint.ToBitString()]
       except Exception as e:
           print(f"Error processing SMILES {smiles}: {e}")
           return [None] * nBits  # Return a list of None if an error occurs
    ecfp_bits_df = df[smiles_col].apply(get_ecfp).apply(pd.Series)
    ecfp_bits_df.columns = [f'ECFP{i}' for i in range(nBits)]
    ecfp_bits_df
    return ecfp_bits_df

def calculate_rdkit(df, smiles_col, nBits=2048):
    '''
    Compute RDKIT fingerprints, nBits = 2048
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_rdkit(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            fingerprint = Chem.RDKFingerprint(mol)
            return [int(bit) for bit in fingerprint.ToBitString()]
        except:
            return [None] * nBits  # Return a list of None if an error occurs

    rdkit_bits_df = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    rdkit_bits_df.columns = [f'RDKit{i}' for i in range(nBits)]
    return rdkit_bits_df

def calculate_maccs(df, smiles_col):
    '''
    Compute MACCS fingerprints, nBits = 167
    ------
    df: DataFrame
    smiles_col: SMILE column
    '''
    def get_maccs(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            return [int(bit) for bit in fingerprint.ToBitString()]
        except:
            return [None] * 167

    maccs_bits_df = df[smiles_col].apply(get_maccs).apply(pd.Series)
    maccs_bits_df.columns = [f'MACCS{i}' for i in range(167)]
    return maccs_bits_df

def calculate_descriptors(df, smiles_col):
    """
    Compute molecular descriptors using RDKit.
    ------
    df: DataFrame
    smiles_col: Column name containing SMILES strings
    """
    descriptor_functions = {
        'molecular_weight': Descriptors.MolWt,
        'log_p': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'CalcTPSA': rdMolDescriptors.CalcTPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'CalcNumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles,
        'CalcNumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles,
        'CalcNumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings,
        'CalcNumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms,
        'CalcNumRings': rdMolDescriptors.CalcNumRings,
        'CalcNumHeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms,
        'CalcNumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
        'CalcNumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles,
        'CalcNumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles,
        'NumValenceElectrons': Descriptors.NumValenceElectrons,
        'CalcNumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms,
        'CalcNumHeterocycles': rdMolDescriptors.CalcNumHeterocycles,
        'CalcNumAmideBonds': rdMolDescriptors.CalcNumAmideBonds,
    }

    def get_descriptors(smiles_col):
        try:
            mol = Chem.MolFromSmiles(smiles_col)
            return [func(mol) for func in descriptor_functions.values()]
        except:
            return [None] * len(descriptor_functions)

    descriptors_df = df[smiles_col].apply(get_descriptors).apply(pd.Series)
    descriptors_df.columns = list(descriptor_functions.keys())
    return descriptors_df


# === 1. SMILES Tokenizer and Vocabulary ===
SMILES_CHARS = [
        '<pad>', '<sos>', '<eos>', '<SEP>', '<MASK>', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '.', 
        'n', '3', 'F', 'Cl', '>>', '~', '-', '4', '[C@H]', 'S', '[C@@H]', '[O-]', 'Br', '#', '/', '[nH]', 
        '[N+]', 's', '5', 'o', 'P', '[Na+]', '[Si]', 'I', '[Na]', '[Pd]', '[K+]', '[K]', '[P]', 'B', '[C@]', 
        '[C@@]', '[Cl-]', '6', '[OH-]', '\\', '[N-]', '[Li]', '[H]', '[2H]', '[NH4+]', '[c-]', '[P-]', '[Cs+]',
        '[Li+]', '[Cs]', '[NaH]', '[H-]', '[O+]', '[BH4-]', '[Cu]', '7', '[Mg]', '[Fe+2]', '[n+]', '[Sn]', 
        '[BH-]', '[Pd+2]', '[CH]', '[I-]', '[Br-]', '[C-]', '[Zn]', '[B-]', '[F-]', '[Al]', '[P+]', '[BH3-]',
        '[Fe]', '[C]', '[AlH4]', '[Ni]', '[SiH]', '8', '[Cu+2]', '[Mn]', '[AlH]', '[nH+]', '[AlH4-]', '[O-2]',
        '[Cr]', '[Mg+2]', '[NH3+]', '[S@]', '[Pt]', '[Al+3]', '[S@@]', '[S-]', '[Ti]', '[Zn+2]', '[PH]', 
        '[NH2+]', '[Ru]', '[Ag+]', '[S+]', '[I+3]', '[NH+]', '[Ca+2]', '[Ag]', '9', '[Os]', '[Se]', '[SiH2]',
        '[Ca]', '[Ti+4]', '[Ac]', '[Cu+]', '[S]', '[Rh]', '[Cl+3]', '[cH-]', '[Zn+]', '[O]', '[Cl+]', '[SH]', 
        '[H+]', '[Pd+]', '[se]', '[PH+]', '[I]', '[Pt+2]', '[C+]', '[Mg+]', '[Hg]', '[W]', '[SnH]', '[SiH3]',
        '[Fe+3]', '[NH]', '[Mo]', '[CH2+]', '%10', '[CH2-]', '[CH2]', '[n-]', '[Ce+4]', '[NH-]', '[Co]', 
        '[I+]', '[PH2]', '[Pt+4]', '[Ce]', '[B]', '[Sn+2]', '[Ba+2]', '%11', '[Fe-3]', '[18F]', '[SH-]', 
        '[Pb+2]', '[Os-2]', '[Zr+4]', '[N]', '[Ir]', '[Bi]', '[Ni+2]', '[P@]', '[Co+2]', '[s+]', '[As]', 
        '[P+3]', '[Hg+2]', '[Yb+3]', '[CH-]', '[Zr+2]', '[Mn+2]', '[CH+]', '[In]', '[KH]', '[Ce+3]', '[Zr]',
        '[AlH2-]', '[OH2+]', '[Ti+3]', '[Rh+2]', '[Sb]', '[S-2]', '%12', '[P@@]', '[Si@H]', '[Mn+4]', 'p', 
        '[Ba]', '[NH2-]', '[Ge]', '[Pb+4]', '[Cr+3]', '[Au]', '[LiH]', '[Sc+3]', '[o+]', '[Rh-3]', '%13', 
        '[Br]', '[Sb-]', '[S@+]', '[I+2]', '[Ar]', '[V]', '[Cu-]', '[Al-]', '[Te]', '[13c]', '[13C]', '[Cl]', 
        '[PH4+]', '[SiH4]', '[te]', '[CH3-]', '[S@@+]', '[Rh+3]', '[SH+]', '[Bi+3]', '[Br+2]', '[La]', 
        '[La+3]', '[Pt-2]', '[N@@]', '[PH3+]', '[N@]', '[Si+4]', '[Sr+2]', '[Al+]', '[Pb]', '[SeH]', '[Si-]', 
        '[V+5]', '[Y+3]', '[Re]', '[Ru+]', '[Sm]', '*', '[3H]', '[NH2]', '[Ag-]', '[13CH3]', '[OH+]', '[Ru+3]',
        '[OH]', '[Gd+3]', '[13CH2]', '[In+3]', '[Si@@]', '[Si@]', '[Ti+2]', '[Sn+]', '[Cl+2]', '[AlH-]', 
        '[Pd-2]', '[SnH3]', '[B+3]', '[Cu-2]', '[Nd+3]', '[Pb+3]', '[13cH]', '[Fe-4]', '[Ga]', '[Sn+4]', 
        '[Hg+]', '[11CH3]', '[Hf]', '[Pr]', '[Y]', '[S+2]', '[Cd]', '[Cr+6]', '[Zr+3]', '[Rh+]', '[CH3]', 
        '[N-3]', '[Hf+2]', '[Th]', '[Sb+3]', '%14', '[Cr+2]', '[Ru+2]', '[Hf+4]', '[14C]', '[Ta]', '[Tl+]', 
        '[B+]', '[Os+4]', '[PdH2]', '[Pd-]', '[Cd+2]', '[Co+3]', '[S+4]', '[Nb+5]', '[123I]', '[c+]', '[Rb+]',
        '[V+2]', '[CH3+]', '[Ag+2]', '[cH+]', '[Mn+3]', '[Se-]', '[As-]', '[Eu+3]', '[SH2]', '[Sm+3]', '[IH+]',
        '%15', '[OH3+]', '[PH3]', '[IH2+]', '[SH2+]', '[Ir+3]', '[AlH3]', '[Sc]', '[Yb]', '[15NH2]', '[Lu]', 
        '[sH+]', '[Gd]', '[18F-]', '[SH3+]', '[SnH4]', '[TeH]', '[Si@@H]', '[Ga+3]', '[CaH2]', '[Tl]', 
        '[Ta+5]', '[GeH]', '[Br+]', '[Sr]', '[Tl+3]', '[Sm+2]', '[PH5]', '%16', '[N@@+]', '[Au+3]', '[C-4]',
        '[Nd]', '[Ti+]', '[IH]', '[N@+]', '[125I]', '[Eu]', '[Sn+3]', '[Nb]', '[Er+3]', '[123I-]', '[14c]',
        '%17', '[SnH2]', '[YH]', '[Sb+5]', '[Pr+3]', '[Ir+]', '[N+3]', '[AlH2]', '[19F]', '%18', '[Tb]', 
        '[14CH]', '[Mo+4]', '[Si+]', '[BH]', '[Be]', '[Rb]', '[pH]', '%19', '%20', '[Xe]', '[Ir-]', '[Be+2]', 
        '[C+4]', '[RuH2]', '[15NH]', '[U+2]', '[Au-]', '%21', '%22', '[Au+]', '[15n]', '[Al+2]', '[Tb+3]', 
        '[15N]', '[V+3]', '[W+6]', '[14CH3]', '[Cr+4]', '[ClH+]', 'b', '[Ti+6]', '[Nd+]', '[Zr+]', '[PH2+]', 
        '[Fm]', '[N@H+]', '[RuH]', '[Dy+3]', '%23', '[Hf+3]', '[W+4]', '[11C]', '[13CH]', '[Er]', '[124I]', 
        '[LaH]', '[F]', '[siH]', '[Ga+]', '[Cm]', '[GeH3]', '[IH-]', '[U+6]', '[SeH+]', '[32P]', '[SeH-]',
        '[Pt-]', '[Ir+2]', '[se+]', '[U]', '[F+]', '[BH2]', '[As+]', '[Cf]', '[ClH2+]', '[Ni+]', '[TeH3]',
        '[SbH2]', '[Ag+3]', '%24', '[18O]', '[PH4]', '[Os+2]', '[Na-]', '[Sb+2]', '[V+4]', '[Ho+3]', '[68Ga]',
        '[PH-]', '[Bi+2]', '[Ce+2]', '[Pd+3]', '[99Tc]', '[13C@@H]', '[Fe+6]', '[c]', '[GeH2]', '[10B]',
        '[Cu+3]', '[Mo+2]', '[Cr+]', '[Pd+4]', '[Dy]', '[AsH]', '[Ba+]', '[SeH2]', '[In+]', '[TeH2]', '[BrH+]',
        '[14cH]', '[W+]', '[13C@H]', '[AsH2]', '[In+2]', '[N+2]', '[N@@H+]', '[SbH]', '[60Co]', '[AsH4+]',
        '[AsH3]', '[18OH]', '[Ru-2]', '[Na-2]', '[CuH2]', '[31P]', '[Ti+5]', '[35S]', '[P@@H]', '[ArH]', 
        '[Co+]', '[Zr-2]', '[BH2-]', '[131I]', '[SH5]', '[VH]', '[B+2]', '[Yb+2]', '[14C@H]', '[211At]', 
        '[NH3+2]', '[IrH]', '[IrH2]', '[Rh-]', '[Cr-]', '[Sb+]', '[Ni+3]', '[TaH3]', '[Tl+2]', '[64Cu]',
        '[Tc]', '[Cd+]', '[1H]', '[15nH]', '[AlH2+]', '[FH+2]', '[BiH3]', '[Ru-]', '[Mo+6]', '[AsH+]',
        '[BaH2]', '[BaH]', '[Fe+4]', '[229Th]', '[Th+4]', '[As+3]', '[NH+3]', '[P@H]', '[Li-]', '[7NaH]',
        '[Bi+]', '[PtH+2]', '[p-]', '[Re+5]', '[NiH]', '[Ni-]', '[Xe+]', '[Ca+]', '[11c]', '[Rh+4]', '[AcH]',
        '[HeH]', '[Sc+2]', '[Mn+]', '[UH]', '[14CH2]', '[SiH4+]', '[18OH2]', '[Ac-]', '[Re+4]', '[118Sn]',
        '[153Sm]', '[P+2]', '[9CH]', '[9CH3]', '[Y-]', '[NiH2]', '[Si+2]', '[Mn+6]', '[ZrH2]', '[C-2]',
        '[Bi+5]', '[24NaH]', '[Fr]', '[15CH]', '[Se+]', '[At]', '[P-3]', '[124I-]', '[CuH2-]', '[Nb+4]',
        '[Nb+3]', '[MgH]', '[Ir+4]', '[67Ga+3]', '[67Ga]', '[13N]', '[15OH2]', '[2NH]', '[Ho]', '[Cn]'
    ]
SMILES_VOCAB_SIZE = len(SMILES_CHARS)
char_to_idx = {char: i for i, char in enumerate(SMILES_CHARS)}
idx_to_char = {i: char for i, char in enumerate(SMILES_CHARS)}
MAX_SMILES_LEN = 200  # Max length for padding

def tokenize_smiles(smiles):
    """Tokenizes a SMILES string."""
    tokens = ['<sos>'] + list(smiles) + ['<eos>']
    return [char_to_idx.get(char, char_to_idx['.']) for char in tokens]

# === 1. Dataset and Collation ===
class MolecularDataset(Dataset):
    """Custom PyTorch Dataset for multimodal molecular data."""
    def __init__(self, desc, ecfp, maccs, rdkit, smiles, labels):
        self.desc = torch.tensor(desc.values, dtype=torch.float32)
        self.ecfp = torch.tensor(ecfp.values, dtype=torch.float32)
        self.maccs = torch.tensor(maccs.values, dtype=torch.float32)
        self.rdkit = torch.tensor(rdkit.values, dtype=torch.float32)
        self.smiles = smiles
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smiles_str = self.smiles[idx]
        
        # Tokenize SMILES
        tokenized = tokenize_smiles(smiles_str)
        padded_tokens = tokenized + [char_to_idx['<pad>']] * (MAX_SMILES_LEN - len(tokenized))
        smiles_tokens = torch.tensor(padded_tokens[:MAX_SMILES_LEN], dtype=torch.long)

        # Convert SMILES to graph
        graph_data = mol_to_graph(smiles_str, self.labels[idx].item())
        if graph_data is None:
            return None

        return {
            'desc': self.desc[idx],
            'ecfp': self.ecfp[idx],
            'maccs': self.maccs[idx],
            'rdkit': self.rdkit[idx],
            'graph_data': graph_data,
            'smiles': smiles_str,
            'smiles_tokens': smiles_tokens,
            'label': self.labels[idx]
        }

def collate_fn(batch):
    """Custom collate function to handle different data types and None values."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Standard tensors
    desc = torch.stack([item['desc'] for item in batch])
    ecfp = torch.stack([item['ecfp'] for item in batch])
    maccs = torch.stack([item['maccs'] for item in batch])
    rdkit = torch.stack([item['rdkit'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    smiles_tokens = torch.stack([item['smiles_tokens'] for item in batch])
    
    # PyG graphs
    graph_data = Batch.from_data_list([item['graph_data'] for item in batch])
    
    # SMILES strings
    smiles = [item['smiles'] for item in batch]

    return {
        'desc': desc, 'ecfp': ecfp, 'maccs': maccs, 'rdkit': rdkit,
        'graph_data': graph_data, 'smiles': smiles,
        'smiles_tokens': smiles_tokens, 'label': labels
    }


# === 1. SMILES to Graph Conversion ===
def atom_features(atom):
    """Extracts features for a single atom."""
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        float(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        float(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)

def bond_features(bond):
    """Extracts features for a single bond."""
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        float(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def mol_to_graph(smiles, label=None):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        feat = bond_features(bond)
        edge_attr.extend([feat, feat])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 4), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data

def plot_umap_sampling(
    all_embeddings_2d, labeled_idx, pool_idx, query_idx,
    strategy_name, round_num, output_dir,
    smiles_train, y_train):
    # --- Plotting ---
    labeled_emb = all_embeddings_2d[labeled_idx]
    pool_emb = all_embeddings_2d[pool_idx]
    query_emb = all_embeddings_2d[query_idx]

    plt.figure(figsize=(3, 3))
    
    # Plot pool data (grey)
    plt.scatter(pool_emb[:, 0], pool_emb[:, 1], c='dimgray', alpha=0.7, label='Pool data')
    
    # Plot labeled data (blue)
    plt.scatter(labeled_emb[:, 0], labeled_emb[:, 1], c='royalblue', alpha=0.7, label='Training data')
    
    # Plot queried data (yellow)
    plt.scatter(query_emb[:, 0], query_emb[:, 1], c='goldenrod', edgecolor='black', linewidth=1, label='Queried data')
    
    plt.title(f'UMAP of Latent Space - Round {round_num}, Strategy: {strategy_name.capitalize()}', fontsize=12, fontweight='bold', style='italic')
    plt.xlabel('UMAP 1', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('UMAP 2', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Create a dedicated folder for UMAP plots
    umap_dir = os.path.join(output_dir, "umap_plots")
    os.makedirs(umap_dir, exist_ok=True)
    
    output_file = os.path.join(umap_dir, f"round_{round_num}_{strategy_name}.svg")
    plt.savefig(output_file, dpi=300, format='svg', bbox_inches='tight')
    plt.close()

    # --- Save Coordinates to CSV ---
    status = np.full(len(all_embeddings_2d), 'pool', dtype=object)
    status[labeled_idx] = 'labeled'
    status[query_idx] = 'queried'

    df = pd.DataFrame({
        'UMAP_1': all_embeddings_2d[:, 0],
        'UMAP_2': all_embeddings_2d[:, 1],
        'SMILES': smiles_train,
        'Label': y_train,
        'Status': status
    })

    # Create a dedicated folder for UMAP data
    umap_data_dir = os.path.join(output_dir, "umap_data")
    os.makedirs(umap_data_dir, exist_ok=True)
    
    csv_output_file = os.path.join(umap_data_dir, f"round_{round_num}_{strategy_name}_coords.csv")
    df.to_csv(csv_output_file, index=False)

# === 1.6. Metrics Calculation ===
def calculate_classification_metrics(y_true, y_prob_positive_class, k=100):
    y_pred = (y_prob_positive_class > 0.5).astype(int)
    # Ensure there are both classes in y_true to avoid errors in metric calculation
    if len(np.unique(y_true)) < 2:
        # Return default values if only one class is present
        return {
            'roc_auc': 0.5, 'auprc': 0.0, 'f1': 0.0, 'balanced_accuracy': 0.0,
            'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, f'hit_rate_at_{k}': 0.0
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = recall_score(y_true, y_pred)  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_prob_positive_class),
        'auprc': average_precision_score(y_true, y_prob_positive_class),
        'f1': f1_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        f'hit_rate_at_{k}': calculate_hit_rate_at_k(y_true, y_prob_positive_class, k)
    }
    return metrics

def calculate_hit_rate_at_k(y_true, y_prob_positive_class, k=100):
    """
    Calculates the hit rate (number of true positives) in the top-k predictions.
    This is useful for evaluating a model's ability to prioritize toxic compounds
    within a fixed screening budget.
    
    Args:
        y_true (np.ndarray): Array of true binary labels (0 or 1).
        y_prob_positive_class (np.ndarray): Array of predicted probabilities for the positive class.
        k (int): The number of top samples to consider (the budget).
        
    Returns:
        float: The number of true positives found in the top k predictions.
    """
    # Ensure k is not larger than the number of samples
    k = min(k, len(y_true))
    
    # Get indices that would sort the probabilities in descending order
    top_k_indices = np.argsort(y_prob_positive_class)[-k:]
    
    # Get the true labels for these top k samples
    top_k_true_labels = y_true[top_k_indices]
    
    # Count how many of them are positive (toxic)
    hits = np.sum(top_k_true_labels)
    
    return float(hits)


# === 2. CNN Module ===
class CNN_Module(nn.Module):
    """CNN for feature extraction from fingerprints"""
    def __init__(self, input_dim, output_dim=128):
        super(CNN_Module, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3. Transformer Module ===
class TransformerModule(nn.Module):
    """Transformer for feature extraction"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, output_dim=64):
        super(TransformerModule, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.embedding(x)  # (batch, 1, d_model)
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, output_dim)
        return x

# === 3.5 GNN Module ===
class GNN_Module(nn.Module):
    """Enhanced GNN with more layers and skip connections"""
    def __init__(self, input_dim=8, feature_dim=256):  # Increase capacity
        super(GNN_Module, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = GCNConv(256, 256)  # Add third layer
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)
        
        # Layer 2
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2)
        
        # Layer 3 with skip connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = x3 + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Global pooling
        x = global_mean_pool(x3, batch)
        x = self.fc(x)
        
        return x
    
class SmilesDecoder(nn.Module):
    """Transformer Decoder for SMILES reconstruction"""
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, latent_dim):
        super(SmilesDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_SMILES_LEN, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.latent_to_memory = nn.Linear(latent_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt_tokens, memory):
        # tgt_tokens shape: (batch, seq_len)
        # memory shape: (batch, latent_dim)
        
        tgt_embed = self.embedding(tgt_tokens) + self.pos_encoder[:, :tgt_tokens.size(1), :]
        
        # Project latent vector to match decoder dimension and repeat for each token
        memory_proj = self.latent_to_memory(memory).unsqueeze(1).repeat(1, tgt_tokens.size(1), 1)
        
        # Generate a mask to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)
        
        output = self.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
        return self.fc_out(output)
# === 4. Combined Model ===
class Multimodal(nn.Module):
    """Enhanced Multimodal with better architecture"""
    def __init__(self, desc_dim, ecfp_dim, maccs_dim, rdkit_dim, feature_dim=64):  # Change to 64
        super(Multimodal, self).__init__()
        
        # --- Encoder Part with Residual Connections ---
        self.cnn_desc  = CNN_Module(desc_dim, feature_dim)
        self.cnn_ecfp  = CNN_Module(ecfp_dim, feature_dim)
        self.cnn_maccs = CNN_Module(maccs_dim, feature_dim)
        self.cnn_rdkit = CNN_Module(rdkit_dim, feature_dim)
        
        # Transformer with more capacity
        self.trans_desc  = TransformerModule(desc_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)
        self.trans_ecfp  = TransformerModule(ecfp_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)
        self.trans_maccs = TransformerModule(maccs_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)
        self.trans_rdkit = TransformerModule(rdkit_dim, d_model=128, nhead=4, num_layers=2, output_dim=feature_dim)

        # Enhanced GNN
        self.gnn = GNN_Module(input_dim=8, feature_dim=feature_dim)
        
        # 🔥 IMPROVED: Deeper fusion with residual connections and attention
        fusion_dim = feature_dim * 9
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 🔥 ADD: Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_dim // 4, 9),  # 9 feature groups
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

        # Decoder
        self.smiles_decoder = SmilesDecoder(
            vocab_size=SMILES_VOCAB_SIZE,
            embed_dim=64,
            nhead=4,
            num_layers=3,
            latent_dim=64
        )
        
    def forward(self, desc, ecfp, maccs, rdkit, graph_data, smiles_tokens):
        # --- Encoder ---
        cnn_desc_feat = self.cnn_desc(desc)
        cnn_ecfp_feat = self.cnn_ecfp(ecfp)
        cnn_maccs_feat = self.cnn_maccs(maccs)
        cnn_rdkit_feat = self.cnn_rdkit(rdkit)
        
        trans_desc_feat = self.trans_desc(desc)
        trans_ecfp_feat = self.trans_ecfp(ecfp)
        trans_maccs_feat = self.trans_maccs(maccs)
        trans_rdkit_feat = self.trans_rdkit(rdkit)

        gnn_feat = self.gnn(graph_data)
        
        # 🔥 Concatenate with attention weighting
        combined = torch.cat([
            cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
            trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
            gnn_feat
        ], dim=1)
        
        # 🔥 Apply feature attention
        attention_weights = self.feature_attention(combined).unsqueeze(2)  # (batch, 9, 1)
        feature_groups = combined.view(combined.size(0), 9, -1)  # (batch, 9, feature_dim)
        weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
        
        # Fusion
        latent_features = self.fusion(weighted_features)
        
        # Classification
        logits = self.classifier(latent_features)
        
        # Decoder
        decoder_input = smiles_tokens[:, :-1]
        reconstruction_logits = self.smiles_decoder(decoder_input, latent_features)
        
        return logits, latent_features, reconstruction_logits
    
    def predict_proba(self, desc, ecfp, maccs, rdkit, graph_data):
        """Get probability predictions. Ignores decoder for prediction."""
        with torch.no_grad():
            # Create dummy tokens for forward pass during prediction
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(desc.device)
            logits, _, _ = self.forward(desc, ecfp, maccs, rdkit, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()


# === 4.5. Focal Loss ===
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are the logits from the model (batch_size, C)
        # targets are the ground truth labels (batch_size)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# === 5. Training Function ===
def train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device, recon_weight=0.2):
    """Train Multimodal for one epoch with combined loss"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        if batch is None: continue
        desc = batch['desc'].to(device)
        ecfp = batch['ecfp'].to(device)
        maccs = batch['maccs'].to(device)
        rdkit = batch['rdkit'].to(device)
        graph_data = batch['graph_data'].to(device)
        labels = batch['label'].to(device)
        smiles_tokens = batch['smiles_tokens'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        cls_logits, _, recon_logits = model(desc, ecfp, maccs, rdkit, graph_data, smiles_tokens)
        
        # --- Calculate Losses ---
        # 1. Classification Loss
        loss_cls = criterion_cls(cls_logits, labels)
        
        # 2. Reconstruction Loss
        # Target is tokens shifted by one, excluding the first one (<sos>)
        recon_target = smiles_tokens[:, 1:]
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) and (Batch * SeqLen)
        loss_recon = criterion_recon(
            recon_logits.reshape(-1, SMILES_VOCAB_SIZE),
            recon_target.reshape(-1)
        )
        
        # Combined Loss
        loss = loss_cls + recon_weight * loss_recon
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = cls_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


# === 5.5. Train with Validation ===
def train_with_validation(model, full_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device):
    """
    Trains a model, using a validation split to find the best model state.
    Returns the best model state dict and training history.
    """
    if len(full_train_dataset) < 5: # Cannot split if too small
        train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        best_model_state = None
        history = {'loss': [], 'acc': [], 'val_auprc': []}
        for epoch in range(epochs):
            loss, acc = train_multimodal(model, train_loader, criterion_cls, criterion_recon, optimizer, device)
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['val_auprc'].append(0.0) # No validation
        best_model_state = model.state_dict()
        return best_model_state, history

    # Split the full training data set nto sub-train and sub-validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    sub_train_dataset, sub_val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    sub_train_loader = DataLoader(sub_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    sub_val_loader = DataLoader(sub_val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    best_val_auprc = 0.0
    best_model_state = None
    history = {'loss': [], 'acc': [], 'val_auprc': []}

    for epoch in range(epochs):
        loss, acc = train_multimodal(model, sub_train_loader, criterion_cls, criterion_recon, optimizer, device)
        history['loss'].append(loss)
        history['acc'].append(acc)

        # Evaluate on the sub-validation set
        val_probs, val_labels = evaluate_multimodal(model, sub_val_loader, device)
        if len(np.unique(val_labels)) < 2:
            val_auprc = 0.0
        else:
            precision, recall, _ = precision_recall_curve(val_labels, val_probs[:, 1])
            val_auprc = auc(recall, precision)
        history['val_auprc'].append(val_auprc)

        scheduler.step(val_auprc)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = model.state_dict()
            
    # If no improvement was seen, save the last state
    if best_model_state is None:
        best_model_state = model.state_dict()

    return best_model_state, history


# === 6. Evaluation Function ===
def evaluate_multimodal(model, data_loader, device):
    """Evaluate Multimodal and return predictions"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            # Create dummy tokens for evaluation pass
            batch_size = desc.size(0)
            dummy_tokens = torch.zeros((batch_size, 2), dtype=torch.long).to(device)
            
            logits, _, _ = model(desc, ecfp, maccs, rdkit, graph_data, dummy_tokens)
            probs = F.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            if 'label' in batch:
                all_labels.append(batch['label'].cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    if all_labels:
        all_labels = np.concatenate(all_labels)
        return all_probs, all_labels
    return all_probs


# === 7. Active Learning Sampling Strategies ===

def get_latent_embeddings(model, dataset, device):
    """Helper function to get latent space embeddings for a dataset."""
    model.eval()
    embeddings = []
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            # Encoder pass
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_ecfp_feat = model.cnn_ecfp(ecfp)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            cnn_rdkit_feat = model.cnn_rdkit(rdkit)
            trans_desc_feat = model.trans_desc(desc)
            trans_ecfp_feat = model.trans_ecfp(ecfp)
            trans_maccs_feat = model.trans_maccs(maccs)
            trans_rdkit_feat = model.trans_rdkit(rdkit)
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
                trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
                gnn_feat
            ], dim=1)

            # 🔥 Apply feature attention
            attention_weights = model.feature_attention(combined).unsqueeze(2)
            feature_groups = combined.view(combined.size(0), 9, -1)
            weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
            
            latent_features = model.fusion(weighted_features)
            embeddings.append(latent_features.cpu().numpy())
            
    return np.vstack(embeddings)


def uncertainty_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with highest uncertainty (closest to 0.5 probability)"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # For binary classification, uncertainty is highest when probability is close to 0.5
    uncertainty = np.abs(probs[:, 1] - 0.5)  # Distance from 0.5
    
    # Select samples with the smallest distance to 0.5 (highest uncertainty)
    # We use argsort which sorts in ascending order, so we want the smallest values.
    selected_idx = np.argsort(uncertainty)[:n_samples]
    return selected_idx


def entropy_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with highest prediction entropy"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    entropy_scores = entropy(probs.T)
    selected_idx = np.argsort(entropy_scores)[-n_samples:]
    return selected_idx


def margin_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples with the smallest margin between the top two class probabilities"""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # Calculate the margin (difference between top 2 class probabilities)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    # Select samples with the smallest margin
    selected_idx = np.argsort(margin)[:n_samples]
    return selected_idx


def confidence_sampling_toxic(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Selects samples with the highest predicted probability for the toxic class (exploitation)."""
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    probs, _ = evaluate_multimodal(model, pool_loader, device)
    
    # Assuming class 1 is the "toxic" class
    toxic_class_prob = probs[:, 1]
    
    # Get the indices of the samples with the highest probability for the toxic class
    selected_idx = np.argsort(toxic_class_prob)[-n_samples:]
    return selected_idx


def random_sampling(pool_size: int, n_samples: int) -> np.ndarray:
    """Random sampling"""
    return np.random.choice(pool_size, n_samples, replace=False)


def novelty_sampling(model, pool_dataset, labeled_dataset, n_samples: int, device) -> np.ndarray:
    """Select samples most different from the labeled set in the latent space."""
    pool_embeddings = get_latent_embeddings(model, pool_dataset, device)
    labeled_embeddings = get_latent_embeddings(model, labeled_dataset, device)
    
    # Find the distance of each pool sample to its nearest neighbor in the labeled set
    distances = pairwise_distances(pool_embeddings, labeled_embeddings, metric='euclidean').min(axis=1)
    
    # Select the samples with the largest minimum distances
    selected_idx = np.argsort(distances)[-n_samples:]
    return selected_idx


def diversity_sampling(model, pool_dataset, n_samples: int, device) -> np.ndarray:
    """Select diverse samples using k-means++ like approach in the latent space."""
    pool_embeddings = get_latent_embeddings(model, pool_dataset, device)
    
    selected_idx = []
    # Select the first point randomly
    first_idx = np.random.randint(0, len(pool_embeddings))
    selected_idx.append(first_idx)
    
    for _ in range(n_samples - 1):
        selected_features = pool_embeddings[selected_idx]
        # Calculate distance from all points to the already selected points
        distances = pairwise_distances(pool_embeddings, selected_features, metric='euclidean')
        # Find the minimum distance for each point to any of the selected points
        min_distances = distances.min(axis=1)
        
        # Avoid re-selecting already chosen samples
        min_distances[selected_idx] = -1
        # Select the point that is furthest from any already selected point
        next_idx = np.argmax(min_distances)
        selected_idx.append(next_idx)
    
    return np.array(selected_idx)




# === 8.6. Generate Reconstructions ===
def generate_reconstructions(model, test_loader, idx_to_char, device, output_dir, n_samples_to_show=20):
    """
    Generates reconstructed SMILES from the test set using the trained autoencoder.
    """
    model.eval()
    original_smiles, reconstructed_smiles, predictions, true_labels = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Generating Reconstructions")):
            if batch is None: continue
            desc = batch['desc'].to(device)
            ecfp = batch['ecfp'].to(device)
            maccs = batch['maccs'].to(device)
            rdkit = batch['rdkit'].to(device)
            graph_data = batch['graph_data'].to(device)
            labels = batch['label']
            
            # --- 1. Encoder Pass to get latent features ---
            cnn_desc_feat = model.cnn_desc(desc)
            cnn_ecfp_feat = model.cnn_ecfp(ecfp)
            cnn_maccs_feat = model.cnn_maccs(maccs)
            cnn_rdkit_feat = model.cnn_rdkit(rdkit)
            trans_desc_feat = model.trans_desc(desc)
            trans_ecfp_feat = model.trans_ecfp(ecfp)
            trans_maccs_feat = model.trans_maccs(maccs)
            trans_rdkit_feat = model.trans_rdkit(rdkit)
            gnn_feat = model.gnn(graph_data)
            
            combined = torch.cat([
                cnn_desc_feat, cnn_ecfp_feat, cnn_maccs_feat, cnn_rdkit_feat,
                trans_desc_feat, trans_ecfp_feat, trans_maccs_feat, trans_rdkit_feat,
                gnn_feat
            ], dim=1)

            # 🔥 Apply feature attention
            attention_weights = model.feature_attention(combined).unsqueeze(2)
            feature_groups = combined.view(combined.size(0), 9, -1)
            weighted_features = (feature_groups * attention_weights).view(combined.size(0), -1)
            
            latent_features = model.fusion(weighted_features)
            # --- Get classification prediction ---
            logits = model.classifier(latent_features)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # --- 2. Autoregressive Decoding ---
            batch_size = latent_features.size(0)
            # Start with the <sos> token for each sequence in the batch
            decoder_input = torch.full((batch_size, 1), char_to_idx['<sos>'], dtype=torch.long, device=device)
            
            # Project the latent features once to create the memory for the decoder
            memory = model.smiles_decoder.latent_to_memory(latent_features)
            
            # Generate sequence step-by-step
            for _ in range(MAX_SMILES_LEN - 1):
                # The memory shape for the decoder should be (seq_len, batch, embed_dim)
                # but since we generate one token at a time, we can adapt.
                # Let's make memory (batch, 1, embed_dim) for simplicity with batch_first=True
                memory_for_step = memory.unsqueeze(1)

                tgt_embed = model.smiles_decoder.embedding(decoder_input) + model.smiles_decoder.pos_encoder[:, :decoder_input.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

                # The memory needs to be repeated for each token in the target sequence
                memory_proj = memory.unsqueeze(1).repeat(1, decoder_input.size(1), 1)

                output = model.smiles_decoder.transformer_decoder(tgt_embed, memory_proj, tgt_mask=tgt_mask)
                
                # Get the prediction for the very last token
                last_token_logits = model.smiles_decoder.fc_out(output[:, -1, :])
                next_token = torch.argmax(last_token_logits, dim=1).unsqueeze(1)
                
                # Append the predicted token to the input for the next iteration
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # --- 3. Convert tokens to SMILES strings ---
            for j in range(batch_size):
                original_smiles.append(batch['smiles'][j])
                true_labels.append(labels[j].item())
                predictions.append(probs[j])
                
                # Convert sequence of indices to string
                seq = ""
                for token_idx in decoder_input[j, :]:
                    char = idx_to_char.get(token_idx.item())
                    if char == '<eos>': break
                    if char not in ['<sos>', '<pad>']:
                        seq += char
                reconstructed_smiles.append(seq)

    # Create and save DataFrame
    df = pd.DataFrame({
        'Original_SMILES': original_smiles,
        'Reconstructed_SMILES': reconstructed_smiles,
        'True_Label': true_labels,
        'Predicted_Proba': predictions
    })
    
    output_path = os.path.join(output_dir, "baseline_reconstructions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} SMILES reconstructions to {output_path}")
    
    # Print a few examples
    print("\n🔍 Example Reconstructions:")
    print(df.head(n_samples_to_show).to_string())
    
    return df


# === 8.5. Baseline: Train with initial data and evaluate ===
def run_initial_model_evaluation(
    desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
    desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
    initial_idx, pool_idx,
    output_dir,
    epochs=10,
    n_acquire=50  # Number of samples to acquire from the pool
):
    """
    Trains a model on the initial samples.
    1. Evaluates on the test set for AUPRC.
    2. Predicts on the pool set, acquires the top n_acquire samples, and calculates "Hit 100".
    """
    print(f"\n{'='*80}")
    print(f"🎯 Running Baseline Evaluation on Initial {len(initial_idx)} Samples")
    print(f"{'='*80}\n")

    # Create datasets
    initial_train_dataset = MolecularDataset(
        desc_train.iloc[initial_idx], ecfp_train.iloc[initial_idx], maccs_train.iloc[initial_idx],
        rdkit_train.iloc[initial_idx], smiles_train[initial_idx], y_train[initial_idx]
    )
    pool_dataset = MolecularDataset(
        desc_train.iloc[pool_idx], ecfp_train.iloc[pool_idx], maccs_train.iloc[pool_idx],
        rdkit_train.iloc[pool_idx], smiles_train[pool_idx], y_train[pool_idx]
    )
    test_dataset = MolecularDataset(desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test)

    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # Create and train model
    model = Multimodal(
        desc_dim=desc_train.shape[1], ecfp_dim=ecfp_train.shape[1],
        maccs_dim=maccs_train.shape[1], rdkit_dim=rdkit_train.shape[1]
    ).to(device)

    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    print(f"📊 Training on {len(initial_idx)} initial samples for {epochs} epochs...")
    best_model_state, history = train_with_validation(
        model, initial_train_dataset, epochs, criterion_cls, criterion_recon, optimizer, scheduler, device
    )
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(output_dir, 'best_initial_50_model.pth'))

    # 1. Evaluate on the unseen test set
    test_probs, test_labels = evaluate_multimodal(model, test_loader, device)
    test_metrics = calculate_classification_metrics(test_labels, test_probs[:, 1], k=100)
    test_auprc = test_metrics['auprc']
    test_hit_rate = test_metrics['hit_rate_at_100']
    print(f"✅ Initial Model Test AUPRC: {test_auprc:.4f}")
    print(f"✅ Initial Model Test Hit Rate @ 100: {test_hit_rate:.0f}")

    # 2. Predict on the pool to simulate screening and calculate "Hit 100"
    print(f"\n🔍 Simulating screening on the pool of {len(pool_idx)} samples to find top {n_acquire}...")
    pool_probs, pool_labels = evaluate_multimodal(model, pool_loader, device)
    
    # Get indices of top n_acquire predicted toxic samples from the pool
    top_acquire_indices_local = np.argsort(pool_probs[:, 1])[-n_acquire:]
    top_acquire_indices_global = pool_idx[top_acquire_indices_local]

    # Save predictions for the acquired samples
    acquired_df = pd.DataFrame({
        'SMILES': smiles_train[top_acquire_indices_global],
        'True_Label': y_train[top_acquire_indices_global],
        'Predicted_Proba_Toxic': pool_probs[top_acquire_indices_local, 1],
        'Source': 'Acquired_Baseline'
    })

    # Combine with initial data for a full 100-sample set
    initial_df = pd.DataFrame({
        'SMILES': smiles_train[initial_idx],
        'True_Label': y_train[initial_idx],
        'Predicted_Proba_Toxic': -1,  # No prediction, as it was in training
        'Source': 'Initial_Random'
    })
    
    hit_100_df = pd.concat([initial_df, acquired_df], ignore_index=True)
    hit_100_csv_path = os.path.join(output_dir, "baseline_hit_100_samples.csv")
    hit_100_df.to_csv(hit_100_csv_path, index=False)
    print(f"✅ Saved baseline's 100 selected samples to {hit_100_csv_path}")

    # Calculate "Hit 100"
    total_hits_100 = hit_100_df['True_Label'].sum()
    print(f"✅ Baseline 'Hit 100': Found {total_hits_100:.0f} toxic compounds in the combined 100 samples.")

    # Save summary results
    baseline_results = {
        'initial_samples': len(initial_idx),
        'test_auprc': test_auprc,
        'test_hit_rate_100': test_hit_rate,
        'baseline_hit_100': total_hits_100,
    }

    summary_df = pd.DataFrame([baseline_results])
    summary_csv_path = os.path.join(output_dir, "initial_model_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved initial model summary to {summary_csv_path}")

    return baseline_results


# === 9. Active Learning Experiment ===
def run_al_experiment_for_strategy(
    strategy_name,
    desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
    test_loader,
    initial_idx, pool_idx,
    output_dir,
    n_queries, n_instances, epochs_per_round,
    device
):
    """Runs a full active learning loop for a single strategy."""
    print(f"\n{'='*30} Running Strategy: {strategy_name.capitalize()} {'='*30}")

    # Initialize indices for this strategy
    labeled_idx = initial_idx.copy()
    pool_idx = pool_idx.copy()

    # Performance tracking for this strategy
    test_auprc_history = []
    cumulative_hits_history = [y_train[initial_idx].sum()]
    acquired_samples_list = []

    # Initial dataset and model
    initial_dataset = MolecularDataset(
        desc_train.iloc[initial_idx], ecfp_train.iloc[initial_idx],
        maccs_train.iloc[initial_idx], rdkit_train.iloc[initial_idx],
        smiles_train[initial_idx], y_train[initial_idx]
    )
    
    model = Multimodal(
        desc_dim=desc_train.shape[1], ecfp_dim=ecfp_train.shape[1],
        maccs_dim=maccs_train.shape[1], rdkit_dim=rdkit_train.shape[1]
    ).to(device)
    
    criterion_cls = FocalLoss()
    criterion_recon = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)

    # Initial training and evaluation
    print(f"  Round 0: Training on initial {len(initial_idx)} samples...")
    best_initial_state, _ = train_with_validation(
        model, initial_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
    )
    model.load_state_dict(best_initial_state)
    
    probs, labels = evaluate_multimodal(model, test_loader, device)
    metrics = calculate_classification_metrics(labels, probs[:, 1], k=100)
    test_auprc_history.append(metrics['auprc'])
    print(f"    - Initial Test AUPRC: {metrics['auprc']:.4f}, Initial Hits: {cumulative_hits_history[0]}")

    # Active learning loop
    for i in range(n_queries):
        print(f"\n  Query round {i+1}/{n_queries}")

        if len(pool_idx) == 0:
            print("    ⚠️  No more samples in pool. Stopping.")
            break
        
        n_instances_round = min(n_instances, len(pool_idx))

        # Create current datasets
        labeled_dataset = MolecularDataset(
            desc_train.iloc[labeled_idx], ecfp_train.iloc[labeled_idx],
            maccs_train.iloc[labeled_idx], rdkit_train.iloc[labeled_idx],
            smiles_train[labeled_idx], y_train[labeled_idx]
        )
        pool_dataset = MolecularDataset(
            desc_train.iloc[pool_idx], ecfp_train.iloc[pool_idx],
            maccs_train.iloc[pool_idx], rdkit_train.iloc[pool_idx],
            smiles_train[pool_idx], y_train[pool_idx]
        )

        # Train model on current labeled set to guide sampling
        best_state, _ = train_with_validation(
            model, labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
        )
        model.load_state_dict(best_state)

        # Select samples
        pool_loader_for_sampling = DataLoader(pool_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
        pool_probs, _ = evaluate_multimodal(model, pool_loader_for_sampling, device)

        if strategy_name == 'random':
            query_idx_local = random_sampling(len(pool_idx), n_instances_round)
        elif strategy_name == 'uncertainty':
            uncertainty_scores = np.abs(pool_probs[:, 1] - 0.5)
            query_idx_local = np.argsort(uncertainty_scores)[:n_instances_round]
        elif strategy_name == 'entropy':
            entropy_scores = entropy(pool_probs.T)
            query_idx_local = np.argsort(entropy_scores)[-n_instances_round:]
        elif strategy_name == 'margin':
            sorted_probs = np.sort(pool_probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            query_idx_local = np.argsort(margin)[:n_instances_round]
        elif strategy_name == 'novelty':
            query_idx_local = novelty_sampling(model, pool_dataset, labeled_dataset, n_instances_round, device)
        elif strategy_name == 'diversity':
            query_idx_local = diversity_sampling(model, pool_dataset, n_instances_round, device)
        else: # confidence_toxic
            query_idx_local = np.argsort(pool_probs[:, 1])[-n_instances_round:]

        # Get absolute indices and update sets
        query_idx_abs = pool_idx[query_idx_local]
        
        # Save acquired samples info
        acquired_this_round_df = pd.DataFrame({
            'SMILES': smiles_train[query_idx_abs],
            'True_Label': y_train[query_idx_abs],
            'Predicted_Proba_Toxic': pool_probs[query_idx_local, 1],
            'Strategy': strategy_name,
            'Round': i + 1
        })
        acquired_samples_list.append(acquired_this_round_df)

        # Update cumulative hits
        new_hits = y_train[query_idx_abs].sum()
        cumulative_hits_history.append(cumulative_hits_history[-1] + new_hits)

        # Update indices for the next round
        labeled_idx = np.concatenate([labeled_idx, query_idx_abs])
        pool_idx = np.setdiff1d(pool_idx, query_idx_abs)
        
        # Create updated dataset for final evaluation this round
        updated_labeled_dataset = MolecularDataset(
            desc_train.iloc[labeled_idx], ecfp_train.iloc[labeled_idx],
            maccs_train.iloc[labeled_idx], rdkit_train.iloc[labeled_idx],
            smiles_train[labeled_idx], y_train[labeled_idx]
        )
        
        # Retrain on the newly expanded set
        print(f"    Retraining on {len(labeled_idx)} samples...")
        best_final_state, _ = train_with_validation(
            model, updated_labeled_dataset, epochs_per_round, criterion_cls, criterion_recon, optimizer, scheduler, device
        )
        model.load_state_dict(best_final_state)

        # Evaluate performance on test set
        final_probs, final_labels = evaluate_multimodal(model, test_loader, device)
        metrics = calculate_classification_metrics(final_labels, final_probs[:, 1], k=100)
        test_auprc_history.append(metrics['auprc'])
        
        print(f"    - Test AUPRC: {metrics['auprc']:.4f}, Cumulative Hits: {cumulative_hits_history[-1]}")

    # Save the final model after the last round
    torch.save(model.state_dict(), os.path.join(output_dir, f"al_{strategy_name}_last_round.pth"))

    return test_auprc_history, cumulative_hits_history, pd.concat(acquired_samples_list, ignore_index=True)


def active_learning_multimodal(
    desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
    desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
    initial_idx, pool_idx,
    output_dir,
    n_queries=9,
    n_instances=50,
    epochs_per_round=20
):
    """Run active learning experiment with Multimodal, starting from a pre-defined split."""
    
    strategies = [
        'random', 'uncertainty', 'entropy',
        'margin', 'novelty', 'diversity', 'confidence_toxic'
    ]
    
    # Track performance and hits across all strategies
    test_performance = {s: [] for s in strategies}
    cumulative_hits = {s: [] for s in strategies}
    
    # Dataframe to store all acquired samples for all strategies
    all_acquired_samples_df = pd.DataFrame()

    print(f"\n📊 Initial training set size: {len(initial_idx)} samples")
    print(f"📊 Initial pool size: {len(pool_idx)} samples")
    print(f"📊 Initial hits: {y_train[initial_idx].sum()}")
    
    # Store initial samples (once)
    initial_samples_df = pd.DataFrame({
        'SMILES': smiles_train[initial_idx],
        'True_Label': y_train[initial_idx],
        'Predicted_Proba_Toxic': -1,
        'Strategy': 'Initial',
        'Round': 0
    })
    all_acquired_samples_df = pd.concat([all_acquired_samples_df, initial_samples_df], ignore_index=True)

    # Define test dataset (once)
    test_dataset = MolecularDataset(desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # Loop through each strategy and run the full experiment
    for strategy in strategies:
        test_perf_hist, hits_hist, acquired_samples = run_al_experiment_for_strategy(
            strategy_name=strategy,
            desc_train=desc_train, ecfp_train=ecfp_train, maccs_train=maccs_train, rdkit_train=rdkit_train,
            smiles_train=smiles_train, y_train=y_train,
            test_loader=test_loader,
            initial_idx=initial_idx, pool_idx=pool_idx,
            output_dir=output_dir,
            n_queries=n_queries, n_instances=n_instances, epochs_per_round=epochs_per_round,
            device=device
        )
        
        test_performance[strategy] = test_perf_hist
        cumulative_hits[strategy] = hits_hist
        all_acquired_samples_df = pd.concat([all_acquired_samples_df, acquired_samples], ignore_index=True)

    # Calculate sample sizes for plotting
    sample_sizes = [len(initial_idx) + i * n_instances for i in range(n_queries + 1)]

    # Save all acquired samples to a single CSV
    acquired_samples_csv_path = os.path.join(output_dir, "active_learning_acquired_samples.csv")
    all_acquired_samples_df.to_csv(acquired_samples_csv_path, index=False)
    print(f"\n✅ Saved all acquired samples across all strategies to {acquired_samples_csv_path}")

    return test_performance, cumulative_hits, sample_sizes


# === 10. Plotting Function ===
def plot_learning_curves_separate(test_perf, hit_rate_perf, sample_sizes, total_train_size, output_dir):
    """Plots separate learning curves for test AUPRC and Cumulative Hits."""
    
    percent_of_train = [100.0 * s / total_train_size for s in sample_sizes]
    
    # Plot 1: Test AUPRC
    plt.figure(figsize=(6, 4))
    for strategy, scores in test_perf.items():
        # Ensure scores list has the same length as percent_of_train
        if len(scores) == len(percent_of_train):
            plt.plot(percent_of_train, scores, marker='o', linestyle='-', label=strategy.capitalize())
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('Test AUPRC', fontsize=12, fontweight='bold', style='italic')
    plt.title('Active Learning: Test AUPRC', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "al_test_auprc_curves.svg"), format='svg', dpi=300)
    plt.close()

    # Plot 2: Cumulative Hits
    plt.figure(figsize=(6, 4))
    for strategy, scores in hit_rate_perf.items():
        # Ensure scores list has the same length as percent_of_train
        if len(scores) == len(percent_of_train):
            plt.plot(percent_of_train, scores, marker='o', linestyle='-', label=strategy.capitalize())
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12, fontweight='bold', style='italic')
    plt.ylabel('Cumulative Toxic Hits Found', fontsize=12, fontweight='bold', style='italic')
    plt.title('Active Learning: Cumulative Hits', fontsize=12, fontweight='bold', style='italic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.7, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "al_cumulative_hits_curves.svg"), format='svg', dpi=300)
    plt.close()
    
    print(f"✅ Saved active learning plots to {output_dir}")


# === 11. Main Function ===
def main(random_seed=0):
    print(f"\n{'='*80}")
    print(f"🚀 Starting Active Learning with Multimodal")
    print(f"SEED: {random_seed}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    output_dir = f"repeat_active_learning_{random_seed}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: '{output_dir}'")
    train = pd.read_csv(f'train_{random_seed}.csv')
    test  = pd.read_csv(f'test_{random_seed}.csv')
    y_train = train['Class'].values
    y_test  =  test['Class'].values
    smiles_train = train['canonical_smiles'].values
    smiles_test  = test['canonical_smiles'].values
    print(f"✅ Loaded labels and SMILES: {len(y_train)} train, {len(y_test)} test")

    # Calculate features
    print("⚙️  Calculating molecular features...")
    train_ecfp = calculate_ecfp(train, smiles_col="canonical_smiles", radius=10, nBits=4096)
    test_ecfp  = calculate_ecfp(test, smiles_col="canonical_smiles", radius=10, nBits=4096)
    train_maccs = calculate_maccs(train, smiles_col="canonical_smiles")
    test_maccs  = calculate_maccs(test, smiles_col="canonical_smiles")
    train_rdkit = calculate_rdkit(train, smiles_col="canonical_smiles")
    test_rdkit  = calculate_rdkit(test, smiles_col="canonical_smiles")
    train_desc  = calculate_descriptors(train, smiles_col="canonical_smiles")
    test_desc   = calculate_descriptors(test, smiles_col="canonical_smiles")
    print(f"✅ Computed ECFP, MACCS, RDKit, and Descriptor features.")

    # Handle potential NaN values from feature calculation


    desc_train, ecfp_train, maccs_train, rdkit_train = train_desc, train_ecfp, train_maccs, train_rdkit
    desc_test, ecfp_test, maccs_test, rdkit_test = test_desc, test_ecfp, test_maccs, test_rdkit
    
    print(f"\n📊 Training samples: {len(y_train)}")
    print(f"📊 Test samples: {len(y_test)}")

    # === Create the initial 30-sample split ===
    n_initial = 30
    np.random.seed(random_seed)
    initial_idx = np.random.choice(len(desc_train), n_initial, replace=False)
    pool_idx = np.setdiff1d(np.arange(len(desc_train)), initial_idx)
    
    # === BASELINE: Train with initial 30 data and evaluate ===
    baseline_results = run_initial_model_evaluation(
        desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
        desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
        initial_idx, pool_idx,
        output_dir=output_dir,
        epochs=20,
        n_acquire=70
    )
    
    # === ACTIVE LEARNING ===
    print(f"\n{'='*80}")
    print(f"🎯 Starting Active Learning Experiments")
    print(f"{'='*80}")
    
    # Run active learning
    test_perf, cumulative_hits, sample_sizes = active_learning_multimodal(
        desc_train, ecfp_train, maccs_train, rdkit_train, smiles_train, y_train,
        desc_test, ecfp_test, maccs_test, rdkit_test, smiles_test, y_test,
        initial_idx, pool_idx,
        output_dir=output_dir,
        n_queries=7,
        n_instances=10,
        epochs_per_round=20
    )
    
    # Save active learning results
    df = pd.DataFrame({"n_samples": sample_sizes})
    total_train_size = len(y_train)
    df["percent_of_full_train"] = [100.0 * s / total_train_size for s in sample_sizes]
    for strategy, scores in test_perf.items():
        if len(scores) == len(sample_sizes):
            df[f"{strategy}_test_auprc"] = scores
    for strategy, hits in cumulative_hits.items():
        if len(hits) == len(sample_sizes):
            df[f"{strategy}_cumulative_hits"] = hits
    
    csv_path = os.path.join(output_dir, "performance_active_learning.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved active learning performance to {csv_path}")
    
    # Plot learning curves
    plot_learning_curves_separate(test_perf, cumulative_hits, sample_sizes, total_train_size, output_dir)

    # === COMPARISON SUMMARY ===
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON: Baseline vs Active Learning")
    print(f"{'='*80}\n")
    
    print(f"BASELINE (Initial {n_initial} + Acquired {baseline_results.get('n_acquire', 70)}):")
    print(f"  Test AUPRC: {baseline_results['test_auprc']:.4f}")
    print(f"  Total Hits in 100 samples: {baseline_results['baseline_hit_100']:.0f}")
    
    print(f"\nACTIVE LEARNING (Final results after {sample_sizes[-1]} total samples):")
    for strategy in test_perf:
        if len(test_perf[strategy]) == len(sample_sizes):
            test_final = test_perf[strategy][-1]
            hits_final = cumulative_hits[strategy][-1]
            samples_used = sample_sizes[-1]
            percent_used = 100.0 * samples_used / len(y_train)
            print(f"  {strategy.capitalize():<14s} ({percent_used:5.1f}% data): Test AUPRC = {test_final:.4f}, Cumulative Hits = {hits_final:<2.0f}")
    
    # Create comparison table
    comparison_data = [{
        'Method': 'Baseline',
        'Samples': n_initial + baseline_results.get('n_acquire', 70),
        'Percent': 100.0 * (n_initial + baseline_results.get('n_acquire', 70)) / len(y_train),
        'Test_AUPRC': baseline_results['test_auprc'],
        'Cumulative_Hits': baseline_results['baseline_hit_100'],
    }]
    
    for strategy in test_perf:
        if len(test_perf[strategy]) == len(sample_sizes):
            comparison_data.append({
                'Method': f'AL_{strategy.capitalize()}',
                'Samples': sample_sizes[-1],
                'Percent': 100.0 * sample_sizes[-1] / len(y_train),
                'Test_AUPRC': test_perf[strategy][-1],
                'Cumulative_Hits': cumulative_hits[strategy][-1],
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(output_dir, "comparison_baseline_vs_al.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n✅ Saved comparison to {comparison_csv_path}")
    
    print(f"\n{'='*80}")
    print("✅ Multimodal Experiments Completed!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    for seed in [0, 10, 20]:
        main(random_seed=seed)
