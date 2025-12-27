import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import rdmolops
import os
from tqdm import tqdm

# Allowed atom types etc.
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H']

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom):
    # Features: Symbol (one-hot), Degree (one-hot), Hybridization (one-hot), Aromatic (bool), Charge (int->one-hotish/scaled)
    # Simplified features:
    # 1. Atomic Num / Symbol
    features = one_hot_encoding(atom.GetSymbol(), ATOM_TYPES)
    
    # 2. Degree
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    
    # 3. Hybridization
    features += one_hot_encoding(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
    
    # 4. Aromatic
    features += [1 if atom.GetIsAromatic() else 0]
    
    # 5. Formal Charge (just value)
    features += [atom.GetFormalCharge()]
    
    return np.array(features, dtype=np.float32)

def get_edge_features(bond):
    # Features: BondType (one-hot), Conjugated (bool), Aromatic (bool)
    bt = bond.GetBondType()
    features = one_hot_encoding(str(bt), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    features += [1 if bond.GetIsConjugated() else 0]
    features += [1 if bond.GetIsAromatic() else 0]
    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles, y_val=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    
    # Edge features and adjacency
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        e_feat = get_edge_features(bond)
        
        # Add undirected edge (i,j) and (j,i)
        edge_indices.append([i, j])
        edge_attrs.append(e_feat)
        edge_indices.append([j, i])
        edge_attrs.append(e_feat)
        
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float) # 6 is feature dim
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    
    # Optional labels can be added here
    if y_val is not None:
        data.y = torch.tensor([y_val], dtype=torch.float)
        
    return data

class PFASDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PFASDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['master_chem_table.csv']

    @property
    def processed_file_names(self):
        return ['pfas_graphs.pt']

    def download(self):
        # We assume master_chem_table.csv is already in root
        pass

    def process(self):
        input_path = os.path.join(self.root, '../data/processed/master_chem_table.csv')
        df = pd.read_csv(input_path)
        # Filter valid SMILES
        df = df[df['SMILES_valid'] == True]
        
        data_list = []
        print(f"Converting {len(df)} SMILES to graphs...")
        
        # Debugging: track failures
        failures = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smi = row['SMILES_canonical']
            ld50 = row.get('tox21_oral_rat_ld50_mol_kg', float('nan'))
            
            try:
                graph = smiles_to_graph(smi, y_val=ld50)
                if graph is not None:
                    graph.dtxsid = row.get('DTXSID', 'unknown')
                    data_list.append(graph)
            except Exception as e:
                failures += 1
                if failures < 5:
                    print(f"Error processing SMILES {smi}: {e}")
        
        print(f"Total failures: {failures}")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    # Create dataset in current folder
    dataset = PFASDataset(root='.')
    print(f"Created dataset with {len(dataset)} graphs.")
    print(f"Node feature dim: {dataset.num_node_features}")
    print(f"Edge feature dim: {dataset.num_edge_features}")
    print(f"Sample graph: {dataset[0]}")
