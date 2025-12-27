from rdkit import Chem
import numpy as np
import torch

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

smi = "C1=CC=CC=C1" # Benzene
mol = Chem.MolFromSmiles(smi)

print("--- Atom Hybridization ---")
for atom in mol.GetAtoms():
    hyb = atom.GetHybridization()
    print(f"Object: {hyb}, Type: {type(hyb)}, Str: '{str(hyb)}'")
    break

print("\n--- Bond Type ---")
for bond in mol.GetBonds():
    bt = bond.GetBondType()
    print(f"Object: {bt}, Type: {type(bt)}, Str: '{str(bt)}'")
    break

print("\n--- Feature Vectors ---")
# Test the logic
try:
    # Hybridization check
    h = str(mol.GetAtoms()[0].GetHybridization())
    res = one_hot_encoding(h, ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
    print(f"Hyb encoding: {res}")
    
    # Bond check
    b = str(mol.GetBonds()[0].GetBondType())
    res_b = one_hot_encoding(b, ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    print(f"Bond encoding: {res_b}")

except Exception as e:
    print(f"Error in encoding: {e}")

# Test Tensor creation
try:
    edge_indices = [[0, 1], [1, 0]]
    t = torch.tensor(edge_indices, dtype=torch.long)
    print(f"Tensor shape: {t.shape}")
except Exception as e:
    print(f"Error in tensor: {e}")
