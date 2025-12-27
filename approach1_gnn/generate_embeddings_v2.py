import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from build_graph_dataset import PFASDataset
import os
import pandas as pd
import numpy as np

# Same config
BATCH_SIZE = 128
EMBED_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_dim):
        super(GNNEncoder, self).__init__()
        # GIN with MLPs
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)
        h = global_add_pool(h, batch)
        h = self.lin(h)
        return h

def main():
    if not os.path.exists("pfas_gnn_encoder.pth"):
        print("Model file not found!")
        return

    dataset = PFASDataset(root='.')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    encoder = GNNEncoder(dataset.num_node_features, EMBED_DIM, EMBED_DIM).to(DEVICE)
    encoder.load_state_dict(torch.load("pfas_gnn_encoder.pth"))
    encoder.eval()
    
    print("Generating embeddings...")
    all_embeds = []
    all_rows = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            h = encoder(data.x, data.edge_index, data.batch)
            all_embeds.append(h.cpu())
            
            # Extract metadata safely
            # Assuming batch size N
            # data.smiles is list of N
            # data.dtxsid is list of N
            current_batch_size = h.size(0)
            
            smiles_batch = data.smiles if hasattr(data, 'smiles') else [''] * current_batch_size
            dtxsid_batch = data.dtxsid if hasattr(data, 'dtxsid') else ['unknown'] * current_batch_size
            
            # Ensure they are lists
            if not isinstance(smiles_batch, list):
                smiles_batch = [smiles_batch] * current_batch_size # Should not happen if collated
            if not isinstance(dtxsid_batch, list):
                dtxsid_batch = [dtxsid_batch] * current_batch_size

            for i in range(current_batch_size):
                all_rows.append({
                    'SMILES': smiles_batch[i] if i < len(smiles_batch) else '',
                    'DTXSID': dtxsid_batch[i] if i < len(dtxsid_batch) else 'unknown'
                })

    embeddings = torch.cat(all_embeds, dim=0).numpy()
    np.save("embeddings_pfas.npy", embeddings)
    
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv("embeddings_meta.csv", index=False)
    
    print(f"Saved embeddings shape: {embeddings.shape}")
    print(f"Saved metadata shape: {df_out.shape}")

if __name__ == "__main__":
    main()
