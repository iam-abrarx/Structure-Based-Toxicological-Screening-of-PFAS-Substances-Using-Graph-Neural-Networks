import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from build_graph_dataset import PFASDataset
import os
import copy
import random
import pandas as pd

# Configuration
BATCH_SIZE = 128
EPOCHS = 75 # Updated to 75 as per user request
LR = 0.001
EMBED_DIM = 128
PROJECTOR_DIM = 64
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ... (rest of imports)
import matplotlib.pyplot as plt

# ... (after imports)

# --- Augmentations ---
def mask_nodes(data, mask_prob=0.1):
    data = copy.deepcopy(data)
    num_nodes = data.num_nodes
    mask = torch.rand(num_nodes) < mask_prob
    # Mask features by zeroing them out
    data.x[mask] = 0.0
    return data

def drop_edges(data, drop_prob=0.1):
    data = copy.deepcopy(data)
    edge_index = data.edge_index
    mask = torch.rand(edge_index.size(1)) >= drop_prob
    data.edge_index = edge_index[:, mask]
    data.edge_attr = data.edge_attr[mask]
    return data

# --- Model ---
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
        
        # Global Pooling
        h = global_add_pool(h, batch)
        h = self.lin(h)
        return h

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index, batch)
        z = self.projector(h)
        return h, z

# --- Loss ---
def nt_xent_loss(z1, z2, temperature=0.5):
    # z1, z2 are [batch, dim]
    batch_size = z1.size(0)
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix
    out = torch.cat([z1, z2], dim=0) # [2N, dim]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim_matrix = sim_matrix.masked_fill(mask, 0) # mask self-sim
    
    # Positive pairs are (i, i+N) and (i+N, i)
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) # [N]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # [2N]
    
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
    return loss.mean()

def main():
    set_seed(SEED)
    
    if not os.path.exists("pfas_graphs.pt") and not os.path.exists("./processed/pfas_graphs.pt"):
        print("Dataset file not found! Please run build_graph_dataset.py first.")
        # Try to run it?
        # return

    dataset = PFASDataset(root='.')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    encoder = GNNEncoder(dataset.num_node_features, EMBED_DIM, EMBED_DIM)
    model = ContrastiveModel(encoder, EMBED_DIM, PROJECTOR_DIM).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Starting Pretraining...")
    model.train()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # View 1: Mask Nodes
            view1 = mask_nodes(data)
            h1, z1 = model(view1.x, view1.edge_index, view1.batch)
            
            # View 2: Drop Edges
            view2 = drop_edges(data)
            h2, z2 = model(view2.x, view2.edge_index, view2.batch)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # Save History
    df_hist = pd.DataFrame({'epoch': range(1, EPOCHS+1), 'loss': loss_history})
    df_hist.to_csv("loss_history.csv", index=False)
    
    # Plot Loss 
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.title('GNN Pretraining Loss Curve')
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Loss curve saved.")

    # Save Encoder
    torch.save(model.encoder.state_dict(), "pfas_gnn_encoder.pth")
    print("Pretraining complete. Encoder saved to pfas_gnn_encoder.pth")

    # Generate Embeddings for all data
    print("Generating embeddings...")
    model.eval()
    all_embeds = []
    all_smi = []
    all_dtxsid = []
    
    loader_eval = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for data in loader_eval:
            data = data.to(DEVICE)
            h, _ = model(data.x, data.edge_index, data.batch)
            all_embeds.append(h.cpu())
            all_smi.extend(data.smiles)
            if hasattr(data, 'dtxsid'):
                all_dtxsid.extend(data.dtxsid)
                
    embeddings = torch.cat(all_embeds, dim=0).numpy()
    
    # Save embeddings
    import numpy as np
    np.save("embeddings_pfas.npy", embeddings)
    
    # Save meta data
    df_out = pd.DataFrame({
        'DTXSID': all_dtxsid,
        'SMILES': all_smi
    })
    df_out.to_csv("embeddings_meta.csv", index=False)
    print("Embeddings saved to embeddings_pfas.npy and embeddings_meta.csv")

if __name__ == "__main__":
    main()
