import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
import os
import random
import matplotlib.pyplot as plt

# Config
DATA_PATH = "../data/processed/master_chem_table.csv"
OUTPUT_GEN = "generated_safepfas_candidates.csv"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 70 
LR = 0.001
MAX_LEN = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char_to_idx, max_len):
        self.smiles_list = smiles_list
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        # Add Start/End tokens
        smi = "<" + smi + ">"
        input_seq = [self.char_to_idx[c] for c in smi[:-1]]
        target_seq = [self.char_to_idx[c] for c in smi[1:]]
        
        # Pad
        input_seq = input_seq + [0] * (self.max_len - len(input_seq))
        target_seq = target_seq + [0] * (self.max_len - len(target_seq))
        
        return torch.tensor(input_seq[:self.max_len], dtype=torch.long), \
               torch.tensor(target_seq[:self.max_len], dtype=torch.long)

class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    smiles = df[df['SMILES_valid'] == True]['SMILES_canonical'].tolist()
    # Filter length
    smiles = [s for s in smiles if len(s) < MAX_LEN - 2]
    print(f"Training on {len(smiles)} SMILES.")
    
    # Build Vocab
    chars = set()
    for s in smiles:
        chars.update(s)
    chars = sorted(list(chars))
    # Special tokens: 0=PAD, 1=<, 2=> (Start, End)
    # Map chars starting from 3
    char_to_idx = {c: i+3 for i, c in enumerate(chars)}
    char_to_idx['<'] = 1
    char_to_idx['>'] = 2
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(char_to_idx) + 1 # +1 mostly for PAD=0
    
    print(f"Vocab size: {vocab_size}")
    max_idx = max(char_to_idx.values())
    print(f"Max index in mapping: {max_idx}")
    
    if max_idx >= vocab_size:
        print(f"WARNING: Max index {max_idx} >= Vocab Size {vocab_size}. Adjusting vocab size.")
        vocab_size = max_idx + 1
        print(f"New Vocab Size: {vocab_size}")
    
    dataset = SMILESDataset(smiles, char_to_idx, MAX_LEN)
    # Use batch_size=1 for safety against shape mismatches
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    model = RNNGenerator(vocab_size, 128, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Starting training...")
    model.train()
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            output, _ = model(x)
            loss = criterion(output.transpose(1, 2), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        if batches > 0:
            avg_loss = total_loss/batches
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
            
    # Save History
    df_hist = pd.DataFrame({'epoch': range(1, EPOCHS+1), 'loss': loss_history})
    df_hist.to_csv("gen_loss_history.csv", index=False)
    
    # Plot Loss 
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Generative Model Training Curve')
    plt.legend()
    plt.savefig("gen_loss_curve.png")
    print("Loss curve saved.")
        
    torch.save(model.state_dict(), "pfas_generator.pth")
    print("Model saved.")
    
    # Sampling
    print("Generating candidates...")
    model.eval()
    generated = []
    
    # Generate 100
    for _ in range(100):
        start_token = torch.tensor([[1]], dtype=torch.long).to(DEVICE) # <
        hidden = None
        smi_chars = []
        
        curr_token = start_token
        for _ in range(MAX_LEN):
            out, hidden = model(curr_token, hidden)
            # Sample
            probs = torch.softmax(out[0, 0], dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == 2: # >
                break
            if next_idx == 0: # PAD (should trigger break usually or ignore)
                break
            
            if next_idx in idx_to_char:
                c = idx_to_char[next_idx]
                smi_chars.append(c)
            
            curr_token = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
            
        gen_smi = "".join(smi_chars)
        generated.append(gen_smi)
        
    # Filter Validity
    valid_cands = []
    for s in generated:
        mol = Chem.MolFromSmiles(s)
        if mol:
            # Check fluorine (PFAS criteria roughly)
            if "F" in s:
                valid_cands.append(s)
                
    print(f"Generated {len(generated)} sequences.")
    print(f"Valid PFAS-like SMILES: {len(valid_cands)}")
    
    df_gen = pd.DataFrame({'SMILES': valid_cands})
    df_gen.to_csv(OUTPUT_GEN, index=False)
    print("Saved candidates.")

if __name__ == "__main__":
    main()
