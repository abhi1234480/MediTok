import os, sys, warnings, ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ptbxl_gnn_hybrid_final.pth"
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']  # adjust if different
NUM_CLASSES = len(CLASSES)
warnings.filterwarnings("ignore")

def create_anatomical_adjacency():
    rel = [
        (0,1),(1,2),
        (0,4),(1,4),(2,5),(0,3),(3,4),(4,5),
        (6,7),(7,8),(8,9),(9,10),(10,11),
        (1,6),(5,10),(4,11)
    ]
    edges = []
    for a,b in rel:
        edges.append([a,b]); edges.append([b,a])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

EDGE_INDEX = create_anatomical_adjacency()

class Residual1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = (kernel-1)//2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x): return self.act(self.conv(x) + self.res(x))

class PerNodeCNN(nn.Module):
    def __init__(self, out_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            Residual1D(1, 16),
            Residual1D(16, 32),
            Residual1D(32, 48)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, out_dim)
    def forward(self, x):
        x = x.unsqueeze(1)        # (nodes, 1, seq_len)
        x = self.net(x)           # (nodes, ch, seq_len)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class GNNHybrid(nn.Module):
    def __init__(self, num_classes, node_feat_dim=48, gnn_hidden=64):
        super().__init__()
        self.node_cnn = PerNodeCNN(node_feat_dim)
        self.gnn1 = SAGEConv(node_feat_dim, gnn_hidden)
        self.gnn2 = SAGEConv(gnn_hidden, gnn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    def forward(self, data):
        x = self.node_cnn(data.x.to(DEVICE))
        edge_index = data.edge_index.to(DEVICE)
        batch = data.batch.to(DEVICE) if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.classifier(g)

def predict(basefile_noext, model, thresholds=None):
    base = os.path.splitext(basefile_noext)[0]
    rec, _ = wfdb.rdsamp(base)
    sig = rec.astype(np.float32)
    if sig.ndim == 1: sig = sig[:, None]
    if sig.shape[1] != 12 and sig.shape[0] == 12: sig = sig.T
    if sig.shape[1] != 12:
        sig = np.pad(sig, ((0,0),(0, max(0,12-sig.shape[1]))))
        sig = sig[:, :12]
    sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0)+1e-8)
    processed = sig.T
    data = Data(x=torch.tensor(processed, dtype=torch.float), edge_index=EDGE_INDEX)
    batch = Batch.from_data_list([data]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > 0.5).astype(int) if thresholds is None else (probs > np.array(thresholds)).astype(int)
    result = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    predicted = [CLASSES[i] for i in range(NUM_CLASSES) if preds[i] == 1]
    return result, predicted

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_record_without_extension>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Train first.")
        sys.exit(1)
    model = GNNHybrid(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    res, preds = predict(path, model)
    print("Probabilities per class:", res)
    print("Predicted superclasses:", preds)
