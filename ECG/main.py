import argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import os, sys
# Add project root (parent of ECG) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.Fetal_data_loader import data_loader
from models import Real_FITS

class RealFITS_PeakWrapper(nn.Module):
    def __init__(self, seq_len, cut_freq, enc_in=1, pred_len=0, individual=False):
        super().__init__()
        class Cfg:
            def __init__(self, seq_len, pred_len, enc_in, cut_freq, individual):
                self.seq_len=seq_len
                self.pred_len=pred_len
                self.enc_in=enc_in
                self.cut_freq=cut_freq
                self.individual=individual
        cfg = Cfg(seq_len, pred_len, enc_in, cut_freq, individual)
        self.backbone = Real_FITS.Model(cfg)
        # Head maps reconstructed signal -> peak probability
        self.head = nn.Sequential(
            nn.Conv1d(enc_in, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, L, C)
        y, _ = self.backbone(x)          # (B, L(+pred), C)
        y = y[:, :x.size(1), :]          # ensure same L
        mask = self.head(y.transpose(1,2)).transpose(1,2)  # (B,L,1)
        return mask

class FetalWindowDataset(Dataset):
    def __init__(self, records, seq_len, stride, loader_obj):
        self.samples = []
        total_peaks = sum(r.labels.sum() for r in records)
        print("Records:", len(records), "Total peaks:", int(total_peaks))
        for rec in records:
            X, Y = loader_obj.windowize(rec, seq_len, stride)
            if X.size == 0:
                continue
            X = X[..., None]  # add channel
            Y = Y[..., None]
            for i in range(X.shape[0]):
                self.samples.append( (X[i], Y[i]) )
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

def metrics(pred, true, thresh=0.5):
    pred_bin = (pred >= thresh).astype(np.float32)
    tp = (pred_bin * true).sum()
    fp = (pred_bin * (1-true)).sum()
    fn = ((1-pred_bin) * true).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1 = 2*prec*rec / (prec+rec+1e-8)
    return prec, rec, f1

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x,y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / max(1,len(loader.dataset))

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    all_p, all_t = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            total += loss.item() * x.size(0)
            all_p.append(pred.cpu().numpy())
            all_t.append(y.cpu().numpy())
    all_p = np.concatenate(all_p,0)
    all_t = np.concatenate(all_t,0)
    mse = total / max(1,len(loader.dataset))
    prec, rec, f1 = metrics(all_p, all_t)
    return mse, prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Path to abdominal-and-direct-fetal-ecg dataset folder')
    ap.add_argument('--seq_len', type=int, default=2000)
    ap.add_argument('--stride', type=int, default=500)
    ap.add_argument('--cut_freq', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--no_preprocess', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_path', type=str, default='fetal_real_fits.pt')
    args = ap.parse_args()

    loader_obj = data_loader()
    print("Loading EDF records...")
    records = loader_obj.load_directory(args.data_dir, channel=args.channel, preprocess=not args.no_preprocess)
    if not records:
        print("No EDF files found.")
        return
    dataset = FetalWindowDataset(records, args.seq_len, args.stride, loader_obj)
    if len(dataset) == 0:
        print("No windows extracted.")
        return

    val_len = max(1, int(len(dataset)*args.val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RealFITS_PeakWrapper(seq_len=args.seq_len, cut_freq=args.cut_freq, enc_in=1, pred_len=0, individual=False).to(args.device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, args.device)
        val_mse, prec, rec, f1 = eval_epoch(model, val_loader, loss_fn, args.device)
        print(f"Epoch {epoch}/{args.epochs} | Train MSE {tr_loss:.6f} | Val MSE {val_mse:.6f} | P {prec:.3f} R {rec:.3f} F1 {f1:.3f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model -> {args.save_path}")

    # Sample inference save
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    with torch.no_grad():
        x,y = next(iter(val_loader))
        x = x.to(args.device).float()
        pred = model(x).cpu().numpy()
        np.save('sample_input.npy', x.cpu().numpy())
        np.save('sample_label.npy', y.numpy())
        np.save('sample_pred.npy', pred)
        print("Saved sample_input.npy / sample_label.npy / sample_pred.npy")

if __name__ == '__main__':
    main()