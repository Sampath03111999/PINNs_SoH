# -*- coding: utf-8 -*-
"""
Battery SOH Prediction using Physics-Informed Neural Networks (PINN)
Master's Thesis – Mohan Vijay Sampath Vurukuti

Framework for multi-cell LFP battery capacity degradation prediction
using a shared PINN architecture with cell-specific embeddings.

Dataset: CATL 314 Ah LFP prismatic cells (19 usable cells)
Physics: SPM-based Particle Aging Model (PAM) — proprietary Bosch model
         (physics functions are stubs; full implementation not included)

Dependencies:
    Python 3.10.8, PyTorch 2.0.1, NumPy 1.24.3, SciPy 1.10.1,
    Matplotlib 3.7.1, scikit-learn 1.2.2
"""

# =============================================================================
# Imports
# =============================================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d

# =============================================================================
# Hyperparameters
# =============================================================================
N_TRAIN_CELLS = 16
N_TEST_CELLS  = 3
BATCH_SIZE    = 512
EPOCHS        = 200
LR            = 5e-4
EMBED_DIM     = 8
HIDDEN        = 64
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# File paths  (update to match your local data location)
# =============================================================================
DATA_FILE = "path/to/PAM_StandardFormat_data.mat"
PAM_JSON  = "path/to/PAMparams.json"
FPM_MAT   = "path/to/FPMfit_CATL_314Ah.mat"

# =============================================================================
# Utility: differentiable 1-D linear interpolation
# =============================================================================
def torch_lininterp(x, xp, fp):
    """
    Differentiable 1-D linear interpolation of fp(xp) evaluated at x.

    Args:
        x  : torch.Tensor — query points (any shape)
        xp : torch.Tensor — 1-D grid, monotonic (ascending or descending)
        fp : torch.Tensor — 1-D values at xp

    Returns:
        torch.Tensor — interpolated values, same shape as x
    """
    x  = x.clone()
    xp = xp.to(x.device)
    fp = fp.to(x.device)

    if xp[0] > xp[-1]:          # ensure ascending
        xp = torch.flip(xp, dims=[0])
        fp = torch.flip(fp, dims=[0])

    x_flat = x.reshape(-1).clamp(xp[0], xp[-1])
    idx    = torch.searchsorted(xp, x_flat, right=False).clamp(1, xp.numel() - 1)

    x_lo, x_hi = xp[idx - 1], xp[idx]
    y_lo, y_hi = fp[idx - 1], fp[idx]

    denom  = (x_hi - x_lo).clamp(min=1e-12)
    t      = (x_flat - x_lo) / denom
    y_flat = y_lo + t * (y_hi - y_lo)

    return y_flat.reshape(x.shape)


# =============================================================================
# Utility: data subsampling
# =============================================================================
def subsample_cell_series(inputs_list, caps_list, cid_list,
                          frac=0.05, min_points=50):
    """
    Randomly subsample each cell's time series to a given fraction.
    Uses a fixed random seed (42) for reproducibility.

    Args:
        inputs_list : list of dicts, one per cell
        caps_list   : list of 1-D capacity arrays
        cid_list    : list of 1-D cell-ID arrays
        frac        : fraction of points to keep (default 0.05 = 5 %)
        min_points  : minimum points to keep per cell

    Returns:
        Subsampled (inputs_list, caps_list, cid_list)
    """
    rng = np.random.default_rng(42)
    new_inputs, new_caps, new_cids = [], [], []

    for inp, cap, cid in zip(inputs_list, caps_list, cid_list):
        cap = np.asarray(cap)
        cid = np.asarray(cid)
        n   = len(cap)

        if n <= min_points:
            new_inputs.append(inp)
            new_caps.append(cap)
            new_cids.append(cid)
            continue

        k   = max(int(frac * n), min_points)
        idx = np.sort(rng.choice(n, size=k, replace=False))

        inp_sub = {k_in: (np.asarray(v)[idx] if len(np.asarray(v)) == n
                          else np.asarray(v))
                   for k_in, v in inp.items()}
        new_inputs.append(inp_sub)
        new_caps.append(cap[idx])
        new_cids.append(cid[idx])

    return new_inputs, new_caps, new_cids


# =============================================================================
# Dataset
# =============================================================================
class MultiCellDataset(Dataset):
    """
    PyTorch Dataset that concatenates time-series data from multiple cells.

    Each sample is a tuple (features, capacity, cell_id).
    Features are ordered according to input_keys_order.
    Data are sorted chronologically within each cell.
    """

    def __init__(self, inputs_list, caps_list, cellid_list, input_keys_order):
        X_parts, y_parts, cid_parts = [], [], []
        self.cell_sizes = []

        for inputs, caps, cids in zip(inputs_list, caps_list, cellid_list):
            sort_idx = np.argsort(np.asarray(inputs['time']).flatten())
            cols     = [np.asarray(inputs[k], dtype=np.float32).flatten()[sort_idx]
                        for k in input_keys_order]
            min_len  = min(len(c) for c in cols)
            if min_len == 0:
                continue
            cols    = [c[:min_len] for c in cols]
            y       = np.asarray(caps,  dtype=np.float32).flatten()[sort_idx][:min_len]
            cid_arr = np.asarray(cids,  dtype=np.int64  ).flatten()[:min_len]

            X_parts.append(np.column_stack(cols))
            y_parts.append(y)
            cid_parts.append(cid_arr)
            self.cell_sizes.append(min_len)

        self.X    = np.concatenate(X_parts)
        self.y    = np.concatenate(y_parts)
        self.cids = np.concatenate(cid_parts)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx],    dtype=torch.float32),
            torch.tensor(self.y[idx],    dtype=torch.float32),
            torch.tensor(self.cids[idx], dtype=torch.long),
        )


# =============================================================================
# Neural network
# =============================================================================
class SharedBatteryPINN(nn.Module):
    """
    Shared multi-layer perceptron with learnable cell-specific embeddings.

    Architecture:
        [8 features | 8 embedding] -> Linear(16, 64) -> ReLU
                                   -> Linear(64, 64) -> ReLU
                                   -> Linear(64,  1)

    Args:
        n_inputs  : number of input features (default 8)
        n_cells   : total number of cells in embedding table (default 20)
        embed_dim : embedding vector dimension (default 8)
        hidden    : hidden layer width (default 64)
    """

    def __init__(self, n_inputs=8, n_cells=20, embed_dim=8, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_cells,
                                  embedding_dim=embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)

        self.net = nn.Sequential(
            nn.Linear(n_inputs + embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, cell_id):
        """
        Args:
            x       : (batch, n_inputs) feature tensor
            cell_id : (batch,) integer cell indices

        Returns:
            (batch,) predicted normalised capacity
        """
        e   = self.embed(cell_id)
        inp = torch.cat([x, e], dim=1)
        return self.net(inp).squeeze(1)


# =============================================================================
# Physics stubs
# (Full implementations contain proprietary Bosch PAM equations
#  and are not included in this repository.)
# =============================================================================

def generateAuxPAMInputs_red(inputAM, p_jj, AMS, pSPM,
                              return_history=False, debug=False):
    """
    Generate auxiliary PAM inputs (Uneg, OCV, Dsoc) from raw measurements.

    Cleans duplicate timestamps, computes anode SOC limits via Newton
    iteration on the SPM equilibrium equations, and interpolates electrode
    potentials using the SPM OCP lookup tables.

    NOTE: Full implementation contains proprietary PAM equations.
          This stub raises NotImplementedError.
    """
    raise NotImplementedError(
        "generateAuxPAMInputs_red contains proprietary Bosch PAM equations "
        "and is not included in this public repository."
    )


def physics_residual(ts, x0, I, T, V, Uneg, OCV, Dsoc, params):
    """
    Propagate the nine PAM degradation states forward by one time step
    using explicit Euler integration.

    States: x1=SEI thickness, x2=Li loss, x3=solvent concentration,
            x4=neg. electrode capacity, x5=pos. electrode capacity (=0 LFP),
            x6=filtered current density, x7/x8=resistance SOH factors,
            x9=SOC accumulator.

    Args:
        ts     : time step [s]
        x0     : (batch, 9) initial state tensor
        I,T,V,Uneg,OCV,Dsoc : (batch,) input tensors
        params : LearnableBatteryParams instance

    Returns:
        xnew : (batch, 9) updated states
        rhs  : (batch, 9) state derivatives

    NOTE: Full implementation contains proprietary Bosch PAM equations
          and is not included in this public repository.
    """
    raise NotImplementedError(
        "physics_residual contains proprietary Bosch PAM equations "
        "and is not included in this public repository."
    )


def torch_outputC_batch(x_batch, p0_np, fpm_tensors, AMS_dict):
    """
    Map PAM internal states to observable cell capacity (Ah).

    Algorithm:
        1. Convert states to electrode equilibrium parameters (p_k)
        2. Compute anode/cathode capacities Qa, Qc from SPM OCP tables
        3. Find anode SOC limits at Vmin/Vmax via OCV curve
        4. Return C = Qa * (SOCa_max - SOCa_min) / 3600

    Args:
        x_batch     : (B, 9) state tensor
        p0_np       : numpy array of nominal equilibrium parameters
        fpm_tensors : precomputed SPM tensor dict (from build_fpm_tensors)
        AMS_dict    : physical parameters dict

    Returns:
        dict with keys 'C' (B,), 'SOC', 'OCV', 'Qa', 'Qc', 'QLi'

    NOTE: Full implementation contains proprietary Bosch PAM equations
          and is not included in this public repository.
    """
    raise NotImplementedError(
        "torch_outputC_batch contains proprietary Bosch PAM equations "
        "and is not included in this public repository."
    )


# =============================================================================
# Adaptive loss weight schedules
# =============================================================================
def get_physics_weight(epoch, epochs_total=200):
    """Ramp physics loss weight from 0 to 0.10 over first 30 % of training."""
    return 0.10 * min(1.0, epoch / (epochs_total * 0.3))


def get_consistency_weight(epoch):
    """Exponential decay of neural-physics consistency weight."""
    return 1.0 * np.exp(-0.03 * epoch)


# =============================================================================
# Training
# =============================================================================
def train_multicell(model, params_model, train_loader, AMS, p_0,
                    C_nom_dict, val_loader,
                    epochs=200, lr=5e-4, ts=1.0, device="cpu",
                    patience=20, lambda_phys_base=0.1,
                    lambda_consistency=1.0, lambda_smooth=1e-4,
                    cap_mean=None, cap_std=None, fpm_tensors=None):
    """
    Train the shared PINN with adaptive physics-informed loss weighting.

    Loss components (8 total):
        L_data_nn   : MSE between NN prediction and true capacity
        L_data_phys : MSE between physics capacity and true capacity
        L_data_soh  : MSE between physics SOH and true SOH
        L_phys      : normalised PAM state residual (physics constraint)
        L_cons      : MSE between NN and physics capacity (consistency)
        L_smooth    : temporal smoothness of NN predictions
        L_mono      : monotonicity penalty
        L_reg       : L1 state regularisation

    Adaptive weights:
        lambda_phys ramps from 0.01 to 0.10 over first 60 epochs
        lambda_cons decays exponentially from 1.0 toward 0

    Early stopping: patience = 20 epochs on validation loss.

    Args:
        model          : SharedBatteryPINN instance
        params_model   : LearnableBatteryParams instance
        train_loader   : training DataLoader (batch_size=512, shuffle=False)
        AMS            : physical parameters dict
        p_0            : nominal equilibrium parameter array
        C_nom_dict     : dict mapping cell_id -> nominal capacity (Ah)
        val_loader     : validation DataLoader
        epochs         : maximum training epochs (200)
        lr             : Adam learning rate (5e-4)
        ts             : physics time step [s]
        device         : 'cpu' or 'cuda'
        patience       : early stopping patience (20 epochs)
        lambda_phys_base : base physics loss weight
        lambda_consistency : initial consistency loss weight
        lambda_smooth  : smoothness/monotonicity loss weight (1e-4)
        cap_mean       : capacity normalisation mean
        cap_std        : capacity normalisation std
        fpm_tensors    : precomputed SPM tensors for torch_outputC_batch

    Returns:
        (model, params_model) with best validation weights restored
    """
    assert cap_mean is not None and cap_std is not None
    assert fpm_tensors is not None

    model.to(device)
    params_model.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(params_model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        params_model.train()

        total_sum = data_sum = phys_sum = cons_sum = 0.0
        n_batches = 0

        for batch_x, batch_y, batch_cid in train_loader:
            batch_x   = batch_x.to(device)
            batch_y   = batch_y.to(device)
            batch_cid = batch_cid.to(device)
            bs        = batch_x.shape[0]

            optimizer.zero_grad()

            # --- NN prediction ---
            C_nn = model(batch_x, batch_cid).view(-1)

            # --- Physics inputs ---
            I    = batch_x[:, 1]
            T    = batch_x[:, 2]
            V    = batch_x[:, 3]
            Uneg = batch_x[:, 4]
            OCV  = batch_x[:, 5]
            Dsoc = batch_x[:, 6]

            # --- PAM state evolution ---
            x0           = params_model.x0_init.unsqueeze(0).repeat(bs, 1).to(device)
            xnew, rhs    = physics_residual(ts, x0, I, T, V, Uneg, OCV, Dsoc, params_model)

            # --- Physics residual loss (auto-normalised per state) ---
            rhs_scale = rhs.abs().detach().mean(dim=0) + 1e-8
            phys_loss = torch.mean((rhs / rhs_scale) ** 2)

            # --- Physics capacity ---
            out_batch    = torch_outputC_batch(xnew, p_0, fpm_tensors, AMS)
            C_phys       = out_batch['C'].to(device).float().view(-1)
            C_phys_norm  = (C_phys - cap_mean) / (cap_std + 1e-12)

            # --- True capacities ---
            cap_true_denorm = batch_y * cap_std + cap_mean
            C_nom_batch     = torch.tensor(
                [C_nom_dict[int(cid.item())] for cid in batch_cid],
                dtype=torch.float32, device=device)
            SOHc_true = cap_true_denorm / (C_nom_batch + 1e-12)

            # --- Loss components ---
            data_loss_nn    = ((C_nn - batch_y) ** 2).mean()
            data_loss_physC = ((C_phys_norm - batch_y) ** 2).mean()
            data_loss       = data_loss_nn + data_loss_physC

            consistency_loss = ((C_nn - C_phys_norm) ** 2).mean()
            smooth_loss = (torch.mean((C_nn[1:] - C_nn[:-1]) ** 2)
                           if bs > 1 else torch.tensor(0.0, device=device))
            mono_loss   = (torch.mean(torch.relu(C_nn[1:] - C_nn[:-1]) ** 2)
                           if bs > 1 else torch.tensor(0.0, device=device))
            state_reg   = 1e-6 * torch.mean(torch.abs(xnew))

            # --- Adaptive weights ---
            lp = get_physics_weight(epoch, epochs)
            lc = get_consistency_weight(epoch)

            total_loss = (data_loss
                          + lp * phys_loss
                          + lc * consistency_loss
                          + lambda_smooth * (smooth_loss + mono_loss)
                          + state_reg)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(params_model.parameters()), 1.0)
            optimizer.step()

            total_sum += total_loss.item()
            data_sum  += data_loss.item()
            phys_sum  += phys_loss.item()
            cons_sum  += consistency_loss.item()
            n_batches += 1

        # --- Validation ---
        train_loss = total_sum / n_batches
        val_result = evaluate_multicell(model, val_loader, AMS,
                                        cap_mean, cap_std, device, plot=False)
        val_loss   = float(np.mean((val_result[0] - val_result[1]) ** 2))

        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train={train_loss:.3e} | Data={data_sum/n_batches:.3e} | "
              f"Phys={phys_sum/n_batches:.3e} | Cons={cons_sum/n_batches:.3e} | "
              f"λ_phys={lp:.3f} | λ_cons={lc:.3f} | Val={val_loss:.3e}")

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            no_improve    = 0
            best_state    = {"model":  model.state_dict(),
                             "params": params_model.state_dict()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        params_model.load_state_dict(best_state["params"])

    return model, params_model


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_multicell(model, dataloader, AMS_dict, cap_mean, cap_std,
                       device=DEVICE, plot=False, trim=5,
                       test_inputs_list=None, test_ids=None,
                       feature_means=None, feature_stds=None):
    """
    Evaluate the model and compute RMSE, R², MAPE metrics.

    Args:
        model          : trained SharedBatteryPINN
        dataloader     : DataLoader with test/validation data
        AMS_dict       : physical parameters dict
        cap_mean/std   : capacity normalisation statistics
        device         : torch device
        plot           : whether to show prediction plots
        trim           : samples to trim from plot edges
        test_inputs_list, test_ids, feature_means, feature_stds :
            optional — enables per-cell time-domain plots

    Returns:
        (C_pred_denorm, C_true_denorm, cell_ids) as numpy arrays
    """
    model.to(device)
    model.eval()
    all_pred, all_true, all_cid = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_cid in dataloader:
            batch_x   = batch_x.to(device)
            batch_cid = batch_cid.to(device)
            all_pred.append(model(batch_x, batch_cid).cpu().numpy())
            all_true.append(batch_y.numpy())
            all_cid.append(batch_cid.numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    all_cid  = np.concatenate(all_cid)

    C_pred = (all_pred * cap_std + cap_mean).astype(np.float64)
    C_true = (all_true * cap_std + cap_mean).astype(np.float64)

    mse  = mean_squared_error(C_true, C_pred)
    r2   = r2_score(C_true, C_pred)
    mask = C_true != 0
    mape = np.mean(np.abs((C_true[mask] - C_pred[mask]) / C_true[mask])) * 100

    print(f"Eval → RMSE: {np.sqrt(mse):.4f} Ah | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    if plot:
        def ema(c, a=0.15):
            s = np.zeros_like(c)
            s[0] = c[0]
            for i in range(1, len(c)):
                s[i] = a * c[i] + (1 - a) * s[i - 1]
            return s
        t = np.arange(len(C_true))
        plt.figure(figsize=(10, 4))
        plt.plot(t[trim:-trim], C_true[trim:-trim], label="True", lw=2)
        plt.plot(t[trim:-trim], ema(C_pred)[trim:-trim], "--", label="Predicted", lw=2)
        plt.xlabel("Sample index"); plt.ylabel("Capacity / Ah")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return C_pred, C_true, all_cid


# =============================================================================
# Embedding fine-tuning (new unseen cell)
# =============================================================================
def fine_tune_new_cell(model, model_path, new_loader, new_cell_id,
                       lr=5e-3, steps=150, device="cpu"):
    """
    Fine-tune only the embedding vector for a new unseen cell.
    All shared network weights are frozen.

    Args:
        model        : trained SharedBatteryPINN
        model_path   : path to saved model weights (.pt file)
        new_loader   : DataLoader for the new cell (normalised)
        new_cell_id  : integer cell ID
        lr           : embedding learning rate (5e-3)
        steps        : number of optimisation steps (150)
        device       : torch device

    Returns:
        fine-tuned model
    """
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = ("embed" in name)

    if new_cell_id >= model.embed.num_embeddings:
        old_w = model.embed.weight.data
        new_w = torch.cat([old_w, torch.zeros(1, old_w.shape[1], device=device)])
        model.embed = nn.Embedding.from_pretrained(new_w, freeze=False)

    optimizer = torch.optim.Adam(model.embed.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    model.train()

    for step in range(steps):
        total, nb = 0.0, 0
        for batch_x, batch_y, batch_cid in new_loader:
            batch_x   = batch_x.to(device)
            batch_y   = batch_y.to(device)
            batch_cid = torch.full_like(batch_cid, new_cell_id)
            loss      = loss_fn(model(batch_x, batch_cid).view(-1), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item(); nb += 1
        if step % 30 == 0 or step == steps - 1:
            print(f"  FT step {step:03d}/{steps} | Loss={total/nb:.4e}")

    return model


# =============================================================================
# Sequential future-prediction fine-tuning
# =============================================================================
def fine_tune_embedding_sequential(model, test_dataset, cell_id,
                                   lr=5e-3, frac_past=0.7, steps=150):
    """
    Fine-tune embedding on the first frac_past of a cell's history,
    then evaluate predictions on the remaining (1 - frac_past) portion.

    Args:
        model        : trained SharedBatteryPINN
        test_dataset : MultiCellDataset containing the target cell
        cell_id      : integer cell ID
        lr           : embedding learning rate (5e-3)
        frac_past    : fraction of history used for adaptation (0.70)
        steps        : number of optimisation steps (150)

    Returns:
        (preds_future, y_future) — both as numpy arrays (normalised)
    """
    model.eval()
    mask     = test_dataset.cids == cell_id
    Xc, yc  = test_dataset.X[mask], test_dataset.y[mask]
    sort_idx = np.argsort(Xc[:, 0])
    Xc, yc  = Xc[sort_idx], yc[sort_idx]

    split    = int(frac_past * len(Xc))
    Xpast, ypast      = Xc[:split],  yc[:split]
    Xfuture, yfuture  = Xc[split:],  yc[split:]

    if len(Xpast) < 5 or len(Xfuture) < 5:
        print(f"Not enough samples for Cell {cell_id}.")
        return np.array([]), np.array([])

    for p in model.parameters():
        p.requires_grad = False

    embed_vec = model.embed.weight[cell_id].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([embed_vec], lr=lr)
    loss_fn   = nn.MSELoss()

    Xp_t = torch.tensor(Xpast,  dtype=torch.float32)
    yp_t = torch.tensor(ypast,  dtype=torch.float32)

    for step in range(steps):
        optimizer.zero_grad()
        e    = embed_vec.expand(len(Xp_t), -1)
        pred = model.net(torch.cat([Xp_t, e], dim=1)).squeeze(1)
        loss_fn(pred, yp_t).backward()
        optimizer.step()
        if step % 30 == 0:
            print(f"  Step {step:03d} | Loss={loss_fn(pred, yp_t).item():.4e}")

    model.eval()
    with torch.no_grad():
        Xf_t = torch.tensor(Xfuture, dtype=torch.float32)
        ef   = embed_vec.expand(len(Xf_t), -1)
        preds_future = model.net(torch.cat([Xf_t, ef], dim=1)).squeeze(1)
        mse  = loss_fn(preds_future, torch.tensor(yfuture, dtype=torch.float32)).item()
        r2   = 1 - mse / (np.var(yfuture) + 1e-12)
        print(f"Future R²={r2:.4f}")

    model.embed.weight.data[cell_id] = embed_vec.detach()
    return preds_future.numpy(), yfuture


# =============================================================================
# Main
# =============================================================================
def main():
    """
    Main training and evaluation pipeline.

    Steps:
        1. Load data from MATLAB .mat files
        2. Load fixed train/test cell split from JSON
        3. Preprocess: generate auxiliary PAM inputs, interpolate capacity
        4. Subsample training cells (5 % default)
        5. Normalise features and capacity
        6. Train shared PINN (up to 200 epochs, early stopping patience=20)
        7. Evaluate zero-shot generalisation on test cells
        8. Fine-tune embedding for one unseen test cell
        9. Sequential future capacity prediction
    """
    # --- Load data ---
    mat             = scipy.io.loadmat(DATA_FILE, struct_as_record=False, squeeze_me=True)
    all_cell_list   = list(mat['CellData'].flatten())
    targets_list    = list(mat['Targets'].flatten())

    with open("cell_split.json") as f:
        split = json.load(f)
    train_ids = split["train_ids"]   # e.g. [0,1,2,4,5,7,9,10,11,12,13,14,15,17,18,19]
    test_ids  = split["test_ids"]    # e.g. [6, 8, 16]

    with open(PAM_JSON) as f:
        params_json = json.load(f)
    AMS    = params_json['AMS']
    p_opt  = params_json['p_opt']
    x0_init = params_json['x0_init']
    p_0    = np.array(AMS['p_0'], dtype=np.float32)

    input_keys = ['time', 'current', 'temperature', 'voltage',
                  'Uneg', 'OCV', 'Dsoc', 'soc']

    # --- Preprocess (physics stubs raise NotImplementedError) ---
    # Replace the call below with your own generateAuxPAMInputs_red
    # implementation once you have access to the full PAM library.
    #
    # train_inputs_list = [generateAuxPAMInputs_red(...) for cid in train_ids]
    # test_inputs_list  = [generateAuxPAMInputs_red(...) for cid in test_ids]
    print("Physics stubs active — replace with full PAM implementation to run.")
    return


if __name__ == "__main__":
    main()
