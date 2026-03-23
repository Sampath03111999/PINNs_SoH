# PINNs_SoH
# Battery Aging Prediction using Physics-Informed Neural Networks

**Master's Thesis — Mohan Vijay Sampath Vurukuti**   
[Schmalkalden University of Applied Sciences]  
Supervised by: [Prof. Dr.-Ing. Maria Schweigel] | Industry Partner: Robert Bosch GmbH 

---

## Overview

This repository contains the public implementation of the Physics-Informed
Neural Network (PINN) framework developed in the Master's thesis:

> *Hybrid Modelling for Aging of Lithium-ion Batteries Using Physics-Informed
> Neural Networks*

The framework predicts capacity degradation trajectories of lithium-ion
battery cells by combining a shared deep learning backbone with
cell-specific embedding vectors and physics-informed loss terms derived
from an electrochemical aging model.

---

## Key Features

- **Shared multi-cell architecture** — a single model trained across 16
  CATL 314 Ah LFP cells generalises to unseen cells via learnable
  cell-specific embeddings
- **Physics-informed training** — degradation physics constrain the
  network through an eight-component composite loss function with
  adaptive dual-annealing weight schedules
- **Two physics evaluation modes** — per-sample (exact) and per-batch
  (3.9× speedup, 0.031 % residual accuracy)
- **Embedding fine-tuning** — unseen test cells achieve R² ≥ 0.989
  after adapting only the 8-dimensional embedding vector
- **Sequential future prediction** — 70 % past data used for adaptation,
  30 % held out for evaluation; R² = 0.997 on held-out future data

---

## Repository Structure

```
PINNs_SoH/
│
├── battery_pinn_soh.py   # Main implementation (see note below)
├── README.md                # This file
└── cell_split.json          # Fixed train/test cell split (seed 42)
```

---

## What is Included

| Module | Description |
|---|---|
| `MultiCellDataset` | PyTorch Dataset concatenating multi-cell time-series |
| `SharedBatteryPINN` | 2-layer MLP with learnable cell embeddings |
| `torch_lininterp` | Differentiable 1-D linear interpolation |
| `subsample_cell_series` | Reproducible per-cell time-series subsampling |
| `train_multicell` | Full training loop with adaptive loss weighting and early stopping |
| `evaluate_multicell` | Evaluation with RMSE, R², MAPE metrics |
| `fine_tune_new_cell` | Embedding-only fine-tuning for new cells |
| `fine_tune_embedding_sequential` | Sequential past/future fine-tuning and prediction |

## What is Not Included

The three physics functions listed below contain proprietary
electrochemical equations from the Bosch Particle Aging Model (PAM)
and are replaced with documented stubs in this repository:

| Function | Reason |
|---|---|
| `physics_residual` | Proprietary nine-state PAM ODE system |
| `torch_outputC_batch` | Proprietary SPM capacity mapping |
| `generateAuxPAMInputs_red` | Proprietary SPM auxiliary variable computation |

The stubs include full docstrings describing inputs, outputs, and the
algorithm, so the framework structure is fully reproducible with access
to the PAM library.

---

## Dataset

The experiments use the **CATL 314 Ah LFP** prismatic cell dataset:

- 20 cells total; Cell 3 excluded (incomplete data); 19 usable cells
- 16 cells for training, 3 cells (Cell 6, Cell 8, Cell 16) for testing
- Each cell: approximately 200,000 time-series samples and 8 RPT
  capacity measurements
- Fixed train/test split stored in `cell_split.json`

The dataset is proprietary and not included in this repository.

---

## Results Summary

| Configuration | Zero-Shot R² (Cell 16) | Fine-Tune R² | Future R² | Train Time |
|---|---|---|---|---|
| 5 % per-sample | 0.480 | 0.989 | 0.741 | ~1,900 min |
| 10 % per-sample | 0.393 | 0.999 | 0.997 | ~3,800 min |
| 100 % per-batch | 0.763 | 0.999 | 0.997 | ~493 min |

---

## Requirements

```
Python      3.10.8
PyTorch     2.0.1
NumPy       1.24.3
SciPy       1.10.1
Matplotlib  3.7.1
scikit-learn 1.2.2
```

Install dependencies:
```bash
pip install torch==2.0.1 numpy==1.24.3 scipy==1.10.1 \
            matplotlib==3.7.1 scikit-learn==1.2.2
```

---

## Citation

If you use this code in your work, please cite:

```
Vurukuti, M. V. S. (2025). Hybrid Modelling of Ageing of Li-ion Batteries
Using Physics-Informed Neural Networks. Master's Thesis,
[Schmalkalden University of Applied Sciences].
```

---

## Contact

**Mohan Vijay Sampath Vurukuti**  
[mohanvijaysampath@gmail.com] 
