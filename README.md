# OFI Feature Builder

> Generate Best‑Level, Multi‑Level, Integrated, and Cross‑Asset **Order‑Flow Imbalance (OFI)** features.

---

### ✨ Features

| Feature | Column name | Description |
|---------|-------------|-------------|
| Best‑Level OFI | `OFI_best` | Signed size change at level 1 (best bid / ask). |
| Multi‑Level OFI | `OFI_Lk` | Level‑by‑level OFI for *k* = 1…*L*. |
| Integrated OFI | `OFI_integrated` | Weighted sum of Multi‑Level OFI (equal weights ⬇ default, or learned via LASSO). |
| Cross‑Asset OFI | `OFI_crossasset` | Weighted spill‑over from other stocks’ Integrated OFI. |
