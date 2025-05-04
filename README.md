# Order‑Flow Imbalance (OFI) Feature Engineering

This repository contains a toolkit for recreating the **best‑level, multi‑level, integrated and cross‑asset OFI** features described in Cont, Cucuringu & Zhang (2024) *“Cross‑Impact of Order Flow Imbalance in Equity Markets.”*

## Key design points

### Multiple depth levels
The toolkit reproduces the paper’s ten‑level specification, yielding an event‑level OFI vector `(ofi_00 … ofi_09)`.

### PCA‑based integrated OFI
`integrated_ofi_daily` fits the first PC **separately for every stock‑day**.

### Cross‑asset design matrix
`cross_asset_ofi` converts a long table of per‑stock predictors into a wide matrix where each row `(t, i)` contains the OFIs of *all* stocks `j ≠ i`, ready for Lasso cross‑impact estimation.
