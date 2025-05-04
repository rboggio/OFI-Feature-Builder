from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class OFIFeatureEngineer:
    """
    Object for computing best‑level, multi‑level, integrated and
    cross‑asset Order‑Flow Imbalance (OFI) features.

    Parameters
    ----------
    levels : int, default 10
        Number of depth levels expected in the LOB snapshot (0 … levels‑1).
    col_fmts : dict[str, str], optional
        Format strings for the column names.  Defaults assume the schema
        'bid_px_00', 'bid_sz_00', ..., 'ask_sz_09'.
        Keys = 'bid_px', 'bid_sz', 'ask_px', 'ask_sz'.
        Each format string **must** accept one integer argument.
    """

    def __init__(
        self,
        levels: int = 10,
        col_fmts: dict | None = None,
    ) -> None:
        self.levels = levels
        self.col_fmts = col_fmts or {
            "bid_px": "bid_px_{:02d}",
            "bid_sz": "bid_sz_{:02d}",
            "ask_px": "ask_px_{:02d}",
            "ask_sz": "ask_sz_{:02d}",
        }

    # ------------------------------------------------------------------ #
    #  low‑level
    # ------------------------------------------------------------------ #
    @staticmethod
    def _single_level_ofi(px_bid: pd.Series,
                          sz_bid: pd.Series,
                          px_ask: pd.Series,
                          sz_ask: pd.Series) -> pd.Series:
        """Vectorised OFI for one depth level."""
        prev_px_bid, prev_px_ask = px_bid.shift(), px_ask.shift()
        prev_sz_bid, prev_sz_ask = sz_bid.shift(), sz_ask.shift()

        bid_contrib = (
            (px_bid == prev_px_bid) * (sz_bid - prev_sz_bid) +
            (px_bid > prev_px_bid) * sz_bid -
            (px_bid < prev_px_bid) * prev_sz_bid
        )

        ask_contrib = (
            (px_ask == prev_px_ask) * (sz_ask - prev_sz_ask) +
            (px_ask < prev_px_ask) * sz_ask -
            (px_ask > prev_px_ask) * prev_sz_ask
        )

        return bid_contrib - ask_contrib

    # ------------------------------------------------------------------ #
    #  feature builders
    # ------------------------------------------------------------------ #
    def best_level_ofi(self, df: pd.DataFrame,
                       level: int = 0,
                       prefix: str = "ofi_best") -> pd.Series:
        """Best‑level OFI (for single‑level OFI at depth `level`)."""
        px_b = df[self.col_fmts["bid_px"].format(level)]
        sz_b = df[self.col_fmts["bid_sz"].format(level)]
        px_a = df[self.col_fmts["ask_px"].format(level)]
        sz_a = df[self.col_fmts["ask_sz"].format(level)]
        ofi = self._single_level_ofi(px_b, sz_b, px_a, sz_a)
        ofi.name = prefix
        return ofi

    def multi_level_ofi(self, df: pd.DataFrame,
                        prefix: str = "ofi_") -> pd.DataFrame:
        """Compute per‑level OFI for levels 0…levels‑1."""
        out = {}
        for lvl in range(self.levels):
            out[f"{prefix}{lvl:02d}"] = self._single_level_ofi(
                df[self.col_fmts["bid_px"].format(lvl)],
                df[self.col_fmts["bid_sz"].format(lvl)],
                df[self.col_fmts["ask_px"].format(lvl)],
                df[self.col_fmts["ask_sz"].format(lvl)],
            )
        return pd.DataFrame(out, index=df.index)

    # ---- integrated OFI ------------------------------------------------ #
    @staticmethod
    def _normalise_pca_weights(w: np.ndarray) -> np.ndarray:
        if w.sum() < 0:
            w = -w
        return w / w.sum()

    def fit_pca_weights(self, ofi_levels: pd.DataFrame) -> np.ndarray:
        """
        Estimate PCA, returns a weight vector of length = n_levels.
        """
        clean = ofi_levels.dropna()
        pca = PCA(n_components=1).fit(clean.values)
        return self._normalise_pca_weights(pca.components_[0])

    def integrated_ofi(self,
                       ofi_levels: pd.DataFrame,
                       weights: np.ndarray | None = None,
                       col_name: str = "ofi_integrated"
                       ) -> pd.Series:
        """
        Collapse multi‑level OFI to a single series using PCA weights.
        If `weights` is None they are estimated on the passed data.
        """
        if weights is None:
            weights = self.fit_pca_weights(ofi_levels)
        if len(weights) != ofi_levels.shape[1]:
            raise ValueError("Weight vector length ≠ number of OFI columns")
        integ = ofi_levels.dot(weights)
        integ.name = col_name
        return integ

    # ---- cross‑asset OFI ---------------------------------------------- #
    @staticmethod
    def cross_asset_ofi(ofi_long: pd.DataFrame,
                        time_col: str = "timestamp",
                        symbol_col: str = "symbol",
                        ofi_col: str = "ofi_integrated",
                        prefix: str = "ofi_",
                        drop_self: bool = True) -> pd.DataFrame:
        """
        Create a wide matrix with rows (time, target_symbol) and columns
        'ofi_<ticker>' for every asset in the universe.
        """
        wide = ofi_long.pivot(index=time_col,
                              columns=symbol_col,
                              values=ofi_col).add_prefix(prefix)

        # broadcast to multi‑index (time, target_symbol)
        stacked_blocks = []
        for tgt in wide.columns:
            block = wide.copy()
            block["target_symbol"] = tgt[len(prefix):]
            stacked_blocks.append(block)

        stacked = pd.concat(stacked_blocks)
        stacked.set_index("target_symbol", append=True, inplace=True)
        stacked.index.names = [time_col, symbol_col]

        if drop_self:
            for col in wide.columns:
                tgt = col[len(prefix):]
                stacked.loc[(slice(None), tgt), col] = np.nan

        return stacked.sort_index()





if __name__ == "__main__":


    # 1. Load your sample
    df = pd.read_csv("data/first_25000_rows.csv")

    # 2. Instantiate the feature engineer
    fe = OFIFeatureEngineer(levels=10)

    # 3. Compute multi‑level OFI
    ofi_levels = fe.multi_level_ofi(df)

    # 4. Fit PCA weights
    weights = fe.fit_pca_weights(ofi_levels)

    # 5. Integrated OFI
    df["ofi_integrated"] = fe.integrated_ofi(ofi_levels, weights)


