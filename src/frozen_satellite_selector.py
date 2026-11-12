"""Frozen satellite selector of the Core35_Satellite65 strategy.

This module contains the weekly satellite selection rule used for the
development freeze and for the single final out-of-sample evaluation.
The logic must not be modified: ``src/submission_validation.py`` checks
its source against the release contract.
"""
import numpy as np
import pandas as pd


def generate_dynamic_growth_matrix(
        asset_dfs,
        df_features=None,
        signal_col='signal',
        conf_col='confidence',
        vol_col='volatility',
        default_vol=0.01,
        min_confidence=0.35,
        hysteresis_buffer=0.015
):
    """Select one satellite ETF per decision date from the orchestrator signals.

    For each date, every asset with a non-zero signal and a confidence of at
    least ``min_confidence`` is scored as ``signal * confidence / volatility``.
    When SPY trades above its 30-week simple moving average, SPY shorts are
    disabled and the SPY score is multiplied by ``price / SMA``, bounded
    between 1.2 and 2.5. The asset with the highest absolute score is
    selected, unless the current holding is still valid and the challenger
    does not exceed its score by more than ``hysteresis_buffer``.

    Args:
        asset_dfs: Mapping ``ticker -> DataFrame`` with signal, confidence and,
            optionally, volatility columns.
        df_features: Optional DataFrame containing the SPY close, used for the
            trend filter and the trend-strength multiplier.
        signal_col: Name of the signal column (-1, 0, +1).
        conf_col: Name of the confidence column.
        vol_col: Name of the volatility column.
        default_vol: Volatility used when the column is missing.
        min_confidence: Minimum confidence required for a signal to qualify.
        hysteresis_buffer: Relative score advantage required to switch asset.

    Returns:
        DataFrame of positions (-1, 0, +1) with one non-zero entry per date
        at most.
    """
    signals_dict = {}
    confidences_dict = {}
    volatilities_dict = {}

    for asset, df in asset_dfs.items():
        if signal_col not in df.columns or conf_col not in df.columns:
            raise ValueError(f"Il DataFrame per '{asset}' deve contenere le colonne '{signal_col}' e '{conf_col}'.")

        sig = df[signal_col].copy()

        # Long-only filter for SPY when it trades above its trend.
        if asset == 'SPY' and df_features is not None:
            close_col = 'SPY_Close' if 'SPY_Close' in df_features.columns else 'SPY_close'
            if close_col in df_features.columns:
                sma_trend = df_features[close_col].rolling(window=30).mean()
                bullish_trend = df_features[close_col] > sma_trend
                sig[(sig == -1) & bullish_trend] = 0

        signals_dict[asset] = sig
        confidences_dict[asset] = df[conf_col]

        if vol_col in df.columns:
            volatilities_dict[asset] = df[vol_col]
        else:
            volatilities_dict[asset] = pd.Series(default_vol, index=df.index)

    signals_df = pd.DataFrame(signals_dict).dropna(how='all')
    confidences_df = pd.DataFrame(confidences_dict).reindex(signals_df.index).fillna(0)
    volatilities_df = pd.DataFrame(volatilities_dict).reindex(signals_df.index).fillna(default_vol)

    assets = list(asset_dfs.keys())
    positions_matrix = pd.DataFrame(0.0, index=signals_df.index, columns=assets)

    current_holding = None

    for t in signals_df.index:
        S = signals_df.loc[t]
        C = confidences_df.loc[t]
        V = volatilities_df.loc[t]

        valid_mask = (S != 0) & (C >= min_confidence)

        if not valid_mask.any():
            current_holding = None
            continue

        scores = (S * C) / (V + 1e-8)

        # Dynamic boost of the SPY score based on trend strength.
        if 'SPY' in scores.index and valid_mask['SPY'] and df_features is not None:
            close_col = 'SPY_Close' if 'SPY_Close' in df_features.columns else 'SPY_close'
            try:
                price = df_features.loc[t, close_col]
                sma = df_features[close_col].rolling(window=30).mean().loc[t]
                if price > sma:
                    trend_strength = min(2.5, max(1.2, price / sma))
                    scores['SPY'] *= trend_strength
            except KeyError:
                scores['SPY'] *= 1.5

        abs_scores = scores.abs()
        candidate_best = abs_scores[valid_mask].idxmax()
        candidate_score = abs_scores[valid_mask][candidate_best]

        chosen_asset = candidate_best

        if current_holding is not None and current_holding in abs_scores.index:
            if valid_mask[current_holding]:
                prev_score = abs_scores[current_holding]
                required_score = prev_score * (1.0 + hysteresis_buffer)

                if candidate_score < required_score:
                    chosen_asset = current_holding

        positions_matrix.loc[t, chosen_asset] = np.sign(S[chosen_asset])
        current_holding = chosen_asset

    return positions_matrix
