import numpy as np
import pandas as pd
import os


def process_features_to_market_data(df_features, asset_names, vol_window=20):
    returns_dict = {}
    volatilities_dict = {}

    for asset in asset_names:
        if f'{asset}_return' in df_features.columns:
            rets = df_features[f'{asset}_return']
        elif f'{asset}_Close' in df_features.columns:
            rets = df_features[f'{asset}_Close'].pct_change()
        elif f'{asset}_close' in df_features.columns:
            rets = df_features[f'{asset}_close'].pct_change()
        else:
            raise ValueError(
                f"Attenzione: Impossibile trovare una colonna di prezzo o rendimento per l'asset '{asset}'.")

        returns_dict[asset] = rets

        if f'{asset}_volatility' in df_features.columns:
            vols = df_features[f'{asset}_volatility']
        else:
            vols = rets.rolling(window=vol_window).std()

        volatilities_dict[asset] = vols

    returns_df = pd.DataFrame(returns_dict)
    volatilities_df = pd.DataFrame(volatilities_dict)

    returns_df = returns_df.dropna()
    volatilities_df = volatilities_df.reindex(returns_df.index).bfill()

    return returns_df, volatilities_df


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
    signals_dict = {}
    confidences_dict = {}
    volatilities_dict = {}

    for asset, df in asset_dfs.items():
        if signal_col not in df.columns or conf_col not in df.columns:
            raise ValueError(f"Il DataFrame per '{asset}' deve contenere le colonne '{signal_col}' e '{conf_col}'.")

        sig = df[signal_col].copy()

        # Filtro Long-Only per SPY
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

        # Boost dinamico su SPY basato sul trend
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


def run_backtest(positions_matrix, returns_df, transaction_cost_pct=0.001):
    common_index = positions_matrix.index.intersection(returns_df.index)
    pos = positions_matrix.loc[common_index]
    rets = returns_df.loc[common_index]

    executed_positions = pos.shift(1).fillna(0)

    position_changes = pos.diff().abs().fillna(0)
    turnover = position_changes.sum(axis=1) / 2.0
    transaction_costs = turnover * transaction_cost_pct

    strategy_gross_returns = (executed_positions * rets).sum(axis=1)
    strategy_net_returns = strategy_gross_returns - transaction_costs
    equity_curve = (1.0 + strategy_net_returns).cumprod()

    return pd.DataFrame({
        'Gross_Return': strategy_gross_returns,
        'Net_Return': strategy_net_returns,
        'Transaction_Costs': transaction_costs,
        'Equity_Curve': equity_curve
    })


def calculate_performance_metrics(backtest_results, periods_per_year=52):
    net_returns = backtest_results['Net_Return']
    equity = backtest_results['Equity_Curve']

    total_return = equity.iloc[-1] - 1.0
    n_periods = len(equity)
    annualized_return = (equity.iloc[-1]) ** (periods_per_year / n_periods) - 1.0
    annualized_vol = net_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    total_trades = (backtest_results['Transaction_Costs'] > 0).sum()

    return pd.Series({
        "Rendimento Totale (%)": total_return * 100,
        "Rendimento Annualizzato (%)": annualized_return * 100,
        "Volatilità Annualizzata (%)": annualized_vol * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown (%)": max_drawdown * 100,
        "Numero di Trade": total_trades
    })


# --- ESECUZIONE FINALE: CORE 35% & SATELLITE 65% ---
if __name__ == "__main__":
    folder_path = "../data/signals/"
    asset_names = ['SHY', 'IEF', 'GLD', 'SPY']
    asset_dfs = {}

    for asset in asset_names:
        file_name = f"signals_orchestrator_{asset.lower()}.csv"
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        asset_dfs[asset] = df

    features_path = "../data/processed/features_weekly.csv"
    df_features = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date')

    print("--- Generazione Matrice Satellite Dinamica ---")
    satellite_raw = generate_dynamic_growth_matrix(
        asset_dfs=asset_dfs,
        df_features=df_features,
        signal_col='signal',
        conf_col='confidence',
        vol_col='volatility',
        default_vol=0.1,
        min_confidence=0.35,
        hysteresis_buffer=0.015
    )

    returns_df, _ = process_features_to_market_data(df_features, asset_names=asset_names)

    # Configurazione Fissa Ottimale: Core 35% / Satellite 65%
    core_pct = 0.35
    satellite_pct = 0.65

    print(f"\n--- ESECUZIONE PORTAFOGLIO FINALE (Core {int(core_pct * 100)}% / Sat {int(satellite_pct * 100)}%) ---")

    core_matrix = pd.DataFrame(0.0, index=satellite_raw.index, columns=asset_names)
    core_matrix['SPY'] = core_pct

    scaled_satellite = satellite_raw * satellite_pct
    final_portfolio_matrix = core_matrix + scaled_satellite

    # Backtest della strategia ottimizzata
    bt_res = run_backtest(final_portfolio_matrix, returns_df, transaction_cost_pct=0.001)
    metrics_strategy = calculate_performance_metrics(bt_res, periods_per_year=52)

    # Benchmark SPY Puro (Buy & Hold 100%)
    spy_benchmark_matrix = pd.DataFrame(0.0, index=satellite_raw.index, columns=asset_names)
    spy_benchmark_matrix['SPY'] = 1.0
    spy_bt_res = run_backtest(spy_benchmark_matrix, returns_df, transaction_cost_pct=0.001)
    metrics_spy = calculate_performance_metrics(spy_bt_res, periods_per_year=52)

    # Tabella di confronto finale
    final_comparison = pd.DataFrame({
        f"Core {int(core_pct * 100)}% + Satellite": metrics_strategy,
        "Benchmark (SPY 100%)": metrics_spy
    })

    print("\n" + "=" * 70)
    print("--- RISULTATI FINALI A CONFRONTO ---")
    print("=" * 70)
    print(final_comparison.round(2).to_string())
    print("=" * 70)