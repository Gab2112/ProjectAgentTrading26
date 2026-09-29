"""
evaluator.py - Modulo per il calcolo delle metriche quantitative di performance e rischio.
"""

import pandas as pd
import numpy as np


class ModelEvaluator:
    """Calcola le metriche finanziarie essenziali su rendimenti ed equity curve."""

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute_drawdown(self, equity_series: pd.Series) -> tuple[float, pd.Series]:
        """
        Calcola la serie dei drawdown percentuali e il Maximum Drawdown (MDD).
        """
        # Picco storico progressivo (running peak)
        running_max = equity_series.cummax()
        # Serie di drawdown percentuale da ciascun picco
        drawdown_series = (equity_series - running_max) / running_max
        # Il max drawdown è il minimo (valore più negativo) registrato
        max_drawdown = float(drawdown_series.min())
        return max_drawdown, drawdown_series

    def compute_metrics(self, backtest_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola le metriche chiave a confronto tra Strategia e Benchmark.

        Parameters:
            backtest_results (pd.DataFrame): DataFrame contenente 'Strategy_Equity',
                                             'Benchmark_Equity', 'Strategy_Net_Returns',
                                             'Market_Returns'.

        Returns:
            pd.DataFrame: Tabella riassuntiva delle metriche.
        """
        print("[5/5] Calcolo delle metriche di performance e rischio...")

        n_periods = len(backtest_results)
        if n_periods < 2:
            raise ValueError("Serie storica troppo breve per calcolare le metriche.")

        # --- Max Drawdown ---
        strat_mdd, _ = self.compute_drawdown(backtest_results["Strategy_Equity"])
        bench_mdd, _ = self.compute_drawdown(backtest_results["Benchmark_Equity"])

        # --- Rendimento Cumulativo Totale ---
        strat_tot_ret = (backtest_results["Strategy_Equity"].iloc[-1] / backtest_results["Strategy_Equity"].iloc[0]) - 1.0
        bench_tot_ret = (backtest_results["Benchmark_Equity"].iloc[-1] / backtest_results["Benchmark_Equity"].iloc[0]) - 1.0

        # --- Rendimento Annualizzato (CAGR) ---
        years = n_periods / self.periods_per_year
        strat_cagr = ((1.0 + strat_tot_ret) ** (1.0 / years)) - 1.0 if years > 0 else np.nan
        bench_cagr = ((1.0 + bench_tot_ret) ** (1.0 / years)) - 1.0 if years > 0 else np.nan

        # --- Volatilità Annualizzata ---
        strat_vol = backtest_results["Strategy_Net_Returns"].std() * np.sqrt(self.periods_per_year)
        bench_vol = backtest_results["Market_Returns"].std() * np.sqrt(self.periods_per_year)

        # --- Sharpe Ratio Annualizzato ---
        # (R_ann - Rf) / Vol_ann
        strat_sharpe = (strat_cagr - self.risk_free_rate) / strat_vol if strat_vol > 0 else np.nan
        bench_sharpe = (bench_cagr - self.risk_free_rate) / bench_vol if bench_vol > 0 else np.nan

        # Organizzazione in tabella a confronto
        metrics_df = pd.DataFrame(
            {
                "Rendimento Totale": [f"{strat_tot_ret * 100:.2f}%", f"{bench_tot_ret * 100:.2f}%"],
                "CAGR (Ann.)": [f"{strat_cagr * 100:.2f}%", f"{bench_cagr * 100:.2f}%"],
                "Volatilità (Ann.)": [f"{strat_vol * 100:.2f}%", f"{bench_vol * 100:.2f}%"],
                "Sharpe Ratio": [f"{strat_sharpe:.2f}", f"{bench_sharpe:.2f}"],
                "Max Drawdown": [f"{strat_mdd * 100:.2f}%", f"{bench_mdd * 100:.2f}%"],
            },
            index=["Strategia SMA", "Benchmark (B&H)"]
        )

        return metrics_df

    def print_report(self, metrics_df: pd.DataFrame, title: str = "Report Performance") -> None:
        """Stampa la tabella delle metriche formattata a console."""
        print(f"\n==================== {title.upper()} ====================")
        print(metrics_df.to_string())
        print("============================================================\n")