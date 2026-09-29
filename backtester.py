"""
backtester.py - Modulo per la simulazione dell'esecuzione e visualizzazione dell'equity curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Backtester:
    """
    Simula l'esecuzione a mercato di una strategia quantitativa
    considerando commissioni, slippage e visualizzando l'equity curve.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005          # 0.05% per trade
    ):
        self.initial_capital = initial_capital
        self.total_cost_per_trade = commission_rate + slippage

    def run_simulation(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        """Esegue il backtest vettorizzato sui segnali forniti."""
        print("[4/5] Simulazione esecuzione a mercato e calcolo Equity Curve...")

        if df_signals is None or df_signals.empty:
            raise ValueError("Il DataFrame passato al backtester è vuoto o nullo.")

        for col in ["Date", "Close", "Signal"]:
            if col not in df_signals.columns:
                raise KeyError(f"Colonna obbligatoria '{col}' mancante in df_signals.")

        results = df_signals.copy()

        # 1. Rendimenti del mercato (Close-to-Close)
        results["Market_Returns"] = results["Close"].pct_change().fillna(0.0)

        # 2. Posizione a mercato: shift del segnale a t-1 per evitare look-ahead bias
        results["Position"] = results["Signal"].shift(1).fillna(0).astype(int)

        # 3. Rendimento lordo della strategia
        results["Strategy_Gross_Returns"] = results["Position"] * results["Market_Returns"]

        # 4. Transazioni e costi proporzionali al turnover
        results["Trades"] = (results["Position"].diff().abs()).fillna(0.0)
        results["Transaction_Costs"] = results["Trades"] * self.total_cost_per_trade

        # 5. Rendimento netto
        results["Strategy_Net_Returns"] = results["Strategy_Gross_Returns"] - results["Transaction_Costs"]

        # 6. Calcolo del capitale nel tempo (Equity Curve)
        results["Strategy_Cumulative_Return"] = (1.0 + results["Strategy_Net_Returns"]).cumprod()
        results["Strategy_Equity"] = self.initial_capital * results["Strategy_Cumulative_Return"]

        results["Benchmark_Cumulative_Return"] = (1.0 + results["Market_Returns"]).cumprod()
        results["Benchmark_Equity"] = self.initial_capital * results["Benchmark_Cumulative_Return"]

        # Log riassuntivo
        tot_strat = (results["Strategy_Equity"].iloc[-1] / self.initial_capital - 1.0) * 100
        tot_bench = (results["Benchmark_Equity"].iloc[-1] / self.initial_capital - 1.0) * 100
        print(f"Simulazione completata:")
        print(f" - Capitale Finale Strategia: {results['Strategy_Equity'].iloc[-1]:,.2f} € ({tot_strat:+.2f}%)")
        print(f" - Capitale Finale Benchmark: {results['Benchmark_Equity'].iloc[-1]:,.2f} € ({tot_bench:+.2f}%)")

        return results

    def plot_equity_curves(
        self,
        results: pd.DataFrame,
        title: str = "Confronto Capitale Totale nel Tempo (Strategia vs Benchmark)"
    ) -> None:
        """
        Genera il grafico di confronto dell'evoluzione del capitale nel tempo.

        Parameters:
            results (pd.DataFrame): Output di run_simulation contenente 'Date',
                                    'Strategy_Equity' e 'Benchmark_Equity'.
            title (str): Titolo del grafico.
        """
        print("[4/5] Generazione del grafico di confronto capitale...")

        dates = pd.to_datetime(results["Date"])
        strat_equity = results["Strategy_Equity"]
        bench_equity = results["Benchmark_Equity"]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Linee di capitale
        ax.plot(dates, strat_equity, label="Strategia (SMA Filter)", color="#1f77b4", linewidth=2.0)
        ax.plot(dates, bench_equity, label="Benchmark (Buy & Hold)", color="#7f7f7f", linewidth=1.5, linestyle="--")

        # Evidenzia la linea del capitale iniziale
        ax.axhline(self.initial_capital, color="red", linestyle=":", alpha=0.7, label=f"Capitale Iniziale ({self.initial_capital:,.0f} €)")

        # Formattazione assi e label
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Data", fontsize=11)
        ax.set_ylabel("Valore del Portafoglio (€)", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f} €"))

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()