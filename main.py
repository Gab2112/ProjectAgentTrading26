"""
main.py - Grid Search su SMA e Plot Esclusivo della Soluzione Vincente
"""

from src.data_acquisition import get_data
from src.data_transformation import DataTransformer
from src.signal_generation import SignalGenerator
from src.backtester import Backtester
from src.evaluator import ModelEvaluator

d
def run_pipeline():
    # =========================================================================
    # Configurazione Parametri
    # =========================================================================
    config = {
        "tickers": "SPY",
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "interval": "1d",
        "train_ratio": 0.8,
        "initial_capital": 100_000.0,
        "commission_rate": 0.001,
        "slippage": 0.0005,
        "risk_free_rate": 0.0,
        # Griglia di parametri per la ricerca
        "window_min": 30,
        "window_max": 150,
        "window_step": 5,
    }

    print("=== AVVIO PIPELINE ===")

    # =========================================================================
    # 1. Acquisizione Dati
    # =========================================================================
    raw_data = get_data(
        tickers=config["tickers"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        interval=config["interval"]
    )

    # =========================================================================
    # 2. Pulizia e Split Cronologico
    # =========================================================================
    transformer = DataTransformer()
    cleaned_data = transformer.clean_data(raw_data)

    train_clean, test_clean = transformer.split_train_test(
        data=cleaned_data,
        train_ratio=config["train_ratio"]
    )

    backtester = Backtester(
        initial_capital=config["initial_capital"],
        commission_rate=config["commission_rate"],
        slippage=config["slippage"]
    )

    # =========================================================================
    # 3. Addestramento / Grid Search sul Train Set (Senza Plot)
    # =========================================================================
    windows_range = range(
        config["window_min"],
        config["window_max"] + 1,
        config["window_step"]
    )

    best_window = None
    best_final_equity = -float("inf")
    best_train_results = None

    print(f"\n--- Ricerca Finestra Ottimale sul Train ({config['window_min']} -> {config['window_max']}, step {config['window_step']}) ---")

    for w in windows_range:
        # Calcolo feature ed esecuzione simulazione
        train_feat = transformer.engineer_features(train_clean, window=w)
        signal_gen = SignalGenerator(sma_col_name=f"SMA_{w}")
        train_sig = signal_gen.generate_signals(train_feat)
        train_res = backtester.run_simulation(train_sig)

        current_equity = train_res["Strategy_Equity"].iloc[-1]

        # Selezione della finestra con ricavo maggiore
        if current_equity > best_final_equity:
            best_final_equity = current_equity
            best_window = w
            best_train_results = train_res

    print(f"\n================ RISULTATO SELEZIONE ================")
    print(f"Finestra Vincente: SMA_{best_window}")
    print(f"Capitale Finale su Train: {best_final_equity:,.2f} €")
    print(f"======================================================\n")

    # =========================================================================
    # 4. Esecuzione Out-Of-Sample sul Test Set (Solo per la Finestra Vincente)
    # =========================================================================
    print(f"--- Valutazione Test Set con Finestra SMA_{best_window} ---")
    test_feat = transformer.engineer_features(test_clean, window=best_window)
    best_signal_gen = SignalGenerator(sma_col_name=f"SMA_{best_window}")
    test_sig = best_signal_gen.generate_signals(test_feat)
    test_results = backtester.run_simulation(test_sig)

    # =========================================================================
    # 5. Valutazione Metriche (Train vs Test)
    # =========================================================================
    evaluator = ModelEvaluator(risk_free_rate=config["risk_free_rate"])

    train_metrics = evaluator.compute_metrics(best_train_results)
    evaluator.print_report(
        train_metrics,
        title=f"Metriche In-Sample (Train - SMA {best_window})"
    )

    test_metrics = evaluator.compute_metrics(test_results)
    evaluator.print_report(
        test_metrics,
        title=f"Metriche Out-Of-Sample (Test - SMA {best_window})"
    )

    # =========================================================================
    # 6. Grafici Esclusivi per la Soluzione Vincente
    # =========================================================================
    # Grafico 1: Curva del Capitale sul Train Set (In-Sample)
    backtester.plot_equity_curves(
        best_train_results,
        title=f"Soluzione Vincente: Train Set (SMA {best_window} vs Buy & Hold)"
    )

    # Grafico 2: Curva del Capitale sul Test Set (Out-Of-Sample)
    backtester.plot_equity_curves(
        test_results,
        title=f"Soluzione Vincente: Test Set (SMA {best_window} vs Buy & Hold)"
    )

    print("=== PIPELINE COMPLETATA ===")


if __name__ == "__main__":
    run_pipeline()