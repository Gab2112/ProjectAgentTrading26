"""Group 9 reproducible systematic-trading research interface (Simplified)."""


from src.test_oos_agent import run_comparison_assets
from src.agent_backtest import run_comparison_assets_is
from src.plots import run_full_train_test_comparison
from src.agents.train import run_all_assets



def main() -> None:
    print("\n=======================================================")
    print("  ESECUZIONE TRAIN")
    print("=======================================================\n")
    # =====================================================================
    # COMMAND-LINE ENTRY POINT
    # =====================================================================
    target_assets = ['SHY', 'SPY', 'GLD', 'IEF']
    execute_train = (input("Eseguire il train degli agenti? Y/N  \n"))


    if execute_train == 'Y':
        run_all_assets(
            assets=target_assets,
        )
    else:
        print("caricamento parametri preesistenti")


    print("\n=======================================================")
    print("  CONFRONTO ASSET MULTI-AGENTE (IN-SAMPLE & OUT-OF-SAMPLE)")
    print("=======================================================\n")
    run_comparison_assets_is()
    run_comparison_assets()

    print("\n=========================================================================================")
    print("  CONFRONTO GLOBALE TRAIN (IN-SAMPLE) VS TEST (OUT-OF-SAMPLE) & PLOT EQUITY E DRAWDOWN")
    print("=========================================================================================\n")
    run_full_train_test_comparison()



if __name__ == "__main__":
    main()