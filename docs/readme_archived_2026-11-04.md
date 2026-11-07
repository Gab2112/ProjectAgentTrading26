> Historical project README saved during the 7 November release. The 4 November status below is intentionally archival.

# ProjectAgentTrading26

**Allocazione multi-asset consapevole dei regimi, trading sistematico e consenso tra agenti di intelligenza artificiale**
LUISS | *Data-Driven Models for Investment*

> **Stato della ricerca al 4 novembre 2026.** Sono implementati A1/A2, B_v1/B_v2 e le versioni sperimentali C_v1/C_v2. Dopo la ridivisione dei moduli e la sostituzione della SVM (4 novembre) sono superati **266 test automatici**. Il confronto omogeneo tra nove portafogli e l'analisi di sensibilità all'esposizione riguardano **soltanto il 2010–2016**. I risultati di C sono **in-sample**, poiché l'Orchestrator è stato ottimizzato sullo stesso intervallo. La validation 2017–2021 è stata già esaminata separatamente per B_v2; una validation indipendente di C non è ancora stata completata. **Il test finale dal 2022 resta bloccato.**

## 1. Domanda di ricerca e architettura

L'aggiunta di complessità (controllo della volatilità, regimi macroeconomici e agenti AI) migliora il **rendimento netto corretto per il rischio** e la protezione dai drawdown rispetto a portafogli passivi semplici?

**Universo investibile:** SPY (azioni USA), IEF (Treasury USA intermedi), GLD (oro) e SHY (Treasury USA brevi). Le decisioni sono generate **dopo la chiusura** dell'ultima seduta settimanale ed eseguite **alla prima apertura successiva disponibile**. Il motore condiviso valorizza giornalmente NAV, quantità e cassa.

Le ipotesi di simulazione sono: capitale iniziale di **100.000 USD**, commissioni di **2 bps** e slippage di **1 bp** sul nozionale negoziato, costo annuo del prestito titoli dell'**1%** dove applicabile, cassa non remunerata e tasso privo di rischio posto a zero per le metriche diagnostiche. Sono assunzioni di ricerca, non condizioni di esecuzione garantite.

| Livello | Strategia | Implementazione |
|---|---|---|
| **A1** | SPY Buy & Hold; 60/40 SPY-IEF settimanale; Equal Weight settimanale e Buy & Hold | `src/portfolio.py` |
| **A2** | Portafoglio Long-only pienamente investito, covarianza a 126 sedute, limite di volatilità annua stimata all'8% | `src/portfolio.py` |
| **B_v1** | Regimi macro deterministici, momentum con segno a 21/63 sedute e filtro direzionale macro | `src/regime_strategy.py` |
| **B_v2** | Budget per regime e momentum Long/Flat, senza filtro macro direzionale aggiuntivo | `src/regime_strategy.py` |
| **C_v1** | Budget B_v2 e consenso Buy tra momentum e Orchestrator; Long/Flat | `src/level_c_strategy.py` |
| **C_v2** | Budget B_v2 e consenso direzionale tra momentum e Orchestrator; Long/Short/Flat | `src/level_c_strategy.py` |

In C il vincolo di rischio usa una covarianza storica a 126 sedute e un limite dell'8% di volatilità annua **stimata**, applicato **una sola volta** ai pesi iniziali. Il vincolo riduce i pesi rischiosi ma **non investe automaticamente la cassa inutilizzata**. Per C_v2, i proventi contabili delle vendite allo scoperto non equivalgono a liquidità inutilizzata.

## 2. Periodi e interpretazione

| Segmento | Date | Utilizzo |
|---|---|---|
| Preparazione degli indicatori (*warm-up*) | Osservazioni precedenti al 20 maggio 2005 | Solo storia per gli indicatori |
| Training | 20 maggio 2005 – 31 dicembre 2016 (ultima seduta: 30 dicembre) | Costruzione dei modelli e ricerca sulle strategie |
| Confronto omogeneo A/B/C | 8 gennaio 2010 – 30 dicembre 2016 | 365 date settimanali di decisione; stesso calendario giornaliero dei NAV |
| Validation | 2017–2021 | Già osservata per B_v2; **non completata indipendentemente per C** |
| Test finale | Dal 1° gennaio 2022 | **Bloccato** |

**Avvertenza metodologica:** le previsioni RF/SVM *out-of-fold* non rendono i risultati C *out-of-sample*, perché l'Orchestrator è stato ottimizzato sul campione 2010–2016. Neppure gli orari storici di disponibilità e le vintage delle serie macro FRED sono stati certificati *point-in-time*. Non è ancora stato dimostrato un vantaggio predittivo investibile.

## 3. Preparazione dell'ambiente

È previsto **Python 3.11**. Il file Conda installa l'interprete e pip, **non tutte le dipendenze Python**.

```bash
# Opzione 1: Conda
conda env create -f environment-multiasset.yml
conda activate multiasset-opt
python -m pip install -r requirements.txt

# Opzione 2: ambiente virtuale Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Test automatici
python -m unittest discover -s tests -p 'test_*.py'
```

L'agente opzionale `src/agents/FinBERT_agent.py` richiede `torch` e `transformers`, da installare e verificare separatamente per la piattaforma. **FinBERT non fa parte del confronto dei nove portafogli.**

## 4. Dati: sorgenti versionate e prodotti generati

In Git vengono conservati gli snapshot dei file originali in **`data/raw/market/`** e **`data/raw/macro/`**. È una scelta intenzionale.

- `data/processed/`: sei CSV generati, tra cui `market_daily.csv`, `weekly_decision_dates.csv`, `master_weekly.csv` e `decision_features_training.csv`.
- `data/signals/`: segnali prodotti dagli agenti. Per C servono i quattro file `signals_orchestrator_{spy,ief,gld,shy}.csv`.
- `data/results/`: backtest, NAV, costi, diagnostiche e confronti.
- `src/agents/config/`: alcuni JSON storici tracciati. Nuovi file di parametri e modelli `.joblib` devono seguire una politica di conservazione e provenienza concordata; **non aggiungere indiscriminatamente questa cartella a Git**.

Il file `data/raw/market/TLT.csv` e alcuni parametri degli agenti SHY/IEF/TLT appartengono alla ricerca precedente sui Treasury; TLT **non** fa parte dell'universo investibile corrente. Consultare [`data/README.md`](data/README.md).

## 5. Riproduzione delle analisi

Eseguire dalla radice del repository, lasciando attivi i vincoli `validation.development_only: true` ed `evaluation.evaluate_oos: false` in `config.yaml`.

```bash
# Dataset e strategie A1/A2, solo training
python main.py --level ALL

# Riproduzione dei due Level B (artefatti congelati non sovrascritti)
python main.py --level B --b-version v1
python main.py --level B --b-version v2

# SOLO se mancano i segnali: rigenerazione controllata degli agenti.
# Attenzione: produce/scrive JSON, modelli joblib e CSV; non eseguirla
# su artefatti congelati senza averne verificato la provenienza.
python -m src.agents.train ALL

# Confronti disponibili per il Level C, esclusivamente sul training
python -m src.run_training_comparison
python -m src.run_abc_training_comparison --save
python -m src.run_c_exposure_sensitivity

# Regimi con clustering (GMM), affiancati a B come confronto/regolatore di rischio
python -m src.regime_clustering

# Diagnostica manuale della validation di B_v2 (non è un test automatico)
python scripts/holdout_diagnostic_bv2.py --through 2021-12-31
```

**Limite della CLI attuale:** `python main.py --level C` **non è ancora collegato** e restituisce `NotImplementedError`. I moduli C funzionano attraverso gli script di ricerca indicati. `--level ALL` significa ancora **A1+A2**, non tutte le strategie. In `config.yaml`, `strategies.C.enabled` e `agents.enabled` restano `false` finché dispatcher, configurazione e test non saranno aggiornati insieme.

Il comando di training degli agenti stima i modelli e ottimizza l'Orchestrator sul training; un nuovo clone del repository non può eseguire C senza aver prima generato i CSV di segnale. Per riprodurre esattamente i risultati storici occorre registrare commit, versioni software, dati di input, parametri e hash dei segnali. Evitare training accidentali sopra gli output congelati.

## 6. Risultati del confronto omogeneo, 2010–2016

| Strategia | CAGR | Sharpe | Drawdown massimo |
|---|---:|---:|---:|
| SPY Buy & Hold | 12,260% | 0,826 | −18,606% |
| 60/40 SPY-IEF settimanale | 9,566% | **1,143** | −8,470% |
| Equal Weight settimanale | 4,688% | 0,819 | −7,571% |
| Equal Weight Buy & Hold | 5,031% | 0,838 | −7,678% |
| A2, controllo della volatilità | 4,653% | 0,816 | −7,571% |
| B_v1 | −1,073% | −0,194 | −17,553% |
| B_v2 | 1,366% | 0,393 | −7,011% |
| C_v1 | 1,688% | 0,585 | **−5,054%** |
| C_v2 | 1,953% | 0,615 | **−5,054%** |

I valori di C sono quelli ottenuti dopo la sostituzione dell'agente SVM (4 novembre, vedi `docs/decision_log.md`); con la SVM precedente erano C_v1 1,793% / 0,607 / −3,693% e C_v2 2,091% / 0,649 / −3,693%. Le cifre di C dipendono dai segnali rigenerati con `src.agents.train` e possono variare leggermente tra ambienti software diversi.

C_v2 migliora B_v2 **nel campione di sviluppo**, ma non supera il benchmark 60/40 per Sharpe né SPY Buy & Hold per CAGR. L'esposizione media di C è bassa: un drawdown inferiore non dimostra automaticamente la capacità predittiva degli agenti.

Nell'esperimento separato di sensibilità, **16 scenari** variano il moltiplicatore dei pesi (×1, ×1,25, ×1,5, ×2) e il limite di volatilità (8%, 12%). L'esposizione maggiore aumenta il rendimento assoluto, ma **non migliora lo Sharpe di base di C_v2**. Questo esercizio **non è uno stress test di mercato** né una validation indipendente.

Per dettagli e risultati completi: [`docs/level_c_research.md`](docs/level_c_research.md). Le decisioni storiche A/B restano in [`docs/research_design.md`](docs/research_design.md) e [`docs/decision_log.md`](docs/decision_log.md). La precedente ricerca sui Treasury è archiviata in [`docs/treasury_research_review.md`](docs/treasury_research_review.md).

## 7. Struttura del codice

Ogni responsabilità ha un solo modulo (ridivisione del 4 novembre, nessuna regola economica modificata):

- **Condivisi:** `src/utils.py` (universo unico `ASSETS`, `get_tickers`, `load_config`, ritardi di pubblicazione, scrittura dei file congelati) e `src/time_split.py` (confini training/validation/OOS e calendario delle decisioni).
- **Dati:** `src/data_acquisition.py` (download), `src/data_loader.py` (tutte le letture di file: `load_dataset`, prezzi ETF giornalieri, rendimenti Treasury, dataset degli agenti), `src/data_transformation.py` (costruzione di `data/processed/`), `src/macro_availability.py` (allineamento temporale delle osservazioni macro).
- **Feature:** `src/features.py`, unica fonte delle variabili dei modelli (feature di mercato alle date di decisione, livelli macro di B, variabili del clustering, feature degli agenti).
- **A1/A2:** `src/portfolio.py` (pesi A1, ottimizzatore CVXPY di A2 e limite di volatilità per pesi con segno usato da B e C), `src/backtester.py`, `src/evaluator.py` (tutte le metriche, anche quelle degli agenti).
- **Esecuzione:** `src/execution_engine.py` (ex `longshort_backtester.py`): simulazione giornaliera condivisa, esecuzione all'apertura successiva.
- **B:** `src/regime_allocator.py` (classificazione dei regimi, budget, segnali momentum, filtro macro, contratto per etichette esterne), `src/regime_strategy.py` (B_v1/B_v2 e `run_level_b`), `src/regime_clustering.py` (regimi GMM).
- **C:** `src/level_c_allocator.py` (regole dei pesi C_v1 e C_v2) e `src/level_c_strategy.py` (costruzione di C_v1 e C_v2); confronti sul training in `src/run_training_comparison.py`, `src/run_abc_training_comparison.py`, `src/run_c_exposure_sensitivity.py`.
- **Agenti:** `src/agents/` contiene solo la logica dei modelli: `BaseAgent`, SMA/EMA, Z-Score, macro, Random Forest, SVM (`SVM_agent.py`: walk-forward annuale con purge, impostazioni in `strategies.C.svm_agent`), FinBERT, orchestratore con algoritmo genetico, `train.py` (segnali RF/SVM out-of-fold) e `agent_execution_bridge.py`. Dati, feature e metriche arrivano dai moduli sopra; `src/plots.py` produce i grafici diagnostici dei segnali.
- **Script e test:** `scripts/holdout_diagnostic_bv2.py` (diagnostica manuale della validation di B_v2); `tests/test_*.py` (test di regressione e invarianti).

## 8. Collaborazione e integrazione

- Il branch `modulo-C` comprende i commit di audit degli agenti ereditati da `gianluca`, le strategie C e i runner comparativi.
- Il clustering proposto da Giovanni (`src/regime_clustering.py`) è una **possibile variante separata di gestione del rischio per B_v2**, non una sostituzione della baseline congelata. La regola cluster → budget va fissata prima di osservare i risultati.
- Il refactoring dei moduli in `src/` (sezione 7) e la sostituzione dell'agente SVM sono stati integrati dal branch `refactor-moduli` dopo `modulo-C`, verificando che A, B, gli altri agenti e i confronti producano file identici.
- La validation C 2017–2021 richiede un protocollo specifico. Il test finale 2022+ resta bloccato.

Per le verifiche ancora aperte: [`docs/pre_pr_audit_2026-11-04.md`](docs/pre_pr_audit_2026-11-04.md).
