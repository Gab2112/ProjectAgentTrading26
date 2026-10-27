# ProjectAgentTrading26

**Regime-Aware Multi-Asset Allocation: Dynamic Risk-Adjusted Performance and Downside Protection**

Progetto di ricerca in trading sistematico e finanza quantitativa per il corso **Data-Driven Models for Investment**, LUISS Guido Carli (docenti: Antonio Simeone e Villy De Luca).

> **Stato al 26 ottobre 2026:** **Level A implementato e riprodotto localmente sul solo training**. A1 comprende quattro benchmark; A2 introduce un vincolo di volatilità stimata. La pipeline `main.py` esegue entrambi. **Validation 2017–2021 non valutata; OOS dal 2022 bloccato.** I livelli B (regimi) e C (ML/agenti) sono progettati ma **non implementati**. Nessun risultato riportato qui prova superiorità fuori campione.

Il repository documenta anche la precedente ricerca Treasury, conservata nello storico Git e in [`docs/treasury_research_review.md`](docs/treasury_research_review.md). Il riferimento metodologico aggiornato è [`docs/research_design.md`](docs/research_design.md); la cronologia delle decisioni è in [`docs/decision_log.md`](docs/decision_log.md).

## 1. Domanda di ricerca

**Un sistema di allocazione multi-asset può migliorare il rendimento corretto per il rischio e la protezione dai drawdown rispetto a portafogli più semplici, quando vengono aggiunti progressivamente controllo del rischio, informazione macro-finanziaria e previsioni?**

La ricerca è volutamente incrementale:

| Livello | Contenuto | Stato |
|---|---|---|
| **A1** | Benchmark con pesi prestabiliti | Implementato, training riprodotto |
| **A2** | Equal Weight modificato quando il rischio stimato supera l'8% annualizzato | Implementato, training riprodotto |
| **B** | Allocazione condizionata a inflazione e stress finanziario; clustering opzionale | Non implementato |
| **C** | Forecasting ML e, se i dati lo consentono, agenti/notizie con timestamp verificabili | Non implementato |

L'obiettivo non è ottenere la curva più redditizia sul campione già osservato. Ogni livello dovrà dimostrare il proprio **valore incrementale** con dati, convenzioni e costi confrontabili.

## 2. Universo investibile e Level A

| ETF | Esposizione | Ruolo nel portafoglio |
|---|---|---|
| **SPY** | Azioni statunitensi large cap | Crescita e rischio azionario |
| **IEF** | Treasury USA a duration intermedia | Esposizione ai tassi e diversificazione potenziale |
| **GLD** | Oro | Diversificazione potenziale tra asset class |
| **SHY** | Treasury USA a duration breve | Componente relativamente difensiva |

**Regole operative A:** long-only, senza leva o vendite allo scoperto; pesi target non negativi e somma pari a 1. Il cash serve alla contabilità e alla fase iniziale, **non** è un asset di allocazione strategica. Nessun ETF è garantito come copertura o privo di rischio.

### A1: quattro benchmark

| Identificatore negli output | Pesi target | Ribilanciamento |
|---|---|---|
| `spy_buy_and_hold` | SPY 100% | Solo allocazione iniziale |
| `sixty_forty` | SPY 60%, IEF 40% | Settimanale |
| `equal_weight` | 25% ciascuno dei quattro ETF | Settimanale |
| `equal_weight_buy_and_hold` | 25% ciascuno dei quattro ETF | Solo allocazione iniziale |

**Importante:** un portafoglio *buy-and-hold* non viene riportato ai pesi iniziali dopo il primo acquisto. I portafogli settimanali, invece, ricevono un nuovo target a ogni decisione. Non vanno confusi i pesi **target** con i pesi **effettivi** dopo il movimento dei prezzi.

### A2: Volatility Control

A2 parte dal benchmark `equal_weight` (25% per ETF):

1. Per ogni decisione settimanale calcola i rendimenti giornalieri storicamente disponibili degli ultimi **126 giorni di negoziazione**.
2. Stima la covarianza campionaria e la annualizza moltiplicandola per **252**.
3. Se la volatilità **stimata** dell'Equal Weight è al massimo **8%**, mantiene i quattro pesi al 25%.
4. Altrimenti minimizza la distanza quadratica dall'Equal Weight con vincoli long-only, somma dei pesi pari a 1 e volatilità stimata al massimo dell'8%, tramite **CVXPY + CLARABEL**.
5. Se il limite è matematicamente non fattibile, la regola prevista è il portafoglio a **varianza minima**, con gli stessi vincoli di investimento.

L'8% è un **budget di rischio previsto dalla stima**, non un limite alla volatilità effettivamente realizzata. Non è un rendimento obiettivo, una previsione del mercato o una garanzia sul drawdown. La soglia è stata definita nel percorso di ricerca sul training; non va trattata come un parametro selezionato indipendentemente dai dati.

## 3. Dati, calendario e simulazione

- **Input principali:** CSV locali in `data/raw/market/` per SPY, IEF, GLD e SHY, con prezzi giornalieri Open/Close rettificati in modo coerente.
- **Preparazione:** allineamento delle sedute effettivamente comuni ai quattro ETF, controlli su prezzi e date, senza riempire artificiosamente le quotazioni mancanti.
- **Finestra preliminare di storico:** dal 2004 per il warm-up delle misure rolling; non costituisce performance valutata.
- **Training:** dal **20 maggio 2005** al **31 dicembre 2016**; l'ultima seduta di trading è il **30 dicembre 2016**.
- **Validation:** dal **1° gennaio 2017** al **31 dicembre 2021**, **non valutata nell'attuale Level A**.
- **OOS finale:** dal **1° gennaio 2022**, bloccato durante lo sviluppo.
- **Decisione:** dopo la chiusura dell'ultima seduta negoziabile della settimana, normalmente venerdì.
- **Esecuzione:** all'**Open rettificato della seduta successiva**, senza possibilità di negoziare retroattivamente al Close che genera il segnale.
- **Contabilità:** quantità di quote, liquidità e NAV giornaliero; nessun ribilanciamento giornaliero implicito.
- **Costi modellati:** 2 bps di commissioni + 1 bp di slippage sul **nozionale lordo negoziato** (somma degli acquisti e delle vendite in valore assoluto).
- **Capitale iniziale:** 100.000 USD. Il risk-free è impostato a **0** per le metriche diagnostiche iniziali.

Il pannello processato può contenere osservazioni del periodo di sviluppo necessarie alla preparazione; **pesi, backtest e metriche dell'attuale A sono circoscritti al training**. L'ultima decisione settimanale del training non viene eseguita se la prima apertura successiva ricadrebbe nella validation.

L'uso di Open/Close **rettificati** è una proxy pratica del total return e dell'esecuzione, non una ricostruzione perfetta di prezzi realmente negoziabili, dividendi, spread o qualità di esecuzione storica.

## 4. Installazione ed esecuzione locale

Ambiente verificato sul Mac di sviluppo: **Python 3.11**, `pandas==3.0.6`, `numpy==2.4.6`, `scipy==1.17.1`, `cvxpy==1.9.3`, `clarabel==0.11.1`; le altre dipendenze sono in `requirements.txt`.

Dalla radice del repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip check
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 main.py --level ALL
```

Esecuzioni alternative:

```bash
python3 main.py --level A1
python3 main.py --level A2
python3 main.py                 # usa pipeline.active_level del config.yaml
```

**Per default non viene eseguito alcun download.** Impostare `pipeline.run_data_acquisition: true` soltanto se si intende aggiornare deliberatamente i CSV grezzi: dati diversi possono cambiare i risultati. `main.py --level B` e `--level C` falliscono esplicitamente perché tali livelli non esistono ancora.

La configurazione è centralizzata in `config.yaml`; la versione Python/pip effettivamente verificata è quella di `requirements.txt`. Il file alternativo `environment-multiasset.yml` va allineato prima del push, se si vuole mantenerlo come seconda modalità di installazione.

## 5. Risultati Level A: training soltanto

Backtest giornaliero sul periodo **20/05/2005–30/12/2016**, costi inclusi, Sharpe con `rf = 0` e metriche annualizzate con **252 sedute**. Valori arrotondati come nell'esecuzione locale del 26/10/2026.

| Strategia | CAGR | Volatilità | Sharpe | Max drawdown | NAV finale (USD) |
|---|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 7,77% | 19,68% | 0,479 | −55,19% | 238.298,26 |
| 60/40 settimanale | 7,13% | 10,88% | 0,688 | −32,00% | 222.430,62 |
| Equal Weight settimanale | 6,54% | 6,99% | 0,942 | −15,69% | 208.608,24 |
| Equal Weight Buy & Hold | 6,16% | 8,75% | 0,727 | −17,63% | 200.164,62 |
| **A2 Volatility Control** | **6,24%** | **6,41%** | **0,977** | **−14,28%** | **201.948,33** |

Rispetto all'Equal Weight settimanale, A2 registra **−0,30 punti percentuali di CAGR**, **−0,58 punti di volatilità**, **+0,035 circa di Sharpe** e **+1,41 punti di drawdown massimo** (meno negativo). Non si è verificato un miglioramento del rendimento assoluto né una superiorità statisticamente dimostrata.

**Diagnostica A2 verificata localmente:** 607 date decisionali nel training; `equal_weight = 520`, `ottimizzato = 87`, `fallback_minimum_variance = 0`. Le strategie settimanali hanno **606 esecuzioni**, perché l'ultima decisione non può essere negoziata nel training; i buy-and-hold ne hanno una. Le 87 ottimizzazioni non equivalgono automaticamente a 87 operazioni: riguardano il calcolo del target alla data decisionale.

## 6. Struttura e output

```text
ProjectAgentTrading26/
├── config.yaml
├── main.py                        # punto d'ingresso A1/A2
├── requirements.txt
├── environment-multiasset.yml     # alternativa da allineare
├── README.md
├── docs/
│   ├── research_design.md
│   ├── decision_log.md
│   └── treasury_research_review.md
├── data/
│   ├── raw/market/                 # CSV di input versionati
│   ├── processed/                  # output generati, esclusi da Git
│   ├── signals/                    # output generati, esclusi da Git
│   └── results/                    # output generati, esclusi da Git
├── src/
│   ├── data_acquisition.py
│   ├── data_loader.py
│   ├── data_transformation.py
│   ├── time_split.py
│   ├── utils.py
│   ├── portfolio.py
│   ├── multiasset_allocator.py
│   ├── multiasset_backtester.py
│   ├── backtester.py
│   └── evaluator.py
└── tests/
    ├── test_pipeline_invariants.py
    └── test_time_split.py
```

File principali prodotti da `python3 main.py --level ALL`:

| Contenuto | Percorso |
|---|---|
| Prezzi giornalieri allineati | `data/processed/market_daily.csv` |
| Calendario decisionale | `data/processed/weekly_decision_dates.csv` |
| Pesi target | `data/signals/A1/<strategia>/target_weights.csv` e `data/signals/A2/volatility_control/target_weights.csv` |
| Stato dell'ottimizzatore | `data/signals/A2/volatility_control/diagnostics.csv` |
| Simulazione giornaliera | `data/results/<livello>/<strategia>/backtest_daily.csv` |
| Metriche e curve per livello | `data/results/A1/evaluation_metrics.csv`, `data/results/A2/evaluation_metrics.csv` (e rispettivi `evaluation_curves.csv`) |
| Confronto A1/A2 | `data/results/multiasset_level_a_comparison.csv` |

`data/processed/`, `data/signals/` e `data/results/` sono **rigenerabili e ignorati da Git**. I dati raw necessari alla replica devono invece essere mantenuti e tracciabili. Gli esperimenti e i vecchi moduli Treasury non partecipano alla pipeline attiva.

## 7. Verifiche e limiti

- **60/60 test superati** sul Mac prima della rimozione del codice legacy: 40 test su pipeline e invarianti, 20 test sui confini temporali. Inclusi test numerici con CVXPY/CLARABEL.
- **`main.py --level ALL` rieseguito con successo dopo la rimozione dei moduli legacy**, senza differenze nelle metriche riportate.
- Il conteggio diagnostico **520/87** è stato confermato leggendo `diagnostics.csv`.
- **Rimane da ripetere la suite da 60 test dopo la pulizia** come ultimo controllo prima del commit.

**Limiti metodologici:** risultati solo in-sample; soglia 8% definita durante esplorazioni sul training; dati di mercato rettificati e costi semplificati; assenza di confronto con serie storiche risk-free; nessun test di significatività, analisi a parità di rischio o robustezza fuori campione; dati macro *point-in-time* non ancora ricostruiti per B. Una suite verde verifica gli invarianti implementati, non l'esistenza di un *trading edge*.

## 8. Stato del lavoro e prossimo ordine operativo

1. **Finitura locale:** ripetere i 60 test dopo la rimozione legacy; controllare i file versionati e allineare `environment-multiasset.yml` a `requirements.txt` oppure rimuoverne l'ambiguità; correggere eventuali riferimenti testuali obsoleti.
2. **Revisione della documentazione:** assicurare coerenza tra README, Research Design, Decision Log, codice e output.
3. **Git:** esaminare `git diff --check` e `git status --short`, verificare i tag storici, poi effettuare **commit/push manuale sul branch `gianluca`**, senza toccare il branch condiviso `main`.
4. **Punto metodologico personale:** comprendere A1/A2, matematica, implementazione, assunzioni e interpretazione dei risultati.
5. **Solo dopo:** avviare l'implementazione del Level B, con un protocollo di confronto e validazione deciso prima di consultare nuovi risultati.

La precedente ricerca Treasury è descritta in [`docs/treasury_research_review.md`](docs/treasury_research_review.md). I tag di archivio menzionati nella documentazione storica (`archive/treasury-gianluca-final` e `archive/treasury-forecasting`) devono essere verificati con `git tag --list 'archive/treasury*'` sul clone locale prima di considerarli disponibili.
