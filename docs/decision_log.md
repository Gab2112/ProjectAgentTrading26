
# Registro delle decisioni di ricerca (Decision Log)

> **Addendum 4 novembre 2026:** il completamento delle versioni C_v1 e C_v2, i confronti A/B/C sul 2010–2016 e la sensibilità all'esposizione sono riportati in [Level C: consenso momentum–agenti AI](level_c_research.md). Le decisioni A/B qui sotto sono mantenute come cronologia della ricerca, non come stato corrente dell'intera pipeline.

**Progetto:** ProjectAgentTrading26 | *Regime-Aware Multi-Asset Allocation*\
**Aggiornato al:** 28 ottobre 2026\
**Scopo:** registrare le decisioni metodologiche e architetturali rilevanti, **non** ogni modifica al codice o ogni risultato intermedio. Metodologia completa e tabelle delle performance sono riportate in [`research_design.md`](research_design.md).

## 10–23 ottobre 2026 | Ricerca Treasury precedente (archiviata)

**Impostazione iniziale.** La prima ricerca utilizzava **SHY, IEF e TLT**, con regimi macroeconomici e dei tassi, esposizioni alla duration sia Long sia Short, segnali settimanali, ritardi prudenziali di pubblicazione dei dati e regole deterministiche di scoring. Sono stati introdotti progressivamente costi di transazione e controlli temporali; sono stati inoltre studiati forecasting Ridge, diversi indicatori dei rendimenti Treasury, allocazioni difensive e una regola momentum semplice.

**Evidenze raccolte.** La strategia originaria sulla duration e le alternative predittive testate non hanno dimostrato una superiorità robusta rispetto ai benchmark passivi. Alcune candidate erano state selezionate dopo aver osservato risultati sul training e, nel corso della ricerca Treasury, era stata esaminata anche la validation 2017–2021. Non è stata dimostrata una capacità generale di generare alpha sui Treasury.

**Decisione del 23 ottobre:** interrompere la ricerca di nuovi parametri specifici sui Treasury e conservare lo studio come archivio. Cronologia, specifiche, esperimenti e limiti sono documentati in [`treasury_research_review.md`](treasury_research_review.md). L'architettura SHY/IEF/TLT **non** coincide con la pipeline Multi-Asset attuale; i risultati precedenti non possono essere trasferiti automaticamente al nuovo modello SPY/IEF/GLD/SHY.

## 23–26 ottobre 2026 | Definizione del Level A

**Decisione: universo investibile.** Adottare **SPY, IEF, GLD e SHY**, per rappresentare azioni, Treasury intermedi, oro e Treasury brevi. Conservare quattro benchmark A1: SPY Buy & Hold; SPY/IEF 60/40 settimanale; Equal Weight settimanale; Equal Weight Buy & Hold.

**Decisione: baseline dinamica A2.** Utilizzare una **covarianza a 126 sedute**, soglia di volatilità annua **stimata all'8%**, vincoli Long-only e pieno investimento, con ottimizzazione CVXPY/CLARABEL rispetto ai pesi del 25% per ETF. Se il vincolo di volatilità è irrealizzabile, applicare il portafoglio a varianza minima. L'8% è un'ipotesi di ricerca, non un valore ottimale validato fuori campione né una garanzia sul rischio realizzato.

**Decisione: convenzione causale unificata.** Generare i target dopo l'ultimo **Close** settimanale ed eseguire al **successivo Open disponibile**. Monitorare quantità, liquidità, NAV giornaliero e nozionale negoziato. Applicare **2 bps di commissioni + 1 bp di slippage**, annualizzazione su 252 sedute e risk-free diagnostico pari a zero. Escludere le osservazioni oltre i confini del periodo. I confronti devono utilizzare regole contabili coerenti.

**Risultati e stato al 26 ottobre.** A1/A2 riprodotti sul training 2005–2016. A2 registra **CAGR 6,24%, Sharpe 0,977 e maximum drawdown −14,28%**, contro **6,54%, 0,942 e −15,69%** dell'Equal Weight settimanale. A2 riduce il rischio, ma non migliora il rendimento assoluto. Su **607** decisioni di training, il vincolo A2 ha attivato **87** ottimizzazioni. Una suite di 60 test risultava superata prima dell'ultima pulizia locale del codice; le performance A1/A2 sono state riprodotte anche dopo. La suite dovrà essere rieseguita sul codice integrato definitivo: i test precedenti non equivalgono a una verifica dell'eventuale merge.

**Decisione: separazione dei campioni.** Training **20 maggio 2005 – 31 dicembre 2016**, validation **2017–2021**, test finale **dal 2022**. La pipeline ordinaria `main.py` resta limitata al training. Ogni consultazione della validation deve essere registrata e svolta separatamente; il periodo dal 2022 resta riservato al progetto integrato.

## 27 ottobre 2026 | Level B: allocazione macro deterministica

**Decisione: costruire B_v1 come baseline verificabile.** Utilizzare VIXCLS e inflazione implicita a 10 anni (**T10YIE**) per produrre quattro regimi, mediante soglie alta/bassa basate sulla **mediana delle 52 settimane precedenti**. Associare ai regimi budget SPY/IEF/GLD/SHY, segnali momentum con segno a **21/63 sedute**, filtro macro aggiuntivo e limite di volatilità **stimata all'8%**. Il backtester Long/Short mantiene contabilità autofinanziante, liquidità, controlli di margine e costo semplificato del prestito titoli pari all'**1% annuo**.

**Risultato iniziale:** **607** decisioni nel training, **604** esecuzioni. B_v1 registra **CAGR 0,24%, Sharpe 0,071 e drawdown −17,55%**. L'attribuzione del P&L indica circa **−20.300 USD** dalle posizioni Short, contro **+31.900 USD** dalle posizioni Long, prima dei costi. Short, costi di negoziazione e interessi sul prestito titoli hanno penalizzato fortemente i risultati. B_v1 resta riproducibile, ma non è la strategia selezionata.

## 27 ottobre 2026 | Ablation study sul training e scelta di B_v2

**Decisione: non ottimizzare retroattivamente gli Short.** Gli esperimenti controllati sul training hanno eliminato le posizioni Short, valutato il filtro macro direzionale separato e confrontato i budget dinamici con allocazioni fisse. L'eliminazione degli Short ha migliorato nettamente i risultati; nella configurazione Long/Flat esaminata è risultata favorevole anche la rimozione del filtro macro direzionale.

**Decisione: selezionare B_v2 Long/Flat.** Un ETF è Long **soltanto quando i momentum a 21 e 63 sedute sono entrambi positivi**; negli altri casi è Flat. Conservare i quattro **budget condizionati al regime**, eliminando filtro macro aggiuntivo e posizioni Short. Applicare il **limite di volatilità stimata all'8%, basato su 126 sedute, dopo la costruzione dei pesi Long/Flat**. L'allocazione non investita rimane in liquidità a interesse zero. La strategia è **Long/Flat, non market-neutral**.

**Evidenze che hanno guidato la scelta:** nel training B_v2 produce **CAGR 2,64%, volatilità 4,17%, Sharpe 0,6464 e drawdown −7,01%**; NAV finale **135.371,33 USD** su 100.000 iniziali, **591** esecuzioni ed esposizione media target del **54,18%**. Nell'ultimo confronto tra budget, i regimi producono soltanto **+0,037 di Sharpe** rispetto ai budget medi fissi: la differenza non isola il contributo predittivo del timing da variazioni di esposizione e turnover. A2 mantiene CAGR e Sharpe superiori sul training.

**Decisione: congelare e conservare entrambe le versioni.** B_v2 è stata selezionata utilizzando soltanto risultati del training; B_v1 rimane baseline storica. Output separati, test di riproducibilità e controlli del contenuto dei file verificano che `main.py --level B --b-version v1|v2` non sovrascriva artefatti congelati differenti. L'esecuzione ordinaria B resta limitata al training; la CLI usa v1 come default per compatibilità.

## 27 ottobre 2026 | Consultazione della validation (soltanto B_v2)

**Azione, dopo il congelamento di B_v2:** eseguire un replay diagnostico una tantum fino al **31 dicembre 2021**, senza modificarne la parametrizzazione. Lo script ha riprodotto i **pesi e il NAV del training 2005–2016**, superato un controllo d'invarianza causale fino al 2019 e mantenuto invariati gli output congelati.

**Risultati osservati sulla validation 2017–2021:** rendimento cumulato **+8,919%**, **CAGR 1,7248%**, volatilità annualizzata **3,3727%**, **Sharpe 0,5239**, maximum drawdown **−4,6963%**, su **1.259 sedute**. La crescita normalizzata da 100.000 a **108.919 USD** riguarda il solo periodo di validation; **non** indica un azzeramento o un riavvio del NAV originario.

**Conseguenza metodologica:** la validation di B_v2 è stata **effettivamente esaminata**, anche se le regole non sono state modificate successivamente. Questo risultato non dimostra alpha, disponibilità FRED *point-in-time* o validazione indipendente dei futuri sistemi di clustering/ML. **Non sono stati esaminati risultati B Multi-Asset dal 2022 in poi.** L'audit diagnostico resta in `data/results/B/heldout_diagnostic/`, escluso da Git.

## 27 ottobre 2026 | Interfaccia d'integrazione e prossimi passi

**Decisione: separare la fonte dei regimi.** L'interfaccia, ora consolidata in `src/regime_allocator.py` (`validate_regime_series` e `build_regime_budget_panel`), verifica una serie di etichette associate a date e le trasforma in vettori di budget SPY/IEF/GLD/SHY attraverso una politica esplicita. I test hanno confermato i budget deterministici già esistenti e accettato input sintetici `cluster_0`/`cluster_1`. Un'etichetta di cluster **non ha un significato economico intrinseco**; questa interfaccia **non** implementa né valida di per sé il clustering.

**Decisione: conservare il backtester condiviso.** `src/longshort_backtester.py` rimane necessario anche per B_v2: Long/Flat è un caso particolare della simulazione autofinanziante esistente. Eliminare adesso B_v1 o il motore compatibile con gli Short introdurrebbe rischio di refactoring senza rafforzare le evidenze scientifiche.

**Prossime attività (non ancora concluse):**

1. Aggiornare README, Research Design e Decision Log esistenti; includere nel commit soltanto i moduli riproducibili, i dati raw tracciabili e i test/script diagnostici utili. Escludere backup locali, output generati e script esplorativi usa e getta.
2. Integrare le modifiche con il `main` remoto aggiornato, **senza forzare il push** e senza sovrascrivere i contributi dei colleghi. Rieseguire A1/A2/B e i test sul codice integrato prima della pubblicazione.
3. Integrare il classificatore di regimi basato sul clustering di Giovanni tramite l'interfaccia definita; successivamente valutare forecasting, ML e agenti con protocollo walk-forward e disponibilità temporale degli input verificata. Conservare il **test finale dal 2022** fino al congelamento del sistema integrato e dei criteri di valutazione.

**Limiti scientifici aperti:** le date di osservazione macro non certificano l'effettiva disponibilità storica o le vintage; dati rettificati e costi sono semplificati; B_v2 è stata scelta osservando il training; la liquidità rende lo 0%; A2 e B_v2 assumono esposizioni e rischi molto diversi. Questi limiti devono accompagnare qualsiasi affermazione sulle performance.

## 28 ottobre 2026 | Consolidamento e verifiche della pipeline A/B

**Decisione architetturale.** Ridurre la frammentazione del codice senza modificare le regole economiche: `src/multiasset_backtester.py` è stato assorbito in `src/longshort_backtester.py`; `src/multiasset_allocator.py` in `src/portfolio.py`; `src/data_loader.py` in `src/data_transformation.py`; `src/macro_features.py` in `src/macro_availability.py`; `src/regime_signals.py` e `src/regime_contract.py` in `src/regime_allocator.py`. Le interfacce pubbliche utilizzate dalla pipeline sono state riallineate. Non sono stati eliminati i componenti `src/agents/` dei collaboratori.

**Verifiche locali.** Dopo l'ultimo consolidamento, **123 test su 123** superati sul Mac. La pipeline A1/A2 ha riprodotto i risultati di training precedentemente congelati, incluso A2 con NAV **201.948,33 USD** e Sharpe **0,977**. Le esecuzioni B_v1 e B_v2 avevano riprodotto rispettivamente NAV **102.826,95 USD** e **135.371,33 USD** con output congelati verificati prima dell'ultimo intervento di consolidamento A2; la verifica finale resta da ripetere sul commit risultante e dopo il merge.

**Diagnostica validation.** Lo script `holdout_diagnostic_bv2.py` è stato spostato in `tests/holdout_diagnostic_bv2.py` come programma manuale (non incluso nella discovery `test_*.py`). È stata rimossa l'opzione che consentiva la valutazione al 2025: lo script accetta esclusivamente `--through 2021-12-31` e non deve essere usato per consultare il test finale dal 2022. La validation 2017–2021 era già stata ispezionata il 27 ottobre.

**Archivio Treasury e responsabilità dei collaboratori.** I tre file generati `data/signals/signals_statistical_{shy,ief,tlt}.csv` del precedente universo SHY/IEF/TLT sono stati esclusi dal tracciamento e le copie locali eliminate; non fanno parte della pipeline A/B. Il codice degli agenti resta invariato e può rigenerare i segnali statistici in presenza degli input appropriati. Prima di integrare gli agenti occorre concordare un contratto dati e uno split temporale coerenti con il sistema Multi-Asset.

**Prossima decisione operativa.** Aggiornare la documentazione agli import e ai percorsi correnti, poi integrare `refactor/cleanup` in `main` senza forzare la cronologia remota, rieseguire test e pipeline sul risultato del merge e soltanto dopo eliminare il branch temporaneo. La disponibilità point-in-time delle serie macro rimane non certificata.

## 4 novembre 2026 | Ridivisione delle funzioni nei moduli

**Decisione architetturale.** Ogni responsabilità ha ora un solo modulo, senza modificare regole economiche, parametri o risultati. La ridivisione è stata applicata sopra il branch `modulo-C` (algoritmo genetico, `SVM_agent.py`, correzioni metodologiche degli agenti, livello C e confronti), mantenendone intatto il comportamento.

- `src/data_loader.py` è stato ripristinato come unico punto di lettura dei file: `load_dataset`, prezzi giornalieri ETF, rendimenti Treasury del clustering e i loader degli agenti (prima metodi di `BaseAgent`, ora `load_agent_market_data`, `load_agent_macro_data`, `load_agent_dataset`, `prepare_asset_data`).
- `src/features.py` (nuovo) raccoglie le feature: `build_decision_feature_panel`, `build_training_decision_features`, `build_macro_feature_levels`, `build_cluster_features`, `build_asset_features` e `build_agent_macro_features` (estratta dal loader macro degli agenti).
- `src/utils.py` contiene l'unica definizione dell'universo `ASSETS` (prima ripetuta in cinque moduli), `load_config`, `get_project_root` e `write_frozen_outputs`; `data_acquisition.py` usa `get_tickers` invece di una copia.
- `build_weekly_decision_dates` è passato in `src/time_split.py`; `classify_macro_regimes` in `src/regime_allocator.py`; il limite di volatilità per pesi con segno in `src/portfolio.py`.
- `calculate_performance_metrics` e `compute_objective_score` sono passati da `base_agent.py` a `src/evaluator.py` (le definizioni di Sharpe restano diverse e andranno unificate).
- `_run_level_b` è passato da `main.py` a `src/regime_strategy.py` come `run_level_b`.
- `src/longshort_backtester.py` è stato rinominato `src/execution_engine.py`; `tests/holdout_diagnostic_bv2.py` è stato spostato in `scripts/`.
- Reintegrato `src/regime_clustering.py` (con test e sezione `strategies.B.clustering` del config), che era rimasto fuori da `main`.
- Livello C: `build_level_c_v2_target_weights` è passato in `src/level_c_allocator.py` e `build_level_c_v2_training` in `src/level_c_strategy.py` (stessa divisione regole/costruzione del livello B); `src/level_c_v2_strategy.py` è stato eliminato. I loader macro degli agenti mantengono i ritardi di pubblicazione da `config.yaml` introdotti il 3 novembre (FEDFUNDS e CPI), ora in `load_agent_macro_data` e `build_agent_macro_features`.

**Verifiche.** 255 test superati; output di A1, A2, B_v1, B_v2, segnali e parametri dei quattro ETF prodotti da `src/agents/train.py`, confronti A/B/C, analisi di sensibilità di C e diagnostica di esecuzione degli agenti identici file per file alla versione di `modulo-C`.

**Limiti noti (da evidenziare, senza nuovi esperimenti).** Sul training 2010-2016 l'orchestratore su SPY è Long in circa il 99% delle settimane, con Sharpe vicino al buy & hold; i risultati dell'orchestratore e del livello C sul training sono in-sample, anche se i segnali RF/SVM sono out-of-fold. Durante la revisione del 3 novembre alcuni numeri 2017-2021 sono stati calcolati solo per diagnosticare errori di codice, non per scegliere parametri: la validation va comunque considerata osservata.

## 4 novembre 2026 | Sostituzione dell'agente SVM

**Decisione.** `src/agents/SVM_agent.py` è stato sostituito con la versione walk-forward progettata il 28 ottobre (Giovanni), mantenendo i nomi `SVMAgent`, `SVMOptimizer` e `SVMWalkForwardTrainer` usati da `train.py` e dall'orchestratore. Motivo: la versione precedente era Long nel 96-99% delle settimane su SPY e SHY (soglie di probabilità 0,51-0,53 attorno a 0,5), usava solo variabili macro uguali per tutti gli ETF e la calibrazione di Platt (`probability=True`), che stima la probabilità con fold interni non temporali.

**Come funziona.** Obiettivo: prezzo più alto tra 4 settimane. Variabili retrospettive per ETF (`src/features.build_asset_features`: rendimento a 1 settimana, momentum a 4/13/26/52 settimane, volatilità a 13 settimane, distanza dal massimo a 52 settimane, momentum a 13 settimane degli altri ETF) più tre variazioni dei tassi dal modulo dei regimi. Per ogni anno dal 2010 al 2016 il modello è allenato solo sulle settimane la cui etichetta è nota prima dell'inizio dell'anno (purge di 4 settimane); gli iperparametri (lineare o RBF, C, gamma) sono scelti con `TimeSeriesSplit` (gap 4) e balanced accuracy. Il punteggio è la `decision_function` divisa per la sua deviazione standard sul training, con banda neutra 0,25. Le previsioni annuali sono i segnali out-of-fold per l'orchestratore; il modello finale è allenato su tutto il training (fino al 2016) e salvato in `src/agents/config/svm_params_<etf>.json/.joblib`. Impostazioni in `config.yaml` (`strategies.C.svm_agent`), fissate il 28 ottobre e non modificate dopo aver visto i risultati.

**Risultati sul training (OOF 2010-2016, rendimento della settimana successiva, senza costi).** Quota Long: SPY 93% (prima 96%), IEF 61% (88%), GLD 52% (93%), SHY 81% (99%). Sharpe del segnale: SPY 0,75 (0,79), IEF −0,29 (0,52), GLD 0,03 (−0,06), SHY 1,02 (0,93). Nel confronto A/B/C sul training: C_v1 Sharpe 0,585 (prima 0,630), C_v2 0,615 (0,669). La nuova SVM non è più quasi sempre Long, ma non migliora le prestazioni: il potere previsivo resta debole (balanced accuracy della CV interna tra 0,50 e 0,59). Il risultato va presentato come limite, senza ulteriori ottimizzazioni.

**Verifiche.** 266 test superati (test della SVM riscritti per la nuova interfaccia: etichetta futura, cutoff applicato prima delle etichette, purge, nessun riempimento all'indietro, segnali out-of-fold solo dagli anni di test, salvataggio e ricaricamento del modello). Tutti gli altri segnali (SMA/EMA, Z-Score, macro, Random Forest) e gli output di A e B sono identici a prima; cambiano solo SVM, orchestratore e livello C.

