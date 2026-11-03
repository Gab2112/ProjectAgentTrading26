
# Registro delle decisioni di ricerca (Decision Log)

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
