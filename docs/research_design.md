# Disegno della ricerca: Regime-Aware Multi-Asset Allocation

**Progetto:** ProjectAgentTrading26 | LUISS, *Data-Driven Models for Investment*\
**Stato al 28 ottobre 2026:** A1, A2 e B implementati e consolidati localmente; 123 test unitari superati dopo il refactoring; B_v2 congelata dopo la selezione sul training; validation 2017–2021 esaminata successivamente; test finale dal 2022 non eseguito per questo sistema Multi-Asset.\
**Archivio della ricerca precedente:** [`treasury_research_review.md`](treasury_research_review.md) documenta lo studio su SHY/IEF/TLT. Le sue regole e i suoi risultati non descrivono la strategia Multi-Asset attuale, basata su SPY/IEF/GLD/SHY.

## 1. Obiettivo e ipotesi di ricerca

**Domanda principale:** l'aggiunta progressiva di controllo del rischio, regimi macro-finanziari e modelli predittivi può migliorare il rendimento netto corretto per il rischio e la protezione dai drawdown rispetto a semplici portafogli multi-asset?

La ricerca procede per livelli incrementali. I confronti devono mantenere, per quanto possibile, gli stessi dati di mercato, calendari di negoziazione e costi:

- **H1 / A2:** il controllo della volatilità stimata aggiunge valore rispetto a un portafoglio Equal Weight con ribilanciamento settimanale.
- **H2 / B:** condizionare l'allocazione ai regimi economici aggiunge valore rispetto a controlli del rischio più semplici e a budget di allocazione fissi.
- **H3 / C:** il forecasting produce un beneficio economico incrementale rispetto a B, non soltanto un errore di previsione inferiore.
- **H4 / C, condizionata alla disponibilità dei dati:** notizie e sentiment con timestamp storici verificabili aggiungono valore rispetto alle previsioni prive di informazioni testuali.

Queste sono **ipotesi da verificare, non risultati dimostrati**. In particolare, nel training B_v2 non supera A2 né per Sharpe né per CAGR. Un drawdown minore, preso isolatamente, non dimostra la presenza di alpha o una performance corretta per il rischio superiore.

## 2. Dati e protocollo temporale

**Universo investibile:** SPY (azioni statunitensi), IEF (Treasury USA a 7–10 anni), GLD (oro) e SHY (Treasury USA a 1–3 anni). Gli input comprendono CSV giornalieri locali con prezzi OHLC rettificati e osservazioni FRED originali; non si applica un riempimento artificiale dei prezzi mancanti. Un nuovo download o una revisione delle rettifiche del provider può modificare i risultati e richiede un nuovo controllo dei dati.

| Segmento | Periodo | Utilizzo |
|---|---|---|
| Warm-up | Date disponibili prima del 20 maggio 2005 | Solo calcolo degli indicatori rolling |
| **Training** | **20 maggio 2005 – 31 dicembre 2016** (ultima seduta: 30 dicembre) | Costruzione dei modelli, ablation study e selezione di B_v2 |
| **Validation** | **1° gennaio 2017 – 31 dicembre 2021** | Diagnostica una tantum su B_v2 **già esaminata**; nessuna modifica successiva a B_v2 |
| **Test finale (OOS)** | **Dal 1° gennaio 2022** | Riservato al sistema integrato; nessuna valutazione B Multi-Asset effettuata |

La pipeline ordinaria `main.py` per A1/A2/B valuta **soltanto il training**, con `evaluation.evaluate_oos: false` e `validation.development_only: true`. La validation di B_v2 è stata effettuata tramite uno **script diagnostico separato e limitato esplicitamente al 2021**; non significa che la pipeline principale includa automaticamente la validation. Il periodo dal 2022 era già stato discusso durante la precedente ricerca Treasury: non va quindi presentato come totalmente sconosciuto ai ricercatori, sebbene non sia stato valutato per questa strategia Multi-Asset.

**Regola informativa:** il target settimanale viene generato dopo il Close dell'ultima seduta della settimana; l'ordine è eseguito soltanto al primo Open disponibile, mai al Close che ha prodotto il segnale. Indicatori rolling e stime di volatilità utilizzano osservazioni non successive alla decisione. L'allineamento per data delle osservazioni FRED **non certifica l'orario storico di pubblicazione né la vintage del dato**. La verifica *point-in-time* rimane aperta per B e per i futuri modelli C.

## 3. Negoziazione, contabilità e metriche

Il simulatore mantiene quantità detenute e liquidità, ribilancia soltanto alle aperture programmate e valorizza il NAV giornalmente al Close. Le ipotesi di base sono: **capitale iniziale di 100.000 USD**, **2 bps di commissioni + 1 bp di slippage** sul nozionale negoziato, liquidità non remunerata e esposizione lorda complessiva non superiore al 100%. Prezzi Open rettificati, costi costanti in bps e gestione semplificata delle garanzie e del prestito titoli per gli Short sono assunzioni del modello, non esecuzioni realistiche garantite da un broker.

Le metriche derivano dai rendimenti giornalieri netti del NAV, annualizzati su **252 sedute**, con tasso risk-free **posto a zero a fini diagnostici**. Il CAGR è calcolato come `(NAV_final / NAV_initial) ** (252 / sessions) - 1`; la volatilità annualizzata è la deviazione standard campionaria dei rendimenti giornalieri moltiplicata per `sqrt(252)`. Il maximum drawdown è misurato rispetto al precedente massimo del NAV, includendo il capitale iniziale.

**A1:** quattro benchmark: SPY Buy & Hold; SPY 60%/IEF 40% con ribilanciamento settimanale; 25% per ciascun ETF con ribilanciamento settimanale; Equal Weight Buy & Hold. Nei portafogli Buy & Hold i pesi iniziali non vengono ripristinati periodicamente.

**A2:** portafoglio Long-only, sempre interamente investito. Utilizza la covarianza campionaria annualizzata delle ultime **126 sedute**: se la volatilità prevista del portafoglio Equal Weight è al massimo **8%**, mantiene il 25% per ETF; altrimenti minimizza la distanza quadratica dall'Equal Weight imponendo pesi non negativi, somma pari a uno e volatilità stimata non superiore all'8%, mediante CVXPY/CLARABEL. Se il vincolo di volatilità è irrealizzabile, utilizza il portafoglio a varianza minima con gli stessi vincoli Long-only e di pieno investimento. L'8% riguarda la volatilità **stimata**, non quella effettivamente realizzata.

## 4. Level B: regimi deterministici e segnali

### 4.1 Identificazione dei regimi economici

Il classificatore deterministico utilizza **VIXCLS** (stress di mercato) e **T10YIE** (inflazione implicita a 10 anni, *breakeven inflation*). Ciascuna variabile è classificata come alta o bassa rispetto alla propria **mediana rolling delle 52 settimane precedenti**, ritardata per escludere l'osservazione corrente dalla soglia. Nelle etichette, il primo termine riguarda il VIX e il secondo il breakeven.

| Regime | Budget SPY | IEF | GLD | SHY |
|---|---:|---:|---:|---:|
| `low_low` | 25% | 30% | 20% | 25% |
| `low_high` | 25% | 20% | 30% | 25% |
| `high_low` | 15% | 30% | 20% | 35% |
| `high_high` | 15% | 20% | 30% | 35% |

Questi budget sono **ipotesi di allocazione fissate per la ricerca**, non pesi ottimali dimostrati né rendimenti attesi stimati. La media semplice dei quattro stati è `(20%, 25%, 25%, 30%)` per SPY/IEF/GLD/SHY.

### 4.2 B_v1: baseline Long/Short conservata

Per ciascun ETF e a ogni decisione settimanale: **Long +1** se il momentum a 21 e 63 sedute è positivo in entrambi gli orizzonti; **Short −1** se entrambi sono negativi; **Flat 0** negli altri casi. Un filtro macro aggiuntivo impedisce posizioni Long su SPY nei regimi di VIX elevato e su IEF nei regimi di breakeven elevato. I pesi obiettivo con segno sono il prodotto del budget di regime per il segnale. Una riduzione proporzionale basata sulla covarianza a 126 sedute limita la volatilità annua stimata del portafoglio all'**8%**, senza aumentare l'esposizione quando il rischio stimato è inferiore.

Il backtester include controlli sulle garanzie e sui margini Short, un costo di prestito titoli ipotizzato all'**1% annuo**, interessi sulla liquidità pari a zero e i normali costi di negoziazione. B_v1 è conservata per **riproducibilità**, non perché i suoi risultati nel training giustifichino il mantenimento delle posizioni Short.

### 4.3 B_v2: candidata Long/Flat congelata

B_v2 mantiene gli stessi quattro regimi macro e relativi budget, ma utilizza il **solo momentum**, eliminando le posizioni Short e il filtro macro direzionale aggiuntivo:

- `signal_i(t) = 1` soltanto quando i momentum a **21 e 63 sedute** sono entrambi strettamente positivi; altrimenti il segnale vale `0`.
- Peso Long/Flat iniziale: `raw_weight_i(t) = regime_budget_i(t) * signal_i(t)`.
- Il limite di volatilità stimata, calcolato su **126 sedute**, si applica **dopo** aver eliminato i pesi negativi, quindi al portafoglio Long/Flat effettivo.
- I pesi non vengono mai aumentati per sfruttare la capacità di rischio inutilizzata; l'esposizione investita complessiva non supera 1 e la parte residua rimane in **liquidità non remunerata**.

Si tratta di una strategia **Long/Flat, non market-neutral**. L'utilizzo di quattro regimi macro predefiniti non costituisce di per sé un modello di forecasting. B_v2 è stata selezionata attraverso analisi sul training e congelata **prima** di esaminare le metriche della validation 2017–2021.

## 5. Evidenze sul training e selezione della candidata

**Training osservato:** 20 maggio 2005 – 30 dicembre 2016, al netto dei costi modellati, NAV iniziale di 100.000 USD.

| Strategia | CAGR | Volatilità | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| SPY Buy & Hold | 7,77% | 19,68% | 0,479 | −55,19% |
| 60/40 settimanale | 7,13% | 10,88% | 0,688 | −32,00% |
| Equal Weight settimanale | 6,54% | 6,99% | 0,942 | −15,69% |
| Equal Weight Buy & Hold | 6,16% | 8,75% | 0,727 | −17,63% |
| **A2: Volatility Control** | **6,24%** | **6,41%** | **0,977** | **−14,28%** |
| B_v1: regime Long/Short | 0,24% | 5,55% | 0,071 | −17,55% |
| **B_v2: regime Long/Flat** | **2,64%** | **4,17%** | **0,646** | **−7,01%** |

Il training comprende **607 decisioni settimanali**. B_v1 esegue **604 ribilanciamenti** (NAV finale **102.826,95 USD**); B_v2 ne esegue **591** (NAV finale **135.371,33 USD**). L'esposizione media target di B_v2 è **54,18%**: una quota rilevante di liquidità senza interessi riduce il rischio osservato, ma limita anche il rendimento e rende necessario un confronto corretto per esposizione.

**Diagnostiche e ablation study (B6.1–B6.6), esclusivamente sul training:**

1. L'attribuzione del P&L mostra circa **+31.900 USD** dalle posizioni Long e **−20.300 USD** dalle posizioni Short, prima dei costi. Dopo circa **6.580 USD** di costi di negoziazione e **2.170 USD** di interessi sul prestito titoli, il NAV di B_v1 aumenta di soli **2.830 USD**.
2. L'eliminazione degli Short migliora notevolmente lo Sharpe nel training. Anche la rimozione del filtro macro direzionale separato migliora il risultato Long/Flat nella configurazione esaminata. Sono **confronti osservati sul training**, non verifiche indipendenti.
3. I budget macro condizionati ai regimi migliorano modestamente i risultati rispetto a budget fissi. Con il limite di volatilità effettivamente usato in B_v2, lo Sharpe è **0,6464**, contro **0,6094** per i budget fissi medi della politica e **0,5955** per il 25% fisso per ETF. Il vantaggio di **+0,037** rispetto ai budget medi è contenuto; differenze di esposizione media e costi impediscono di attribuirlo interamente, in senso causale, al timing dei regimi.
4. In B_v2 il limite di rischio è ricalcolato **dopo** la conversione dei segnali Long/Short in Long/Flat: non equivale a porre semplicemente a zero i pesi Short finali di B_v1.

**Decisione:** conservare B_v1 come baseline negativa e verificabile; selezionare e congelare B_v2 come candidata più semplice per il Level B. Né uno Sharpe positivo né un drawdown inferiore dimostrano superiorità economica incrementale rispetto ad A2. Restano necessari confronti a parità di rischio/esposizione e una quantificazione dell'incertezza statistica.

## 6. Validation già esaminata: 2017–2021

Dopo il congelamento della versione selezionata sul training, uno script diagnostico separato ha esteso B_v2 fino al **31 dicembre 2021**, senza cambiarne i parametri. Il replay ha verificato la **corrispondenza esatta dei pesi target del training** con quelli congelati, la corrispondenza del NAV del training, l'invarianza causale del prefisso fino al 2019 e l'assenza di sovrascritture degli artefatti congelati. I risultati riferiti **soltanto alla validation** sono:

| Sedute | Rendimento cumulato | CAGR | Volatilità annua | Sharpe | Maximum drawdown |
|---:|---:|---:|---:|---:|---:|
| **1.259** | **+8,919%** | **1,725%** | **3,373%** | **0,5239** | **−4,696%** |

Il valore di **108.919 USD** indica la crescita normalizzata di 100.000 USD all'inizio della validation, **non** un riavvio del NAV originario del portafoglio. Il replay è stato continuo e ha mantenuto le posizioni già detenute; nella validation risultano **256 esecuzioni**. I dati della validation sono ormai **stati esaminati**. Le regole di B_v2 non sono state modificate in seguito, ma i futuri sviluppi di clustering e ML non potranno presentare lo stesso campione come completamente inesplorato per l'intero sistema integrato.

L'audit locale è salvato in `data/results/B/heldout_diagnostic/bv2_through_2021-12-31.json`, percorso escluso da Git. Il test di validation **non** certifica le vintage FRED *point-in-time* né elimina gli eventuali bias del provider dei dati. **Non è stata eseguita alcuna valutazione Multi-Asset dal 2022 in poi.**

## 7. Riproducibilità, interfacce e limiti aperti

**Comandi principali**, dalla radice del repository:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 main.py --level ALL
python3 main.py --level B --b-version v1
python3 main.py --level B --b-version v2
python3 tests/holdout_diagnostic_bv2.py --through 2021-12-31
```

`main.py --level B` utilizza v1 come default per compatibilità con il codice precedente. Entrambe le versioni B sono state riprodotte senza sovrascrivere gli output congelati (B6.8); l'interfaccia dei regimi ha riprodotto i budget esistenti e superato test con etichette sintetiche di clustering (B6.9). La suite locale di 123 test è stata superata dopo il consolidamento A/B del 28 ottobre 2026. I test e i backtest dovranno comunque essere rieseguiti sul codice risultante dall'integrazione in `main`: la verifica del branch locale non sostituisce la verifica successiva al merge.

**Moduli principali (dopo il consolidamento del 28 ottobre):** `src/data_transformation.py` contiene anche l'API `load_dataset`; `src/macro_availability.py` realizza l'allineamento delle osservazioni macro, le feature e la classificazione deterministica dei regimi. `src/regime_allocator.py` comprende la validazione delle etichette di regime, il contratto esplicito cluster-budget, i segnali momentum, i filtri e i vincoli di volatilità. `src/regime_strategy.py` compone B_v1 e B_v2; `src/longshort_backtester.py` esegue la simulazione giornaliera con contabilità autofinanziante, anche per Long/Flat. `src/portfolio.py` contiene A1/A2 e l'ottimizzatore CVXPY; `src/backtester.py` coordina il percorso A; `src/evaluator.py` calcola le metriche e `src/time_split.py` gestisce i confini temporali. `main.py` seleziona le versioni. I vecchi moduli `multiasset_allocator.py`, `macro_features.py`, `regime_signals.py`, `regime_contract.py` e `data_loader.py` sono stati assorbiti nei moduli indicati e non devono essere importati.

**Interfaccia per il clustering:** una serie temporale di etichette deve essere allineata alle date di decisione e ai confini del periodo; è necessaria una politica di associazione **esplicita e non anticipativa** fra etichette e budget. `cluster_0` non equivale automaticamente a `low_low`. L'interfaccia è stata testata, ma **non** addestra i cluster, non ne certifica la stabilità e non valida la politica di associazione appresa.

**Questioni aperte:**

- Orari di pubblicazione e revisioni storiche FRED **non certificati point-in-time**: l'allineamento alla data di osservazione è soltanto una proxy.
- Costi di transazione, prezzi di esecuzione rettificati, finanziamento degli Short, garanzie e **rendimento della liquidità pari a zero** sono semplificazioni.
- La selezione del modello e gli ablation study dipendono dai risultati del **training**; una sola validation non dimostra significatività statistica o alpha investibile.
- A2 e B_v2 presentano esposizioni lorde molto diverse: mancano confronti a rischio/esposizione comparabili, sensibilità agli interessi sul cash e quantificazione dell'incertezza.
- Il **test finale dal 2022** va conservato per il sistema integrato, dopo aver definito e congelato le regole di clustering, forecasting ed eventuali agenti.

## 8. Prossime fasi della ricerca

1. **Stabilizzare e pubblicare l'infrastruttura comune:** aggiornare README, Research Design e Decision Log esistenti; proteggere gli output congelati; eseguire test di regressione e di invarianti; integrare il lavoro con il `main` remoto più recente senza sovrascrivere i contributi dei colleghi.
2. **Clustering:** fornire una nuova serie di regimi tramite il contratto `validate_regime_series` / `build_regime_budget_panel` in `src/regime_allocator.py`, mantenendo invariati momentum, riduzione del rischio e backtester; misurare se il nuovo classificatore aggiunge valore rispetto ai regimi deterministici e ai budget fissi.
3. **Forecasting e agenti:** definire **prima dell'addestramento** variabile target, orizzonte, benchmark e protocollo walk-forward. Valutare separatamente capacità predittiva e risultati economici netti. Le notizie storiche richiedono timestamp verificabili. Il test finale dal 2022 è destinato alla valutazione conclusiva, non allo sviluppo.
