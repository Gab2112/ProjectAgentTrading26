# Registro delle decisioni (Decision Log)

Questo file registra esclusivamente le **decisioni sostanziali relative alla ricerca e all'architettura del progetto**. Le modifiche ordinarie al codice sono intenzionalmente escluse.

---

## 10 Ottobre - Direzione di ricerca preliminare: regimi macroeconomici e dei tassi d'interesse

**Decisione**

Adottare un framework basato sui regimi macroeconomici e dei tassi d'interesse statunitensi come direzione di ricerca.

**Domanda di ricerca**

> Le combinazioni di inflazione, tassi d'interesse, forma della curva dei rendimenti e volatilità dei tassi possono identificare regimi ricorrenti associati a differenti esposizioni ai titoli obbligazionari (USA)?

---

## 10 Ottobre - Universo degli strumenti negoziati: SHY / IEF / TLT

**Decisione**

- SHY: Treasury USA con scadenza residua di 1-3 anni, duration breve;
- IEF: Treasury USA con scadenza residua di 7-10 anni, duration intermedia;
- TLT: Treasury USA con scadenza residua superiore a 20 anni, duration lunga.

---

## 10 Ottobre - Frequenza settimanale della strategia

**Decisione**

Utilizzare dati grezzi di mercato a frequenza giornaliera, dati macroeconomici alla loro frequenza originale e una strategia con ribilanciamento settimanale.

---

## 10 Ottobre - Gestione dei ritardi di pubblicazione dei dati macroeconomici

**Decisione**
```
CPI        45 giorni di calendario

Core CPI   45 giorni di calendario

FEDFUNDS   35 giorni di calendario
```
**Limitazione nota**

Le serie storiche attualmente disponibili su FRED potrebbero contenere revisioni effettuate successivamente alla pubblicazione originale.

**Azione futura**

Valutare, se fattibile, l'utilizzo di dati storici **point-in-time** tramite ALFRED oppure dei calendari esatti delle pubblicazioni storiche, prima della valutazione sul Test Set definitiva.

---

## 11 Ottobre - Calcolo dell'inflazione CPI YoY sui dati mensili originali

**Decisione**

Calcolare CPI YoY e Core CPI YoY utilizzando un confronto effettivo a 12 mesi sulle serie mensili originali, applicando successivamente il ritardo di pubblicazione e l'allineamento alla frequenza settimanale.

---

## 11 Ottobre - Modello deterministico di riferimento basato su quattro dimensioni

**Decisione**
```
Inflazione      crescente / decrescente

Tassi           crescenti / decrescenti

Curva           normale / invertita

Volatilità      alta / bassa
```
---

## 11 Ottobre - Soglie per la classificazione dei regimi

**Decisione**

- Finestra temporale per il trend dell'inflazione: 13 settimane;
- Finestra temporale per il trend dei tassi: 13 settimane;
- Soglia di inversione della curva: rendimento a 10 anni meno rendimento a 2 anni = 0;
- Soglia di volatilità: mediana storica cumulativa, calcolata utilizzando almeno 52 osservazioni e ritardata di una settimana.

---

## 11 Ottobre - Punteggio deterministico della duration

**Decisione**
```
Tassi in diminuzione       +2

Tassi in aumento           -2

Inflazione in diminuzione  +1

Inflazione in aumento      -1

Curva invertita            +1

Curva normale               0
```
In presenza di elevata volatilità, il punteggio viene avvicinato allo zero di un punto.

**Stato**

Esclusivamente modello di riferimento (**baseline**). Non si sostiene che i pesi adottati siano ottimali. Da sottoporre ad analisi approfondite successivamente.

---

## 11 Ottobre - Soglie dei segnali della baseline

**Decisione**
```
score >= +2  -> regime di duration lunga

score <= -2  -> regime di duration breve

altrimenti   -> regime di duration intermedia
```
Associazione agli strumenti finanziari:
```
Duration lunga       -> TLT +1

Duration intermedia  -> IEF +1

Duration breve       -> SHY +1 e TLT -1
```
---

## 11 Ottobre - Allocazione del portafoglio

**Decisione**

Utilizzare `equal_active_signal`, con esposizione lorda massima pari a 1,0 e senza leva finanziaria aggiuntiva.

Esempio:
```
SHY +1 e TLT -1

-> SHY +0.50

-> TLT -0.50
```
---

## 11 Ottobre - Tempistica di esecuzione

**Decisione**

Applicare i pesi obiettivo con un ritardo di una settimana.
```
Segnale al tempo t

-> pesi obiettivo al tempo t

-> posizione effettivamente applicata al tempo t+1
```
---

## 11 Ottobre - Suddivisione dei dati per la validazione

**Decisione**
```
TRAIN

2005-05-27 -> 2016-12-30

VALIDATION

2017-01-06 -> 2021-12-31

OOS

Dal 2022-01-01 in avanti
```
**Regola fondamentale**

Il periodo OOS deve rimanere escluso da qualsiasi valutazione fino a quando il modello e le relative scelte progettuali non saranno stati definiti e congelati.

---

## 11 Ottobre - Mantenimento della baseline deterministica nonostante le prestazioni insufficienti

**Decisione**

Mantenere l'attuale sistema economico deterministico come baseline ufficiale dello Stage 1, nonostante abbia registrato prestazioni inferiori rispetto al benchmark obbligazionario sia nel periodo di training sia in quello di validazione.

**Motivazione**

Il suo scopo è fornire un riferimento per il confronto e garantire interpretabilità, non essere ottimizzato retroattivamente sulla base dei risultati ottenuti.

---

## 11 Ottobre - Introduzione degli agenti rinviata

**Decisione**

Mantenere gli agenti disabilitati fino a quando non saranno stati sviluppati il nucleo quantitativo e almeno un modello di previsione / Machine Learning.

**Motivazione**

I calcoli deterministici devono rimanere deterministici. Gli agenti potranno successivamente apportare valore aggiunto nelle attività di interpretazione, coordinamento, analisi critica o aggregazione delle decisioni.

---

## 12 Ottobre - Backtest limitato al periodo di sviluppo finché l'OOS rimane bloccato

**Decisione**

Quando `evaluation.evaluate_oos` è impostato su `false`, il backtest si interrompe alla data `validation.validation_end` (2021-12-31).

**Motivazione**

In precedenza, il backtest proseguiva fino all'ultima settimana disponibile e mostrava il valore finale dell'equity, rivelando indirettamente il rendimento della strategia nel periodo OOS (dal 2022 in avanti).

Il modulo di valutazione escludeva già correttamente il periodo OOS; ora anche il modulo di backtesting rispetta la stessa regola.

---

## 12 Ottobre - Introduzione dei costi di transazione

**Decisione**
```
Commissioni   2 bps

Slippage      1 bps
```
I costi vengono applicati al turnover del portafoglio, calcolato come somma delle variazioni assolute dei pesi.

**Motivazione**

Gli ETF sui Treasury statunitensi sono generalmente liquidi. I 3 punti base per unità di turnover costituiscono tuttavia un'ipotesi iniziale da verificare con analisi di sensibilità; non rappresentano una stima completa dei costi di negoziazione, finanziamento e prestito titoli per eventuali posizioni short.

L'applicazione dei pesi con un ritardo di una settimana limita il rischio di look-ahead bias, ma resta da verificare la coerenza tra momento di generazione del segnale, primo prezzo negoziabile e intervallo dei rendimenti attribuiti alla posizione. Lo slippage ipotizzato non sostituisce tale verifica.

**Effetto sulla baseline deterministica**

Rendimento annualizzato:

- Training: da 0,34% a -0,27%;
- Validation: da 0,93% a 0,54%.

---

## 13 Ottobre - Ridenominazione del benchmark ribilanciato

**Decisione**

Rinominare la serie a pesi uguali con ribilanciamento settimanale presente in `backtest_weekly.csv` in `benchmark_equal_weight_rebalanced\_*`.

**Motivazione**

Il benchmark ufficiale riportato in `evaluation_metrics.csv` è un autentico portafoglio **buy-and-hold**, senza ribilanciamento.

Utilizzare lo stesso nome per due portafogli differenti potrebbe generare incoerenze nei grafici e nelle tabelle.

---

## 13 - 15 Ottobre - Sviluppo sperimentale degli agenti

**Sviluppi effettuati**

Tra il 13 e il 15 ottobre sono stati introdotti i primi moduli agent-based:

- Agenti tecnici SMA/EMA, basati su logiche trend-following.
- Agente Z-Score, basato su logiche mean-reversion.
- Agente macroeconomico basato su yield curve e inflazione.
- Agente FinBERT per l'analisi del sentiment finanziario.
- Componenti sperimentali di ottimizzazione e walk-forward training.

Il 16 ottobre il branch di Gabriele sullo sviluppo degli agenti è stato integrato in `main`.

**Decisione**

Conservare questi componenti come base sperimentale, senza attivarli nella pipeline principale.

**Motivazione**

La presenza degli agenti nel repository non implica che siano già integrati, validati o economicamente utili. La loro introduzione deve seguire la stabilizzazione della strategia quantitativa e delle procedure di valutazione.

**Stato**

Implementazione preliminare. Integrazione operativa e valutazione comparativa ancora da completare.

---

## 15 - 16 Ottobre - Centralizzazione del caricamento dati e gestione del repository

**Decisioni**

- Introdurre `src/data_loader.py` come punto centralizzato per il caricamento e la validazione dei dataset.
- Centralizzare alcune funzionalità comuni nel modulo `src/utils.py`.
- Mantenere in Git i dati grezzi necessari alla riproducibilità.
- Escludere dal versionamento i CSV generati automaticamente dalla pipeline (`data/processed/`, `data/signals/`, `data/results/`).

**Motivazione**

Ridurre la duplicazione del codice, prevenire inconsistenze nella lettura dei dati ed evitare conflitti Git causati da output rigenerabili.

La pipeline deve continuare a essere riproducibile tramite `python3 main.py`.

**Nota**

I dataset elaborati e i risultati dovranno essere riproducibili e disponibili nel pacchetto finale di consegna, secondo le istruzioni dell'esame.

---

## 16 Ottobre - Prima integrazione dei contributi dei collaboratori

**Evento**

Effettuata una prima integrazione su `main` dei branch `GCafa` e `Gabs`.

**Decisione**

Mantenere `main` come riferimento condiviso per la pipeline economica, preservando lo sviluppo dei componenti sperimentali nei rispettivi branch.

---

## 19 Ottobre - Primo briefing di coordinamento e definizione delle priorità

**Decisione**

Consolidare l'impostazione del progetto come sistema quantitativo modulare per il trading sui Treasury statunitensi, con particolare attenzione alla previsione dei rendimenti, all'identificazione dei regimi macroeconomici e alla gestione delle esposizioni.

**Impostazione concordata**

- Conservare SHY, IEF e TLT come universo iniziale.
- Mantenere una baseline deterministica interpretabile.
- Sviluppare progressivamente modelli di regressione e forecasting.
- Valutare l'apporto dei modelli di clustering per l'identificazione dei regimi.
- Considerare gli agenti come livello complementare, senza sostituire i calcoli quantitativi deterministici.
- Utilizzare procedure comuni di backtesting, valutazione e controllo del rischio.

**Priorità metodologiche**

1. Completare la revisione della qualità dei dati e della pipeline esistente.
2. Garantire la corretta separazione temporale fra train, validation e Test Set.
3. Definire target e orizzonte predittivo prima di implementare nuovi modelli.
4. Confrontare ogni estensione con la baseline e con strategie passive.
5. Valutare rendimenti netti, Sharpe Ratio, drawdown, turnover e robustezza temporale.

**Obiettivo condiviso**

Sviluppare una strategia che possa generare rendimenti superiori ai benchmark passivi, mantenendo rigore metodologico, riproducibilità e plausibilità economica.

**Stato**

Impostazione di lavoro per le successive attività individuali e per la futura integrazione dei contributi.

---

## 19 Ottobre - Centralizzazione della gestione temporale e protezione del Test Set

**Decisione**

Centralizzare la gestione dei periodi train, validation e out-of-sample (OOS) nel modulo `src/time_split.py`.

Quando `evaluation.evaluate_oos` è impostato su `false`, il master dataset viene limitato al periodo di sviluppo prima delle successive elaborazioni della pipeline.

**Motivazione**

Evitare che le osservazioni OOS entrino nei dataset di sviluppo, nei report diagnostici e nelle procedure di selezione dei modelli.

Mantenere comunque lo storico necessario al calcolo delle feature rolling e degli indicatori macroeconomici.

**Verifiche effettuate**

- Pipeline end-to-end eseguita correttamente.
- Risultati economici della baseline rimasti invariati.
- Superati i test sintetici sui confini temporali.
- Nessuna osservazione successiva al 31 dicembre 2021 negli output di sviluppo.

**Limitazioni**

La separazione temporale non garantisce automaticamente l'assenza di data leakage.

---

## 20 Ottobre - Selezione sperimentale del Treasury 30Y e revisione delle esposizioni

**Problema di ricerca**

Valutare se la scelta dell'indicatore utilizzato per identificare i regimi dei tassi e una diversa allocazione nei regimi sfavorevoli alla duration possano migliorare le prestazioni della baseline deterministica.

**Analisi degli indicatori dei tassi**

Sono stati confrontati Treasury 2Y, 10Y e 30Y, mantenendo invariata la struttura della strategia e utilizzando esclusivamente il periodo di training di 606 settimane.

| Indicatore | Rendimento cumulato | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Treasury 2Y | −3,11% | −0,102 | −36,67% |
| Treasury 10Y | +19,20% | 0,084 | −25,21% |
| Treasury 30Y | +46,96% | 0,267 | −19,89% |

La configurazione basata sul Treasury 30Y ha mostrato le migliori prestazioni tra le tre alternative considerate, con rendimenti positivi anche nei due sottoperiodi del train precedentemente analizzati.

**Analisi delle esposizioni difensive**

Mantenendo il Treasury 30Y come indicatore dei tassi, sono state confrontate tre allocazioni nei regimi sfavorevoli alla duration lunga.

| Configurazione difensiva | Rendimento cumulato | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| A. 50% SHY / −50% TLT | +46,96% | 0,267 | −19,89% |
| B. 100% SHY | +58,20% | 0,340 | −19,89% |
| C. −100% TLT | +35,10% | 0,184 | −22,22% |

Tutti i risultati si riferiscono esclusivamente al periodo di training. I confronti sono stati riprodotti mediante uno script sperimentale senza utilizzare nuove osservazioni di validation o OOS.

**Decisione provvisoria**

Mantenere la baseline deterministica originale, basata sul Treasury 2Y, come riferimento ufficiale.

Conservare parallelamente una candidata sperimentale caratterizzata da:

- **Regime favorevole alla duration lunga:** 100% TLT.
- **Regime intermedio:** 100% IEF.
- **Regime difensivo:** 100% SHY.
- **Indicatore dei regimi dei tassi:** Treasury 30Y.

Questa candidata rappresenta una strategia di allocazione tattica della duration, con esposizioni long-only nelle tre configurazioni considerate.

**Motivazione**

La soluzione consente di esplorare il ruolo informativo dei rendimenti Treasury a lunga scadenza, distinguendo la selezione del regime dalla successiva allocazione degli ETF.

L'eliminazione della posizione short su TLT nei regimi difensivi costituisce inoltre un'ipotesi alternativa più semplice dal punto di vista operativo, che richiede comunque un confronto economico rigoroso.

**Stato e implementazione**

- Parametrizzato l'indicatore dei tassi tramite `rate_state_source`.
- Conservato `rates__treasury_2y` come valore predefinito della baseline.
- Implementato uno script dedicato al confronto delle esposizioni.
- Riprodotti i risultati sperimentali sul train.
- Modifiche salvate sul branch personale `gianluca`, senza alterare la baseline condivisa su `main`.

**Limitazioni e criticità aperte**

- La candidata è stata selezionata utilizzando risultati in-sample: esiste un rischio concreto di overfitting.
- La validation è stata parzialmente consultata nel percorso di ricerca e non può essere considerata completamente indipendente da tutte le successive decisioni.
- La configurazione selezionata rimane inferiore al benchmark equal-weight buy-and-hold nel training.
- È necessario verificare la stabilità delle prestazioni nei sottoperiodi e in differenti condizioni di mercato.
- Le esposizioni nominali non garantiscono un rischio di duration uniforme fra SHY, IEF e TLT.
- Restano da approfondire i costi effettivi di esecuzione, la robustezza dei segnali e il rischio macroeconomico.

**Regola metodologica**

La candidata Treasury 30Y non sostituisce la baseline ufficiale e non autorizza l'apertura anticipata del Test Set OOS.

---

## 21 Ottobre - Introduzione dei test automatici e verifica degli invarianti della pipeline

**Decisione**

Introdurre una suite permanente di test automatici basata su `unittest`, per verificare le proprietà temporali, economiche e operative della pipeline prima di sviluppare nuovi modelli predittivi.

**Ambiti di verifica**

Sono stati introdotti due moduli:

- `tests/test_time_split.py`: 15 test sui confini temporali, sulla separazione train/validation, sulla protezione OOS e sull'integrità dei DataFrame.
- `tests/test_pipeline_invariants.py`: 16 test sui segnali, sui pesi di portafoglio, sul ritardo di esecuzione, sui rendimenti, sui costi e sul caricamento dei dati.

I controlli utilizzano dataset sintetici e, quando necessario, file CSV temporanei, senza accedere ai rendimenti reali del Test Set.

**Motivazione**

La successiva introduzione di forecasting, clustering e agenti aumenterà la complessità del sistema.

È quindi necessario disporre di controlli riproducibili che rilevino eventuali regressioni del codice, violazioni delle regole di portafoglio o errori nella gestione temporale dei dati.

**Problema identificato e correzione**

Durante i test è stata individuata una criticità nella rappresentazione numerica dei pesi del portafoglio.

La sostituzione degli zeri con `pd.NA` nel calcolo del denominatore poteva produrre colonne di tipo `object`, anziché valori numerici coerenti.

È stata applicata una correzione minima in `src/portfolio.py`, mantenendo invariata la formula economica dell'allocazione.

**Verifiche effettuate**

- 31 test automatici superati senza errori.
- Pipeline completa eseguita correttamente in tutti i 7 passaggi.
- Valutazione OOS mantenuta disabilitata.

**Decisione metodologica**

Mantenere i test come parte permanente del processo di sviluppo e rieseguirli dopo ogni modifica sostanziale alla pipeline.

**Limitazione**

Il superamento dei test non dimostra automaticamente l'assenza di qualsiasi forma di data leakage.

Rimangono necessarie verifiche specifiche sulla disponibilità storica delle informazioni macroeconomiche, sulla costruzione delle feature e sulla coerenza tra segnali, prezzi ed esecuzione.

---

## 21 Ottobre - Adozione della validazione rigorosa nel caricamento dei dati

**Problema identificato**

Il modulo centralizzato `src/data_loader.py` prevede due modalità:

- `strict=True`: interrompe il caricamento quando rileva date invalide o duplicate.
- `strict=False`: elimina le date invalide e conserva l'ultima osservazione in presenza di duplicati.

Sul branch `gianluca`, alcune operazioni di caricamento dei dati raw utilizzavano esplicitamente `strict=False`, diversamente da `main`.

**Analisi effettuata**

È stato eseguito un audit sulla colonna delle date dei 13 file CSV raw utilizzati dal progetto.

| Controllo | Risultato |
|---|---:|
| CSV raw esaminati | 13 |
| Date invalide | 0 |
| Date duplicate | 0 |
| Date duplicate fino al 2021 | 0 |

Non sono quindi emerse anomalie nelle date che giustifichino l'utilizzo sistematico di una modalità permissiva.

**Decisione**

Adottare `strict=True` come politica ordinaria di caricamento sia per i dati raw sia per quelli processati.

Sono stati uniformati:

- `src/data_transformation.py`, nella lettura dei CSV raw.
- `src/feature_engineering.py`, nella lettura delle serie mensili originali.

Non è stato necessario modificare `src/data_loader.py`, che utilizza già `strict=True` come comportamento predefinito.

**Motivazione**

La validazione rigorosa evita che osservazioni finanziarie vengano eliminate o sostituite silenziosamente durante la preparazione dei dati.

Eventuali anomalie future dovranno essere esaminate e risolte mediante procedure esplicite, documentate e riproducibili.

**Verifica finale**

Dopo l'uniformazione del caricamento:

- Tutti i 31 test automatici sono stati nuovamente superati.
- La pipeline ha completato correttamente tutti i passaggi.
- I risultati della baseline sono rimasti invariati.
- L'OOS è rimasto bloccato.

**Politica futura**

La modalità permissiva potrà essere utilizzata soltanto in casi esplicitamente giustificati, accompagnati da controlli sulla quantità e sulla natura delle osservazioni eliminate.

---

## Prossime decisioni da prendere

Completata la stabilizzazione iniziale della pipeline, le successive decisioni riguarderanno prevalentemente la correttezza informativa delle feature, il forecasting e la gestione del rischio.

1. **Disponibilità delle informazioni macroeconomiche:** verificare i ritardi di pubblicazione, le revisioni storiche dei dati e l'eventuale utilizzo di serie point-in-time.
2. **Coerenza temporale dell'esecuzione:** verificare il rapporto fra chiusura settimanale, disponibilità del segnale, primo prezzo negoziabile e rendimento attribuito alla posizione.
3. **Variabile target del forecasting:** scegliere tra rendimento degli ETF, rendimento relativo rispetto a SHY o variazioni dei rendimenti Treasury, definendo prima la relativa interpretazione economica.
4. **Orizzonte predittivo:** considerare inizialmente un orizzonte di una settimana, coerente con la frequenza della strategia.
5. **Modello iniziale:** privilegiare un modello di regressione interpretabile e confrontarlo con benchmark predittivi semplici.
6. **Procedura di validazione:** definire training, eventuali finestre walk-forward e criteri di selezione prima dell'addestramento dei modelli.
7. **Gestione del rischio:** confrontare esposizioni nominali, volatility targeting e dimensionamento basato sulla duration.
8. **Integrazione dei componenti:** stabilire come utilizzare successivamente clustering e agenti senza compromettere la riproducibilità del sistema.

**Regole metodologiche permanenti**

- La baseline deterministica Treasury 2Y rimane il riferimento ufficiale.
- La candidata Treasury 30Y resta sperimentale e non ancora validata in modo indipendente.
- Le alternative devono essere confrontate utilizzando metriche e ipotesi economiche coerenti.
- Ogni modifica sostanziale deve essere accompagnata dall'esecuzione dei test automatici.
- Il Test Set OOS dal 2022 deve rimanere bloccato fino al congelamento definitivo della strategia, dei parametri e del protocollo di valutazione.
- Nessuna scelta progettuale deve essere giustificata retroattivamente utilizzando i risultati OOS.


## 22 Ottobre - Verifica della disponibilità temporale dei dati e del backtesting

**Decisione**

Mantenere l'attuale trattamento dei release lag macroeconomici e la metodologia di esecuzione settimanale della baseline, dopo averne verificato la coerenza nei controlli effettuati.

**Verifiche**

- Copertura delle serie FRED esaminata.
- Allineamento temporale di CPI, Core CPI e FEDFUNDS verificato.
- Rendimenti degli ETF calcolati correttamente come variazioni settimanali dei prezzi adjusted.
- Pesi applicati con un ritardo di una settimana.
- Calcolo di rendimenti lordi, turnover, costi e rendimenti netti verificato.
- Test Set OOS non utilizzato.

**Limitazioni accettate**

Le serie macroeconomiche FRED possono contenere revisioni storiche successive. Inoltre, l'applicazione dei pesi al rendimento della settimana successiva assume un'esecuzione al prezzo di chiusura utilizzato per generare il segnale.

Queste approssimazioni saranno dichiarate nel report e potranno essere oggetto di analisi di robustezza successive.

**Stato**

Audit concluso per questa fase. Nessuna modifica alle regole economiche o ai parametri della baseline.






# Treasury Regime Trading: Research Review and Phase Closure

**Project:** ProjectAgentTrading26 | LUISS, *Data-Driven Models for Investment*\
**Research phase:** 8-23 October 2026\
**Status:** Treasury research paused; change of market/universe **under consideration**, not yet agreed.\
**Version:** Research archive; not a final out-of-sample performance report.

## Executive summary

The team developed a reproducible systematic trading research pipeline for three U.S. Treasury ETFs: SHY (short duration), IEF (intermediate duration), and TLT (long duration). The research hypothesis was that macroeconomic/rates regimes, and later predictive models, could improve the allocation of duration exposure relative to passive benchmarks.

The implementation works end to end and its core temporal and portfolio invariants are covered by 31 automated tests. However, **the tested rules did not demonstrate robust economic outperformance of passive alternatives**. The original four-component deterministic duration score underperformed benchmarks. A simple Ridge model did not improve over an expanding historical-mean forecast on an internal train-only walk-forward test. A simple 30-year-yield trend rule performed well during the train diagnostic window but failed to sustain that advantage during 2017-2021 validation.

**Decision:** Pause further Treasury-specific parameter searches and modelling. Preserve the code, data pipeline, experiments, and results as a research baseline; evaluate alternative markets and investable universes jointly before changing the shared branch. No 2022+ OOS performance was consulted.

## 1. Research question and economic motivation

**Initial question:** Can changes in inflation, Treasury yields, the yield-curve shape, and interest-rate volatility identify market regimes that help rotate between short-, intermediate-, and long-duration Treasury ETF exposure?

Economic basis:

- Bond total returns depend on carry/income, changing yields, duration, convexity, and price dynamics. Higher yield sensitivity is not synonymous with higher expected risk-adjusted return.
- The Treasury yield curve can carry information about rates expectations and risk premia, but a macro indicator associated with long-horizon bond returns need not forecast **next week's** ETF relative return.
- Short-term returns may be dominated by news and shocks not captured by monthly macroeconomic data.

The project's first operational goal was to test this premise on three liquid ETF proxies, not to claim that U.S. bond markets are predictably inefficient.

## 2. Scope and data

| Item | Implementation |
| --- | --- |
| Universe | SHY, IEF, TLT |
| Trading/portfolio frequency | Weekly |
| Market data | Daily adjusted ETF OHLCV from Yahoo Finance, aggregated to weeks ending Friday |
| Macroeconomic data | FRED Treasury yields, CPI/Core CPI, FEDFUNDS, inflation breakevens |
| Feature families | ETF returns; Treasury yield levels and changes; yield-curve slopes and changes; inflation YoY; realized rates volatility |
| Baseline rate trend proxy | 2-year Treasury yield, backward 13-week change |
| Other regime components | CPI YoY trend; 10Y-2Y inversion; 13-week 10Y yield-change volatility relative to expanding historical median |
| Signal-to-position | Long TLT / long IEF / 50% long SHY + 50% short TLT |
| Original score thresholds | +2 and -2 |
| Execution approximation | Signal computed from week-t information; target weights shifted one weekly period; adjusted-close-to-adjusted-close returns |
| Turnover cost assumption | 2 bps commission + 1 bp slippage per unit of turnover |
| Temporal protection | `evaluation.evaluate_oos: false`; data and model development stop by 2021-12-31 |

The combined regime model has 16 theoretical binary combinations; 13 distinct states were observed in the available development sample. This is a **rule-based taxonomy**, not a statistically identified collection of latent economic states.

### Data partitions

- **Development dataset:** 940 weekly rows, 2004-01-02 to 2021-12-31, before excluding initial feature warm-up periods.
- **Active strategy train evaluation:** 606 weeks, 2005-05-27 to 2016-12-30.
- **Strategy validation evaluation:** 261 weeks, 2017-01-06 to 2021-12-31.
- **Final OOS period:** starts 2022-01-01, never activated in this phase.

These counts are different because the preprocessing and indicators have warm-up requirements. Some standalone diagnostics start in 2008; compare figures **only over matching sample windows**.

## 3. Architecture and technical verification

The pipeline comprises data acquisition/loading, transformation, feature engineering, regime model, signal generation, portfolio weights, backtest, and evaluation. It is runnable with `python3 main.py` when local prerequisites and raw datasets are present.

As reported after the 21 October stabilization:

- **31/31** automated unit tests passed (15 temporal tests; 16 pipeline/portfolio/execution/data tests).
- The pipeline completed its seven configured execution stages.
- Baseline end-of-development equity remained **$99,528.20** from $100,000 initial capital; total turnover **303** under the 3-bps convention.
- An audit of 13 raw CSVs found no invalid or duplicate dates; the normal loading policy is `strict=True`.
- A sample audit confirmed the configured monthly release lags (45 calendar days for CPI/Core CPI, 35 for FEDFUNDS) and expected year-over-year calculations.
- Checks confirmed `.shift(1)` in portfolio weights and internal consistency of close-to-close returns, gross/net returns, and stated costs.

**Boundaries of these checks:** Passing code tests does not prove the economic assumptions or eliminate every potential form of information leakage.

## 4. Experiments and results

### 4.1 Original deterministic baseline and passive comparators

All percentages in this table are **cumulative returns in their respective evaluation windows**, not CAGR.

| Strategy | Train 2005-2016 | Validation 2017-2021 |
| --- | ---: | ---: |
| Deterministic baseline: 2Y trend, original +/-2 thresholds | -3.11% | +2.72% |
| Exploratory threshold variant: +/-3 | +34.55% | +5.32% |
| SHY buy-and-hold | +27.94% | +7.58% |
| IEF buy-and-hold | +72.03% | +18.98% |
| TLT buy-and-hold | +93.50% | +38.18% |
| Equal-weight true buy-and-hold | +64.49% | +21.58% |

Changing thresholds from +/-2 to +/-3 reduced switching and improved observed backtest returns, but did **not** beat passive alternatives in validation. That modification was made after inspecting results and must not be treated as a clean independently tested selection.

### 4.2 Rate-state proxy: 2Y vs 10Y vs 30Y

With the rest of the original score and allocation unchanged, the **train-only** comparisons were:

| Yield used for the 13-week rate state | Cumulative return | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| 2Y | -3.11% | -0.102 | -36.68% |
| 10Y | +19.20% | 0.084 | -25.21% |
| 30Y | +46.96% | 0.267 | -19.89% |

Because TLT has greater sensitivity to long-term rates than short-term rates, the 30Y proxy has an intelligible economic rationale. However, it was also chosen after comparisons on train data: the gains do not establish independent forecasting power.

### 4.3 Defensive allocation alternatives under the 30Y rate state

**Train only**, all other deterministic rules preserved:

| Bearish duration configuration | Cumulative return | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| A: +50% SHY, -50% TLT | +46.96% | 0.267 | -19.89% |
| B: +100% SHY | +58.20% | 0.340 | -19.89% |
| C: -100% TLT | +35.10% | 0.184 | -22.22% |

These are **not equal-risk portfolios**. Gross nominal exposure does not equate duration exposure, and the short ETF leg adds lending, financing, collateral, and implementation assumptions that the backtester does not fully model. The long-only candidate B was retained as an experiment, never promoted to the official baseline.

### 4.4 Score diagnostic: next-week TLT minus SHY

Train-only diagnostic, using the original +/-2 score groups:

| Regime-score class | 2Y proxy: mean future TLT-SHY | 30Y proxy: mean future TLT-SHY |
| --- | ---: | ---: |
| Defensive | +14.27 bps | +1.19 bps |
| Intermediate | +18.68 bps | +13.83 bps |
| Long duration | -16.25 bps | +6.32 bps |

The 2Y score exhibited a **directional inconsistency** with the intended long-duration ranking over this train sample. The 30Y version was more plausible for the rate component but did not generate a clean ordering across all score classes. These are conditional means, not statistically validated causal effects or proof of an executable trading edge.

### 4.5 Ridge walk-forward forecasting: next-week TLT minus SHY

**Train only:** 619 complete feature/target rows, initial 156-row fit, 463 expanding walk-forward predictions (2008-02-15 to 2016-12-23). Predictors: 13-week 30Y yield change, 10Y-2Y spread, CPI YoY, 13-week 10Y rate volatility. Pipeline: within-fit standardization + Ridge regression (`alpha=10`, fixed ex ante). Comparator: expanding mean of previously observed target values.

| Metric | Ridge | Expanding historical mean |
| --- | ---: | ---: |
| RMSE, bps | 200.360 | 199.103 |
| MAE, bps | 154.316 | 151.931 |
| Directional accuracy | 50.54% | 54.43% |

**Relative predictive R2:** -0.01267, meaning Ridge's mean squared error was about 1.27% higher than the expanding-mean benchmark. No validation or 2022+ OOS observations were used in the Ridge experiment. This is a negative result for **this specification and horizon**, not a proof that bond returns cannot be predicted.

The experimental Ridge script `experiments/forecast_ridge_walkforward.py` was run locally; its presence in the remote `gianluca` branch was **not confirmed as of this archive draft**. Preserve its local source before any checkout, reset, clean, or branch changes.

### 4.6 Simple 30Y momentum: discrete long-only timing

Exploratory rule: if the 30Y yield has declined over the previous 13 weeks, allocate 100% to TLT; otherwise allocate 100% to SHY. Weekly signal decisions are applied to subsequent adjusted-close returns using the existing approximate timing convention; assumed costs are 3 bps per turnover unit.

| Period | Strategy cumulative return | TLT buy-and-hold | Strategy annualized volatility | Strategy max drawdown | Strategy turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train diagnostic, 2008-02-15 to 2016-12-23 (463 weeks) | +91.80% | +73.19% | 11.51% | -16.47% | 101 |
| Validation, 2017-01-06 to 2021-12-31 (261 weeks) | +7.15% | +38.18% | 10.95% | -13.76% | 94 |

Within the train diagnostic, the rule improved return, volatility, and drawdown relative to passive TLT over that **matched interval**. In validation, it underperformed passive SHY (+7.58%), IEF (+18.98%), TLT (+38.18%), and equal-weight buy-and-hold (+21.58%). Annualized volatility of equal-weight buy-and-hold in validation was 6.54%, versus 10.95% for the momentum strategy.

The chosen momentum rule was identified after other train explorations. The 2017-2021 validation period was also inspected in earlier development and is not an untouched independent test of all subsequent decisions. **No formal statistical-significance or full realistic execution analysis has been performed.**

## 5. Methodological limitations

1. **Predictive horizon mismatch.** Slow-moving macro signals may be poorly aligned with a one-week return target. This was hypothesized, not exhaustively tested.
2. **Feature construction and discretionary scores.** The score weights (+/-2 for yield direction, +/-1 for CPI trend and curve bonus) are economically motivated heuristics, not estimated out of sample. An inverted 10Y-2Y curve does not mechanically imply next-week gains in long-duration bonds.
3. **Risk asymmetry.** Equal gross dollar weights across SHY, IEF, and TLT imply materially different interest-rate/duration exposures.
4. **Trading implementation.** Friday adjusted-close-to-Friday adjusted-close returns with `shift(1)` approximate an execution price at the information cutoff. A trade placed after seeing Friday's close cannot be guaranteed at that same close. Monday-open execution has not been modelled.
5. **Short-position realism.** ETF shares can in principle be shorted subject to borrowing and broker conditions, but the model omits explicit borrow charges, financing/collateral treatment, and some practical rebalancing effects.
6. **Revised macro history.** Conservative CPI/FEDFUNDS release lags do not guarantee historical point-in-time/vintage correctness. The input is not a complete ALFRED vintage reconstruction.
7. **Selection and repeated testing.** Comparison of proxy maturities, position rules, thresholds, and post-hoc diagnostics creates model-selection risk; validation was consulted more than once. The final 2022+ holdout remains locked.
8. **Benchmark and sample alignment.** Train subwindows and validation spans differ in length; comparisons require matched dates, consistent costing, and risk-aware metrics. Strong cumulative returns alone are insufficient evidence of alpha.

## 6. Overall conclusions

- **Technical conclusion:** The pipeline and its main invariants are stable and largely reusable. Research execution and historical audit methods improved materially during the project.
- **Economic conclusion:** No evaluated Treasury strategy demonstrated a convincing and sufficiently stable net-of-cost advantage over simple passive portfolios across development periods. The apparent 30Y trend advantage was not sustained in validation.
- **Forecasting conclusion:** A small, interpretable, weekly Ridge specification failed to outperform a naive expanding-mean predictor on internal train walk-forward data. More complex algorithms are not warranted solely by this negative result.
- **Scope conclusion:** The findings **do not** imply that the bond market is untradeable or that all bond forecasting is impossible. They justify pausing this specific approach under the project's time and collaboration constraints.

## 7. Decision, preservation, and handoff

**Research status: PAUSED / ARCHIVED AS A CANDIDATE, NOT DELETED.** The team has not yet finalized a replacement market, asset universe, or project thesis. Discussion candidates include a 15-20-stock cross-sectional equity portfolio, U.S. sector ETFs, or a small liquid FX universe. Each would require its own historical universe construction, data-availability audit, trading/cost assumptions, and benchmarks. None is automatically easier to outperform.

**Preserve before pivoting:**

- `main` as the stable historical Treasury baseline; do not overwrite it with speculative changes.
- Current personal and teammate branches and existing stashes until their contents are accounted for.
- `docs/decision_log.md`, this archive, raw datasets, test modules, and code for experimental exposure comparisons.
- The **local** Ridge experiment script and its console results; the script was not found in the remote branch when this report was drafted.
- The 2022+ OOS lock. Do not run a final Treasury OOS merely to decide whether to abandon the market.

**Proposed next team meeting:** Choose the investment problem before developing signals. For each candidate market, establish (a) a clear economic hypothesis, (b) a point-in-time feasible asset universe and data sources, (c) realistic implementability, (d) simple passive and technical benchmarks, and (e) complementary roles for forecasting, clustering, and agents. Record a single agreed decision before reworking the pipeline.

## 8. Selected literature for interpretation (not replication claims)

- Campbell, J. Y., and R. J. Shiller (1991), *Yield Spreads and Interest Rate Movements: A Bird's Eye View*. Term-structure signals, expectations, and return dynamics.
- Cochrane, J. H., and M. Piazzesi (2005), *Bond Risk Premia*. Predictive information in bond forward rates, mainly on longer horizons.
- Moskowitz, T. J., Y. H. Ooi, and L. H. Pedersen (2012), *Time Series Momentum*. Cross-asset evidence for trend strategies, with caution about horizons and transfer to ETFs.
- Moreira, A., and T. Muir (2017), *Volatility-Managed Portfolios*. Motivation for distinguishing direction forecasts from risk exposure sizing.
- Bianchi, D., M. Büchner, and A. Tamoni (2021), *Bond Risk Premiums with Machine Learning*. Evidence and methodological limitations in more flexible bond-return prediction; see also the published correction.

These papers motivate testable hypotheses. They do not validate the project's specific feature weights, weekly trading rule, or observed backtest performance.

---

## 23 Ottobre 2026 - Chiusura provvisoria della fase di ricerca Treasury

**Contesto e decisione**

Concluse le verifiche tecniche ed economiche iniziali della strategia sistematica basata sugli ETF SHY, IEF e TLT, si decide di **sospendere nuovi esperimenti specifici sul mercato Treasury** e di valutare con il gruppo un eventuale cambiamento di mercato e universo investibile. Il cambiamento non è ancora una decisione collegiale definitiva: la pipeline e i dati Treasury rimangono conservati come baseline storica e archivio di ricerca.

**Evidenze raccolte**

- Baseline deterministica Treasury 2Y (score originale, soglie +/-2): rendimento cumulato -3,11% nel train e +2,72% nella validation; sottoperformance rispetto ai benchmark passivi.
- Soglie esplorative +/-3: +34,55% nel train e +5,32% nella validation, ancora inferiori ai benchmark passivi. Non sostituiscono le soglie ufficiali.
- Indicatore dei tassi 30Y e allocazione difensiva 100% SHY: risultati train migliori della baseline originale, ma configurazione selezionata in modo esplorativo e non validata come strategia finale.
- Primo forecasting Ridge (target TLT meno SHY, una settimana): 463 previsioni walk-forward solo sul train, RMSE 200,360 bps contro 199,103 bps della media storica, R2 relativo -0,01267. Nessun vantaggio predittivo dimostrato per questa specificazione.
- Semplice regola di momentum 30Y long-only TLT/SHY: +91,80% nel sottoperiodo train 2008-2016, ma soltanto +7,15% nella validation 2017-2021, contro +21,58% del benchmark equal-weight buy-and-hold; il risultato non conferma una superiorità stabile.
- Stabilizzazione tecnica: 31 test automatici superati e pipeline 7/7, con OOS dal 2022 non utilizzato.

**Limitazioni principali**

La strategia usa segnali macro a frequenza lenta per rendimenti settimanali; score e soglie non hanno evidenza indipendente di ottimalità; le esposizioni SHY/IEF/TLT non sono equiparabili per rischio di duration; l'esecuzione Close-to-Close e i costi delle posizioni short sono semplificati. I ritardi di pubblicazione FRED non ricostruiscono le vintage storiche. La validation è stata consultata durante la ricerca e non va considerata un test completamente indipendente di tutte le scelte successive.

**Conseguenze operative**

1. Non modificare la baseline storica su `main` e non aprire il Test Set OOS dal 2022.
2. Conservare i risultati, gli esperimenti, i test, i documenti e i file locali ancora non pubblicati, incluso lo script Ridge.
3. Documentare nel file `docs/treasury_research_review.md` il quadro completo delle evidenze, distinguendo campioni e limiti.
4. Valutare collegialmente un nuovo mercato (equity cross-sectional, ETF settoriali, FX) prima di cambiare codice, feature o strategia.
5. Riutilizzare gli elementi generali della pipeline soltanto dopo avere definito la nuova ipotesi economica e il relativo protocollo di backtest.

**Stato:** fase Treasury sospesa, in attesa della decisione del gruppo; nessuna conclusione generale sull'impossibilità di fare systematic trading obbligazionario.

---

## 2026-10-23 — Consolidamento Level A: Multi-Asset Allocation

### Decisione 1 — Universo investibile

Adottato un universo di quattro ETF statunitensi:

- SPY: mercato azionario.
- IEF: Treasury a duration intermedia.
- GLD: oro.
- SHY: Treasury a breve duration.

L'universo è long-only, senza leva finanziaria.

### Decisione 2 — Strategia dinamica A2

Scelto un approccio di allocazione con vincolo di volatilità:

- Riferimento: Equal Weight, 25% per ETF.
- Finestra di covarianza: 126 sedute giornaliere.
- Budget di volatilità annualizzata stimata: 8%.
- Ottimizzazione: minima distanza quadratica dall'Equal Weight.
- Vincoli: pesi non negativi e somma pari a 100%.

La soglia dell'8% è stata scelta durante la ricerca sul training,
valutandone frequenza di attivazione e fattibilità operativa.
Non è stata selezionata massimizzando lo Sharpe Ratio.

Il parametro viene mantenuto fisso nelle successive valutazioni.

### Decisione 3 — Risolutore numerico

SLSQP ha mostrato instabilità numerica in alcune settimane
caratterizzate da vincolo di volatilità attivo.

Adottati CVXPY e CLARABEL.

Risultato: 87 ottimizzazioni valide su 87 nel training.

### Decisione 4 — Convenzione di esecuzione

- Generazione dei segnali dopo l'ultima chiusura settimanale.
- Esecuzione alla prima apertura disponibile successiva.
- Portafoglio rappresentato mediante quantità detenute.
- Pesi lasciati variare naturalmente fra ribilanciamenti.
- Turnover calcolato sui valori effettivamente negoziati.
- Costi: 2 bps commissioni e 1 bp slippage.

La simulazione utilizza prezzi giornalieri rettificati.

### Decisione 5 — Risultati preliminari

Periodo del backtest: 2005-05-20 / 2016-12-30.

A2:
- CAGR: 6,24%.
- Volatilità annualizzata: 6,41%.
- Maximum Drawdown: -14,28%.
- Sharpe Ratio con rf=0: 0,9773.

Equal Weight settimanale:
- CAGR: 6,54%.
- Volatilità annualizzata: 6,99%.
- Maximum Drawdown: -15,69%.
- Sharpe Ratio con rf=0: 0,9419.

A2 presenta un contenimento del rischio e un miglioramento
modesto delle metriche corrette per il rischio, accompagnati
da un rendimento annualizzato inferiore.

Non è stata dimostrata una superiorità statisticamente robusta.

### Decisione 6 — Limiti della ricerca

Tutti i risultati sono esclusivamente in-sample.

Non sono stati utilizzati dati di validation o OOS
per valutare le performance della strategia.

I test sono stati eseguiti mediante script diagnostici;
la loro integrazione in una suite automatizzata resta da completare.

Il confronto con tassi risk-free storici, la verifica
point-in-time dei dati macroeconomici e la validazione
fuori campione appartengono a fasi successive.

### Decisione 7 — Passaggio al Level B

Il Level A viene consolidato come insieme di benchmark
statici e baseline dinamico per la ricerca successiva.

La prossima fase introdurrà regimi macro-finanziari
interpretabili, verificandone il contributo incrementale.

Non verranno modificati retroattivamente i parametri
di A2 per migliorarne i risultati storici.
