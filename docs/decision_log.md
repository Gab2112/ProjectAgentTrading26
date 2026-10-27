# Registro delle decisioni (Decision Log)

**Progetto:** ProjectAgentTrading26, *Regime-Aware Multi-Asset Allocation*\
**Aggiornato al:** 26 ottobre 2026\
**Branch locale di lavoro:** `gianluca`; nessun push della migrazione Level A risulta effettuato in questa fase.

Questo registro conserva le **decisioni sostanziali di ricerca e di architettura**, comprese quelle della precedente esperienza Treasury. Gli interventi ordinari al codice sono esclusi salvo che cambino il protocollo di ricerca, la riproducibilità o i controlli metodologici.

> **Come leggere la cronologia:** le decisioni sul vecchio universo **SHY/IEF/TLT** documentano lo stato della ricerca alla data originale e **non** descrivono la pipeline Multi-Asset attuale. L'adozione di SPY/IEF/GLD/SHY, della regola A2 e dei nuovi controlli è registrata **dal 23 ottobre in poi**. Le voci storiche restano intatte; le successive le completano o sostituiscono sul piano operativo.

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

## Prossime decisioni da prendere (promemoria storico della ricerca Treasury)

Completata la stabilizzazione iniziale della pipeline, le successive decisioni riguarderanno prevalentemente la correttezza informativa delle feature, il forecasting e la gestione del rischio.

1. **Disponibilità delle informazioni macroeconomiche:** verificare i ritardi di pubblicazione, le revisioni storiche dei dati e l'eventuale utilizzo di serie point-in-time.
2. **Coerenza temporale dell'esecuzione:** verificare il rapporto fra chiusura settimanale, disponibilità del segnale, primo prezzo negoziabile e rendimento attribuito alla posizione.
3. **Variabile target del forecasting:** scegliere tra rendimento degli ETF, rendimento relativo rispetto a SHY o variazioni dei rendimenti Treasury, definendo prima la relativa interpretazione economica.
4. **Orizzonte predittivo:** considerare inizialmente un orizzonte di una settimana, coerente con la frequenza della strategia.
5. **Modello iniziale:** privilegiare un modello di regressione interpretabile e confrontarlo con benchmark predittivi semplici.
6. **Procedura di validazione:** definire training, eventuali finestre walk-forward e criteri di selezione prima dell'addestramento dei modelli.
7. **Gestione del rischio:** confrontare esposizioni nominali, volatility targeting e dimensionamento basato sulla duration.
8. **Integrazione dei componenti:** stabilire come utilizzare successivamente clustering e agenti senza compromettere la riproducibilità del sistema.

**Regole metodologiche permanenti per la ricerca Treasury allora attiva**

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

---

## Nota archivistica — Ricerca Treasury, review separata

La *Treasury Regime Trading: Research Review and Phase Closure*, precedentemente ricopiata integralmente anche in questo file, è mantenuta **una sola volta** in [`treasury_research_review.md`](treasury_research_review.md). Questa modifica elimina una duplicazione editoriale, **non** i risultati o le scelte storiche: le decisioni del 10–23 ottobre restano nel registro, e la review completa rimane consultabile come documento autonomo.

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

---

## 26 Ottobre 2026 — Stabilizzazione tecnica locale del Level A

**Contesto**

La ricerca è stata migrata dalla precedente architettura Treasury a **Regime-Aware Multi-Asset Allocation**, fase **A**. Prima di documentare il consolidamento, sono stati confrontati gli output A1/A2 prodotti dal punto d'ingresso unico e sono stati aggiornati i test automatici. Questa voce registra gli esiti effettivamente comunicati dall'esecuzione locale, non un nuovo esperimento di performance.

### Decisione 8 — Pipeline A1/A2 unica e archivio del codice precedente

- **Punto d'ingresso ufficiale:** `main.py`, con opzioni `--level A1`, `--level A2` e `--level ALL`.
- **Universo investibile attivo:** SPY, IEF, GLD, SHY, long-only, completamente investito nei quattro ETF dopo l'allocazione; nessuna leva o posizione short.
- **A1:** quattro benchmark: SPY Buy & Hold; 60/40 SPY–IEF settimanale; Equal Weight settimanale; Equal Weight Buy & Hold.
- **A2:** controllo della volatilità stimata all'8%, su covarianza a 126 sedute annualizzata a 252, con CVXPY/CLARABEL e fallback Minimum Variance se il vincolo è non fattibile.
- **B/C:** non implementati e disabilitati. Il nome complessivo *Regime-Aware* descrive la direzione futura, non implica che A2 utilizzi già regimi macroeconomici.
- **Moduli legacy:** `src/feature_engineering.py`, `src/regime_model.py`, `src/signal_generation.py`, `src/agents/` e lo script diagnostico `run_level_a1.py` rimossi dalla cartella locale attiva, come riferito durante l'audit; la cronologia Treasury deve restare recuperabile dallo storico Git e dal documento dedicato.

**Precisazione di versione:** la rimozione locale non equivale a commit/push. I tag archivio menzionati nel README precedente (`archive/treasury-gianluca-final`, `archive/treasury-forecasting`) devono essere confermati sul clone con `git tag --list 'archive/treasury*'` prima della pubblicazione.

### Decisione 9 — Risultati riprodotti e diagnostica del vincolo A2

L'esecuzione locale del **26 ottobre**, `python3 main.py --level ALL`, ha riprodotto per la sola finestra di training **20/05/2005–30/12/2016** i seguenti valori arrotondati, già comprensivi dei costi modellati:

| Strategia | CAGR | Volatilità | Sharpe, rf=0 | Maximum Drawdown | NAV finale USD |
|---|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 7,77% | 19,68% | 0,479 | −55,19% | 238.298,26 |
| 60/40 settimanale | 7,13% | 10,88% | 0,688 | −32,00% | 222.430,62 |
| Equal Weight settimanale | 6,54% | 6,99% | 0,942 | −15,69% | 208.608,24 |
| Equal Weight Buy & Hold | 6,16% | 8,75% | 0,727 | −17,63% | 200.164,62 |
| A2 Volatility Control | 6,24% | 6,41% | 0,977 | −14,28% | 201.948,33 |

La diagnostica `data/signals/A2/volatility_control/diagnostics.csv`, letta direttamente sul Mac, riporta:

- `equal_weight`: **520** decisioni;
- `ottimizzato`: **87** decisioni;
- `fallback_minimum_variance`: **0** decisioni;
- **totale:** 607 date decisionali nel training.

Gli output del backtester registrano **606 eventi di esecuzione** per le strategie a decisione settimanale, poiché l'ultima decisione non può essere eseguita prima della validation. Le due strategie Buy & Hold hanno una sola esecuzione iniziale. Una decisione con status `ottimizzato` **non coincide necessariamente** con un nuovo trade economicamente distinto.

**Interpretazione limitata:** A2 ha ridotto volatilità e drawdown e aumentato leggermente lo Sharpe rispetto all'Equal Weight settimanale, ma il CAGR è diminuito di circa 0,30 punti percentuali. Non sono disponibili evidenze che confermino significatività statistica, robustezza fuori campione o vantaggio per unità di rischio comparabile.

### Decisione 10 — Test automatici e protezione temporale

- **Prima della pulizia legacy:** eseguita sul Mac la suite `python3 -m unittest discover -s tests -p 'test_*.py' -v`, con **60 test superati su 60** e nessun errore. Comprende **40 test** per la pipeline e gli invarianti, **20** per la gestione temporale, inclusi test numerici reali CVXPY/CLARABEL.
- **Dopo la rimozione legacy:** rieseguito con successo `python3 main.py --level ALL`, che ha riprodotto le metriche A1/A2; è stato confermato separatamente il conteggio `520/87` del file diagnostico.
- **Ancora da fare prima del commit:** rieseguire l'intera suite di 60 test **dopo** la rimozione legacy, come controllo finale sul repository effettivamente pubblicato.
- **Protezione:** costruzione dei pesi, backtest e valutazione Level A circoscritti al training. La data di decisione della settimana finale non autorizza una negoziazione nella validation. Validation 2017–2021 non valutata per A; OOS dal 2022 bloccato.

**Limite:** la separazione temporale e il successo dei test non garantiscono che un dataset esterno contenga informazioni rigorosamente *point-in-time* né attestano l'assenza di ogni forma di data leakage.

### Decisione 11 — Convenzioni di valutazione e pubblicazione dei risultati

- Annualizzazione dei rendimenti giornalieri con **252 sedute**.
- **CAGR** computato sui rendimenti giornalieri annualizzati per osservazione; il risultato corrente di **SPY è 7,77%**, che sostituisce il **7,76%** riportato in una precedente diagnostica basata su una convenzione lievemente diversa.
- Sharpe e Sortino della Fase A con **risk-free annuo pari a zero**, solo come diagnostica preliminare.
- Costi di negoziazione modellati con **2 bps commissioni e 1 bp slippage** sul nozionale lordo effettivamente negoziato; simulazione per quote e liquidità, con operazioni alla prima apertura dopo il segnale.
- I dati dei prezzi **rettificati** e il modello di costi non equivalgono a una replica esatta del mercato reale.

**Regola:** nessun confronto tra numeri ottenuti con finestre, prezzi, costi o convenzioni metriche diverse senza un'esplicita riconciliazione.

### Decisione 12 — Ordine delle attività prima del Level B

L'ordine precedentemente indicato come "passaggio al Level B" viene **specificato** come segue:

1. Concludere la stabilizzazione locale, ripetendo i test dopo la pulizia.
2. Armonizzare `README.md`, `docs/research_design.md` e il presente `docs/decision_log.md` con i risultati e il codice attuali.
3. Verificare gli eventuali disallineamenti fra `requirements.txt` e `environment-multiasset.yml`, controllare `git diff --check`/`git status` e conservare i dati raw necessari alla replica.
4. Effettuare **commit e push manuale** sul branch personale `gianluca`, senza modifiche automatiche al branch condiviso `main`.
5. Dedicare **una sessione metodologica e di comprensione personale** al modello A1/A2 e ai suoi limiti.
6. **Solo successivamente** iniziare il Level B, fissandone le nuove regole prima di esaminarne i risultati.

Rimangono non decisi per i futuri studi: metrica primaria della validation, confronto a parità di rischio, sensibilità dei costi, controllo della qualità delle serie rettificate, dati macroeconomici con disponibilità storica verificata e eventuali requisiti del corso sui segnali discreti `-1/0/+1`.

**Stato finale di questa voce:** la Fase A è **operativa e verificata sul training**; la pulizia/documentazione locale è in fase di completamento e **non è ancora stato effettuato il push della migrazione**. Nessun risultato B/C, validation A o OOS viene dichiarato disponibile.
