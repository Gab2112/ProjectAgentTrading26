# ProjectAgentTrading26

**Progetto universitario di trading sistematico e finanza quantitativa**

**Corso:** Data-Driven Models for Investment, LUISS Guido Carli

**Docenti:** Antonio Simeone e Villy De Luca

> **Stato attuale: Fase 0 - progettazione della nuova ricerca Multi-Asset.**
> La strategia descritta qui non è ancora implementata. Il codice Python presente nel repository esegue tuttora la precedente pipeline sui Treasury statunitensi. Non interpretare i suoi output come risultati del nuovo progetto.

## 1. Panoramica

Il progetto studia se un'**allocazione dinamica tra azioni, Treasury statunitensi e oro** possa migliorare la performance corretta per il rischio e la protezione dalle perdite rispetto a portafogli diversificati statici.

La nuova direzione di ricerca è denominata:

**Regime-Aware Multi-Asset Allocation: Dynamic Risk-Adjusted Performance and Downside Protection**

L'obiettivo tecnico è sviluppare, con complessità incrementale, tre livelli della medesima strategia: una baseline trasparente (**A**), un'allocazione condizionata ai regimi macro-finanziari (**B**) e una versione con **previsioni ML e agenti AI** (**C**).

Il sistema deve essere valutabile e riproducibile. Non si assume che una strategia dinamica o un modello più complesso batta necessariamente un benchmark passivo.

**Documento metodologico di riferimento:** [`docs/research_design.md`](docs/research_design.md). Il presente README fornisce una panoramica del progetto e dello stato del repository; il Research Design contiene ipotesi, vincoli e questioni ancora aperte.

## 2. Domande di ricerca e obiettivi

**Domanda principale:** un'allocazione dinamica basata su stati macro-finanziari identificabili con informazioni disponibili al momento della decisione può migliorare il rendimento corretto per il rischio di un portafoglio SPY-IEF-GLD-SHY rispetto a benchmark statici e a una semplice regola dinamica, senza compromettere la protezione dalle perdite?

**Domanda incrementale:** previsioni di machine learning e segnali strutturati derivati da notizie aggiungono valore economico misurabile rispetto all'allocazione già condizionata ai regimi?

- **Obiettivo primario:** migliorare la performance corretta per il rischio, non massimizzare soltanto il rendimento assoluto.
- **Obiettivo secondario:** contenere il *downside*, studiando drawdown, volatilità al ribasso, fasi di stress e partecipazione ai recuperi.
- **Principio di valutazione:** una strategia che detiene quasi sempre l'asset meno rischioso non è automaticamente superiore; quando opportuno, i confronti devono tenere conto di budget di rischio comparabili.

## 3. Universo investibile proposto

| ETF | Esposizione | Ruolo economico ipotizzato |
| --- | --- | --- |
| **SPY** | Azioni statunitensi large cap | Crescita ed esposizione al rischio azionario |
| **IEF** | Treasury USA a scadenza intermedia | Duration e possibile diversificazione in recessioni disinflazionistiche |
| **GLD** | Oro | Diversificazione in alcune condizioni di stress o shock monetari |
| **SHY** | Treasury USA a breve durata | Componente difensiva con rischio di duration ridotto |

**Ipotesi operative iniziali:** posizioni long-only, nessuna leva, nessuno short selling, capitale investito nei quattro ETF salvo decisione esplicita sulla liquidità, ribilanciamento mensile, limiti di concentrazione e costi di transazione.

**Attenzione:** obbligazioni e oro non sono coperture universali. IEF può perdere valore quando i rendimenti salgono; GLD può scendere anche durante una crisi; SHY non equivale a una posizione in contanti priva di rischio. Le caratteristiche difensive degli asset devono essere verificate sui dati.

L'universo e le regole operative sono **proposte di Fase 0**, ancora soggette ad audit e approvazione metodologica.

## 4. Architettura sperimentale: A -> B -> C

I tre livelli non sono strategie indipendenti: ogni livello deve permettere di misurare il valore incrementale dei moduli aggiunti.

### Livello A - Benchmark e baseline semplice

- **A1, benchmark statici:** SPY buy-and-hold; portafoglio 60% SPY / 40% IEF; portafoglio equal-weight sui quattro ETF.
- **A2, baseline dinamica:** una regola interpretabile da scegliere *prima* della valutazione, candidata tra controllo del trend azionario e gestione della volatilità.
- **Domanda:** una semplice gestione dinamica migliora un'allocazione diversificata statica?

### Livello B - Allocazione basata sui regimi

- **B1:** classificazione macro-finanziaria interpretabile, inizialmente fondata su stress finanziario e pressioni inflazionistiche.
- **B2, opzionale:** clustering statistico, per esempio K-means o GMM, se motivato e valutabile senza informazioni future.
- **Domanda:** conoscere il regime aggiunge valore rispetto alla baseline A2, al netto dei costi?

### Livello C - Forecasting ML e agenti AI

- **Previsione:** target primario da definire; candidati sono il rischio azionario nel mese successivo o la probabilità di una perdita significativa definita ex ante. Modelli semplici e benchmark predittivi vengono prima dei modelli complessi.
- **Modulo regime:** classificazioni o probabilità di regime costruite con informazioni disponibili alla data della decisione.
- **Agente notizie/sentiment:** segnali ottenuti da notizie storiche con timestamp verificabile. Senza un archivio adeguato, soltanto una dimostrazione contemporanea, senza attribuirle performance di backtest.
- **Controllo del rischio:** vincoli quantitativi deterministici su pesi, concentrazione, rischio e turnover; nessun agente può superarli.
- **Allocatore:** conversione esplicita e riproducibile degli output in pesi di portafoglio.

**Confronto incrementale previsto:** portafogli statici -> A2 -> B -> C senza notizie -> C con notizie, se i dati lo consentono.

## 5. Dati e metodo di valutazione

**Fonti candidate:** prezzi giornalieri rettificati degli ETF, VIX, serie Treasury, CPI e altre misure di stress finanziario. Yahoo Finance e FRED sono fonti possibili, da verificare prima dell'utilizzo.

Prima dei backtest devono essere controllati:

- disponibilità storica e negoziabilità effettiva degli ETF;
- trattamento di distribuzioni, split e rendimenti *total return*;
- calendari di mercato, dati mancanti, duplicati e anomalie;
- date di pubblicazione e revisioni delle serie macroeconomiche (*point-in-time*);
- assenza di informazioni future in feature, preprocessing, target e classificazioni di regime;
- riproducibilità dei risultati a partire dai dati originali.

**Frequenza proposta:** dati giornalieri per indicatori e monitoraggio; decisioni di allocazione mensili.

### Separazione temporale

| Campione | Periodo candidato | Stato |
| --- | --- | --- |
| Training | 2005-2016 | Da confermare dopo l'audit dei dati |
| Validation | 2017-2021 | Da confermare e utilizzare con disciplina |
| Test finale | Dal 2022 | **Bloccato durante lo sviluppo** |

Questi intervalli non sono ancora congelati per la nuova ricerca. Il periodo dal 2022 è già stato osservato in precedenti analisi Treasury: pertanto non va presentato come storicamente ignoto ai ricercatori. Le eventuali conoscenze pregresse pertinenti dovranno essere dichiarate e non utilizzate per ottimizzare il nuovo modello.

**Esecuzione:** i segnali devono usare dati disponibili a un timestamp esplicito ed essere negoziati soltanto nella prima finestra realistica successiva. Il semplice ritardo di una riga nei pesi non garantisce, da solo, un backtest eseguibile.

**Costi:** commissioni, slippage/spread e turnover devono essere inclusi. Il protocollo e i valori operativi definitivi restano da fissare.

**Metriche:** rendimento cumulato, CAGR, volatilità annualizzata, Sharpe, Sortino, massimo drawdown, turnover, costi, allocazioni e risultati per sottoperiodo. Le metriche previsionali ML devono rimanere distinte dalle metriche economiche.

I requisiti didattici sui segnali discreti `-1`, `0`, `+1` devono essere verificati: nell'architettura iniziale long-only sono previsti soltanto segnali `0` e `+1`, distinti dai pesi effettivi del portafoglio.

## 6. Stato effettivo del codice

| Componente | Stato corrente |
| --- | --- |
| Research Design Multi-Asset | Documento di Fase 0 presente |
| Dati SPY e GLD | Non ancora acquisiti e verificati nella nuova pipeline |
| Pipeline `main.py` | **Ancora basata sulla ricerca Treasury precedente** |
| Configurazione `config.yaml` | **Ancora specifica per Treasury SHY/IEF/TLT** |
| Backtesting e valutazione | Moduli da verificare e adattare |
| Test automatici | Test della pipeline precedente, non validazione Multi-Asset |
| Clustering, forecasting e agenti Multi-Asset | Non ancora integrati nella nuova strategia |

**Avvertenza operativa:** eseguire `python3 main.py` allo stato attuale significa eseguire la **vecchia strategia Treasury**, non il nuovo sistema multi-asset. Non utilizzarne i risultati per valutare il nuovo progetto.

### Struttura essenziale del repository

```text
ProjectAgentTrading26/
|-- README.md
|-- config.yaml
|-- main.py
|-- requirements.txt
|-- docs/
|   |-- research_design.md
|   |-- decision_log.md
|   `-- treasury_research_review.md
|-- data/
|   `-- raw/
|-- src/
|   |-- data_loader.py
|   |-- time_split.py
|   |-- data_acquisition.py
|   |-- data_transformation.py
|   |-- feature_engineering.py
|   |-- regime_model.py
|   |-- signal_generation.py
|   |-- portfolio.py
|   |-- backtester.py
|   |-- evaluator.py
|   `-- agents/
`-- tests/
```

Questa struttura rappresenta i **componenti principali ancora presenti**, non l'architettura definitiva. Durante la migrazione alcuni moduli saranno adattati o sostituiti; la presenza di un agente nel repository non implica che sia già connesso alla pipeline corrente.

### Ambiente e verifiche

Le dipendenze sono dichiarate in `requirements.txt`. Per preparare un ambiente locale:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Per eseguire i test esistenti:

```bash
python3 -m unittest discover -s tests -v
```

I 31 test della precedente pipeline sono stati superati prima della transizione documentale. Ciò **non** certifica la nuova strategia Multi-Asset: i test andranno aggiornati quando cambieranno i moduli.

## 7. Ricerca Treasury precedente e conservazione Git

La precedente fase studiava una strategia su SHY, IEF e TLT. Le sue verifiche e i suoi risultati, inclusi gli esiti negativi del forecasting Ridge e della validazione Momentum 30Y, sono documentati in [`docs/treasury_research_review.md`](docs/treasury_research_review.md) e nel registro decisionale [`docs/decision_log.md`](docs/decision_log.md).

Il lavoro è stato conservato su GitHub tramite i tag:

- `archive/treasury-gianluca-final`: stato completo della ricerca Treasury al commit `0f13c76`;
- `archive/treasury-forecasting`: precedente sviluppo sperimentale, commit `a8bd531`.

**Branch di lavoro:** `gianluca`. **Branch ufficiale condiviso:** `main`. La transizione non richiede un nuovo branch, riscritture della cronologia o modifiche a `main`.

La pulizia dei file Treasury sul branch attivo avverrà in modo selettivo, tramite modifiche revisionate e commit espliciti; non saranno eliminati i moduli generali soltanto perché utilizzati nella ricerca precedente.

## 8. Prossime attività (Fase 0)

1. Completare l'audit conoscitivo del repository e identificare il codice realmente riutilizzabile.
2. Confermare l'universo investibile, la disponibilità dei dati e il calendario storico comune.
3. Scegliere una sola regola dinamica A2 e definire la metrica primaria di valutazione.
4. Definire limiti di portafoglio, convenzione di esecuzione, costi e disciplina temporale.
5. Congelare un protocollo di ricerca prima di iniziare i nuovi backtest.
6. Implementare progressivamente A, B e C, conservando anche i risultati negativi.

**Regola di sviluppo:** nessuna modifica automatica a `main`, nessun accesso anticipato ai risultati del test, nessun commit o push non revisionato. Il primo obiettivo è un problema economico ben definito e falsificabile, non una curva dei rendimenti attraente a posteriori.
