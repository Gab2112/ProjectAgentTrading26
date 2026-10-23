# Disegno della ricerca

**Progetto:** Regime-Aware Multi-Asset Allocation: Dynamic Risk-Adjusted Performance and Downside Protection\
**Corso:** Data-Driven Models for Investment — LUISS Guido Carli\
**Docenti:** Antonio Simeone, Villy De Luca\
**Repository di riferimento:** `Gab2112/ProjectAgentTrading26`\
**Stato:** bozza di Fase 0; nessuna implementazione autorizzata da questo documento\
**Data:** 23 ottobre 2026

## 1. Contesto e motivazione

Il progetto studia una nuova direzione di ricerca rispetto alle precedenti strategie su ETF Treasury (SHY, IEF, TLT). I risultati della fase precedente non hanno fornito evidenza sufficientemente robusta di valore incrementale rispetto ai benchmark. In particolare, una regressione Ridge sui rendimenti relativi futuri TLT–SHY non ha superato la previsione basata sulla media storica, mentre una strategia momentum TLT/SHY ha mostrato risultati di validazione inferiori a diversi benchmark passivi.

**Implicazione metodologica:** non assumere che la maggiore complessità migliori i risultati; valutare ogni modulo aggiuntivo contro un benchmark più semplice, mantenendo separati qualità previsionale e utilità economica.

## 2. Domanda di ricerca e ipotesi

### 2.1 Domanda principale

> Un'allocazione dinamica basata su stati macro-finanziari identificabili con informazioni disponibili al momento della decisione può migliorare il rendimento corretto per il rischio di un portafoglio diversificato SPY–IEF–GLD–SHY rispetto a benchmark statici e a una regola dinamica semplice, senza compromettere la protezione dalle perdite?

### 2.2 Domanda incrementale

> Modelli previsionali di machine learning e segnali strutturati derivati da notizie aggiungono valore economico misurabile rispetto all'allocazione già condizionata ai regimi?

### 2.3 Meccanismo economico ipotizzato

La volatilità, le correlazioni e le distribuzioni dei rendimenti delle diverse asset class variano al mutare delle condizioni di crescita, inflazione, tassi e stress finanziario. Se tali cambiamenti sono identificabili in tempo utile, una riallocazione disciplinata potrebbe migliorare il trade-off rischio-rendimento rispetto a pesi statici. La premessa non implica che i regimi siano prevedibili, che i rapporti storici siano stabili o che la riallocazione sia redditizia al netto dei costi.

### 2.4 Ipotesi falsificabili

- **H1:** una strategia dinamica semplice presenta prestazioni corrette per il rischio migliori di un portafoglio diversificato statico su dati non utilizzati per definirne le regole.
- **H2:** l'informazione sul regime macro-finanziario apporta valore incrementale rispetto alla strategia dinamica semplice, dopo costi e con un confronto del rischio coerente.
- **H3:** le previsioni ML apportano valore incrementale rispetto alla strategia condizionata ai regimi.
- **H4, esplorativa e subordinata ai dati:** i segnali da notizie storiche verificabili apportano ulteriore valore rispetto alla versione priva di sentiment.

Le ipotesi non devono essere considerate confermate da risultati in-sample, grafici favorevoli o singoli episodi di crisi.

## 3. Obiettivi e criteri di valutazione

**Obiettivo primario:** miglioramento della performance corretta per il rischio, non semplice massimizzazione del rendimento assoluto.

**Obiettivo secondario:** protezione dinamica dal downside, verificata attraverso drawdown, downside volatility, comportamento in sottoperiodi di stress e capacità di partecipare ai successivi recuperi.

**Vincolo interpretativo:** una strategia non è considerata superiore perché investe quasi sempre nello strumento meno volatile. Quando appropriato, confrontare strategie anche a budget di rischio comparabile.

Metriche da riportare: rendimento cumulato ed equity curve, CAGR, volatilità annualizzata, Sharpe (con convenzione risk-free esplicita), Sortino, maximum drawdown, turnover, costi, esposizioni e prestazioni per sottoperiodo. Per il ML riportare separatamente metriche previsionali e finanziarie.

**Da definire prima di usare la validazione:** metrica primaria di selezione, tolleranze sul downside, metodo di confronto a parità di rischio, ipotesi di costo e soglia minima di utilità economica.

## 4. Universo investibile

| ETF | Esposizione | Ruolo economico ipotizzato |
|---|---|---|
| SPY | Azioni USA large cap | Crescita e rischio azionario |
| IEF | Treasury USA a scadenza intermedia | Duration e possibile diversificazione in recessione/disinflazione |
| GLD | Oro | Diversificazione in alcune condizioni di stress o shock monetari |
| SHY | Treasury USA a breve durata | Componente difensiva con rischio di duration ridotto |

**Scelte iniziali:** long-only, senza leva, niente vendite allo scoperto, capitale interamente investito nei quattro ETF salvo successiva introduzione esplicita della liquidità; riallocazione mensile; limiti di concentrazione e costi realistici.

**Avvertenza:** nessuno dei quattro strumenti è universalmente difensivo. IEF può soffrire shock inflazionistici e rialzi dei rendimenti; GLD può perdere valore durante crisi; SHY non coincide con liquidità priva di rischio. Questi rapporti devono essere verificati, non assunti come regole immutabili.

**Da verificare:** date effettive d'inizio delle serie investibili, prezzi adjusted/total return, distribuzioni, copertura comune e ruolo operativo di SHY come posizione difensiva rispetto al riferimento risk-free per il calcolo dello Sharpe.

## 5. Architettura sperimentale incrementale

### Livello A — Benchmark e regola dinamica semplice

- **A1, statici:** SPY buy-and-hold; 60% SPY e 40% IEF; equal-weight sui quattro ETF; altri ETF individuali solo a fini descrittivi.
- **A2, dinamico:** una sola regola semplice e motivata economicamente, candidata tra trend azionario e gestione della volatilità. La selezione deve avvenire prima dell'esame delle prestazioni di validazione.
- **Risultato atteso:** un baseline riproducibile, non una strategia già dichiarata vincente.

### Livello B — Regimi macro-finanziari

- **B1, interpretabile:** schema concettuale di due dimensioni (stress finanziario alto/basso; pressione inflazionistica alta/bassa), con definizioni operative ancora da stabilire.
- **B2, statistico, opzionale:** clustering (es. K-means/GMM) soltanto se apporta una giustificazione e un test chiaro. Standardizzazione, stima dei cluster e assegnazione devono rispettare l'informazione storicamente disponibile.
- **Test decisivo:** miglioramento rispetto ad A2; attenzione a stabilità, persistenza, turnover, interpretabilità e robustezza.

### Livello C — Previsione e agenti

- **C1, previsione:** selezionare un unico target primario. Candidato prioritario: rischio azionario del mese successivo o probabilità di evento negativo **definito quantitativamente prima della stima**. Confrontare benchmark di persistenza/media storica, modello lineare/logistico regolarizzato e al massimo un modello non lineare giustificato.
- **C2, modulo regime:** restituire una classificazione o probabilità di regime ottenuta senza informazioni future.
- **C3, notizie/sentiment:** classificare soltanto documenti con timestamp verificabile, storico adeguato e contenuti disponibili alla decisione. In assenza di tali dati, realizzare una dimostrazione contemporanea, senza attribuirle prestazioni storiche.
- **C4, controllo del rischio:** limiti e verifiche deterministici; nessun agente può derogare ai vincoli rigidi.
- **C5, allocazione:** trasformazione esplicita e riproducibile dei segnali in pesi, con limiti, turnover e tempi di esecuzione definiti.

**Ablation study:** statico → A2 → B → C senza notizie → C con notizie, se metodologicamente possibile.

## 6. Dati e integrità temporale

Fonti candidate: prezzi giornalieri adjusted degli ETF, VIX, rendimenti Treasury, CPI, indicatori di curva e stress finanziario. Yahoo Finance e FRED sono possibili fonti, ma la decisione richiede un audit delle serie e delle rispettive convenzioni.

Controlli minimi:

1. Esistenza e negoziabilità storica di ciascun ETF.
2. Prezzi corretti per split e distribuzioni e confronto coerente dei rendimenti.
3. Giorni comuni di mercato, valori mancanti, anomalie e duplicati.
4. Disponibilità *point-in-time* di dati macro, date di pubblicazione, ritardi e revisioni.
5. Nessun dato derivato dal futuro nelle feature, nelle etichette o nelle trasformazioni.
6. Tracciabilità dei file originali e risultati riproducibili senza riscaricare dati.

**Frequenza proposta:** prezzi giornalieri per feature e monitoraggio; decisione e riallocazione mensile.

## 7. Protocollo cronologico e protezione dell'OOS

**Suddivisione candidata, non ancora congelata:** training 2005–2016; validation 2017–2021; test finale dal 2022 in avanti. Da confermare dopo l'audit di copertura delle serie e della numerosità effettiva delle osservazioni mensili.

- Iperparametri, feature, regole, pesi e soglie si scelgono nel training e nella validation, eventualmente con walk-forward interno.
- Tutti i preprocessing statistici sono stimati soltanto con dati disponibili al momento della previsione.
- Target con orizzonte che supera un confine train/validation/test richiedono gestione esplicita del *purging* o esclusione delle osservazioni contaminate.
- Non analizzare i risultati del test per modificare la metodologia.
- **Importante:** il periodo dal 2022 è già stato analizzato in precedenti esperimenti Treasury. Non va descritto ingenuamente come “mai osservato” a livello di ricercatori. Per la nuova strategia il test deve essere *locked* per le decisioni della nuova ricerca; qualsiasi conoscenza pregressa pertinente va dichiarata nel report.

**Punto aperto:** definizione concreta del test finale e della sua data conclusiva, inclusi gli effetti della ricerca iterativa effettuata sul medesimo intervallo.

## 8. Segnali, operatività e costi

La frequenza mensile non rende automaticamente un backtest eseguibile. È necessario definire la sequenza:

1. Chiusura dell'intervallo informativo e timestamp dell'ultima osservazione utilizzabile.
2. Calcolo dei segnali a dati effettivamente disponibili.
3. Prima finestra di esecuzione accessibile dopo la generazione del segnale.
4. Applicazione dei pesi e attribuzione dei rendimenti ai periodi successivi.
5. Applicazione di commissioni, spread/slippage e costi indotti dal turnover.

**Scelta ancora aperta:** convenzione di prezzo/esecuzione (ad esempio ordine al primo open negoziabile dopo l'osservazione). Evitare il falso assunto di eseguire al close il segnale calcolato con quello stesso close.

I segnali discreti richiesti dal corso (`-1`, `0`, `+1`) dovranno essere rappresentati in modo verificabile per almeno due asset. La prima architettura long-only può usare solo `0` e `+1`, a condizione di motivare come l'assenza di una posizione sia distinta da un'allocazione difensiva e di verificare la compatibilità con i requisiti di consegna. I pesi di portafoglio sono un livello ulteriore rispetto al segnale discreto.

## 9. Criteri di interpretazione e possibilità di fallimento

L'ipotesi non è supportata se i vantaggi compaiono soltanto in training, se spariscono al netto dei costi, se risultano spiegati da minore esposizione al rischio o se dipendono da scelte retrospettive. Risultati negativi vanno conservati e commentati.

Una previsione accurata non equivale a un sistema di trading profittevole; un buon backtest non prova capacità predittiva né causalità. La valutazione finale deve isolare il contributo di ogni livello.

## 10. Decisioni e questioni aperte

### Decisioni di indirizzo adottate per la Fase 0

- Domanda principale: utilità economica dei regimi nella riallocazione multi-asset.
- Obiettivo primario: performance corretta per il rischio.
- Obiettivo secondario: protezione dal downside.
- Universo iniziale: SPY, IEF, GLD, SHY.
- Frequenza iniziale: mensile.
- Progressione: A → B → C, con confronto incrementale.
- Sviluppo e documentazione interna in italiano; inglese solo per i materiali finali.

### Decisioni da prendere prima dei backtest

- Regola unica del baseline dinamico A2.
- Metrica primaria e criteri di successo predefiniti.
- Soglie di regime e calendario storico effettivamente disponibile.
- Target previsionale primario e sua definizione operativa.
- Limiti di peso, scelta del modello di allocazione, costi e calendario di esecuzione.
- Periodi esatti di split e disciplina OOS, dopo audit dei dati.
- Disponibilità e licenze delle notizie storiche.

## 11. Prossima attività autorizzabile

**Audit esclusivamente conoscitivo del repository**, senza modifiche: verificare stato Git, documentazione esistente, struttura dei moduli e possibili componenti riutilizzabili. Prima di qualsiasi commit, push, merge o modifica ai file condivisi, preparare un elenco delle variazioni proposte e sottoporlo a revisione.

---

*Documento di lavoro: nessun risultato quantitativo della nuova strategia è stato ancora prodotto o validato.*

---

## Level A — Risultati e consolidamento della ricerca

### Stato della fase

Il Level A è stato implementato e verificato sul solo campione di training.
La fase comprende benchmark statici (A1) e un'allocazione dinamica
con vincolo di volatilità (A2).

### Universo e periodo

- Asset: SPY, IEF, GLD, SHY.
- Prezzi: OHLC giornalieri rettificati.
- Training: fino al 31 dicembre 2016.
- Backtest comparativo: 20 maggio 2005 – 30 dicembre 2016.
- Frequenza decisionale: settimanale.
- Esecuzione: prima apertura disponibile dopo il segnale.
- Capitale iniziale: 100.000 USD.
- Commissioni: 2 bps; slippage: 1 bp.

### Strategia A2

- Portafoglio di riferimento: Equal Weight (25% per ETF).
- Stima della covarianza: 126 sedute, annualizzazione a 252.
- Budget massimo di volatilità stimata: 8% annualizzato.
- Obiettivo: minimizzare la distanza quadratica dall'Equal Weight.
- Vincoli: long-only, pesi non negativi, somma dei pesi pari a 1.
- Ottimizzazione: CVXPY con risolutore CLARABEL.
- Fallback previsto: portafoglio Minimum Variance in caso di
  impossibilità di rispettare il budget di rischio.

Il limite dell'8% è stato selezionato durante la ricerca sul training.
Non rappresenta una garanzia sulla volatilità futura realizzata.

### Risultati del training

| Strategia | CAGR | Volatilità | Maximum Drawdown |
|---|---:|---:|---:|
| A2 Volatility Constraint | 6,24% | 6,41% | -14,28% |
| Equal Weight Weekly | 6,54% | 6,99% | -15,69% |
| 60/40 Weekly | 7,13% | 10,88% | -32,00% |
| SPY Buy & Hold | 7,76% | 19,68% | -55,19% |
| Equal Weight Buy & Hold | 6,16% | 8,75% | -17,63% |

Nel confronto diretto A2 versus Equal Weight settimanale:

- Sharpe Ratio (rf = 0): 0,9773 contro 0,9419.
- Sortino Ratio (rf = 0): 1,4073 contro 1,3628.
- Calmar Ratio: 0,4368 contro 0,4166.
- Differenza CAGR: -0,30 punti percentuali.
- Miglioramento Maximum Drawdown: circa 1,40 punti percentuali.

### Verifiche tecniche

- 607 settimane di decisione nel training.
- 520 settimane Equal Weight e 87 con vincolo attivo.
- 87 ottimizzazioni valide su 87, senza errori.
- 606 eventi di esecuzione nel backtest.
- Verificati ritardo di esecuzione, drift dei pesi,
  turnover, costi e identità contabile del NAV.

### Interpretazione e limitazioni

A2 riduce volatilità e drawdown rispetto all'Equal Weight,
ma sacrifica parte del rendimento annualizzato.

I miglioramenti delle metriche corrette per il rischio
sono modesti e non ne è stata dimostrata la significatività
statistica.

Nel 2009 il vincolo di volatilità ha penalizzato il rendimento,
coerentemente con il rischio di ridurre l'esposizione durante
una successiva fase di recupero dei mercati.

I risultati sono esclusivamente in-sample.
Non costituiscono evidenza di superiorità out-of-sample.

La simulazione utilizza prezzi rettificati come proxy dei
prezzi di esecuzione e un modello semplificato dei costi.
Il tasso privo di rischio pari a zero è stato utilizzato
soltanto per le metriche diagnostiche iniziali.

La verifica dei dati point-in-time macroeconomici,
la validazione successiva al training e i modelli di regime
appartengono alle fasi successive.

### Implementazione

- `src/multiasset_allocator.py`
- `src/multiasset_backtester.py`
- `environment-multiasset.yml`

Il precedente progetto Treasury rimane preservato
nei relativi tag Git di archivio.
