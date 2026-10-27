# Disegno della ricerca

**Progetto:** *Regime-Aware Multi-Asset Allocation: Dynamic Risk-Adjusted Performance and Downside Protection*\
**Corso:** Data-Driven Models for Investment, LUISS Guido Carli\
**Docenti:** Antonio Simeone e Villy De Luca\
**Repository di lavoro:** `Gab2112/ProjectAgentTrading26`, branch `gianluca`\
**Aggiornamento:** 26 ottobre 2026\
**Stato:** **Level A1/A2 implementato e riprodotto sul training; validazione e OOS non esaminate per la nuova strategia.** Level B/C non implementati.

> **Disciplina delle evidenze.** In questo documento si distinguono (i) decisioni progettuali già attuate, (ii) risultati osservati sul training e (iii) ipotesi/questioni aperte. I risultati della Fase A sono diagnostici *in-sample*, non una prova di efficacia predittiva o superiorità out-of-sample.

## 1. Contesto e motivazione

La prima ricerca del progetto utilizzava SHY, IEF e TLT per costruire una strategia basata sui regimi dei rendimenti Treasury. Gli esperimenti non hanno evidenziato una superiorità robusta sui benchmark passivi: in particolare, un modello Ridge sul differenziale futuro TLT–SHY non ha battuto la media storica espansiva nel training walk-forward e una regola momentum a 30 anni non ha mantenuto il vantaggio nella validation 2017–2021. Il resoconto completo è in [`treasury_research_review.md`](treasury_research_review.md).

La nuova direzione amplia l'universo da una sola famiglia di duration a **azioni, Treasury a duration intermedia, oro e Treasury brevi**. L'idea economica è che varianze, correlazioni e condizioni di rischio mutino nel tempo; un'allocazione disciplinata potrebbe sfruttare questa variabilità. Ma l'esistenza di regimi osservabili **non implica** che essi siano prevedibili o che una strategia di riallocazione generi valore netto.

**Principio di progettazione:** partire da confronti semplici, riproducibili e con timing realistico; richiedere evidenza incrementale prima di aggiungere complessità.

## 2. Domande, obiettivi e ipotesi

### 2.1 Domanda principale

> Un'allocazione dinamica basata su informazioni macro-finanziarie disponibili al momento della decisione può migliorare rendimento corretto per il rischio e protezione dai drawdown, rispetto a portafogli statici e a un semplice controllo della volatilità, dopo i costi di negoziazione?

### 2.2 Domanda incrementale

> Un modello previsionale ML e, quando sia disponibile uno storico verificabile, l'informazione strutturata derivata dalle notizie apportano valore aggiuntivo rispetto a una strategia già condizionata ai regimi?

### 2.3 Ipotesi falsificabili

- **H1 / Level A:** una regola dinamica semplice può migliorare le metriche corrette per il rischio di un portafoglio diversificato con pesi prestabiliti, senza rinunciare in misura sproporzionata al rendimento. Il confronto sul training **non verifica** H1 fuori campione.
- **H2 / Level B:** le informazioni sui regimi migliorano il portafoglio rispetto ad A2, mantenendo convenzioni comparabili di rischio, dati, esecuzione e costi.
- **H3 / Level C:** il forecasting ML apporta valore economico addizionale rispetto al sistema B e a benchmark predittivi elementari.
- **H4 / Level C, condizionata ai dati:** le notizie storiche con timestamp verificabile forniscono un contributo distinto dal forecasting senza notizie.

**Obiettivo primario:** performance corretta per il rischio; **obiettivo secondario:** gestione dei drawdown. Le metriche economiche e quelle di previsione devono restare distinte. Un portafoglio meno rischioso non è automaticamente superiore: serviranno anche confronti **a rischio comparabile**, oltre a test di robustezza e significatività.

## 3. Universo investibile, dati e ipotesi operative

| ETF | Esposizione | Funzione economica ipotizzata | Rischio da ricordare |
|---|---|---|---|
| **SPY** | Azioni large cap USA | Esposizione alla crescita e al premio azionario | Shock equity e drawdown severi |
| **IEF** | Treasury USA a duration intermedia | Possibile diversificazione in alcune recessioni disinflazionistiche | Aumenti dei rendimenti e shock inflazionistici |
| **GLD** | Oro | Possibile diversificazione da shock monetari/finanziari | Volatilità e assenza di copertura universale |
| **SHY** | Treasury USA a duration breve | Componente relativamente difensiva | Rischio di tasso, non equivale al cash risk-free |

La Fase A è **long-only**, senza leva e senza short; i target dei quattro ETF sommano a 1. La liquidità è gestita nel backtester per ragioni contabili e per l'investimento iniziale, ma non è un quinto asset allocabile. Le quantità detenute restano invariate tra le operazioni; i pesi effettivi derivano dai prezzi.

### 3.1 Dati di mercato

- Origine operativa: CSV giornalieri locali `data/raw/market/<ticker>.csv` con `Date`, `Open`, `Close` coerentemente rettificati; il `config.yaml` indica **yfinance** come fonte prevista per il download opzionale.
- Copertura: prime sedute comuni ai quattro ETF e successivo allineamento sui **calendari negoziabili comuni**; i prezzi mancanti non vengono forward-filled.
- Serie processate: `market_daily.csv`, `weekly_decision_dates.csv`, `master_weekly.csv` e report di qualità sotto `data/processed/`.
- La preparazione osservata nell'esecuzione locale genera **4.310 sedute giornaliere comuni** e **894 date decisionali settimanali** nell'intervallo di sviluppo, comprendente storico pre-training e validation. Sono conteggi di **preparazione dati**, non il numero di osservazioni impiegate per valutare la performance del training.
- La riproduzione ufficiale deve partire dagli **stessi CSV raw locali**. Un nuovo download o la modifica delle convenzioni di aggiustamento richiede un nuovo audit e una nuova etichetta dei risultati.

L'uso di Open/Close rettificati permette una proxy coerente del rendimento complessivo, ma può differire dai prezzi storicamente negoziabili in presenza di eventi societari e distribuzioni. È una **limitazione dichiarata del simulatore**, non prova di esecuzione reale.

### 3.2 Informazioni macro

**Non sono usate in A1/A2.** CPI, tassi Treasury, inflazione implicita e misure di stress sono candidate per B; i ritardi prudenziali di pubblicazione non sostituiscono la ricostruzione *point-in-time* delle versioni pubblicate storicamente. Le future trasformazioni e stime statistiche dovranno usare soltanto le informazioni allora disponibili.

## 4. Protocollo temporale e prevenzione del look-ahead

| Segmento | Intervallo configurato | Uso attuale |
|---|---|---|
| **Warm-up** | Storico disponibile precedente al 20/05/2005, da inizio 2004 circa | Solo per statistiche rolling e dati iniziali |
| **TRAIN** | 20/05/2005–31/12/2016 (ultima seduta 30/12) | **Unico segmento di decisione, backtest e valutazione della Fase A** |
| **VALIDATION** | 01/01/2017–31/12/2021 | Disponibile come segmento di sviluppo, ma **non valutata** nell'attuale pipeline A |
| **OOS** | Dal 01/01/2022 | **Bloccato** |

Il codice centrale impone `evaluation.evaluate_oos: false` e `validation.development_only: true`. Selezionare un altro `--level` non abilita il test; B e C producono un errore esplicito. Le funzioni di costruzione e valutazione di A filtrano fino alla fine del training.

**Sequenza causale per ogni decisione settimanale:**

1. osservare i prezzi rettificati fino alla **chiusura** dell'ultima seduta di quella settimana;
2. costruire con i dati disponibili il target `w_t`;
3. negoziare tale target solo alla **prima apertura negoziabile successiva**;
4. mantenere quantità di quote e liquidità, rivalutando il portafoglio giornalmente al Close;
5. non eseguire l'ultima decisione del training se la successiva apertura ricade nella validation.

**Punto metodologico:** il solo spostamento dei rendimenti di una riga settimanale non garantisce un'esecuzione causalmente corretta. La nuova simulazione separa la decisione dopo il Close dall'esecuzione al successivo Open. I test modificano prezzi e target futuri per verificare che i risultati passati restino invariati. Nessuna suite può però dimostrare automaticamente l'assenza di ogni leakage nei dati originali.

Il periodo OOS dal 2022 era già stato oggetto di conoscenze indirette in esperimenti Treasury precedenti; non va dichiarato ingenuamente "mai visto" dal ricercatore. Per la nuova strategia rimane escluso da selezione e valutazione finché la metodologia non sia congelata.

## 5. Level A1: benchmark

I quattro benchmark sono portafogli effettivamente simulati con lo stesso motore e gli stessi costi di A2, non combinazioni teoriche ricostruite a posteriori dal solo rendimento settimanale:

| Identificatore | Target | Frequenza target |
|---|---|---|
| `spy_buy_and_hold` | SPY 100% | Un'unica allocazione iniziale |
| `sixty_forty` | SPY 60%, IEF 40%, GLD/SHY 0% | Ogni settimana |
| `equal_weight` | SPY/IEF/GLD/SHY al 25% | Ogni settimana |
| `equal_weight_buy_and_hold` | SPY/IEF/GLD/SHY al 25% | Un'unica allocazione iniziale |

La distinzione tra **regola di allocazione costante** e **assenza di ribilanciamento** è essenziale: `equal_weight` riporta settimanalmente i pesi al 25%; `equal_weight_buy_and_hold` lascia che i pesi cambino con i prezzi. Anche `sixty_forty` ha un target costante nel tempo ma operazioni settimanali.

L'Equal Weight settimanale è il **benchmark di riferimento diretto** per A2, perché utilizza tutti e quattro gli ETF e la medesima frequenza decisionale.

## 6. Level A2: modello di ottimizzazione

### 6.1 Stima del rischio

Alla decisione `t`, sia `R_t` la matrice dei rendimenti giornalieri degli ETF nelle **ultime 126 sedute** già osservate. La covarianza annualizzata è:

$$
\widehat{\Sigma}_t = 252\;\operatorname{Cov}_{\mathrm{sample}}(R_t),
\qquad
\widehat{\sigma}_t(w)=\sqrt{w^{\top}\widehat{\Sigma}_t w}.
$$

Si considerano `252` sedute per anno; la covarianza usa la convenzione **campionaria**. La stima non include i rendimenti delle sedute future. Il benchmark è
$w^{\mathrm{EW}}=(0.25,0.25,0.25,0.25)^\top$.

### 6.2 Regola decisionale

Se $\widehat{\sigma}_t(w^{\mathrm{EW}})\leq 0.08$, viene adottato **esattamente** il target Equal Weight.

Se la stima supera l'8%, si risolve:

$$
\begin{aligned}
\min_{w\in\mathbb{R}^{4}}\quad & \lVert w-w^{\mathrm{EW}}\rVert_2^2\\
\text{s.t.}\quad & \sum_{i=1}^{4}w_i=1,\\
& w_i\geq0\quad (i=1,\ldots,4),\\
& w^{\top}\widehat{\Sigma}_t w\leq(0.08)^2.
\end{aligned}
$$

**Interpretazione:** la strategia cerca di modificare il benchmark il meno possibile per ottenere un portafoglio il cui **rischio stimato** rientri nel budget. Non massimizza il rendimento atteso, non usa previsioni direzionali e non è, in senso stretto, un modello di *risk parity*.

Il problema viene risolto tramite **CVXPY/CLARABEL**. Se nessun portafoglio long-only, completamente investito, rispetta il budget, il fallback previsto minimizza $w^\top\widehat{\Sigma}_t w$ sugli stessi vincoli di peso. In tale situazione **non si garantisce** il rispetto dell'8%; l'eventuale superamento deve essere registrato e spiegato. Errori o soluzioni invalide non vanno trasformati silenziosamente in risultati favorevoli.

### 6.3 Scelta del parametro e limiti

La finestra 126 sedute e il limite 8% rappresentano le scelte implementate. L'8% è stato scelto durante lo studio del training osservandone fattibilità/frequenza di attivazione; non è quindi una soglia validata indipendentemente. Il modello può ridurre eccessivamente l'esposizione a un asset volatile prima di un recupero successivo. I vantaggi dipendono anche dalla stabilità delle covarianze stimate e dalla **possibilità di scegliere un portafoglio fattibile senza cash**.

## 7. Backtester, contabilità e costi

Ogni operazione aggiorna quote possedute e liquidità. Alla prima apertura dopo il segnale, il sistema determina il capitale disponibile **dopo** i costi risolvendo coerentemente il ribilanciamento, anziché sottrarre un costo arbitrario a rendimenti settimanali teorici.

Il costo su una data di esecuzione è:

$$
C_t = \frac{2+1}{10\,000}\sum_i
\left|V^{\mathrm{post}}_{i,t}-V^{\mathrm{pre}}_{i,t}\right|,
$$

con valori pre/post riferiti ai prezzi di apertura e con il capitale post-costi determinato congiuntamente. I costi comprendono 2 bps di commissione e 1 bp di slippage per unità di **nozionale lordo negoziato**, non una percentuale fissa del NAV per settimana. Il **turnover** di quell'evento è il nozionale lordo negoziato diviso per il valore del portafoglio all'apertura prima dell'operazione.

Il NAV giornaliero è liquidità più valore di mercato delle quote al Close. Il rendimento netto è $r_t^{net}=NAV_t/NAV_{t-1}-1$, e le metriche si riferiscono a questo rendimento. I pesi effettivi sono calcolati sul NAV di fine giornata, quindi possono differire dai target anche senza nuove operazioni.

**Approssimazioni:** l'esecuzione usa Open aggiustati, non un vero book ordini; non vengono modellati slippage dipendente dalla liquidità, impatto di mercato, tasse, restrizioni operative, tracking error e tutti i dettagli delle distribuzioni. Il valore ipotizzato dei costi richiede sensibilità future. La simulazione non costituisce evidenza di eseguibilità identica nel mercato reale.

## 8. Valutazione e convenzioni metriche

- **Frequenza dei rendimenti:** giornaliera; annualizzazione con **252** sedute.
- **Risk-free:** `risk_free_rate_annual = 0.0` per le diagnostiche di A. Non è una stima dei tassi realmente osservati nel periodo.
- **CAGR implementato:** $(\prod_{t=1}^{N}(1+r_t))^{252/N}-1$. È una convenzione basata sulle sedute, che può differire marginalmente dal calcolo sui giorni di calendario; usare **7,77%** per SPY nel report corrente.
- **Volatilità:** deviazione standard campionaria giornaliera (`ddof=1`) moltiplicata per $\sqrt{252}$.
- **Sharpe:** media dei rendimenti giornalieri in eccesso sul risk-free, divisa per la deviazione standard campionaria dei rendimenti in eccesso, moltiplicata per $\sqrt{252}$.
- **Sortino:** media dei rendimenti in eccesso divisa per $\sqrt{\operatorname{mean}[\min(r_t-r_{f,t},0)^2]}$, quindi annualizzata con $\sqrt{252}$; la semideviazione include contributo nullo nelle giornate positive.
- **Maximum Drawdown:** minima distanza relativa fra NAV e massimo cumulato precedente, includendo il capitale iniziale come riferimento.
- **Metriche aggiuntive:** rendimento cumulato, numero di osservazioni, NAV finale, turnover aggregato, costi in USD, numero di eventi di esecuzione e profit factor basato sui rendimenti giornalieri.

Un **profit factor sui rendimenti giornalieri** non va confuso con un profit factor ricostruito da trade discreti. Non si riportano come metriche ufficiali correnti numeri di Calmar o Sortino provenienti da report diagnostici antecedenti senza verificare il corrispondente output aggiornato.

## 9. Evidenze osservate sul training

**Periodo di backtest:** 20 maggio 2005 – 30 dicembre 2016. **Capitale iniziale:** 100.000 USD. **Costi:** inclusi. **Sharpe:** risk-free 0. Risultati dell'esecuzione locale di `python3 main.py --level ALL` del 26 ottobre 2026, arrotondati.

| Strategia | CAGR | Volatilità annua | Sharpe | MDD | NAV finale (USD) | Costi cumulati (USD) |
|---|---:|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 7,77% | 19,68% | 0,479 | −55,19% | 238.298,26 | 29,99 |
| 60/40 settimanale | 7,13% | 10,88% | 0,688 | −32,00% | 222.430,62 | 281,70 |
| Equal Weight settimanale | 6,54% | 6,99% | 0,942 | −15,69% | 208.608,24 | 357,11 |
| Equal Weight Buy & Hold | 6,16% | 8,75% | 0,727 | −17,63% | 200.164,62 | 29,99 |
| **A2 Volatility Control** | **6,24%** | **6,41%** | **0,977** | **−14,28%** | **201.948,33** | **376,38** |

### 9.1 Confronto diretto con Equal Weight settimanale

- CAGR A2 **inferiore di 0,30 punti percentuali**.
- Volatilità A2 **inferiore di 0,58 punti percentuali**.
- Sharpe A2 **0,977** contro **0,942**, miglioramento modesto.
- Maximum Drawdown A2 **−14,28%** contro **−15,69%**, miglioramento di **1,41 punti percentuali**.
- Costi cumulati A2 **376,38 USD** contro **357,11 USD**: la gestione del rischio ha richiesto costi leggermente maggiori nel campione osservato.

Il risultato descrive un **trade-off rischio/rendimento sul training**, non la dimostrazione di una strategia economicamente superiore. I portafogli non sono confrontati a volatilità ex ante identica e l'incremento di Sharpe non è accompagnato da un test di significatività.

### 9.2 Diagnostica A2 ed esecuzioni

Il file `data/signals/A2/volatility_control/diagnostics.csv` è stato controllato localmente:

| Stato decisionale | Conteggio |
|---|---:|
| `equal_weight` | **520** |
| `ottimizzato` | **87** |
| `fallback_minimum_variance` | **0** |
| **Totale decisioni** | **607** |

Questi stati descrivono **decisioni settimanali**, non singole operazioni o trade indipendenti. Nel backtest giornaliero risultano **606 eventi eseguibili** per ogni strategia settimanale: la decisione finale del training non può essere applicata prima della validation. I due buy-and-hold eseguono solo l'allocazione iniziale. La diagnostica non dimostra che il vincolo dell'8% verrà rispettato nel futuro.

### 9.3 Verifiche tecniche e loro perimetro

- **60 test automatici superati** sul Mac con Python 3.11, inclusi test numerici CVXPY/CLARABEL, test di calcolo finanziario, causality, integrità temporale e controllo OOS.
- Esecuzione **A1/A2 completa** sul training, prima e dopo la rimozione locale del codice legacy; metriche finali riprodotte.
- Conteggio `520/87` di A2 confermato dal file diagnostico.
- **Controllo ancora da ripetere:** suite completa di 60 test dopo la rimozione dei moduli legacy, quale ultimo gate prima di commit e push.

Queste sono verifiche del codice e del protocollo di training, non validazioni su dati non utilizzati nella ricerca.

## 10. Rischi metodologici e limitazioni da dichiarare

1. **In-sample e selezione:** il limite 8% è stato fissato nel contesto del training; performance e frequenze osservate possono influenzare le scelte. Non attribuire ai risultati lo status di prova indipendente.
2. **Assenza di validation nuova:** il periodo 2017–2021 non è stato valutato per A1/A2. La sua eventuale apertura richiederà un protocollo dichiarato prima di consultare i risultati.
3. **Conoscenza pregressa del ricercatore:** la validation era stata consultata nella vecchia ricerca Treasury; l'OOS dal 2022 non va descritto come completamente ignoto al gruppo, benché rimanga bloccato nella nuova pipeline.
4. **Confronto non equal-risk:** A2 può avere un'esposizione effettiva più prudente dell'Equal Weight; per attribuire valore all'allocazione servono analisi a rischio comparabile e robustezza a parametri/costi.
5. **Stabilità della stima:** covarianza campionaria rolling e correlazioni possono cambiare in modo discontinuo; il limite **stimato** non vincola il rischio **realizzato**.
6. **Data quality:** valutare continuità dei prezzi rettificati, origine/vintage dei CSV, corporate actions, calendari e riproducibilità dal grezzo; lo stesso input può essere ricalcolato dal provider con aggiustamenti diversi.
7. **Trading costs:** il modello fisso 3 bps non include tutti i frizionamenti; analisi di sensibilità necessaria prima di affermazioni di implementabilità.
8. **Risk-free:** Sharpe/Sortino con rf=0 sono un riferimento diagnostico. Un confronto economico completo richiede il tasso storicamente osservato e una convenzione coerente.
9. **Significatività:** non sono stati ancora stimati intervalli di confidenza, bootstrap a blocchi o test appropriati sul miglioramento di Sharpe/drawdown.
10. **Requisiti dell'esame:** la rappresentazione di eventuali segnali discreti `-1`, `0`, `+1` per gli asset deve essere verificata sul testo delle istruzioni del corso; un target long-only non è automaticamente equivalente a un sistema di segnali direzionali completo.

## 11. Architettura software e riproducibilità

**Pipeline:** `main.py` → `src/data_transformation.py` → `src/portfolio.py` (A1/A2, con `src/multiasset_allocator.py`) → `src/backtester.py` (motore `src/multiasset_backtester.py`) → `src/evaluator.py`. I moduli trasversali sono `src/data_loader.py`, `src/time_split.py`, `src/utils.py`; `src/data_acquisition.py` è opzionale.

**Input e output determinanti:**

| Tipo | File |
|---|---|
| CSV grezzi | `data/raw/market/<ticker>.csv` |
| Prezzi allineati | `data/processed/market_daily.csv` |
| Calendario | `data/processed/weekly_decision_dates.csv` |
| Pesi | `data/signals/A1/<strategia>/target_weights.csv` e `data/signals/A2/volatility_control/target_weights.csv` |
| Diagnostica A2 | `data/signals/A2/volatility_control/diagnostics.csv` |
| NAV e rendimenti | `data/results/<livello>/<strategia>/backtest_daily.csv` |
| Metriche | `data/results/A1/evaluation_metrics.csv`, `data/results/A2/evaluation_metrics.csv` |
| Confronto aggregato | `data/results/multiasset_level_a_comparison.csv` |

Versione locale verificata: **Python 3.11** con dipendenze bloccate in `requirements.txt`. Le cartelle `data/processed`, `data/signals`, `data/results` sono output rigenerabili e ignorati da Git; i CSV raw devono poter essere riprodotti o tracciati. Il file Conda alternativo `environment-multiasset.yml`, se mantenuto, va reso coerente con le versioni pip.

Comandi per replica del training:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip check
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 main.py --level ALL
```

Nessun commit o push remoto deve essere eseguito automaticamente dal processo di analisi.

## 12. Protocollo delle fasi successive

### 12.1 Prima del push: chiusura locale del Level A

1. Verificare la rimozione selettiva del codice Treasury senza perdere l'archivio Git; controllare eventuali tag di conservazione citati nella documentazione.
2. Ripetere la suite **60/60 dopo la pulizia**, poi `main.py --level ALL` e controlli sui file di diagnostica.
3. Allineare `environment-multiasset.yml` alle dipendenze pip oppure definire una sola fonte ufficiale per l'ambiente.
4. Congelare documenti, configurazione e output di riferimento; controllare `git diff --check` e `git status --short`.
5. Effettuare commit e **push manuale** sul branch `gianluca`, senza modificare il branch condiviso `main` senza una decisione separata.

### 12.2 Dopo il push: punto metodologico personale

Dedicare una sessione specifica alla comprensione di: ruolo economico dei quattro ETF; differenza tra Buy & Hold e ribilanciamento; stima della covarianza; geometria e vincoli dell'ottimizzazione; quando il fallback è necessario; significato di NAV, turnover, costi, CAGR, Sharpe e drawdown; limiti dell'inferenza *in-sample*. Questo passaggio **precede** l'implementazione di B.

### 12.3 Prima di B e della futura validation

Fissare senza guardare nuovi risultati: definizione operativa di inflazione/stress e disponibilità *point-in-time*; regole di classificazione; frequenza; regola di allocazione associata ai regimi; metrica primaria di selezione; confronto a parità di rischio; sensibilità dei costi; politica di selezione/ipotesi statistiche; uso disciplinato di train/validation e blocco OOS. B resta un'estensione incrementale da confrontare con A2, non una modifica retrospettiva di A2.

### 12.4 Level C, solo successivamente

Definire **prima** target, orizzonte previsionale e benchmark del forecasting; usare walk-forward e trasformazioni causalmente corrette; valutare separatamente performance predittiva e performance del portafoglio. Gli agenti e FinBERT richiedono dataset storici coerenti, timestamp verificabili e vincoli quantitativi non derogabili. In assenza di dati storici validi per le notizie, la parte sentiment può essere dimostrata solo come prototipo contemporaneo, senza attribuirle rendimenti passati.

## 13. Stato delle conclusioni

**Conclusione tecnica:** la Fase A è implementata, è eseguibile tramite un solo punto d'ingresso ed è stata riprodotta sul training. I 60 test automatici sono risultati tutti verdi prima dell'ultima pulizia legacy; dopo la pulizia è riuscita una nuova esecuzione completa A1/A2.

**Conclusione economica limitata:** rispetto all'Equal Weight settimanale, A2 ha mostrato nel training un rischio osservato inferiore e un lieve miglioramento delle metriche risk-adjusted, ma rendimento minore. Non è stata dimostrata un'abilità predittiva, un vantaggio netto robusto o una superiorità fuori campione.

**Conclusione procedurale:** stabilizzare e pubblicare localmente/remotamente la Fase A, studiarne criticamente il metodo e solo successivamente sviluppare il Level B. Nessun risultato B o C è disponibile alla data del presente aggiornamento.
