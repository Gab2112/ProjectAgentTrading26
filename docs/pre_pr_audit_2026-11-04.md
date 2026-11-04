# Audit pre-PR | ProjectAgentTrading26, branch `modulo-C`

**Data di verifica:** 4 novembre 2026. **Perimetro:** lettura del repository GitHub (`modulo-C` e `main`), principali sorgenti/configurazioni e log dei test e dei backtest forniti dal gruppo. **Non è stata eseguita la suite sul clone del repository** in questo ambiente; i 244 test superati sono il risultato riportato dall'utente sul Mac. Non sono stati analizzati rendimenti del test finale 2022+.

## Decisione sintetica

**PR in bozza: sì. Merge su `main`: ancora no.** I risultati sperimentali ci sono, ma prima del merge bisogna risolvere il contratto di esecuzione ufficiale del Level C, provare la riproduzione su ambiente pulito e coordinare l'integrazione dei branch con gli altri sviluppatori. Una pulizia estesa del codice `src/` non è consigliata in questa fase; Giovanni ne è il referente dopo il merge.

### Stato remoto verificato

- `main` è ancora sulla baseline A/B; la PR **#1 `gianluca` → `main`** dei fix agenti è ancora **aperta**.
- `modulo-C` è presente sul remoto, include i fix agenti di `gianluca`, i nuovi C e i runner di confronto e sensibilità. Rispetto a `main` risultano **14 commit aggiuntivi** nella verifica del 4 novembre.
- Il branch `clustering` di Giovanni **non risultava visibile** nell'elenco remoto esaminato. Non eseguire `git merge origin/clustering` finché non compare in `git branch -r` dopo `git fetch`.
- In `modulo-C` risultano versionati 95 oggetti tra file e directory (conteggio dell'albero Git, non numero di moduli eseguibili). Non sono tracciati i file generati in `data/processed`, `data/signals` e `data/results`.

## Risultati dell'audit, per priorità

| Priorità | Evidenza verificata | Impatto | Rimedio / owner |
|---|---|---|---|
| **P0** | `main.py` supporta A1/A2/B ma `_select_levels('C')` solleva `NotImplementedError`; `--level ALL` significa solo A1+A2 | La CLI pubblica non corrisponde all'architettura effettivamente realizzata | Implementare un **dispatcher C training-only** che richiami i runner C esistenti senza rifare il backtester. Testare `main --level C` e specificare il significato di `ALL`. **Gianluca prima del merge** |
| **P0** | `config.yaml`: `strategies.C.enabled: false`, `agents.enabled: false`, mentre C funziona da script indipendenti | I flag rappresentano la pipeline integrata non ancora attiva, non la disponibilità del codice. Attivarli senza coerenza romperebbe l'invariante esistente nei test | Decidere e documentare esplicitamente due livelli: **training manuale agenti** vs **consumo dei segnali C**. Cambiare i flag solo in abbinamento al nuovo dispatcher e ai test. **Gianluca** |
| **P0** | C richiede `data/signals/signals_orchestrator_{spy,ief,gld,shy}.csv`, non versionati; sul clone pulito sono assenti | Un collega con i soli raw CSV **non può riprodurre immediatamente C** | Dimostrare una **rigenerazione controllata** dei quattro segnali da `src.agents.train` e verificare date, hash e performance; distinguere il training ex novo dal replay degli output. **Gianluca + responsabili agenti** |
| **P0** | PR #1 ancora aperta; `modulo-C` discende da `gianluca` | Una PR diretta di `modulo-C` verso `main` include anche le modifiche della PR #1, con possibile review duplicata | Aprire temporaneamente una **PR stacked** `modulo-C → gianluca`, oppure aspettare la PR #1 e poi aprire `modulo-C → main`. No force-push. **Team** |
| **P1** | `README.md` datato 28 ottobre dichiara C non integrato e solo 123 test; `docs/research_design.md` e `docs/decision_log.md` fermano la cronologia ad A/B | Un revisore legge una fotografia non aggiornata della ricerca | Sostituire README con lo stato A/B/C, aggiungere nota C dedicata e **aggiornare le aperture dei due documenti storici** senza cancellare le decisioni 2005–2016. **Gianluca** |
| **P1** | `.gitignore` copre `data/processed`, `signals`, `results`, ma non binari `.joblib` o cache comuni; risultano molti nuovi parametri locali non tracciati | `git add .` rischia di includere artefatti o configurazioni non verificate | Estendere `.gitignore`, non ignorare genericamente i JSON esistenti. Scegliere esplicitamente come archiviare i modelli e quali parametri JSON diventino input congelati. **Team** |
| **P1** | `requirements.txt` non dichiara `joblib`, pur importato direttamente da RF/SVM; `FinBERT_agent.py` usa `torch/transformers` ma sono opzionali; `environment-multiasset.yml` crea solo Python + pip | Ambiente che passa sul Mac non dimostra bootstrap riproducibile da zero | Aggiungere `joblib` tra le dipendenze esplicite; documentare **installazione pip manuale dopo conda**; mantenere Torch/Transformers opzionali e versionare separatamente l'ambiente sentiment quando serve. **Gianluca** |
| **P1** | Nessuna GitHub Action nel ramo esaminato | Nessun test automatico al momento delle PR | Aggiungere workflow minimo `unittest discover` su Python 3.11, verificare un run CI verde. **Team** |
| **P1** | C sovraperforma B **solo** nel periodo di ottimizzazione dell'Orchestrator; 2017–2021 è stato esaminato per B_v2, non per C | Le conclusioni AI non possono essere presentate come out-of-sample | Etichettare tutti i report 2010–2016 come **in-sample**; fissare un protocollo C che ricostruisca realmente predizioni senza training sul futuro. **Team** |
| **P1** | Le osservazioni FRED non sono verificate sulle vintage storiche/pubblicazioni intraday | Rischio di disponibilità informativa anticipata | Conservare `macro_point_in_time_verified: false`, non promettere un backtest investibile o privo di leakage macro. **Team** |
| **P2** | `data/raw/market/TLT.csv` e parametri agenti SHY/IEF/TLT permangono dai progetti Treasury precedenti | Possibile confusione, ma nessun errore se non usati come universo investibile | Etichettarli come **legacy archive**, senza cancellazioni affrettate. **Giovanni dopo merge** |
| **P2** | File locali non tracciati: `compare_backtest_engines.py`, `notebooks/`, `reports/`, docs esportati; due file agenti locali risultavano modificati | Stato locale non equivalente allo stato remoto | Inventario e decisione singola per file. Non usare `git add .`, `git clean`, `reset --hard`. **Gianluca** |

## Data: la sola cartella raw su Git è corretta

La directory tracciata contiene `data/raw/market/` con SPY, IEF, GLD, SHY e il legacy TLT; contiene inoltre `data/raw/macro/` con fonti FRED come VIXCLS e T10YIE. I CSV hanno intestazioni coerenti con i loader (`Date`, OHLC, Volume per ETF; `Date`, serie per FRED). È corretto che le directory generate **non compaiano in Git** perché sono ignorate e Git non versiona directory vuote.

`src/data_transformation.py` produce esplicitamente sei file:

1. `data/processed/market_daily.csv` (date di trading e Open/Close rettificati per asset);
2. `data/processed/weekly_decision_dates.csv` (calendario dei segnali);
3. `data/processed/market_daily_quality.csv`;
4. `data/processed/master_weekly.csv`;
5. `data/processed/data_quality_summary.csv`;
6. `data/processed/decision_features_training.csv`.

I quattro `signals_orchestrator_*.csv` nascono invece dal workflow `src/agents/train.py`, dentro `data/signals/`; sono **input necessari** per C. Tenere i raw in Git preserva uno snapshot del dataset, ma **non basta** per provare la riproducibilità del fitting dell'Orchestrator: servono versioni del software, semi casuali effettivi e tracciabilità di parametri/modelli/segnali.

**Raccomandazione:** non aggiungere alcun `data/processed/*.csv` alla PR. Aggiungere `data/README.md` con contratto e processo di generazione; introdurre un manifest di riproducibilità dei segnali prima del rilascio finale, usando hash dei file anziché committare gli output generati.

## Contratto consigliato per `main.py` e `config.yaml`

Non duplicare `simulate_longshort_daily`, né spostare ora classi e funzioni tra i moduli: questo è il refactoring di Giovanni. La nuova interfaccia ufficiale dovrebbe essere un **smistatore leggero (*dispatcher*)**, con le seguenti proprietà verificabili:

- `python main.py --level A1|A2|ALL|B` conserva **esattamente** la semantica e i risultati attuali. `ALL` va rinominato o chiaramente documentato come *A1+A2* finché non cambia formalmente.
- `python main.py --level C` esegue solo la ricerca **limitato al training** e consuma i quattro segnali Orchestrator *già presenti*; non addestra modelli, non scrive sui file B congelati e non accede a 2017+.
- Per la scelta `C_v1` o `C_v2`, usare un'opzione esplicita oppure stampa entrambe come confronto, indicando sempre che sono in-sample. Evitare il nome “OOS” nella relazione.
- Solo dopo questa integrazione si può portare `strategies.C.enabled` a `true` e adeguare il test sul set delle strategie abilitate. Non impostare `agents.enabled: true` come surrogato del training, se il dispatcher non lo implementa.
- `validation.development_only: true`, `evaluation.evaluate_oos: false`, `validation.train_end: 2016-12-31`, `validation.oos_start: 2022-01-01` **non vanno cambiati**.
- Test minimi: confronto identico sui 365 segnali, protezione dei file congelati B, errore esplicito se manca un segnale Orchestrator, stessi intervalli temporali, nessun uso del 2022+, errore in caso di flag incoerenti.

**Per questo audit non è stato modificato `main.py` né cambiata la semantica del YAML:** sarebbe un cambio funzionale, e senza una prova di regressione sul repository reale non sarebbe corretto presentarlo come semplice 'pulizia'. Deve essere un commit successivo e separato dalla documentazione.

## Strategia Git e ordine di lavoro

1. Portare in branch pulito/documentato `README.md`, `docs/level_c_research.md`, `data/README.md`, `.gitignore`, commenti di `environment-multiasset.yml` e dipendenza `joblib`, senza toccare gli asset dei colleghi.
2. Aggiornare `docs/research_design.md` e `docs/decision_log.md` con un collegamento all'addendum C e il risultato storico, lasciando intatte le decisioni precedenti.
3. Implementare e testare il dispatcher `main --level C` in un **commit separato**, insieme al cambio coordinato `config.yaml`/test. Non alterare il percorso B congelato.
4. Verificare `python -m unittest discover -s tests -p 'test_*.py'` e ricalcolare soltanto i report 2010–2016. Esigere che la tabella A/B/C coincida con l'output già fornito.
5. Fare un prova di riproduzione su clone pulito: installazione su Python 3.11, generazione `processed` da `data/raw`, generazione controllata segnali Orchestrator, replica dei target C e del confronto. Registrare hash e versioni dei pacchetti.
6. Coordinare PR #1 e branch `clustering` con Giovanni; soltanto dopo il merge confermare il risultato combinato ed evitare refactoring contemporanei.
7. Validare la nuova PR con una GitHub Action verde. Non aprire il final test 2022+ in alcun passaggio.

### Comandi Git non distruttivi prima della PR

```bash
git fetch origin --prune
git branch -vv
git status --short
git diff --check
git diff --name-only origin/main...HEAD
git ls-files data

# Identifica se un eventuale file è tracciato o ignorato senza modificarlo:
git ls-files src/agents/config/
git check-ignore -v data/processed/market_daily.csv src/agents/config/random_forest_params_spy.joblib || true
```

Usare sempre `git add <path>...` sui file approvati; mai stage indiscriminato di `notebooks/`, `reports/`, joblib, JSON dei modelli, download rinnovati o script diagnostici privati. La documentazione finale e i test devono essere verificati sul **commit remoto effettivo**, non soltanto sui file locali.

## Checklist decisionale prima del merge su `main`

- [ ] PR #1 approvata/integrata, oppure `modulo-C` presentata esplicitamente come PR stacked su `gianluca`.
- [ ] Nessuna modifica non revisionata ai raw CSV tracciati e ai file dei colleghi.
- [ ] README, Research Design e Decision Log puntano al nuovo Level C.
- [ ] `main.py --level C` supportato o, se approvato dal gruppo, esplicitamente dichiarato **non supportato** con runner C standalone come interfaccia ufficiale.
- [ ] `config.yaml` e test riflettono il contratto scelto, senza sbloccare OOS.
- [ ] Fresh clone ricrea i sei file processed e tutti e quattro i segnali di C.
- [ ] Provenienza dei segnali Orchestrator (fit, configurazioni, hash) registrata.
- [ ] Tutti i test della suite e l'eventuale CI risultano verdi dopo il merge.
- [ ] Riproduzione esatta dei nove risultati 2010–2016 e della baseline C_v2.
- [ ] Nessun test o reporting 2022+ eseguito.

**Esito:** codice e comparazioni utili e funzionanti nel contesto locale; **non ancora pronto a un merge definitivo** senza riconciliazione della CLI, della provenienza dei dati e dei segnali di C e delle dipendenze Git tra collaboratori.
