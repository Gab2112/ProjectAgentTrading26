# Organizzazione e provenienza dei dati

Il repository conserva intenzionalmente in Git **soltanto gli snapshot delle sorgenti**. I CSV in `raw/market/` sono osservazioni storiche degli ETF ottenute da yfinance (prezzi OHLC rettificati); `raw/macro/` contiene serie macroeconomiche FRED. L'universo investibile corrente è **SPY, IEF, GLD e SHY**. Il file TLT resta come archivio della precedente ricerca sui Treasury e non è allocato dalle strategie A/B/C.

| Cartella | Versionata in Git? | Origine o creazione | Funzione |
|---|---|---|---|
| `raw/market/` | Sì | Snapshot CSV ETF | Prezzi storici e input riproducibili |
| `raw/macro/` | Sì | Snapshot CSV FRED | Feature macro e regimi; disponibilità *point-in-time* **non verificata** |
| `processed/` | **No** | `python main.py --level ALL` | Dataset normalizzati e calendario settimanale |
| `signals/` | **No** | `python -m src.agents.train ALL` e costruttori dei portafogli | Segnali Orchestrator necessari per C e altri segnali generati |
| `results/` | **No** | Script di backtest e confronto | NAV, costi, diagnostiche e risultati della ricerca |

I sei file attualmente prodotti in `processed/` sono:

- `market_daily.csv`
- `weekly_decision_dates.csv`
- `market_daily_quality.csv`
- `master_weekly.csv`
- `data_quality_summary.csv`
- `decision_features_training.csv`

**Non vanno aggiunti a Git** soltanto per rendere visibili le cartelle: sono prodotti derivati, che la pipeline ricrea quando necessari. Git non conserva le directory vuote.

## Regole di riproducibilità e provenienza

1. Conservare invariati gli snapshot raw per ciascun esperimento. Eventuali revisioni del provider possono cambiare prezzi e rettifiche: un nuovo download costituisce una **nuova versione del dataset**, non una sostituzione silenziosa.
2. Il confronto C sul 2010–2016 richiede quattro file `signals_orchestrator_{spy,ief,gld,shy}.csv`, con colonne `signal` e `confidence` e date settimanali coincidenti. Non sono tracciati in Git. Prima del rilascio finale, registrare in un manifesto: commit del codice, hash dei dati sorgenti, configurazione del training, versioni delle librerie, hash dei modelli e hash dei segnali.
3. Non aggiungere automaticamente `src/agents/config/*.joblib` o nuovi JSON di parametri senza una politica condivisa per gli artefatti dei modelli. I JSON storici già tracciati vanno conservati fino alla decisione del responsabile della migrazione.
4. Non attivare o valutare il test finale dal **2022** durante pulizia, test di avvio o confronti di sviluppo. La validation 2017–2021 di B_v2 è già stata consultata; per C manca ancora una valutazione indipendente.

Per i comandi di esecuzione e i limiti dei modelli, vedere `README.md` nella radice e `docs/level_c_research.md`.
