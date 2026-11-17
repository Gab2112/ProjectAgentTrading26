# Level C: consenso tra momentum e agenti AI

**Historical research snapshot, 4 November 2026.** This file retains the original
2010–2016 in-sample Level C experiment and its historical Italian discussion.
The original final OOS Core-Satellite evaluation was subsequently completed for
2022–2026. Therefore references below to a still-locked final test or pending
work are **statements of the historical November 4 status, not current tasks**.
For the registered candidate and subsequent SMA200 extension see
[`core_satellite_final_2026.md`](core_satellite_final_2026.md) and
[`regime_adaptive_exploratory_2026.md`](regime_adaptive_exploratory_2026.md).
These later results must not be inserted into the old nine-portfolio table.

## Ipotesi

Un segnale prodotto dall'Orchestrator può filtrare le posizioni della strategia deterministica B_v2, migliorando rendimento e protezione dal drawdown a parità di regole di esecuzione. L'Orchestrator utilizza modelli e agenti sottostanti, tra cui RF/SVM, ma la sua parametrizzazione è stata ottimizzata sul periodo oggetto del confronto: **i risultati di Level C sono in-sample**, indipendentemente dall'eventuale generazione OOF dei segnali dei singoli agenti.

## Costruzione dei portafogli

**B_v2 di riferimento.** Budget dipendenti da quattro regimi macro VIXCLS/T10YIE; ogni ETF viene acquistato solo se i momentum a 21 e 63 sedute sono entrambi positivi. Le posizioni Short non sono consentite. La quota residua rimane liquida.

**C_v1, Long/Flat.** Per ogni ETF, dopo il Close della seduta decisionale, il portafoglio apre un Long soltanto se momentum B_v2 = +1 **e** segnale Orchestrator = +1 con confidence > 0. Negli altri casi assume peso zero. L'Orchestrator non modifica il budget macro; funziona come secondo filtro di consenso.

**C_v2, Long/Short/Flat.** Long quando momentum = +1 e agente = +1; Short quando momentum = −1 e agente = −1, sempre con confidence > 0; altrimenti Flat. Risultano 44 settimane su 365 con almeno uno Short.

In entrambi i casi il budget di regime moltiplica i segnali e viene applicato **una sola volta** il cap di volatilità stimata (covarianza 126 sedute; soglia annuale 8%). Esposizione lorda complessiva massima 100%. Per C_v2, liquidità contabile e capitale inutilizzato non sono intercambiabili.

## Confronto omogeneo dei nove portafogli

Le nove strategie A1/A2/B_v1/B_v2/C_v1/C_v2 ripartono tutte da **100.000 USD** sul calendario comune 2010–2016. Segnali settimanali dopo Close, esecuzione successivo Open, NAV giornaliero, costi 2 bps commissioni + 1 bp slippage sul nozionale, borrow fee dell'1% annualizzato per i portafogli Short, risk-free diagnostico zero. I portafogli Buy & Hold effettuano un solo ordine iniziale. Le 365 date di decisione arrivano al 30 dicembre 2016; l'ultima decisione che non ha un Open successivo entro il training non viene eseguita oltre la data di cutoff.

| Strategia | CAGR | Volatilità realizzata | Sharpe | Max DD | Esposizione lorda media |
|---|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 12,260% | 15,441% | 0,826 | −18,606% | 99,943% |
| 60/40 settimanale | 9,566% | 8,296% | **1,143** | −8,470% | 99,943% |
| Equal Weight settimanale | 4,688% | 5,802% | 0,819 | −7,571% | 99,943% |
| Equal Weight Buy & Hold | 5,031% | 6,076% | 0,838 | −7,678% | 99,943% |
| A2, controllo della volatilità | 4,653% | 5,780% | 0,816 | −7,571% | 99,943% |
| B_v1 | −1,073% | 4,925% | −0,194 | −17,553% | 66,680% |
| B_v2 | 1,366% | 3,621% | 0,393 | −7,011% | 51,939% |
| C_v1 | 1,793% | 3,001% | 0,607 | **−3,693%** | 46,302% |
| C_v2 | 2,091% | 3,272% | 0,649 | **−3,693%** | 48,864% |

**Interpretazione:** C_v2 supera B_v2 nel training, ma non il 60/40 sullo Sharpe né SPY Buy & Hold sul CAGR. Il drawdown inferiore di C è anche compatibile con la minore esposizione al mercato; non dimostra di per sé alpha predittivo. Il confronto non isolava a rischio identico l'effetto causale dei singoli filtri.

## Analisi di sensibilità dell'esposizione, non stress test di mercato

Sedici scenari: C_v1/C_v2 × fattori di scala dei pesi 1/1,25/1,5/2 × cap di volatilità 8%/12%. L'amplificazione avviene **prima** del cap, con esposizione lorda finale limitata al 100%, senza creare nuovi segnali e senza leva oltre il 100%.

| C_v2 con cap al 12% | CAGR | Volatilità realizzata | Sharpe | Max DD | Esposizione lorda media |
|---|---:|---:|---:|---:|---:|
| ×1 | 2,092% | 3,272% | **0,649** | **−3,693%** | 48,870% |
| ×1,25 | 2,381% | 3,915% | 0,621 | −4,599% | 59,418% |
| ×1,5 | 2,536% | 4,360% | 0,596 | −5,398% | 67,080% |
| ×2 | **2,780%** | 5,067% | 0,567 | −6,415% | 75,894% |

Aumentare l'esposizione produce un CAGR maggiore ma uno Sharpe più basso e drawdown più ampi. Alzare soltanto il cap dall'8% al 12% modifica poco il portafoglio perché molti pesi sono già piccoli **prima** dell'applicazione del cap. Non sono stati simulati shock esogeni di volatilità, costi straordinari o mercati di crisi non presenti nel dataset: chiamare questo esercizio 'stress test' sarebbe impreciso.

## Riproducibilità

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m src.run_training_comparison
python3 -m src.run_abc_training_comparison --save
python3 -m src.run_c_exposure_sensitivity
```

Prerequisiti: `data/processed/market_daily.csv`, `weekly_decision_dates.csv` e quattro `data/signals/signals_orchestrator_{spy,ief,gld,shy}.csv`. Tutti derivati dai raw CSV versionati; i segnali agenti richiedono un processo di training esplicito. Le directory `processed`, `signals`, `results` sono giustamente escluse da Git. Prima di considerare una ricostruzione indipendente, registrare versioni dei pacchetti, commit del codice e hash dei segnali/modelli.

## Decisioni e limiti scientifici

- Le versioni C_v1 e C_v2 originali sono conservate; nessuno scenario di sensibilità è promosso automaticamente a nuova politica.
- I risultati 2010–2016 sono **diagnostici in-sample**. RF/SVM OOF non significa Orchestrator OOS.
- La validation 2017–2021 è già stata vista per B_v2, ma la valutazione indipendente di C è ancora da definire con parametri congelati.
- Gli input macro non certificano i timestamp effettivi di pubblicazione o le vintage dei dati; costi, slippage, margini e borrow fees sono simulati.
- Il test finale 2022+ rimane bloccato.

Sarà eventualmente valutato a parte il clustering proposto da Giovanni, con mappatura cluster → budget fissata prima dell'osservazione dei risultati. Il riordino dei moduli interni a `src/` rimane separato dalla presente documentazione.
