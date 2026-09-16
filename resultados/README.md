# Resultados — índice de experimentos (regime anual: treino 2024, benchmark 2025)

Cada experimento tem sua pasta `NN-<nome>/` com `README.md` próprio
(tabela de métricas, artefatos e leitura dos resultados), `metricas*.csv`,
`modelos/` e `figs/`. Não commitar fora de pastas de experimento.
Os checkpoints de `modelos/` (`.pt`, `.pkl`, `.pkl.gz`, Prophet `.json`) **não vão ao git** —
vivem no GitHub Release [`modelos-v1`](https://github.com/marcos2872/temporal-model-prediction/releases/tag/modelos-v1)
(baixe com `bash scripts/baixar_modelos.sh`); só `normalizacao.json`/`ensemble.json` ficam versionados.

Protocolo: janelas `L=8640 → H=288` · treino = 2024 menos 4 fatias de val (19–28/abr,
20–29/jul, 15–24/set, 20–24/nov, uma por estação) · benchmark = 2025 inteiro (08).
Números abaixo = MAE na val, salvo indicação.

| Pasta | Experimento | Métricas (val) |
|---|---|---|
| [`00-baseline-ph/`](00-baseline-ph/) | Baselines no pH 2024 (persistência, sazonal-naive, MM, ARIMA-h, Prophet) | sazonal 0,0421; MM 0,0468; ARIMA 0,0565 (≈persist, fallback); Prophet 0,1703 (colapsa) |
| [`01-baseline-od/`](01-baseline-od/) | Mesmo protocolo no OD 2024 | sazonal 0,1579; MM 0,3260; ARIMA 0,3893; Prophet 0,4091 (colapsa) |
| [`02-lstnet-ph/`](02-lstnet-ph/) | LSTNet nativo 5 min no pH (conv + GRU + skip p=1d + AR-288 + RevIN) — **NOVA RÉGUA pH** | **lstnet 0,0373**, −11,4% sobre o sazonal 0,0421 |
| [`03-lstnet-od/`](03-lstnet-od/) | Mesmo método no OD — **NOVA RÉGUA OD (1ª vez)** | **lstnet 0,1380**, −12,6% sobre o sazonal 0,1579 |
| [`04-patchtst-ph/`](04-patchtst-ph/) | PatchTST + DLinear no pH, régua 02 recarregada — régua segue LSTNet | lstnet 0,0373; dlinear 0,0394; patchtst 0,0414 (overfita); sazonal 0,0421 |
| [`05-patchtst-od/`](05-patchtst-od/) | Os três no OD — régua segue LSTNet | lstnet 0,1380; patchtst 0,1432; dlinear 0,1435; sazonal 0,1579 |
| [`06-ensemble-ph/`](06-ensemble-ph/) | Ensemble (piso + 288 LGBM + DLinear-res + NNLS) no pH — **NOVA RÉGUA pH** | **ens 0,0357** (−4,3% sobre lstnet); NNLS zera o LGBM |
| [`07-ensemble-od/`](07-ensemble-od/) | Mesmo método no OD — **NOVA RÉGUA OD** | **ens 0,1325** (−4,0% sobre lstnet); dlres com peso 0,30 |
| [`08-benchmark-2025/`](08-benchmark-2025/) | Todos os campeões × 2025 intocado (+ lag-365) — **veredito final** | pH: ens 0,0509, lstnet 0,0529, saz 0,0597, lag365 0,464, prophet 1,90 · OD: ens 0,2107, lstnet 0,2127, saz 0,2714, lag365 0,925, prophet 5,04 |

Quadro das réguas (benchmark 2025, critério principal): **pH → ensemble 0,0509** · **OD → ensemble 0,2107**.
