# Resultados — índice de experimentos (regime anual: treino 2024, benchmark 2025)

Cada experimento tem sua pasta `NN-<nome>/` com `README.md` próprio
(tabela de métricas, artefatos e leitura dos resultados), `metricas*.csv`,
`modelos/` e `figs/`. Não commitar fora de pastas de experimento.
Os checkpoints de `modelos/` (`.pt`, `.pkl`, `.pkl.gz`, Prophet `.json`) **não vão ao git** —
vivem no GitHub Release [`modelos-v1`](https://github.com/marcos2872/temporal-model-prediction/releases/tag/modelos-v1)
(baixe com `bash scripts/baixar_modelos.sh`); só `normalizacao.json`/`ensemble.json` ficam versionados.

Protocolo: janelas `L=8640 → H=288` (v1, exps 00–08) · `L=2304 → H=288` com
purge/embargo (v2, exps 10–18) · treino = 2024 menos fatias de val (v1: 4 fatias —
19–28/abr, 20–29/jul, 15–24/set, 20–24/nov, uma por estação; v2: 5 fatias, inclui
13–22/dez) · benchmark = 2025 inteiro (08 = v1, 18 = v2, só inferência).
Números abaixo = MAE na val, salvo indicação. O probe barato em 4 períodos de
2025 vive em [`../benchmark-2025/`](../benchmark-2025/) (hierarquia igual à do
rolante; o rolante 08/18 decide).

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
| [`10-v2-baseline-ph/`](10-v2-baseline-ph/) | Baselines no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias incl. dez) — **RÉGUA v2 pH** | **sazonal 0,0406**; MM 0,0497; ARIMA 0,0602 (fallback total nos dias-âncora); Prophet 0,1463 (rodou, colapsa) |
| [`11-v2-baseline-od/`](11-v2-baseline-od/) | Baselines no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias incl. dez) — **RÉGUA v2 OD** | **sazonal 0,1736**; MM 0,3631; ARIMA 0,4282 (fallback total nos dias-âncora); Prophet 0,5243 (rodou, colapsa) |
| [`12-v2-lstnet-ph/`](12-v2-lstnet-ph/) | LSTNet no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, conv 3→8 solar+Fourier, 5 seeds) | **lstnet 0,0365±0,0004**, −10,1% sobre o sazonal v2 0,0406 (seeds 0,0360–0,0370) |
| [`13-v2-lstnet-od/`](13-v2-lstnet-od/) | LSTNet no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, conv 3→8 solar+Fourier, 5 seeds) | **lstnet 0,1467±0,0017**, −15,5% sobre o sazonal v2 0,1736 (seeds 0,1450–0,1495) |
| [`14-v2-patchtst-ph/`](14-v2-patchtst-ph/) | PatchTST + DLinear no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, univariado, 5 seeds por modelo) | **patchtst 0,0352±0,0003 · dlinear 0,0356±0,0001**, ambos abaixo do lstnet v2 0,0365 e do sazonal v2 0,0406 (overfit do v1-04 curado) |
| [`15-v2-patchtst-od/`](15-v2-patchtst-od/) | PatchTST + DLinear no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, univariado, 5 seeds por modelo) | **dlinear 0,1482±0,0027 · patchtst 0,1556±0,0028**, dlinear empata o lstnet v2 0,1467 e ambos batem o sazonal v2 0,1736 (ordem do v1-05 mantida) |
| [`16-v2-ensemble-ph/`](16-v2-ensemble-ph/) | Ensemble v2 no pH 2024 (sazonal + LSTNet-12 + LGBM-nativo + DLinear-res + NNLS fit/report, protocolo v2) | **honesto (dez): ens 0,0324, dlres 0,0302** · in-sample (fit 1–4): ens 0,0346; NNLS sazonal 0,0694 · lstnet 0,7048 · lgbm 0,0889 · dlres 0,1366 |
| [`17-v2-ensemble-od/`](17-v2-ensemble-od/) | Ensemble v2 no OD 2024 (sazonal + LSTNet-13 + LGBM-nativo + DLinear-res + NNLS fit/report, protocolo v2) | **honesto (dez): ens 0,1733, dlres 0,1842 ≈ lstnet 0,1848** · in-sample (fit 1–4): ens 0,1234; NNLS sazonal 0,0038 · lstnet 0,6771 · lgbm 0,0 · dlres 0,3216 (= v1-07) |
| [`18-v2-benchmark-2025/`](18-v2-benchmark-2025/) | Campeões v2 × 2025 intocado (+ lag-365, protocolo v2) — **veredito v1×v2** | pH: patchtst 0,0465, ens 0,0470, saz 0,0565, lag365 0,369, prophet 1,17 · OD: patchtst 0,2056, ens 0,2133, saz 0,2519, lag365 1,14, prophet 5,34 |

Quadro das réguas (benchmark 2025 rolante, critério principal): **v1 → ensemble
pH 0,0509 · ensemble OD 0,2107** (servidos na API até decisão do usuário) ·
**v2 → PatchTST pH 0,0465 · PatchTST OD 0,2056** (coroados no 18, troca pendente —
ver `18-v2-benchmark-2025/README.md` § Veredito, item 5).
