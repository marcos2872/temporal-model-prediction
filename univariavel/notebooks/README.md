# Notebooks — índice e cadeia (regime anual: treino 2024, benchmark 2025)

> **Como rodar (ambiente, comandos, troubleshooting): ver
> [`COMO-RODAR.md`](../../COMO-RODAR.md) (canônico).** Este arquivo é só o índice
> notebook→experimento e a ordem da cadeia. Trilha multivariável: notebooks
> `multivariavel/notebooks/M1…M4`, protocolo em `multivariavel/PLANO.md`.

| Notebook | Experimento | Resultados |
|---|---|---|
| [`00-baseline-ph.ipynb`](00-baseline-ph.ipynb) | Baselines no pH 2024 + val 4 fatias | [`../resultados/00-baseline-ph/`](../resultados/00-baseline-ph/) |
| [`01-baseline-od.ipynb`](01-baseline-od.ipynb) | Mesmo protocolo no OD 2024 | [`../resultados/01-baseline-od/`](../resultados/01-baseline-od/) |
| [`02-lstnet-ph.ipynb`](02-lstnet-ph.ipynb) | LSTNet nativo 5 min no pH (nova régua) | [`../resultados/02-lstnet-ph/`](../resultados/02-lstnet-ph/) |
| [`03-lstnet-od.ipynb`](03-lstnet-od.ipynb) | Mesmo método no OD (nova régua) | [`../resultados/03-lstnet-od/`](../resultados/03-lstnet-od/) |
| [`04-patchtst-ph.ipynb`](04-patchtst-ph.ipynb) | PatchTST + DLinear no pH (régua segue LSTNet) | [`../resultados/04-patchtst-ph/`](../resultados/04-patchtst-ph/) |
| [`05-patchtst-od.ipynb`](05-patchtst-od.ipynb) | Os três no OD (régua segue LSTNet) | [`../resultados/05-patchtst-od/`](../resultados/05-patchtst-od/) |
| [`06-ensemble-ph.ipynb`](06-ensemble-ph.ipynb) | Ensemble residual + LightGBM no pH (nova régua) | [`../resultados/06-ensemble-ph/`](../resultados/06-ensemble-ph/) |
| [`07-ensemble-od.ipynb`](07-ensemble-od.ipynb) | Mesmo método no OD (nova régua) | [`../resultados/07-ensemble-od/`](../resultados/07-ensemble-od/) |
| [`08-benchmark-2025.ipynb`](08-benchmark-2025.ipynb) | Todos os campeões × 2025 intocado + lag-365 (só inferência) | [`../resultados/08-benchmark-2025/`](../resultados/08-benchmark-2025/) |
| [`09-analises-pos-benchmark.ipynb`](09-analises-pos-benchmark.ipynb) | Análises pós-benchmark Fase 1: curva MAE(h)+MASE, Diebold-Mariano, climatologia (só inferência, sem treino; escreve em `../resultados/08-benchmark-2025/`) | [`../resultados/08-benchmark-2025/`](../resultados/08-benchmark-2025/) |
| [`10-v2-baseline-ph.ipynb`](10-v2-baseline-ph.ipynb) | Baselines no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias incl. dez) | [`../resultados/10-v2-baseline-ph/`](../resultados/10-v2-baseline-ph/) |
| [`11-v2-baseline-od.ipynb`](11-v2-baseline-od.ipynb) | Baselines no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias incl. dez) | [`../resultados/11-v2-baseline-od/`](../resultados/11-v2-baseline-od/) |
| [`12-v2-lstnet-ph.ipynb`](12-v2-lstnet-ph.ipynb) | LSTNet no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, conv 3→8, 5 seeds) | [`../resultados/12-v2-lstnet-ph/`](../resultados/12-v2-lstnet-ph/) |
| [`13-v2-lstnet-od.ipynb`](13-v2-lstnet-od.ipynb) | LSTNet no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, conv 3→8, 5 seeds) | [`../resultados/13-v2-lstnet-od/`](../resultados/13-v2-lstnet-od/) |
| [`14-v2-patchtst-ph.ipynb`](14-v2-patchtst-ph.ipynb) | PatchTST + DLinear no pH 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, 5 seeds por modelo) | [`../resultados/14-v2-patchtst-ph/`](../resultados/14-v2-patchtst-ph/) |
| [`15-v2-patchtst-od.ipynb`](15-v2-patchtst-od.ipynb) | PatchTST + DLinear no OD 2024 em protocolo v2 (L=2304, purge/embargo, 5 fatias, 5 seeds por modelo) | [`../resultados/15-v2-patchtst-od/`](../resultados/15-v2-patchtst-od/) |
| [`16-v2-ensemble-ph.ipynb`](16-v2-ensemble-ph.ipynb) | Ensemble v2 no pH 2024 em protocolo v2 (sazonal + LSTNet-12 + LGBM-nativo + DLinear-res + NNLS fit/report) | [`../resultados/16-v2-ensemble-ph/`](../resultados/16-v2-ensemble-ph/) |
| [`17-v2-ensemble-od.ipynb`](17-v2-ensemble-od.ipynb) | Ensemble v2 no OD 2024 em protocolo v2 (sazonal + LSTNet-13 + LGBM-nativo + DLinear-res + NNLS fit/report) | [`../resultados/17-v2-ensemble-od/`](../resultados/17-v2-ensemble-od/) |
| [`18-v2-benchmark-2025.ipynb`](18-v2-benchmark-2025.ipynb) | Campeões v2 × 2025 intocado + lag-365 em protocolo v2 (só inferência) | [`../resultados/18-v2-benchmark-2025/`](../resultados/18-v2-benchmark-2025/) |

> Exceção de nome: o `09-analises-pos-benchmark.ipynb` está fora do padrão
> `NN-<modelo>-<variavel>` (ex.: `00-baseline-ph`, `10-v2-baseline-ph`) porque
> não é um experimento com treino — é a
> consolidação versionada das três análises pós-benchmark (Fase 1).

## 1. Ambiente

Ver [`COMO-RODAR.md`](../../COMO-RODAR.md) §1 (criar `.venv`, instalar
`requirements.txt`, CmdStan para o Prophet). Sempre com o kernel do `.venv`.

## 2. Rodar

Ver [`COMO-RODAR.md`](../../COMO-RODAR.md) §2 (interativo × `nbconvert --inplace`,
um notebook por vez, `nbconvert` sobrescreve os artefatos — copie
`../resultados/<exp>/` antes de comparar versões).

**Ordem e dependências:** 00/01 → 02/03 → 04/05 → 06/07 → 08 (cadeia v1), depois
10/11 → 12/13 → 14/15 → 16/17 → 18 (cadeia v2). Dependências v1: 04/05 recarregam
o LSTNet do 02/03, 06/07 recarregam o 02/03 (assert com mensagem clara se ausente);
o 08 exige os 14 checkpoints de 00–07 — e o 18, os de 10–17. Na v2 vale o
encadeamento análogo (ver o README de cada experimento 10–18).

## 3. O que cada execução gera

Tudo cai em `../resultados/<experimento>/` (criado automaticamente):
`metricas_treino.csv`, `metricas_val.csv`, `metricas_val_diaria.csv`,
`metricas_por_dia.csv`, `modelos/` (checkpoints `.pt`, ARIMA `.pkl`, Prophet `.json`,
LightGBM `.pkl.gz`) e `figs/`. Os checkpoints de `modelos/` **não vão ao git** —
vivem no GitHub Release [`modelos-v1`](https://github.com/marcos2872/temporal-model-prediction/releases/tag/modelos-v1);
para reutilizar sem retreinar, baixe com `bash scripts/baixar_modelos.sh` na raiz do repo (extrai para `univariavel/resultados/*/modelos/`).

## 4. Problemas comuns

Ver [`COMO-RODAR.md`](../../COMO-RODAR.md) §2 (Prophet pulado, ARIMA lento,
`ModuleNotFoundError`, reexecução que apaga outputs).
