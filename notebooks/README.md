# Notebooks — como rodar (regime anual: treino 2024, benchmark 2025)

| Notebook | Experimento | Resultados |
|---|---|---|
| [`00-baseline-ph.ipynb`](00-baseline-ph.ipynb) | Baselines no pH 2024 + val 4 fatias | [`../resultados/00-baseline-ph/`](../resultados/00-baseline-ph/) |
| [`00b-baseline-od.ipynb`](00b-baseline-od.ipynb) | Mesmo protocolo no OD 2024 | [`../resultados/00b-baseline-od/`](../resultados/00b-baseline-od/) |
| [`02-lstnet-ph.ipynb`](02-lstnet-ph.ipynb) | LSTNet nativo 5 min no pH (nova régua) | [`../resultados/02-lstnet-ph/`](../resultados/02-lstnet-ph/) |
| [`02b-lstnet-od.ipynb`](02b-lstnet-od.ipynb) | Mesmo método no OD (nova régua) | [`../resultados/02b-lstnet-od/`](../resultados/02b-lstnet-od/) |
| [`03-patchtst-ph.ipynb`](03-patchtst-ph.ipynb) | PatchTST + DLinear no pH (régua segue LSTNet) | [`../resultados/03-patchtst-ph/`](../resultados/03-patchtst-ph/) |
| [`03b-patchtst-od.ipynb`](03b-patchtst-od.ipynb) | Os três no OD (régua segue LSTNet) | [`../resultados/03b-patchtst-od/`](../resultados/03b-patchtst-od/) |
| [`04-ensemble-ph.ipynb`](04-ensemble-ph.ipynb) | Ensemble residual + LightGBM no pH (nova régua) | [`../resultados/04-ensemble-ph/`](../resultados/04-ensemble-ph/) |
| [`04b-ensemble-od.ipynb`](04b-ensemble-od.ipynb) | Mesmo método no OD (nova régua) | [`../resultados/04b-ensemble-od/`](../resultados/04b-ensemble-od/) |
| [`08-benchmark-2025.ipynb`](08-benchmark-2025.ipynb) | Todos os campeões × 2025 intocado + lag-365 (só inferência) | [`../resultados/08-benchmark-2025/`](../resultados/08-benchmark-2025/) |
| [`08-transfer-ph-od.ipynb`](08-transfer-ph-od.ipynb) | Campeões × dados novos fev–abr/2026, só inferência (ponte CTX=2016 exata) | [`../resultados/08-transfer-ph-od/`](../resultados/08-transfer-ph-od/) |

## 1. Ambiente (uma vez)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

O `requirements.txt` (na raiz) já deixa o Prophet funcionando — o CmdStan compila
na primeira execução (~5 min, só na primeira vez).

## 2. Rodar

**Interativo (recomendado para explorar):**
```bash
source .venv/bin/activate
jupyter lab   # ou: jupyter notebook
```
Abra o `.ipynb` e selecione o kernel do `.venv` (`Python 3 (.venv)`).
Se o kernel não aparecer: com o `.venv` ativo, rode
`.venv/bin/python -m ipykernel install --user --name temporal-model` uma vez.

**Via terminal (reproduzível, regenera tudo):**
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/00-baseline-ph.ipynb
```
Troque o nome do arquivo para o `00b` ou `01`. Tempo típico: 5–10 min
(ARIMA reestimado por origem + ajuste do Prophet; o `01` treina o LSTM em CPU, ~5 min; o `02` treina o LSTNet, ~10 min).

## 3. O que cada execução gera

Tudo cai em `resultados/<experimento>/` (criado automaticamente):
`metricas_baseline.csv`, `metricas_holdout.csv`, `modelos/` (ARIMA `.pkl`, Prophet `.json`) e `figs/`.

## 4. Problemas comuns

- **Prophet pulado na §8:** CmdStan ausente — rode
  `.venv/bin/python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"` (precisa de `g++`/`make`) e reexecute.
- **ARIMA lento:** é esperado (reestimação por origem); reduza `ARIMA_STRIDE` na §1 de setup para mais origens, ou aumente para menos.
- **`ModuleNotFoundError`:** kernel errado — confira no canto superior direito do Jupyter se é o do `.venv`.
- **Reexecutar apaga outputs antigos:** o `nbconvert --inplace` sobrescreve métricas, modelos e figuras. Para comparar versões, copie `resultados/<exp>/` antes.
