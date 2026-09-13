# Notebooks — como rodar (regime anual: treino 2024, benchmark 2025)

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
  --ExecutePreprocessor.timeout=2400 notebooks/00-baseline-ph.ipynb
```
Troque o nome do arquivo. Tempos típicos em 12c livres: baselines (ARIMA + Prophet)
15–30 min; LSTNet/PatchTST ~10 min; ensembles ~5 min; 08-benchmark ~10 min.
Rode **um notebook por vez** (12c/23 GB estouram com jobs concorrentes);
para runs compartilhando a máquina, limite threads (`OMP_NUM_THREADS=4`).

## 3. O que cada execução gera

Tudo cai em `resultados/<experimento>/` (criado automaticamente):
`metricas_treino.csv`, `metricas_val.csv`, `metricas_val_diaria.csv`,
`metricas_por_dia.csv`, `modelos/` (checkpoints `.pt`, ARIMA `.pkl`, Prophet `.json`,
LightGBM `.pkl` gitignored) e `figs/`.

## 4. Problemas comuns

- **Prophet pulado:** CmdStan ausente — rode
  `.venv/bin/python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"` (precisa de `g++`/`make`) e reexecute.
- **ARIMA lento:** é esperado (reestimação por origem); ajuste `ARIMA_STRIDE` na célula de setup.
- **`ModuleNotFoundError`:** kernel errado — confira no canto superior direito do Jupyter se é o do `.venv`.
- **Reexecutar apaga outputs antigos:** o `nbconvert --inplace` sobrescreve métricas, modelos e figuras. Para comparar versões, copie `resultados/<exp>/` antes.
- **Dependências entre notebooks:** 04/05 recarregam o LSTNet do 02/03, 06/07 recarregam o 02/03 (assert com mensagem clara se ausente); o 08 exige os 14 checkpoints de 00–07 — rode na ordem numérica (00/01 → 02/03 → 04/05 → 06/07) antes dele.
