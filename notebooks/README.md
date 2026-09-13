# Notebooks — como rodar

| Notebook | Experimento | Resultados |
|---|---|---|
| [`00-baseline-ph.ipynb`](00-baseline-ph.ipynb) | Baselines no pH, L=30d/H=1d + holdout de 10 dias | [`../resultados/00-baseline-ph/`](../resultados/00-baseline-ph/) |
| [`00b-baseline-od.ipynb`](00b-baseline-od.ipynb) | Mesmo protocolo no OD, segmento 01/06→21/07 | [`../resultados/00b-baseline-od/`](../resultados/00b-baseline-od/) |
| [`01-lstm-ph.ipynb`](01-lstm-ph.ipynb) | LSTM-h no pH, mesmo protocolo (grade horária + ×12) | [`../resultados/01-lstm-ph/`](../resultados/01-lstm-ph/) |
| [`01b-lstm-od.ipynb`](01b-lstm-od.ipynb) | Mesmo método no OD, segmento 01/06→21/07 | [`../resultados/01b-lstm-od/`](../resultados/01b-lstm-od/) |
| [`02-lstnet-ph.ipynb`](02-lstnet-ph.ipynb) | LSTNet nativo 5 min no pH (nova régua) | [`../resultados/02-lstnet-ph/`](../resultados/02-lstnet-ph/) |
| [`02b-lstnet-od.ipynb`](02b-lstnet-od.ipynb) | Mesmo método no OD, segmento 01/06→21/07 | [`../resultados/02b-lstnet-od/`](../resultados/02b-lstnet-od/) |
| [`03-patchtst-ph.ipynb`](03-patchtst-ph.ipynb) | PatchTST + DLinear no pH (régua segue LSTNet) | [`../resultados/03-patchtst-ph/`](../resultados/03-patchtst-ph/) |
| [`03b-patchtst-od.ipynb`](03b-patchtst-od.ipynb) | Os três no OD, segmento 01/06→21/07 (régua segue sazonal) | [`../resultados/03b-patchtst-od/`](../resultados/03b-patchtst-od/) |

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
