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
| [`04-ensemble-ph.ipynb`](04-ensemble-ph.ipynb) | Ensemble residual + LightGBM no pH (**a executar** — ver §5) | `../resultados/04-ensemble-ph/` (a gerar) |

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

## 5. Runbook do `04-ensemble-ph` (executar num PC com mais RAM)

O `04-ensemble-ph.ipynb` está pronto e revisado, mas **não executa nesta máquina** (OOM: a matriz de janelas ~1,2 GB + 288 modelos LightGBM em RAM estouram a memória disponível).

**Pré-requisitos no PC novo:**
- RAM livre ≥ 8 GB (recomendado 16 GB); CPU com 4+ núcleos; ~2 GB livres em disco (`lgbm_steps.pkl` tem dezenas de MB).
- Mesmo ambiente: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt`
  (o `requirements.txt` já inclui `torch` CPU e `lightgbm==4.7.0`).
- O notebook recarrega a régua de `resultados/02-lstnet-ph/modelos/lstnet_ph.pt` — faça checkout do commit `b9ce976` (ou posterior) antes, para o checkpoint existir.

**Rodar (tempo típico 15–25 min em CPU):**
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/04-ensemble-ph.ipynb
```

**Verificar se deu certo:**
- `resultados/04-ensemble-ph/metricas_baseline.csv` e `metricas_holdout.csv` existem, com linhas `persistencia, sazonal_naive_288, media_movel_288, lstnet, lgbm, dlres, ens`.
- No output da §8: `pesos ensemble (nnls na val)` impresso (4 pesos) e `ENS teste` / `ENS holdout` com MAE.
- `resultados/04-ensemble-ph/figs/` tem `01-eda.png` … `06-holdout-dias.png`, `07-importancia-lgbm.png`; `modelos/` tem `lgbm_steps.pkl`, `dlinear_res_ph.pt`, `ensemble.json`, `normalizacao.json`.
- Réguas a bater (pH): rolante LSTNet **0,0456**, holdout LSTNet **0,0446**. O notebook imprime `este exp: <modelo> = <MAE>` nas duas tabelas — é o veredito.

**Se OOM mesmo no PC novo:** alternativas por ordem de economia de RAM — (1) fechar apps antes; (2) no setup, `LGB_EST 150→100`; (3) `LGB_STRIDE 2→4`; (4) rodar interativo por seções (§7 salva o `.pkl` antes da §8 pesada).

## 6. Pós-execução do 04 (checklist)

1. Criar `resultados/04-ensemble-ph/README.md` pelo template abaixo (preencher os `X,XXXX` com os CSVs + outputs do notebook).
2. Adicionar a linha do `04-ensemble-ph` em `resultados/README.md` (tabela + quadro das réguas) e o checkbox §7 no `README.md` principal.
3. Trocar nesta tabela a célula `(a executar — ver §5)` pelo link `../resultados/04-ensemble-ph/` e apagar as seções §5–§6 daqui (o runbook cumpriu seu papel).
4. Commit único `feat(model): ...` + proposta de push (regra: sem commit/push sem confirmação explícita).

**Template do `resultados/04-ensemble-ph/README.md`:**
```markdown
# Experimento 04 — Ensemble residual + LightGBM no pH (EF01)

Ensemble com piso no sazonal-naive + LightGBM com lags, no mesmo desenho do [02-lstnet-ph](../02-lstnet-ph/).
Artefatos gerados por `notebooks/04-ensemble-ph.ipynb` (executado de ponta a ponta, 0 erros).
Reproduzir: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/04-ensemble-ph.ipynb`
(precisa de `torch` CPU e `lightgbm` — ver `requirements.txt`; ~15–25 min em CPU; RAM livre ≥ 8 GB).

## Configuração do experimento

- **Série:** pH, estação EF01 Mogi das Cruzes, passo de 5 min
- **Protocolo travado (igual ao 00–03):** avaliação em janelas `L=8640 → H=288` · split 70/15/15 (teste 2.204, alvos 14/08 → 21/08) · **holdout puro** 21/08 → 31/08 (2.594 origens) + 10 origens diárias · limpeza idêntica
- **Modelos:** **lgbm** — 288 `LGBMRegressor` (150 árvores), um por passo do horizonte, alvo = resíduo `r = Y − snaive(X)`, 27 features (lags recentes 1–144 + sazonais 287/288/289/576/2016, média/desvio da mesma fase em 7 dias, médias/desvios móveis, hora-do-dia do passo) · **dlres** — DLinear-5min no mesmo resíduo (sem +mu na desnormalização) · **lstnet (02)** recarregado (só inferência) · **ens** — NNLS (sazonal + 3) com pesos fitados só na val: PESOS_AQUI
- **Treino:** lgbm em 5.141 janelas (stride 2); dlres stride 4 + early stopping; **avaliação em todas as origens**

## Tabela principal — teste rolante (2.204 origens)

| modelo | MAE | RMSE | MAPE | sMAPE |
|---|---|---|---|---|
| ... | X,XXXX | X,XXXX | X,XXXX | X,XXXX |

(copiar de `metricas_baseline.csv`; ordem: melhor MAE primeiro)

## Tabela do holdout — 10 dias previstos (alvos nunca treinados)

| modelo | MAE | RMSE | MAPE | sMAPE |
|---|---|---|---|---|
| ... | X,XXXX | X,XXXX | X,XXXX | X,XXXX |

(copiar de `metricas_holdout.csv`)

MAE por dia previsto: copiar a tabela `por_dia` do output da §9 (10 linhas × modelos).

## Figuras — o que cada uma mostra

### `01-eda.png`, `02-limpeza.png`, `03-stl.png` — mesmos do 00–03

### `04-forecasts.png` — 3 origens do teste rolante (real, sazonal, lstnet(02), lgbm, ens)

![Forecasts](figs/04-forecasts.png)

### `05-mae.png` — barras de MAE no teste rolante

![MAE](figs/05-mae.png)

### `06-holdout-dias.png` — os 10 dias previstos × real (título mostra MAE ens vs saz)

![Holdout diário](figs/06-holdout-dias.png)

### `07-importancia-lgbm.png` — importância média das features (gain, 288 modelos) — NOVO

![Importância](figs/07-importancia-lgbm.png)

- Leitura: quais lags o modelo usa (esperado: sazonais 287/288/289 + recentes se a forma intra-hora importar).

## Arquivos

| Arquivo | O quê |
|---|---|
| `metricas_baseline.csv` | Tabela do teste rolante em CSV |
| `metricas_holdout.csv` | Tabela do holdout diário em CSV |
| `modelos/lgbm_steps.pkl` | 288 regressores LightGBM (dezenas de MB) |
| `modelos/dlinear_res_ph.pt` | DLinear do resíduo (state_dict) |
| `modelos/ensemble.json` | Pesos NNLS (+ modo) |
| `modelos/normalizacao.json` | Modo do experimento |
| `figs/` | As 7 figuras explicadas acima |

## Leitura dos resultados

1. Veredito vs réguas (pH: LSTNet 0,0456 / 0,0446): ESCREVER — quem vence em cada tabela?
2. `dlres` ≈ sazonal-naive? (esperado: piso; se divergir muito, reportar como bug, cf. histórico do exp.)
3. Features top do LightGBM: o que o modelo usa — nível (rollmean) ou fase (lag288/hora)?
4. Pesos do ensemble: o NNLS zerou algum membro? O que isso diz?
5. Fila: replicar vencedor no OD (`04b`) ou não, e por quê.
```
