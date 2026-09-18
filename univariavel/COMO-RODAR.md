# Como rodar — trilha univariada

Guia de execução: ambiente, notebooks, API e checkpoints **univariados**
(comandos a partir da raiz do repo). Trilha multivariável em
[`../multivariavel/COMO-RODAR.md`](../multivariavel/COMO-RODAR.md).
Teoria e bibliografia em [`METODOLOGIA.md`](../METODOLOGIA.md); apresentação
e resultados em [`README.md`](../README.md).

## 1. Ambiente (uma vez)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

O `requirements.txt` já deixa o Prophet funcionando — o CmdStan compila na
primeira execução (~5 min, só na primeira vez; precisa de `g++`/`make`).
Use sempre `.venv/bin/python` ou `.venv/bin/jupyter` (o `.venv/` é gitignored).

Dados: séries CETESB EF01 Mogi das Cruzes (pH + OD, passo 5 min) em `univariavel/dados/`
— treino = 2024, benchmark = 2025 (intocado até a avaliação final). Formato e
como ler em [`dados/README.md`](dados/README.md).

## 2. Notebooks (trilha univariada)

Detalhes por notebook em [`notebooks/README.md`](notebooks/README.md);
métricas e artefatos em [`resultados/`](resultados/README.md).

**Interativo (recomendado para explorar):**

```bash
source .venv/bin/activate
jupyter lab   # ou: jupyter notebook
```

Abra o `.ipynb` em `univariavel/notebooks/` e selecione o kernel do `.venv`
(`Python 3 (.venv)`). Se o kernel não aparecer: com o `.venv` ativo, rode
`.venv/bin/python -m ipykernel install --user --name temporal-model` uma vez.

**Via terminal (reproduzível, regenera tudo):**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 univariavel/notebooks/00-baseline-ph.ipynb
```

Troque o nome do arquivo (cada experimento tem o comando exato no seu
`resultados/<exp>/README.md`). Tempos típicos em 12c livres:
baselines (ARIMA + Prophet) 15–30 min; LSTNet/PatchTST ~10 min;
ensembles ~5 min; 08-benchmark ~10 min.

Regras:

- Rode **um notebook por vez** (12c/23 GB estouram com jobs concorrentes);
  compartilhando a máquina, limite threads (`OMP_NUM_THREADS=4`).
- **Ordem:** 00/01 → 02/03 → 04/05 → 06/07 → 08 (cadeia v1), depois
  10/11 → 12/13 → 14/15 → 16/17 → 18 (cadeia v2). Dependências: 04/05
  recarregam o LSTNet do 02/03; 06/07 recarregam o 02/03; o 08 exige os 14
  checkpoints de 00–07 (assert com mensagem clara se ausente).
- Cada execução escreve em `univariavel/resultados/<experimento>/`:
  `metricas_*.csv`, `modelos/` (checkpoints), `figs/`.
- **Trilha multivariável:** guia próprio em
  [`../multivariavel/COMO-RODAR.md`](../multivariavel/COMO-RODAR.md)
  (notebooks M1…M4, protocolo em `../multivariavel/PLANO.md`).
- **Reexecutar apaga outputs antigos:** o `nbconvert --inplace` sobrescreve
  métricas, modelos e figuras. Para comparar versões, copie
  `univariavel/resultados/<exp>/` antes.

**Problemas comuns:**

- **Prophet pulado:** CmdStan ausente — rode
  `.venv/bin/python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"`
  e reexecute. Os notebooks tratam Prophet como opcional e continuam sem ele.
- **ARIMA lento:** é esperado (reestimação por origem); ajuste `ARIMA_STRIDE`
  na célula de setup.
- **`ModuleNotFoundError`:** kernel errado — confira no canto superior direito
  do Jupyter se é o do `.venv`.

## 3. Checkpoints (GitHub Release — nunca vão ao git)

Os binários de `univariavel/resultados/*/modelos/` (`.pt`, `.pkl`, `.pkl.gz`,
Prophet `.json`, ARIMA `.pkl`) são gitignored; só `normalizacao.json` /
`ensemble.json` ficam versionados. Para reutilizar sem retreinar:

```bash
bash scripts/baixar_modelos.sh   # extrai para univariavel/resultados/*/modelos/
```

Os checkpoints vivem no GitHub Release
[`modelos-v1`](https://github.com/marcos2872/temporal-model-prediction/releases/tag/modelos-v1)
(verificação via `SHA256SUMS.txt` dentro do script).

## 4. API de previsão (local)

```bash
.venv/bin/uvicorn univariavel.app:app --host 127.0.0.1 --port 8000
# Swagger UI: http://127.0.0.1:8000/docs  (botão "Try it out" no POST /prever)
curl -X POST "http://127.0.0.1:8000/prever?variavel=ph&horizonte_horas=24" \
  -F "arquivo=@teste_api_ph.csv"
```

Envie um CSV CETESB (pH ou OD, ≥ ~8 dias a cada 5 min) e receba 12–288 valores do
período seguinte com timestamps. O pipeline validado roda sempre 24 h e devolve o
prefixo pedido; gaps > 2 h no fim da série retornam `422` em vez de prever no escuro.
Serve os ensembles campeões 06/07 (treino 2024, benchmark 2025).

Paridade notebook↔API (exit 0 = tudo passa): `.venv/bin/python scripts/test_paridade.py`
(carrega os dois ensembles, paridade numérica contra recomputação independente,
pesos × `ensemble.json`, smoke do `POST /prever`, edge cases de CSV).

**API multivariável** (M1/M2/M3): ver
[`../multivariavel/COMO-RODAR.md`](../multivariavel/COMO-RODAR.md) §4.

## 5. Probe versionado (benchmark próprio, só inferência)

Compara versões de modelos nos mesmos 4 períodos de 10 dias em 2025.
Detalhes em [`benchmark-2025/README.md`](benchmark-2025/README.md).

```bash
.venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --tag v1   # congela versão (~2 min CPU)
.venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --tag v2   # após evoluir notebooks, mesmos períodos
.venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --diff v1 v2        # ΔMAE/Δ% no terminal
.venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --plot v1 v2        # evolução
```
