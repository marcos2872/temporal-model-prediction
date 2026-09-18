# Como rodar — trilha multivariável

Guia de execução: ambiente, notebooks M1–M4, API e checkpoints **multivariáveis**
(comandos a partir da raiz do repo). Trilha univariada em
[`../univariavel/COMO-RODAR.md`](../univariavel/COMO-RODAR.md).
Protocolo travado em [`PLANO.md`](PLANO.md); apresentação
em [`README.md`](../README.md).

## 1. Ambiente (uma vez)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Mesmo `.venv` da trilha uni (usa `torch` com CUDA quando há GPU; sem GPU cai
para CPU sozinho via `DEVICE` parametrizável: `M1_DEVICE`/`M2_DEVICE`/
`M3_DEVICE`/`M4_DEVICE`, default `cuda` com fallback `cpu`). ARIMA/Prophet
não entram aqui (ver `PLANO.md` §2, "Fora").

Dados: séries CETESB EF01 Mogi das Cruzes (OD + pH + Temp + Turb, passo 5 min)
em `multivariavel/dados/` — treino = 2022–2024, benchmark = 2025 (intocado até
a avaliação final; nunca em treino/val/early-stopping/tuning/seleção).
Formato CETESB (`windows-1252`, `;`, vírgula decimal, pula linha 1,
`dd/mm/aaaa hh:mm`); Precipitação **excluída** (codificação inconsistente +
outlier); limpeza em `PLANO.md` §1 (interp `time` limite 24 por canal,
descarte conjunto, winsorize da turbidez no p99 do treino, z-score por canal).

## 2. Notebooks (M1 → M4)

Índice em [`resultados/`](resultados/README.md) (tabela + leitura por
experimento). **Interativo:** abra o `.ipynb` em `multivariavel/notebooks/`
com o kernel do `.venv`. **Reproduzível:** `nbconvert --execute --inplace`
(comando exato no README de cada experimento; reexecutar sobrescreve
métricas, modelos e figuras — copie `resultados/M*/` antes de comparar).

| Notebook | Experimento | Resultados | Reproduzir (raiz do repo) |
|---|---|---|---|
| [`M1-dlinear-multi.ipynb`](notebooks/M1-dlinear-multi.ipynb) | DLinear multicanal 4→2 heads, 3 H × 3 seeds — valida o pipeline, fecha os pisos | [`resultados/M1-dlinear-multi/`](resultados/M1-dlinear-multi/) | `CUDA_VISIBLE_DEVICES=0 M1_DEVICE=cuda .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 multivariavel/notebooks/M1-dlinear-multi.ipynb` (~7 min em 1× GPU, 9 treinos) |
| [`M2-patchtst-multi-CI.ipynb`](notebooks/M2-patchtst-multi-CI.ipynb) | PatchTST channel-independent, 3 H × 3 seeds — braço controle da ablação CI vs CD | [`resultados/M2-patchtst-multi-CI/`](resultados/M2-patchtst-multi-CI/) | `CUDA_VISIBLE_DEVICES=0 M2_DEVICE=cuda .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 multivariavel/notebooks/M2-patchtst-multi-CI.ipynb` (~62 min em 1× GPU, 9 treinos) |
| [`M3-patchtst-multi-CD.ipynb`](notebooks/M3-patchtst-multi-CD.ipynb) | PatchTST channel-dependent (joint-attention), 3 H × 3 seeds — braço tratamento | [`resultados/M3-patchtst-multi-CD/`](resultados/M3-patchtst-multi-CD/) | `CUDA_VISIBLE_DEVICES=1 M3_DEVICE=cuda .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 multivariavel/notebooks/M3-patchtst-multi-CD.ipynb` (~21 min em 1× GPU, 9 treinos) |
| [`M4-benchmark-2025.ipynb`](notebooks/M4-benchmark-2025.ipynb) | 27 checkpoints M1/M2/M3 × 2025 intocado (**só inferência**, trava executável anti-treino) — veredito final | [`resultados/M4-benchmark-2025/`](resultados/M4-benchmark-2025/) | `CUDA_VISIBLE_DEVICES=0 M4_DEVICE=cuda .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 multivariavel/notebooks/M4-benchmark-2025.ipynb` (~6 min em 1× GPU; exige os 27 checkpoints) |

Regras (ver `PLANO.md` §§2–3):

- **Ordem:** M1 solo (valida o pipeline); M2+M3 em paralelo, **1 job por GPU**
  (`CUDA_VISIBLE_DEVICES=0/1`, nunca 2 jobs na mesma GPU), threads capadas
  (`OMP/MKL/OpenBLAS_NUM_THREADS=8`); depois M4.
- Sem GPU: cai para CPU sozinho (~5–6 h solo; com 2× GPU em paralelo cai
  para ~1–2 h de parede — DLinear = segundos, PatchTST acelera 5–10×).
- Cada execução escreve em `resultados/M*/`: `metricas*.csv`, `modelos/`
  (checkpoints `.pt`), `figs/`.

## 3. Checkpoints (GitHub Release — nunca vão ao git)

Os `.pt` de `resultados/M*/modelos/` são gitignored (só `normalizacao.json`
fica versionado) e regeneráveis pelos notebooks M1/M2/M3. Para reutilizar sem
retreinar:

```bash
bash scripts/baixar_modelos.sh --tag modelo-multivariavel-v1 --dir /tmp/multimodelos
```

(verificação via `SHA256SUMS.txt` dentro do script).

## 4. API de previsão (local)

Detalhes e contrato em [`src/README.md`](src/README.md) (a partir da raiz):

```bash
.venv/bin/python -m multivariavel.src.app --modelo M2          # M1 | M2 | M3 (default M2)
.venv/bin/python -m multivariavel.src.app --modelo M1 --port 8001
# Swagger UI: http://127.0.0.1:8000/docs  (botão "Try it out" no POST /prever)
curl -X POST "http://127.0.0.1:8000/prever?variavel=ph&horizonte_horas=24" \
  -F "arquivo=@multivariavel/dados/benchmark/ef01-mogi-das-cruzes_multivariavel_2025.csv"
```

`MULTIV_MODELO` faz o mesmo que `--modelo`. Contrato curto: input = CSV CETESB
multivariável com **os 4 canais** e ao menos `L+H` slots a cada 5 min
(Precipitação é ignorada); `variavel` = `ph` | `od`; `horizonte_horas` = `1`
(H=12), `6` (H=72) ou `24` (H=288); resposta = H pontos `{ds, y}` +
`mae_referencia` (MAE seed-mean no benchmark 2025, M4) + `checkpoint_sha`;
**gaps > 2 h na cauda** retornam `422`. Seed-mean fixa (3 seeds) por
(modelo, H) — sem pesos ajustáveis.

Pipeline de teste (`src/teste.py`, via subprocesso uvicorn, contra
`src/dados.csv`, série 2026 unseen): 9 partes × 3 H × ph/od; logs em
`src/.logs/` (`M1.log`, `M2.log`, `M3.log` + `geral.log`; `SEM_COBERTURA` =
parte recusada com 422, fora do placar).
