# Benchmark versionado — probe em 4 períodos de 2025

Compara modelos (e versões de modelos) de forma repetível: carrega os checkpoints de
`resultados/` (00–07) e roda **só inferência** sobre 4 períodos de 10 dias em 2025, um
por estação, só dias-âncora (23:55). Não lê nada de `resultados/08-benchmark-2025/`
(metodologia própria, mais barata que o rolante do ano todo).

## Períodos (fixos — é o que torna v1×v2 comparável)

| Período | Datas | n pH | n OD |
|---|---|---|---|
| P1 | 08–17/mar/2025 | 10 | 10 |
| P2 | 03–12/mai/2025 | 10 | 9 |
| P3 | 22–31/ago/2025 | 9 | 9 |
| P4 | 01–10/out/2025 | 10 | 10 |

Escolhidos por busca (maior cobertura conjunta pH+OD de âncoras por trimestre).
Origens com furo **≤ 3 h (36 slots)** têm o furo interpolado localmente e entram
(`meta.json` registra o máx preenchido por período); gap maior descarta a origem —
gap de horas não invalida 31 dias de contexto para fins de benchmark, mas o modelo
não prevê no escuro. Ex.: P4 pH preenche o furo de 11 slots do gap de 30/set; a
âncora P3 mais antiga cai no gap de 55 slots de 22/jul e fica de fora (n=9).

## Modelos (11 por variável, fiel ao 08)

persistencia, sazonal_naive_288, media_movel_288, sazonal_lag365 (+fallback saz-288),
lstnet, patchtst, dlinear, lgbm (288), dlres, ens (pesos NNLS, **sem refit**),
prophet (opcional — pulado se o artefato sumir). `LSTNet1D`/`DLinearLite` vêm de
`app.py` (fonte única com a API); `PatchTST`/feats são cópia fiel do 08 com pin.

## Como rodar

```bash
.venv/bin/python benchmark/benchmark-2025/benchmark_versoes.py --tag v1   # congela versão (~2 min CPU)
.venv/bin/python benchmark/benchmark-2025/benchmark_versoes.py --tag v2   # após evoluir notebooks, mesmos períodos
.venv/bin/python benchmark/benchmark-2025/benchmark_versoes.py --tag v1 --refazer  # recongela (apaga a tag antes)
.venv/bin/python benchmark/benchmark-2025/benchmark_versoes.py --diff v1 v2        # ΔMAE/Δ% no terminal
.venv/bin/python benchmark/benchmark-2025/benchmark_versoes.py --plot v1           # retrato; --plot v1 v2 = evolução
```

## Saídas (`resultados/`)

- `historico.csv` — append-only: uma linha por (tag, variável, modelo); `MAE_benchmark` = média pooled do probe.
- `<tag>/metricas_periodos_{ph,od}.csv` — modelo × (MAE/RMSE/n por período). É o dado individual.
- `<tag>/metricas_media_{ph,od}.csv` — média pooled por modelo. É o que alimenta o gráfico.
- `<tag>/01-mae-benchmark-<tag>.png` — barras em escala log, eixo "menor é melhor", campeão em vermelho.
- `<tag>/meta.json` — tag, SHA, protocolo, períodos + cobertura + preenchimento, runtime.
- `<tag>/vals/` — cópias das vals 2024 (referência; 2025 nunca entra em tuning aqui).

## v1 (probe, 16/09/2026)

pH (39 origens): **ens 0,0512** · lstnet 0,0524 · saz-288 0,0605 · lag-365 0,4367 · prophet 2,07.
OD (38 origens): **ens 0,1996** · lstnet 0,2119 · saz-288 0,2519 · lag-365 1,27 · prophet 3,81.
Mesma hierarquia do 08 (ens vence nas duas, lag-365 inútil, prophet explode).
