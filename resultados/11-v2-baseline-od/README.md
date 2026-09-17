# Experimento 11 — Baseline v2 no OD (EF01): protocolo L=2304 + purge/embargo + 5 fatias

Baselines clássicos **idênticos ao 01** (persistência · sazonal-naive-288 · média-móvel-288 · ARIMA(2,1,2) em grade horária · Prophet),
agora no **protocolo v2**: `L=2304 → H=288` (8 d → 1 d, passo 5 min), interpolação limite 24, descarte de janelas com NaN,
val em **5 fatias de 2024 por data de fim** (19–28/abr, 20–29/jul, 15–24/set, 20–24/nov [5 d], **13–22/dez [10 d, verão — nova]**)
com **purge/embargo ±H** no treino. 2025 intocado (benchmark futuro). O sazonal lag-365 segue fora (incalculável sem 2023).
Artefatos gerados por `notebooks/11-v2-baseline-od.ipynb` (executado de ponta a ponta, 0 erros;
procedência: `temporal-remote` 192.168.1.6, dir `/home/marcos/temporal-model`, 2026-09-16, solo — 12c, threads unset).
Reproduzir: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/11-v2-baseline-od.ipynb`
(ARIMA + Prophet; ~10–25 min em 12c. Prophet pula sem CmdStan — aqui **rodou**.)

## Protocolo v2

- **Janelas** `L=2304 → H=288` (8 d → 1 d, 5 min), interp `time` limite 24, descarte de janelas com NaN (idêntico ao 01).
- **Val = 5 fatias por data de fim** (19–28/abr, 20–29/jul, 15–24/set, 20–24/nov [5 d], **13–22/dez [10 d, verão]**);
  a 5ª fatia cobre o verão austral (estação sem cobertura na VAL4) e servirá ao NNLS futuro — aqui é só reporte.
- **Purge/embargo:** treino exclui janelas cujo alvo `[fim−H, fim]` intersecte qualquer fatia estendida `±H`
  (gap mín **+289 passos**; no v1 era −288, i.e. sem purge). Lógica `purge_train`/`signed_gap_steps` reaproveitada
  verbatim de `/tmp/v2split/validate_split.py`; travada por `assert` (gap ≥ H+1, overlap zero).
- **Baselines idênticos ao 01**, determinísticos, sem seeds; ARIMA em grade horária 720 h/24 h repeat ×12
  (val com stride 48); Prophet ajustado só no treino (fatias de val removidas da série de ajuste).
- **Diferenças vs v1 (01):** `L` 8640→2304 · val 4→5 fatias (S1 parcial de 657 vira cheia) · split com purge · `metricas_por_fatia.csv` novo.

## Configuração do experimento

- **Série:** OD 2024 (`dados/treino/ef01-mogi-das-cruzes_oxigenio-dissolvido_2024.csv`), 105.121 slots, 0,6% faltantes
- **Limpeza idêntica:** grade 5 min + interpolação limite 24; NaN pós-interp 336 slots em 3 blocos —
  02/fev (13 slots, 1,1 h) · 11/mar (13 slots, 1,1 h) · **25–26/mar (310 slots, 25,8 h)** (só micro-outages, sem o outage ~17 d do pH)
- **Janelas:** treino pós-purge 77.141 + val 12.960 ([2880, 2880, 2880, 1440, 2880]);
  descartadas por NaN 8.109 · purge 4.320 · gap mín +289 · dias-âncora (23:55) na val: 45 ([10, 10, 10, 5, 10])
- **Modelos:** persistência · sazonal-naive-288 · média-móvel-288 · ARIMA(2,1,2) em grade horária (stride 48 na val: 270 origens) · Prophet (ajuste só no treino)

## Tabela principal — val rolante

| modelo | MAE | RMSE |
|---|---|---|
| **sazonal_naive_288** | **0,1736** | 0,2548 |
| media_movel_288 | 0,3631 | 0,4801 |
| persistencia | 0,4280 | 0,6222 |
| arima_212_h | 0,4282 | 0,6233 |
| prophet | 0,5243 | 0,6498 |

(copiado de `metricas_val.csv`; ARIMA avaliado em subamostra stride-48) — **régua v2 do OD: sazonal-naive 0,1736.**

## Treino rolante (referência)

| modelo | MAE | RMSE |
|---|---|---|
| sazonal_naive_288 | 0,2356 | 0,3650 |
| media_movel_288 | 0,3539 | 0,4763 |
| persistencia | 0,3812 | 0,5694 |

(copiado de `metricas_treino.csv`)

## Val dias-âncora (45)

| modelo | MAE | RMSE |
|---|---|---|
| sazonal_naive_288 | 0,1748 | 0,2572 |
| media_movel_288 | 0,3593 | 0,4751 |
| persistencia / arima_212_h | 0,5386 | 0,7313 |
| prophet | 0,5230 | 0,6483 |

(copiado de `metricas_val_diaria.csv`; detalhe por dia em `metricas_por_dia.csv`)

## Val por fatia — MAE (novo no v2)

| fatia | persistencia | sazonal_naive_288 | media_movel_288 | arima_212_h | prophet |
|---|---|---|---|---|---|
| 2024-04-19→2024-04-28 | 0,1536 | 0,1240 | 0,1615 | 0,1552 | 0,3861 |
| 2024-07-20→2024-07-29 | 0,1429 | 0,1316 | 0,1381 | 0,1384 | 0,4848 |
| 2024-09-15→2024-09-24 | 0,5584 | 0,1969 | 0,4498 | 0,5595 | 0,3018 |
| 2024-11-20→2024-11-24 | 0,6852 | 0,1579 | 0,5478 | 0,6828 | 0,6292 |
| 2024-12-13→2024-12-22 | 0,7283 | 0,2499 | 0,6106 | 0,7324 | 0,8723 |

(copiado de `metricas_por_fatia.csv`; cobertura por fatia [2880, 2880, 2880, 1440, 2880])

## Figuras — o que cada uma mostra

### `01-eda.png`, `02-limpeza.png`, `03-stl.png` — dado 2024 (ADF −6,61 / p 6,3e-09: estacionária no trecho 15/jul–31/ago)

### `04-forecasts.png` — 3 origens do treino (real × sazonal × persistência)

![Forecasts](figs/04-forecasts.png)

### `05-mae.png` — barras de MAE na val (5 fatias)

![MAE](figs/05-mae.png)

### `06-val-dias.png` — MAE por dia-âncora (5 fatias sazonais, incl. dez; Prophet fora da escala — omitido)

![Val dias](figs/06-val-dias.png)

## Arquivos

| Arquivo | O quê |
|---|---|
| `metricas_treino.csv` | Treino rolante em CSV (3 modelos baratos) |
| `metricas_val.csv` | Val rolante em CSV (primária) |
| `metricas_val_diaria.csv` | Dias-âncora (45) em CSV |
| `metricas_por_dia.csv` | MAE por dia-âncora em CSV (45 linhas, incl. 10 de dez) |
| `metricas_por_fatia.csv` | MAE/RMSE por (fatia, modelo) em CSV (25 linhas, incl. dez — novo no v2) |
| `modelos/arima212_cauda_treino.pkl` | ARIMA ajustado na cauda do treino (inspeção) |
| `modelos/prophet_od.json` | Prophet ajustado só no treino |
| `figs/` | As 6 figuras explicadas acima |

## Leitura dos resultados

1. **Régua v2 do OD: sazonal-naive MAE 0,1736 (val).** Ordem preservada vs 01 (sazonal > MM > persistência ≈ ARIMA; Prophet pior), em patamar acima (01: 0,1579) — val agora cheia (S1 de 657 vira 2880) mais a fatia dura de dez.
2. **ARIMA com fallback total nos dias-âncora** (idêntico à persistência nos 45/45 dias, maxdiff ~1e-07): o ajuste horário falha nas âncoras 23:55 e o fallback honesto assume. Na val stride-48 o ARIMA (0,4282) ≠ persistência (0,4280) — parte dos ajustes converge; taxa de fallback não logada. Limitação reportada, não bug (mesmo padrão do 01 e do 10).
3. **Prophet NÃO foi pulado** (CmdStan presente; chain ~52 s) e colapsa do mesmo jeito (0,5243, 3,0× a régua) — melhor fatia em **set** (0,3018, única onde bate persistência/MM/ARIMA, ainda 1,5× o sazonal). Fora do jogo como modelo global neste regime.
4. **Dezembro é a fatia dura** (sazonal 0,2499, demais ≥ 0,61); julho é a mais fácil (sazonal 0,1316). Val cheia nas 5 fatias — os micro-outages do OD não atingem a val com `L=2304`, e dez comporta-se como fatia normal em cobertura (2880).
5. Fila: portar o mesmo protocolo v2 para os NNs do OD; a fatia dez servirá ao NNLS futuro — aqui é só reporte.
