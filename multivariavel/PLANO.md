# Plano — Treino multivariável (EF01 Mogi das Cruzes)

Aprovado em 2026-09-17. Modelos recomendados a partir dos vereditos univariáveis
(`univariavel/resultados/`): só 2 deep + piso.

## 1. Dados

Treino = 2022–2024 (`multivariavel/dados/treino/`), benchmark = 2025 intocado
(`multivariavel/dados/benchmark/`). Um CSV por ano, grade 5-min, colunas
OD + pH + Precipitação + Temperatura + Turbidez (formato CETESB: `windows-1252`,
`;`, decimal vírgula, pula linha 1, `dd/mm/aaaa hh:mm`).

Auditoria 2026-09-17 (faltantes por ano):

| Canal | 2022 | 2023 | 2024 | Diagnóstico |
|---|---|---|---|---|
| OD | 211 | 431 | 594 | limpo |
| pH | **14.367 (13,7%)** | 1.768 | 11.665 | 2022 pior que 2024; reportar cobertura por fatia |
| Temperatura | 128 | 194 | 184 | covariável ideal, quase completa |
| Turbidez | 3.323 | 706 | 1.539 | cauda pesada (máx 80–143 NTU vs média 7–12) — RevIN cobre |
| Precipitação | **81.688 (78%)** | 6.679 | 60.690 | **EXCLUÍDA**: codificação inconsistente entre anos (2022 quase tudo vazio, 2023 preenchido com zeros) + outlier 2146 mm/5min em 2023. Binário chuva/não-chuva = extensão futura |

## 2. Desenho travado

- **Alvo: pH + OD (2 heads)**; Temp/Turbidez só covariáveis. Sem usar futuro de nenhum canal.
- **Modelos**: (0) piso `sazonal-naive-288` por canal · (1) `DLinear-multi` (linear enxerga os 4 canais + hora/solar/Fourier) · (2) `PatchTST-multi` com **ablação CI vs CD** (o contraste é o resultado principal).
- **Fora**: ARIMA/Prophet/lag-365 (colapsam no uni) · LGBM solo (perde do sazonal em 2025) · ensemble NNLS+LGBM (pesos não transferem interano; no máximo média simples ao fim) · LSTNet-multi (só extensão).
- **Horizontes dedicados**: `H = 12 (1h) / 72 (6h) / 288 (24h)`, `L=2304` fixo, **split único com purge/embargo ±288 (Hmax)** nos três.
- Val = 5 fatias v2 de 2024 (comparável às réguas uni) + 2 auxiliares (1 verão + 1 inverno, 2022/2023). Limpeza: interp `time` limite 24 por canal, **descarta janela se qualquer dos 4 canais tem NaN**, RevIN per-window, normalização fitada só no treino.
- Seeds: 3 por (modelo, horizonte) = 18 treinos. Hiperparâmetros/early stopping verbatim do 14/15. Um job por vez.
- Métricas por (horizonte, variável): MAE/RMSE pooled + por fatia + dias-âncora + **curva MAE(h)** intra-modelo (checar se o 24h bate o 1h em h≤12 — se sim, o dedicado curto é redundante).
- Sucesso: bater piso sazonal por horizonte + réguas uni-v2 (pH 0,0465 / OD 0,2056 no H=288).

## 3. Ordem de execução

1. `M1-dlinear-multi` (H=12/72/288) — valida pipeline 4-canais, fecha pisos por horizonte.
2. `M2-patchtst-multi-CI` → `M3-patchtst-multi-CD` no mesmo split.
3. Benchmark 2025 (só inferência) + READMEs em `multivariavel/resultados/M*/` + índice.
4. Extensões opcionais: LSTNet-multi, precipitação binária, média NNLS honesta (fit 1–4, report dez).

## 4. Riscos

- Cobertura conjunta 2022 (pH 13,7% faltante); fatias auxiliares <1000 janelas viram qualitativas.
- Spikes de turbidez; clipar só como fix documentado.
- Custo ~3–4h solo no remoto; não paralelizar jobs.
