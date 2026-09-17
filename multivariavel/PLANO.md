# Plano — Treino multivariável (EF01 Mogi das Cruzes)

Aprovado em 2026-09-17. Modelos recomendados a partir dos vereditos univariáveis
(`univariavel/resultados/`): só 2 deep + piso.
Revisão 2026-09-17 (pós-validação): correções de aritmética, loss, CD e
pré-registros — desenho inalterado.

## 1. Dados

Treino = 2022–2024 (`multivariavel/dados/treino/`), benchmark = 2025 intocado
(`multivariavel/dados/benchmark/`). Um CSV por ano, grade 5-min, colunas
OD + pH + Precipitação + Temperatura + Turbidez (formato CETESB: `windows-1252`,
`;`, decimal vírgula, pula linha 1, `dd/mm/aaaa hh:mm`).

Auditoria 2026-09-17 (faltantes por ano, verificada nos CSVs):

| Canal | 2022 | 2023 | 2024 | Diagnóstico |
|---|---|---|---|---|
| OD | 211 | 431 | 594 | limpo |
| pH | **14.367 (13,7%)** | 1.768 | 11.665 | 2022 pior que 2024; reportar cobertura por fatia |
| Temperatura | 128 | 194 | 184 | covariável ideal, quase completa |
| Turbidez | 3.323 | 706 | 1.539 | cauda pesada (máx 80–143 NTU vs média 7–12) — RevIN per-channel + clip pré-registrado (threshold do treino-only, §2); RevIN sozinha não cobre spike |
| Precipitação | **81.688 (78%)** | 6.679 | 60.690 | **EXCLUÍDA**: codificação inconsistente entre anos (2022 quase tudo vazio, 2023 preenchido com zeros) + outlier 2146 mm/5min em 2023. Binário chuva/não-chuva = extensão futura (observed-only, §3) |

Descarte conjunto pós-interp (qualquer dos 4 canais com NaN, limite 24):
2022 16.619 (15,9%) · 2023 1.130 (1,1%) · 2024 6.840 (6,5%).
2022 perde muito mais janela que o uni — ver regras das auxiliares em §2/§4.

## 2. Desenho travado

- **Alvo: pH + OD (2 heads)**; Temp/Turbidez só covariáveis observadas. Nenhum canal usa futuro (observed-only, tripartição TFT da METODOLOGIA §3.6); hora/solar/Fourier são as únicas future-known e podem entrar no horizonte. **Loss: `(MSE_ph + MSE_od)/2` sobre alvos z-scorados** (média/dp por canal fitados só no treino) — MSE cru seria dominado pelo OD. Reporte sempre por (horizonte, variável) em unidade original.
- **Modelos**: (0) piso `sazonal-naive-288` por canal e por horizonte (H<288 = prefixo da cópia do dia anterior no mesmo horário) · (1) `DLinear-multi` (concat dos 4 canais + time-features → linear por canal-alvo + head por variável; **exceção** ao "verbatim 14/15", que era univariado sem solar+Fourier) · (2) `PatchTST-multi` com **ablação CI vs CD** (o contraste é o resultado principal; **CD = joint-attention sobre patches concatenados dos 4 canais**, mesmo backbone 3×64/4 heads, patches 48/24 do 14/15 — variante iTransformer-style token-por-canal só como extensão registrada).
- **Fora**: ARIMA/Prophet/lag-365 (colapsam no uni) · LGBM solo (perde do sazonal em 2025) · ensemble NNLS+LGBM como modelo final (pesos não transferem interano; NNLS só como diagnóstico honesto, §3; único ensemble final permitido = média simples) · LSTNet-multi (só extensão, com gatilho: **se CD não bater CI nem DLinear em nenhum H, rodar LSTNet-multi antes de declarar multivariado inútil**).
- **Horizontes dedicados**: `H = 12 (1h) / 72 (6h) / 288 (24h)` (um treino por H, sem truncar modelo de Hmax), `L=2304` fixo, **split único com purge/embargo ±288 (Hmax)** nos três + `assert gap ≥ Hmax+1`. Tradeoff declarado: purge e L conservadores para H=12 (`L/H=192`), intencional p/ comparabilidade entre Hs.
- Val = 5 fatias v2 de 2024 com as datas do 14/15 (19–28/abr, 20–29/jul, 15–24/set, 20–24/nov [5 d], 13–22/dez — sem overlap com o outage pH 27/mai–13/jun) + 2 auxiliares interanuais fixas (verificação 2026-09-18, `L+H=2592`: cobertura 100% nas duas) — verão **18–27/jan/2023** (2880/2880 janelas viáveis; verão-2022 inviável: outage pH 20/jan–09/mar deixa só 749/2880) + inverno **20–29/jul/2022** (2880/2880; espelha as datas v2-jul). Estações = meteorológicas austrais (verão DJF, outono MAM, inverno JJA, primavera SON) derivadas do timestamp no código — sem reescrever CSVs; uso = reporte estratificado + checagem de balanço do treino, não feature de entrada (hora/solar/Fourier já codificam sazonalidade contínua; estação categórica vira ablação futura). Regra <1000 janelas: fatia vira **qualitativa** (só figura + MAE por dia-âncora, sem MAE pooled na manchete). Limpeza: interp `time` limite 24 por canal, **descarta janela se qualquer dos 4 canais tem NaN**, RevIN **per-channel** per-window, z-score por canal fitado só no treino.
- Seeds: 3 por (config treinável, horizonte) = **27 treinos** (3 configs: DLinear, CI, CD × 3 H × 3 seeds). Backbone/hiperparâmetros/early stopping verbatim do 14/15 por (modelo, seed), exceto camada de fusão multicanal + heads (novas, a especificar no README do M1). Um job por vez.
- Métricas por (horizonte, variável): MAE/RMSE pooled + por fatia + dias-âncora 23:55 (mesmo critério do uni) + **curva MAE(h)** intra-modelo (checar se o 24h bate o 1h em h≤12 — se sim, o dedicado curto é redundante).
- Sucesso: bater piso sazonal do mesmo H em cada (horizonte, variável) + réguas uni-v2 no H=288 (pH 0,0465 / OD 0,2056, rolante do `18-v2-benchmark-2025`). H=12/72 não têm régua uni — o piso é a régua.

## 3. Ordem de execução

1. `M1-dlinear-multi` (H=12/72/288) — valida pipeline 4-canais, fecha pisos por horizonte, publica fusão + nº de params.
2. `M2-patchtst-multi-CI` → `M3-patchtst-multi-CD` no mesmo split.
3. Benchmark 2025 (**só inferência — proibido qualquer tuning/early-stopping/seleção em 2025**) + READMEs em `multivariavel/resultados/M*/` (espelho do uni: tabela + figs + leitura) + `multivariavel/resultados/README.md` + atualizar status "multivariada é futuro" no README principal.
4. Extensões opcionais: LSTNet-multi (gatilho acima), precipitação binária observed-only (chuva-futura desconhecida no ato — só com fonte externa se usar futuro), NNLS fit 1–4 / report dez só como diagnóstico.

## 4. Riscos

- Cobertura conjunta 2022 (15,9% pós-interp); auxiliares condicionadas a pré-registro de datas; <1000 janelas = qualitativa (§2).
- Spikes de turbidez; clip/winsorize com threshold do treino-only (ex. p99) ou Huber, documentado no README do M1 — sem threshold ad-hoc pós-val.
- Custo ~5–6h solo no remoto (27 treinos); não paralelizar jobs.
- Vazamento: 2025 nunca em treino/val/early-stopping/tuning/seleção.
