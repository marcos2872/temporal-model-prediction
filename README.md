# Modelo de Predição Temporal — Base Bibliográfica e Guia Prático

Projeto para criar um modelo de IA que **recebe uma série temporal e gera predição futura**.
Esta pasta organiza a bibliografia fundamental e resume o caminho prático de implementação.

> **Busca bibliográfica sistemática (2026-09-10):** pipeline `rigorous` com 133 registros após dedup → 34 incluídos → `verify` EXIT 0. Relatório completo em [`busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) (+ `.bib`/`.ris`, `evidence_table.csv`, `passport.json`, `prisma.md`). Detalhes na seção 8.

## Estrutura

```
temporal-model/
├── README.md              <- este arquivo (guia + fórmulas + referências)
├── busca_bibliografica/   <- kit da busca sistemática (relatório + .bib/.ris + evidence_table.csv + passport.json + prisma.md)
├── reviews/modelos-predicao-temporal/ <- auditoria do pipeline (protocol.md, corpus.json, verification.json, search_log.md)
└── reports/               <- saídas brutas do build_report.py (espelhadas em busca_bibliografica/)
```

> Os 9 PDFs de `artigos/` foram **removidos do repo**. As referências sobrevivem como metadados verificados em `busca_bibliografica/` (tabela de evidências + `.bib`/`.ris`) e os textos podem ser reobtidos pelos links DOI/arXiv da seção 3.

## 1. O que é predição temporal?

Dada uma sequência histórica `y(1)...y(t)` (possivelmente com covariáveis `x(t)`), prever `y(t+1)...y(t+H)` onde `H` é o horizonte.

Formalmente, um modelo de predição temporal aprende:

$$\hat{y}_{t+1:t+H} = f_\theta\big(y_{1:t},\; x_{1:t+H},\; s\big)$$

onde $s$ são covariáveis estáticas (ex.: loja, sensor), $x$ inclui entradas futuras conhecidas (ex.: feriado, promoção) e séries exógenas observadas — a tripartição formalizada pelo TFT (seção 3.5). Na prática o treino usa **janelamento**: cada amostra é `(lookback=L → horizonte=H)` com split temporal sem shuffle.

Desafios clássicos apontados na literatura:
- **Sazonalidade + tendência + ruído** entrelaçados
- **Não-estacionariedade / distribution shift**
- **Dependência de longo prazo** vs custo quadrático do Transformer
- **Multivariada:** dependência entre canais (channel dependence)
- **Incerteza:** previsão pontual vs probabilística (quantis)

## 2. Taxonomia (síntese dos surveys)

| Família | Exemplos | Quando usar |
|---------|----------|-------------|
| Estatísticos | ARIMA, SARIMA, ETS, Holt-Winters | Série curta, linear, estacionária, baseline interpretável |
| ML com lags | XGBoost, LightGBM + lags/date features | Tabular, multivariada, produção, rápido |
| Recorrentes | LSTM, GRU, DeepAR, Bi-LSTM | Não-linear, sequência média, dados limitados |
| Convolucionais | TCN, CNN-LSTM | Long-range com paralelismo, causal |
| Transformers | Informer, Autoformer, PatchTST, iTransformer, TFT | Long-term, multivariado, multi-horizonte |
| Lineares modernos | DLinear, NLinear | Baseline forte que às vezes vence Transformer |
| Generativos / SSM | Diffusion, Mamba, Koopman + Transformer | Incerteza, sequências muito longas |
| Foundation / LLM | Chronos, TimeGPT, Lag-Llama, LLM4Series | Zero-shot, poucas amostras |

## 3. Artigos baixados em `artigos/`

### Surveys — leitura inicial recomendada

**[01] Kong et al. (2025) — Deep Learning for Time Series Forecasting: A Survey**
Arquivo: `artigos/01-Kong2025-Deep-Learning-TSF-Survey.pdf`
Link: https://arxiv.org/abs/2503.10198
Contribuição: taxonomia dinâmica por arquitetura (RNN, TCN, Transformer, GAN), métodos de extração de features (decomposição sazonal-tendência, modelagem periódica) e compilação de datasets por domínio (energia, saúde, tráfego, clima, economia). Melhor ponto de partida.

**[02] Kim et al. (2024, rev. 2025) — Architectural Diversity and Open Challenges**
Arquivo: `artigos/02-Kim2025-Architectural-Diversity-Open-Challenges.pdf`
Link: https://arxiv.org/abs/2411.05793
Contribuição: mostra a “renascença” arquitetural — por que lineares simples superam Transformers em certos casos. Cobre híbridos, diffusion, Mamba e foundation models. Discute desafios abertos: channel dependency, shift distribucional, causalidade.

**[03] Wen et al. (2022) — Transformers in Time Series: A Survey**
Arquivo: `artigos/03-Wen2022-Transformers-in-Time-Series.pdf`
Link: https://arxiv.org/abs/2202.07125
Contribuição: taxonomia de Transformers por modificação de rede (positional/timestamp encoding, atenção esparsa) e por tarefa (forecasting, anomalia, classificação). Base para entender Informer/Autoformer/FEDformer.

**[09] Liao et al. (2026) — Deep learning for time series forecasting: survey of recent advances**
Arquivo: `artigos/09-Liao2026-Recent-Advances-Frontiers.pdf`
Link: https://www.osti.gov/pages/servlets/purl/3025651
Contribuição: visão 2026 cobrindo RNN, CNN, GNN, Transformer, LLM, MLP e Diffusion. Traz tabela comparativa vantagens/desvantagens e análise de DeepAR, P-sLSTM, iTransformer. Bom para justificar escolha arquitetural.

### Modelos seminais

**[04] Lim et al. (2019/2021, Oxford/Google) — Temporal Fusion Transformer (TFT)**
Arquivo: `artigos/04-Lim2019-Temporal-Fusion-Transformer.pdf`
Link: https://arxiv.org/abs/1912.09363
Contribuição: arquitetura padrão para multi-horizonte interpretável. Combina gating, seleção de variáveis, encoder estático + atenção temporal + saída por quantis (p10/p50/p90). Ideal se seu modelo precisa explicar “quais variáveis importaram” e lidar com estáticas + futuras conhecidas + passadas observadas.

**[05] Wu et al. (NeurIPS 2021) — Autoformer**
Arquivo: `artigos/05-Wu2021-Autoformer.pdf`
Link: https://arxiv.org/abs/2106.13008
Contribuição: troca self-attention por Auto-Correlation baseada em periodicidade + bloco de decomposição progressiva interno. Complexidade O(L log L) e +38% em 6 benchmarks long-term. Leitura obrigatória para long-term forecasting.

### Fronteira 2025-2026

**[06] Zhang et al. (2025) — AutoFormer-TS via Neural Architecture Search**
Arquivo: `artigos/06-Zhang2025-AutoFormer-TS-NAS.pdf`
Link: https://arxiv.org/abs/2502.13721
Contribuição: usa DNAS (AB-DARTS) para buscar atenção, ativação e encoding ótimos para séries. Supera baselines SOTA mantendo eficiência. Mostra que arquitetura “vanilla” de NLP não é ótima para tempo.

**[07] Power of Architecture (2025) — Deep Dive into Transformer Architectures for LTSF**
Arquivo: `artigos/07-Power-of-Architecture-LTSF-2025.pdf`
Link: https://arxiv.org/html/2507.13043v1
Contribuição: compara encoder-only vs encoder-decoder vs decoder-only isolando efeito da arquitetura. Conclusão prática: encoder-only com patching (estilo PatchTST) vence na maioria dos LTSF. Guia direto de qual variante implementar.

### Comparativo clássico vs moderno

**[08] Mahajan (2026) — ARIMA vs LSTM vs Prophet**
Arquivo: `artigos/08-Mahajan2026-ARIMA-LSTM-Prophet.pdf`
Link: https://www.preprints.org/manuscript/202601.1377
Contribuição: síntese 2018-2025 + árvore de decisão prática:
- Linear/estacionário → ARIMA (MAPE 3.2-13.6%)
- Não-linear/volátil/multivariado → LSTM (84-87% menos erro que ARIMA)
- Negócio com sazonalidade forte + feriados → Prophet (MAPE 2.2-24.2%)
- Ensembles ponderados (ex: 50/30/20) como caminho robusto. Use para justificar baseline no seu projeto.

## 4. Como criar seu modelo (roteiro)

1. **Defina:** univariada ou multivariada? Horizonte `H`? Precisa de intervalo de confiança?
2. **Dados:** dataframe longo `id | ds | y`. Cheque ADF (estacionariedade), sazonalidade, missing.
3. **Janelamento:** transforme em supervisionado. Ex: `lookback=30 → H=7`. Split temporal 70/15/15 sem shuffle. Validação por expanding window.
4. **Baseline 1 (1 dia de trabalho):** ARIMA/SARIMA (`statsmodels`) + Prophet. Métricas: MAE, RMSE, MAPE, SMAPE.
5. **Baseline 2:** LightGBM com lags (`mlforecast`, `Darts`) + LSTM/GRU (`PyTorch`).
6. **Avançado:** `pytorch-forecasting` TFT ou PatchTST (`Time-Series-Library`, `GluonTS`, `NeuralProphet`, `sktime`).
7. **Avaliação + deploy:** backtest, quantile loss se probabilístico, API `FastAPI /predict` + retreino.

Ordem sugerida de leitura: 01 → 08 → 04 → 05 → 02 → 07 → 03 → 06 → 09.

## 5. Bibliotecas práticas

- Clássicos: `statsmodels`, `prophet`, `neuralprophet`
- ML: `scikit-learn`, `lightgbm`, `xgboost`, `mlforecast`
- DL: `pytorch`, `pytorch-forecasting`, `darts`, `gluonts`, `sktime`
- Benchmarks: `Time-Series-Library (THUML)`, `Monash / M3 / ETTh / Electricity / Traffic`
- LLMs: `LLM4Series` (ICLR 2026 workshop), `Chronos`, `Lag-Llama`

## 6. Referências complementares (não baixadas — acesso restrito)

- Parmezan et al. (2019) — Evaluation of statistical and ML models, Inf. Sciences. https://www.sciencedirect.com/science/article/pii/S0020025519300945
- Sherly et al. (2025) — Hybrid ARIMA + Prophet. https://www.sciencedirect.com/science/article/pii/S2590123025017748
- Forootani & Khosravi (2026) — Compact transformer variants, Neurocomputing. https://doi.org/10.1016/j.neucom.2026.133140
- Silva et al. (2026) — LLM4Series. https://openreview.net/forum?id=6fbcYFRoUL

## 7. Próximos passos neste repo

- [x] Busca bibliográfica sistemática (2026-09-10) — ver seção 8 e `busca_bibliografica/`
- [ ] Definir dataset (ex: `ETTh`, vendas, energia, ação)
- [ ] Criar `notebooks/00-baseline-arima-prophet.ipynb`
- [ ] Criar `src/windowing.py` + `train_lstm.py`
- [ ] Evoluir para `TFT / PatchTST`
- [ ] Expor `app.py` FastAPI

## 8. Busca bibliográfica sistemática — modelos de predição temporal (2026-09-10)

Pipeline `deep-research-br`, template `rigorous`, framework Decomposição (Problema/Solução/Avaliação/Limitações), recorte 2019–2026. Questão: como criar modelos que recebem `y(1..t)` e geram `y(t+1..t+H)`, e quais trade-offs orientam a escolha entre estatísticos, ML, DL e foundation models.

- **Relatório:** [`busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) — síntese temática (não paper-a-paper), mapa de 5 tensões, roteiro de criação, gaps, níveis de confiança e auto-crítica 14pt. Toda afirmação factual tem âncora `[^rank]` ligada ao `corpus.json`.
- **Números do funil (PRISMA):** 133 identificados após dedup (120 brutos federados em 5 queries × OpenAlex/arXiv/EPMC/Crossref + 9 PDFs locais + 30 achados de chasing forward) → 34 incluídos (24 top por score + 9 locais + TFT canônico) → 1 excluído com motivo → 98 pendentes (skim). Saturação atingida (taxa de novidade 0,0 na 2ª rodada).
- **Integridade:** `verify_citations.py` EXIT 0 — 124 `verified`, 9 `sem-id` (o núcleo local; o `dedupe_rank.py` descarta `arquivo_local` na fusão, documentado no relatório), 0 `not_found/mismatched/retracted` após remover 2 falsos-positivos de astrofísica. Chasing backward sem retorno (melhor esforço, declarado).
- **Achados federados verificados que estendem a seção 3** (todos `verified`, todos incluídos): Informer ([DOI 10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325)), "Are Transformers Effective for Time Series Forecasting?" ([DOI 10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317)), PatchTST "A Time Series is Worth 64 Words" ([arXiv 2211.14730](https://arxiv.org/abs/2211.14730)), iTransformer ([arXiv 2310.06625](https://arxiv.org/abs/2310.06625)), NHITS ([DOI 10.1609/aaai.v37i6.25854](https://doi.org/10.1609/aaai.v37i6.25854)), TSMixer, TimesNet ([arXiv 2210.02186](https://arxiv.org/abs/2210.02186)), MSGNet, TS2Vec, TFT publicado ([DOI 10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012)) e o survey de foundation models (2024). Tabela completa em [`busca_bibliografica/evidence_table.csv`](busca_bibliografica/evidence_table.csv); claims→fontes em [`busca_bibliografica/passport.json`](busca_bibliografica/passport.json).
- **Nota de honestidade:** os números de desempenho citados nas seções 3–4 acima (ex.: MAPEs, "+38%", "O(L log L)") vêm da leitura-guia prévia, **não** foram re-afirmados pelo relatório sistemático — os trechos extraídos nesta sessão (cabeçalhos de 2 páginas via `pdftotext`) não os continham (ver §§8–10 do relatório). Use o relatório para o que é auditável e os PDFs em `artigos/` para os números.

> Todos os PDFs em `artigos/` são open-access (arXiv / Preprints / OSTI) para estudo pessoal. Verifique a licença de cada um antes de redistribuir.
