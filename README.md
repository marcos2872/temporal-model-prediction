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

| Família | Exemplos | Quando usar | Ref. nuclear (`busca_bibliografica/`) |
|---------|----------|-------------|----------------------------------------|
| Estatísticos | ARIMA, SARIMA, ETS, Holt-Winters | Série curta, linear, estacionária, baseline interpretável | [Kontopoulou et al. 2023](https://doi.org/10.3390/fi15080255) · Mahajan 2026 (ex-local, §3.6) |
| ML com lags | XGBoost, LightGBM + lags/date features | Tabular, multivariada, produção, rápido | Ver comparativos em [Kontopoulou et al. 2023](https://doi.org/10.3390/fi15080255) |
| Recorrentes | LSTM, GRU, DeepAR, Bi-LSTM | Não-linear, sequência média, dados limitados | Mahajan 2026 (ex-local, §3.6) · surveys §3.1 |
| Convolucionais | TCN, TimesNet | Long-range com paralelismo, padrões 2D por período | [TimesNet (Wu et al. 2022)](https://arxiv.org/abs/2210.02186) |
| Transformers LTSF | Informer, Autoformer, PatchTST, iTransformer, TFT | Long-term, multivariado, multi-horizonte | [Informer](https://doi.org/10.1609/aaai.v35i12.17325) · [PatchTST](https://arxiv.org/abs/2211.14730) · [iTransformer](https://arxiv.org/abs/2310.06625) · [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012) |
| Lineares / MLP modernos | DLinear, NLinear, NHITS, TSMixer | Baseline forte; eficiente contra atenção plena | [Zeng et al. 2023 (o debate)](https://doi.org/10.1609/aaai.v37i9.26317) · [NHITS](https://doi.org/10.1609/aaai.v37i6.25854) · [TSMixer](https://doi.org/10.1145/3580305.3599533) |
| Representação / GNN | TS2Vec, MSGNet | Pré-treino, correlação entre séries | [TS2Vec](https://doi.org/10.1609/aaai.v36i8.20881) · [MSGNet](https://doi.org/10.1609/aaai.v38i10.28991) |
| Foundation / LLM | Chronos, TimeGPT, Lag-Llama | Zero-shot, poucas amostras | [Liang et al. 2024](https://doi.org/10.1145/3637528.3671451) (confiança média — fronteira) |

## 3. Referências nucleares + a matemática de cada família

Fonte: os 34 incluídos da busca sistemática (`busca_bibliografica/evidence_table.csv`, relatório completo [aqui](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md)). Os 9 PDFs locais foram apagados — abaixo, cada ex-local traz o link para reobter o texto. As fórmulas são a **formulação canônica (livro-texto)** de cada método, para ler os artigos com o ferramental na mão — não são citações literais dos papers.

### 3.0 Janelamento e métricas (vale para todos)

Amostra de treino com `lookback=L`, horizonte `H`: entrada `y[i:i+L]`, alvo `y[i+L:i+L+H]`, split temporal sem shuffle + backtest por janela expansiva.

$$\mathrm{MAE} = \tfrac{1}{H}\sum|y-\hat{y}| \quad\quad \mathrm{RMSE} = \sqrt{\tfrac{1}{H}\sum(y-\hat{y})^2}$$

$$\mathrm{MAPE} = \tfrac{100}{H}\sum\Big|\tfrac{y-\hat{y}}{y}\Big| \quad\quad \mathrm{SMAPE} = \tfrac{100}{H}\sum \tfrac{2|y-\hat{y}|}{|y|+|\hat{y}|}$$

Saída probabilística (quantis $q \in \{0.1, 0.5, 0.9\}$) minimiza a **quantile loss**:

$$\mathcal{L}_q(y, \hat{y}) = \max\big(q\,(y-\hat{y}),\; (q-1)\,(y-\hat{y})\big)$$

### 3.1 Surveys — ponto de partida (sem matemática nova, organizam o campo)

- **Wen et al. (2023) — Transformers in Time Series: A Survey.** [DOI 10.24963/ijcai.2023/759](https://doi.org/10.24963/ijcai.2023/759) · 1103 citações · *verified*. Taxonomia por modificação de rede (positional/timestamp encoding, atenção esparsa) e por tarefa. (Ex-local `03-Wen2022` removido — este registro federado é a versão publicada.)
- **Lim & Zohren (2021) — Time-series forecasting with deep learning: a survey.** [DOI 10.1098/rsta.2020.0209](https://doi.org/10.1098/rsta.2020.0209) · 1770 citações · *verified*.
- **Torres et al. (2020) — Deep Learning for Time Series Forecasting: A Survey.** [DOI 10.1089/big.2020.0159](https://doi.org/10.1089/big.2020.0159) · 816 citações · *verified*.
- **Benidis et al. (2022) — Tutorial and Literature Survey.** [DOI 10.1145/3533382](https://doi.org/10.1145/3533382) · *verified*. Base probabilística (DeepAR/DeepState).
- **Kim et al. (2025) — Architectural Diversity and Open Challenges.** [DOI 10.1007/s10462-025-11223-9](https://doi.org/10.1007/s10462-025-11223-9) · *verified*. A "renascença" arquitetural: por que lineares simples às vezes vencem Transformers; desafios abertos (channel dependency, shift, causalidade). (Ex-local `02-Kim2025`, arXiv:2411.05793, removido — reobter em https://arxiv.org/abs/2411.05793.)
- **Kong et al. (2025) — Deep learning for TSF: a survey.** [DOI 10.1007/s13042-025-02560-w](https://doi.org/10.1007/s13042-025-02560-w) · *verified*. (Ex-local `01-Kong2025`, arXiv:2503.10198, removido — reobter em https://arxiv.org/abs/2503.10198.)
- **Chen et al. (2023) — Long sequence TSF with deep learning.** [DOI 10.1016/j.inffus.2023.101819](https://doi.org/10.1016/j.inffus.2023.101819) · *verified*.

### 3.2 O debate: Transformers funcionam para séries? (leia antes de escolher arquitetura)

- **Zeng et al. (2023) — Are Transformers Effective for Time Series Forecasting?** [DOI 10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317) · 3054 citações · *verified*. Mostra DLinear vencendo Transformers em benchmarks LTSF padrão — a tese que toda a fronteira abaixo tenta rebater. Modelo linear de referência:
$$\hat{y}_{t+1:t+H} = W\,y_{t-L+1:t} + b \quad\text{(+ decomposição tendência/sazonalidade no DLinear)}$$

### 3.3 Núcleo LTSF: atenção eficiente, patches e canais

Atenção padrão (o gargalo que todos atacam), custo quadrático no comprimento $L$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \;\;\Rightarrow\;\; \mathcal{O}(L^2)$$

- **Informer (Zhou et al. 2021).** [DOI 10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325) · 6892 citações · *verified* (+ preprint [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)). **ProbSparse**: mede a esparsidade de cada query e só calcula as top-$u = c\ln L_Q$:
$$M(q_i, K) = \max_j \frac{q_i k_j^\top}{\sqrt{d}} - \frac{1}{L_K}\sum_{j=1}^{L_K}\frac{q_i k_j^\top}{\sqrt{d}} \;\;\Rightarrow\;\; \mathcal{O}(L \log L)$$
- **Autoformer (Wu et al. 2021).** (Ex-local `05-Wu2021` removido — reobter em https://arxiv.org/abs/2106.13008.) Troca atenção por **Auto-Correlation** sobre os top-$k$ lags periódicos (via FFT) + bloco de **decomposição progressiva** interno:
$$y_t = \mathrm{Trend}_t + \mathrm{Seasonal}_t, \quad \mathrm{Trend} = \mathrm{AvgPool}(\mathrm{Pad}(y))$$
- **PatchTST (Nie et al. 2022) — "A Time Series is Worth 64 Words".** [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) · *verified* (preprint). Fatiamento em patches + **channel-independence** (mesmos pesos, forward separado por canal):
$$x \in \mathbb{R}^{L} \;\to\; X_p \in \mathbb{R}^{P \times N}, \quad N = \Big\lfloor\frac{L-P}{S}\Big\rfloor + 1$$
com $P$ = tamanho do patch, $S$ = stride. Reduz tokens de $L$ para $N$ e preserva localidade.
- **iTransformer (Liu et al. 2023).** [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) · *verified* (preprint). **Inversão**: cada série vira um token $h_c \in \mathbb{R}^d$ e a atenção corre sobre os $C$ canais, não sobre o tempo — custo $\mathcal{O}(C^2)$ em vez de $\mathcal{O}(L^2)$:
$$\mathrm{Attention}(H,H,H),\quad H \in \mathbb{R}^{C \times d}$$

### 3.4 Contraponto eficiente: MLP/CNN hierárquicos

- **NHITS (Challú et al. 2023).** [DOI 10.1609/aaai.v37i6.25854](https://doi.org/10.1609/aaai.v37i6.25854) · *verified*. Blocos MLP com pooling de taxas $r_b$ distintas e **interpolação hierárquica** (baixa frequência → alta):
$$\hat{y} = \sum_b \mathrm{Interpolate}(\theta_b), \quad \theta_b = \mathrm{MLP}_b(\mathrm{Pool}(x, r_b))$$
- **TSMixer (Ekambaram et al. 2023).** [DOI 10.1145/3580305.3599533](https://doi.org/10.1145/3580305.3599533) · *verified*. Alterna MLP no eixo do tempo e no eixo das features (mixing), sem atenção.
- **TimesNet (Wu et al. 2022).** [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) · *verified* (preprint). Detecta períodos dominantes por FFT, dobra a série 1D em tensores 2D (período × fase) e aplica conv 2D (Inception).
- **MSGNet (2024).** [DOI 10.1609/aaai.v38i10.28991](https://doi.org/10.1609/aaai.v38i10.28991) · *verified*. Correlações inter-séries multi-escala.

### 3.5 Multi-horizonte interpretável: TFT

- **Lim et al. (2019/2021) — Temporal Fusion Transformer.** Publicado: [DOI 10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012) · *verified*. (Ex-local `04-Lim2019`, arXiv:1912.09363, removido — reobter em https://arxiv.org/abs/1912.09363.) Entradas tripartidas $(s, x, y)$, **GRN** com gating, seleção de variáveis $v_t = \mathrm{softmax}(\mathrm{GRN}(\Xi_t))$, atenção temporal interpretável e saída por quantis com a loss de §3.0:
$$\mathrm{GRN}(a,c) = \mathrm{LayerNorm}\big(a + \mathrm{GLU}(\eta_1)\big), \;\; \eta_2 = \mathrm{ELU}(W_2 a + W_3 c + b_2)$$

### 3.6 Comparativo clássico vs moderno (seus baselines)

ARIMA$(p,d,q)$ — o baseline a bater:

$$\phi(B)\,(1-B)^d\, y_t = \theta(B)\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

com $B$ operador de atraso, $d$ diferenciações até estacionariedade (teste ADF no roteiro §4).

- **Kontopoulou et al. (2023) — ARIMA vs ML.** [DOI 10.3390/fi15080255](https://doi.org/10.3390/fi15080255) · *verified*.
- **Mahajan (2026) — ARIMA vs LSTM vs Prophet.** (Ex-local `08` removido — reobter em https://www.preprints.org/manuscript/202601.1377; registro preservado no `.bib` como metadado local.) Árvore de decisão do guia prévio (full-text removido — confirmar no preprint antes de citar): linear/estacionário → ARIMA; não-linear/volátil → LSTM; sazonalidade + feriados → Prophet; ensembles ponderados como caminho robusto.

### 3.7 Fronteira: representação e foundation (confiança média — evidência escassa no corpus)

- **TS2Vec (2022).** [DOI 10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881) · 763 citações · *verified*. Contraste hierárquico sobre views aumentadas (masking/cropping) para representação universal.
- **Liang et al. (2024) — Foundation Models for Time Series.** [DOI 10.1145/3637528.3671451](https://doi.org/10.1145/3637528.3671451) · *verified*. Tutorial/survey de pré-treino e zero-shot.
- Ex-locais removidos desta trilha: `06-Zhang2025` (NAS para Transformers de séries — reobter em https://arxiv.org/abs/2502.13721), `07-Power-of-Architecture` (tese encoder-only + patching — reobter em https://arxiv.org/html/2507.13043v1, tese marcada como confiança **baixa** no relatório), `09-Liao2026` (survey 2026 — reobter em https://www.osti.gov/pages/servlets/purl/3025651).

### Domínios com evidência no corpus

Tráfego ([Spatio-Temporal Adaptive Embedding](https://doi.org/10.1145/3583780.3615160) · [Meta-Graph](https://doi.org/10.1609/aaai.v37i7.25976)), ações ([Stock index](https://doi.org/10.1016/j.eswa.2022.118128)), trajetórias de pedestres (adjacente, [DOI 10.1007/978-3-030-58610-2_30](https://doi.org/10.1007/978-3-030-58610-2_30)). Energia/saúde/clima **não** têm âncora nominal nos incluídos — gap declarado no relatório.

## 4. Como criar seu modelo (roteiro)

1. **Defina:** univariada ou multivariada? Horizonte `H`? Precisa de intervalo de confiança?
2. **Dados:** dataframe longo `id | ds | y`. Cheque ADF (estacionariedade), sazonalidade, missing.
3. **Janelamento:** transforme em supervisionado. Ex: `lookback=30 → H=7`. Split temporal 70/15/15 sem shuffle. Validação por expanding window.
4. **Baseline 1 (1 dia de trabalho):** ARIMA/SARIMA (`statsmodels`) + Prophet. Métricas: MAE, RMSE, MAPE, SMAPE.
5. **Baseline 2:** LightGBM com lags (`mlforecast`, `Darts`) + LSTM/GRU (`PyTorch`).
6. **Avançado:** `pytorch-forecasting` TFT ou PatchTST (`Time-Series-Library`, `GluonTS`, `NeuralProphet`, `sktime`).
7. **Avaliação + deploy:** backtest, quantile loss se probabilístico, API `FastAPI /predict` + retreino.

Ordem sugerida de leitura (tudo reobtível pelos links da §3): surveys (§3.1, ex.: Wen → Lim/Zohren) → debate Zeng (§3.2) → Mahajan/Kontopoulou para baselines (§3.6) → TFT (§3.5) → Informer → Autoformer → PatchTST/iTransformer (§3.3) → NHITS/TSMixer/TimesNet (§3.4) → TS2Vec + foundation survey (§3.7).

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
- [ ] Reobter full-texts nucleares via links DOI/arXiv da §3 (os PDFs de `artigos/` foram apagados)
- [ ] Definir dataset (ex: `ETTh`, vendas, energia, ação)
- [ ] Criar `notebooks/00-baseline-arima-prophet.ipynb`
- [ ] Criar `src/windowing.py` + `train_lstm.py`
- [ ] Evoluir para `TFT / PatchTST`
- [ ] Expor `app.py` FastAPI

## 8. Busca bibliográfica sistemática — modelos de predição temporal (2026-09-10)

Pipeline `deep-research-br`, template `rigorous`, framework Decomposição (Problema/Solução/Avaliação/Limitações), recorte 2019–2026. Questão: como criar modelos que recebem `y(1..t)` e geram `y(t+1..t+H)`, e quais trade-offs orientam a escolha entre estatísticos, ML, DL e foundation models.

- **Relatório:** [`busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) — síntese temática (não paper-a-paper), mapa de 5 tensões, roteiro de criação, gaps, níveis de confiança e auto-crítica 14pt. Toda afirmação factual tem âncora `[^rank]` ligada ao `corpus.json`.
- **Números do funil (PRISMA):** 133 identificados após dedup (120 brutos federados em 5 queries × OpenAlex/arXiv/EPMC/Crossref + 9 PDFs locais — apagados após a busca, metadados preservados — + 30 achados de chasing forward) → 34 incluídos (24 top por score + 9 locais + TFT canônico) → 1 excluído com motivo → 98 pendentes (skim). Saturação atingida (taxa de novidade 0,0 na 2ª rodada).
- **Integridade:** `verify_citations.py` EXIT 0 — 124 `verified`, 9 `sem-id` (o núcleo local; o `dedupe_rank.py` descarta `arquivo_local` na fusão, documentado no relatório), 0 `not_found/mismatched/retracted` após remover 2 falsos-positivos de astrofísica. Chasing backward sem retorno (melhor esforço, declarado).
- **Achados federados verificados que estendem a seção 3** (todos `verified`, todos incluídos): Informer ([DOI 10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325)), "Are Transformers Effective for Time Series Forecasting?" ([DOI 10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317)), PatchTST "A Time Series is Worth 64 Words" ([arXiv 2211.14730](https://arxiv.org/abs/2211.14730)), iTransformer ([arXiv 2310.06625](https://arxiv.org/abs/2310.06625)), NHITS ([DOI 10.1609/aaai.v37i6.25854](https://doi.org/10.1609/aaai.v37i6.25854)), TSMixer, TimesNet ([arXiv 2210.02186](https://arxiv.org/abs/2210.02186)), MSGNet, TS2Vec, TFT publicado ([DOI 10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012)) e o survey de foundation models (2024). Tabela completa em [`busca_bibliografica/evidence_table.csv`](busca_bibliografica/evidence_table.csv); claims→fontes em [`busca_bibliografica/passport.json`](busca_bibliografica/passport.json).
- **Nota de honestidade (atualizada após apagar os PDFs):** os números de desempenho que constavam no guia prévio (ex.: MAPEs do Mahajan, "+38%" do Autoformer) **não** foram re-afirmados pelo relatório sistemático — os trechos extraídos na sessão (cabeçalhos de 2 páginas via `pdftotext`) não os continham, e os PDFs foram apagados em seguida (ver §§8–10 do relatório). Regra a partir daqui: número de desempenho só entra neste README com citação verificável (DOI/arXiv + tabela/página). As fórmulas da §3 são formulação canônica de livro-texto para estudo, não citação literal dos papers.

> A pasta `artigos/` foi removida do repo. As referências vivem em `busca_bibliografica/` (relatório + `.bib`/`.ris` + `evidence_table.csv`); reobtenha os textos pelos links DOI/arXiv da seção 3 e verifique a licença de cada um antes de redistribuir.
