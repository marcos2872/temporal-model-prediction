# Modelo de Predição Temporal — Base Bibliográfica e Guia Prático

Projeto para criar um modelo de IA que **recebe uma série temporal e gera predição futura**.
Esta pasta organiza a bibliografia fundamental e resume o caminho prático de implementação.

> **Busca bibliográfica sistemática (2026-09-10):** pipeline `rigorous` com 133 registros após dedup → 34 incluídos → `verify` EXIT 0. Relatório completo em [`busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) (+ `.bib`/`.ris`, `evidence_table.csv`, `passport.json`, `prisma.md`). Detalhes na seção 8.

## Estrutura

```
temporal-model/
├── README.md              <- este arquivo (guia + fórmulas + referências)
├── dados/                   <- séries CETESB EF01 Mogi das Cruzes (pH + OD, 5 min) + README
├── notebooks/               <- 9 notebooks 00–08 (ver notebooks/README.md; executados em servidor remoto)
├── resultados/              <- índice + uma pasta por experimento (`00-baseline-ph/`, ...)
├── app.py                   <- API FastAPI (ensembles 06/07, Swagger em /docs)
├── requirements.txt         <- deps (instalar com `uv pip install -r requirements.txt`)
└── busca_bibliografica/   <- kit da busca sistemática (relatório + .bib/.ris + evidence_table.csv + passport.json + prisma.md)
```

> Cobertura: 19 dos 25 citados federados com texto integral open-access (links na §3). Sem OA legal: Chen 2023 (Inf. Fusion), Torres 2020 (Big Data), Kontopoulou 2023 (MDPI, bot-wall), Diagnosisformer, Pedestrian-2020 e Stock-2022 — avaliados por metadados, conforme a regra paywall do relatório §10.

## 1. O que é predição temporal?

Dada uma sequência histórica `y(1)...y(t)` (possivelmente com covariáveis `x(t)`), prever `y(t+1)...y(t+H)` onde `H` é o horizonte.

Formalmente, um modelo de predição temporal aprende:

$$\hat{y}_{t+1:t+H} = f_\theta\big(y_{1:t},\; x_{1:t+H},\; s\big)$$

onde $s$ são covariáveis estáticas (ex.: loja, sensor), $x$ inclui entradas futuras conhecidas (ex.: feriado, promoção) e séries exógenas observadas — a tripartição formalizada pelo TFT (seção 3.6). Na prática o treino usa **janelamento**: cada amostra é `(lookback=L → horizonte=H)` com split temporal sem shuffle.

Desafios clássicos apontados na literatura:
- **Sazonalidade + tendência + ruído** entrelaçados
- **Não-estacionariedade / distribution shift**
- **Dependência de longo prazo** vs custo quadrático do Transformer
- **Multivariada:** dependência entre canais (channel dependence)
- **Incerteza:** previsão pontual vs probabilística (quantis)

## 2. Taxonomia (síntese dos surveys)

| Família | Exemplos | Quando usar | Ref. nuclear (`busca_bibliografica/`) |
|---------|----------|-------------|----------------------------------------|
| Estatísticos | ARIMA, SARIMA, ETS, Holt-Winters | Série curta, linear, estacionária, baseline interpretável | [Kontopoulou et al. 2023](https://doi.org/10.3390/fi15080255) (metadados) · §3.7 |
| ML com lags | XGBoost, LightGBM + lags/date features | Tabular, multivariada, produção, rápido | Ver comparativos em [Kontopoulou et al. 2023](https://doi.org/10.3390/fi15080255) |
| Recorrentes | LSTM, GRU, DeepAR, Bi-LSTM | Não-linear, sequência média, dados limitados | DeepAR/MQCNN em [Benidis et al. 2022](https://doi.org/10.1145/3533382) · §3.1 |
| Convolucionais | TCN, TimesNet | Long-range com paralelismo, padrões 2D por período | [TimesNet (Wu et al. 2022)](https://arxiv.org/abs/2210.02186) |
| Transformers LTSF | Informer, Autoformer, PatchTST, iTransformer, TFT | Long-term, multivariado, multi-horizonte | [Informer](https://doi.org/10.1609/aaai.v35i12.17325) · [PatchTST](https://arxiv.org/abs/2211.14730) · [iTransformer](https://arxiv.org/abs/2310.06625) · [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012) |
| Lineares / MLP modernos | DLinear, NLinear, NHITS, TSMixer | Baseline forte; eficiente contra atenção plena | [Zeng et al. 2023 (o debate)](https://doi.org/10.1609/aaai.v37i9.26317) · [NHITS](https://doi.org/10.1609/aaai.v37i6.25854) · [TSMixer](https://doi.org/10.1145/3580305.3599533) |
| Representação / GNN | TS2Vec, MSGNet | Pré-treino, correlação entre séries | [TS2Vec](https://doi.org/10.1609/aaai.v36i8.20881) · [MSGNet](https://doi.org/10.1609/aaai.v38i10.28991) |
| Foundation / LLM | Chronos, TimeGPT, Lag-Llama | Zero-shot, poucas amostras | [Liang et al. 2024](https://doi.org/10.1145/3637528.3671451) (confiança média — fronteira) |

## 3. Referências nucleares + a matemática de cada família

Fonte: os 34 incluídos da busca sistemática (`busca_bibliografica/evidence_table.csv`, relatório completo [aqui](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md)), com link do artigo em cada entrada. Números de desempenho abaixo foram lidos nos textos integrais (tabelas/seções indicadas). As fórmulas são a **formulação canônica** de cada método — para ler os artigos com o ferramental na mão, não citações literais.

### 3.0 Janelamento e métricas (vale para todos)

Amostra de treino com `lookback=L`, horizonte `H`: entrada `y[i:i+L]`, alvo `y[i+L:i+L+H]`, split temporal sem shuffle + backtest por janela expansiva.

$$\mathrm{MAE} = \tfrac{1}{H}\sum|y-\hat{y}| \quad\quad \mathrm{RMSE} = \sqrt{\tfrac{1}{H}\sum(y-\hat{y})^2}$$

$$\mathrm{MAPE} = \tfrac{100}{H}\sum\Big|\tfrac{y-\hat{y}}{y}\Big| \quad\quad \mathrm{SMAPE} = \tfrac{100}{H}\sum \tfrac{2|y-\hat{y}|}{|y|+|\hat{y}|}$$

Saída probabilística (quantis $q \in \{0.1, 0.5, 0.9\}$) minimiza a **quantile loss**:

$$\mathcal{L}_q(y, \hat{y}) = \max\big(q\,(y-\hat{y}),\; (q-1)\,(y-\hat{y})\big)$$

### 3.1 Surveys — ponto de partida (sem matemática nova, organizam o campo)

- **Wen et al. (2023) — Transformers in Time Series: A Survey.** [IJCAI](https://doi.org/10.24963/ijcai.2023/759) · 1103 citações · *verified*. Taxonomia em dois eixos — modificação de rede (positional/timestamp encoding, atenção esparsa, decomposição sazonal-tendência) × tarefa (forecasting, anomalia, classificação) — e a tabela de complexidades que resume a fronteira eficiente: Transformer $\mathcal{O}(N^2)$ → LogTrans/Informer/Autoformer $\mathcal{O}(N\log N)$ → Pyraformer/FEDformer $\mathcal{O}(N)$. Discute ainda o debate Zeng/DLinear e o papel de timestamps calendáricos + especiais como positional encoding.
- **Lim & Zohren (2021) — Time-series forecasting with deep learning: a survey.** [Phil. Trans. R. Soc. A](https://doi.org/10.1098/rsta.2020.0209) · 1770 citações · *verified*. Organiza por blocos (encoders/decoders), saídas pontuais vs probabilísticas, forecasting iterativo vs direto, e fecha em híbridos (redes que geram parâmetros de modelos clássicos) e interpretabilidade para decisão.
- **Torres et al. (2020) — Deep Learning for Time Series Forecasting: A Survey.** [Big Data](https://doi.org/10.1089/big.2020.0159) · 816 citações · *verified* · texto integral paywall (só metadados).
- **Benidis et al. (2022) — Tutorial and Literature Survey.** [ACM Comput. Surv.](https://doi.org/10.1145/3533382) · *verified*. A distinção que estrutura avaliação em painel: **modelos globais** (um NN treinado em todas as $N$ séries, ex.: DeepAR — RNN que emite $\mu_t,\sigma_t$ de Gaussiana com loss NLL) vs **locais** (ARIMA/ETS/SSM por série, exatos via Kalman) vs **híbridos/global-local** (Deep State Space: RNN parametriza SSM linear-Gaussiano; ES-RNN; N-BEATS). Recomendação explícita: NNs como modelos globais quando há dados suficientes; métricas MASE/sMAPE/quantile loss.
- **Kim et al. (2025) — Architectural Diversity and Open Challenges.** [Artif. Intell. Rev.](https://doi.org/10.1007/s10462-025-11223-9) · [cópia OA arXiv](https://arxiv.org/abs/2411.05793) · *verified*. Taxonomias de híbridos, diffusion, Mamba e foundation models + 4 desafios abertos: channel dependency (CI vs CD), distribution shift (normalização), causalidade e extração de features.
- **Kong et al. (2025) — Deep learning for TSF: a survey.** [IJMLC](https://doi.org/10.1007/s13042-025-02560-w) · [cópia OA arXiv](https://arxiv.org/abs/2503.10198) · *verified*. Foco diferencial em extração de features (decomposição, tempo-frequência, pré-treino, patches) e compilação de datasets por domínio: energia, saúde, transporte, meteorologia, economia.
- **Chen et al. (2023) — Long sequence TSF with deep learning.** [Inf. Fusion](https://doi.org/10.1016/j.inffus.2023.101819) · *verified* · texto integral paywall (só metadados).

### 3.2 O debate: Transformers funcionam para séries? (leia antes de escolher arquitetura)

- **Zeng et al. (2023) — Are Transformers Effective for Time Series Forecasting?** [AAAI](https://doi.org/10.1609/aaai.v37i9.26317) · 3054 citações · *verified*. Tese, em 9 datasets reais: o **LTSF-Linear** (uma camada linear no eixo temporal, com variantes Linear/NLinear/DLinear) supera Transformers complexos — porque self-attention é **permutation-invariant** (perde ordem temporal) e os ganhos vêm de decomposição tendência-sazonalidade + tratamento de distribution shift, não da atenção. Achado prático: aumentar o look-back melhora os lineares e degrada Transformers com ruído. Modelo de referência:
$$\hat{y}_{t+1:t+H} = W\,y_{t-L+1:t} + b \quad\text{(+ decomposição tendência/sazonalidade no DLinear)}$$

### 3.3 Núcleo LTSF: atenção eficiente, patches e canais

Atenção padrão (o gargalo que todos atacam), custo quadrático no comprimento $L$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \;\;\Rightarrow\;\; \mathcal{O}(L^2)$$

- **Informer (Zhou et al. 2021).** [AAAI](https://doi.org/10.1609/aaai.v35i12.17325) · [preprint arXiv:2012.07436](https://arxiv.org/abs/2012.07436) (near-duplicata preprint↔publicado no corpus) · 6892 citações · *verified*. Três ideias: **(i) ProbSparse** — mede a esparsidade de cada query e só calcula as top-$u = c\ln L_Q$; **(ii) distilling** — max-pool stride 2 que comprime o encoder pela metade a cada camada, privilegiando features dominantes; **(iii) decoder generativo** — prevê a sequência longa de uma vez a partir de um start-token, sem rollout autoregressivo. Entrada uniforme = embedding posicional + timestamp + valor (dataset ETT: temperatura do óleo + 6 cargas):
$$M(q_i, K) = \max_j \frac{q_i k_j^\top}{\sqrt{d}} - \frac{1}{L_K}\sum_{j=1}^{L_K}\frac{q_i k_j^\top}{\sqrt{d}} \;\;\Rightarrow\;\; \mathcal{O}(L \log L)$$
- **PatchTST (Nie et al. 2022) — "A Time Series is Worth 64 Words".** [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) · *verified* (preprint). Fatiamento em patches + **channel-independence** (mesmos pesos, forward separado por canal) + instance norm (reversível, contra shift). Números do paper: **−21,0% MSE / −16,7% MAE** (PatchTST/64 vs FEDformer); só o patching (P=16, S=8, L=336) já derruba o MSE de 0,397→0,367; pré-treino mascarado (40% dos patches, 100 épocas) + fine-tuning vence o supervisionado puro:
$$x \in \mathbb{R}^{L} \;\to\; X_p \in \mathbb{R}^{P \times N}, \quad N = \Big\lfloor\frac{L-P}{S}\Big\rfloor + 1$$
com $P$ = tamanho do patch, $S$ = stride. Reduz tokens de $L$ para $N$ e preserva localidade.
- **iTransformer (Liu et al. 2023).** [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) · *verified* (preprint). **Inversão**: cada série inteira vira um token $h_c \in \mathbb{R}^d$; a atenção modela correlações **entre canais** e o FFN + LayerNorm codifica a série — custo $\mathcal{O}(C^2)$ em vez de $\mathcal{O}(L^2)$, sem positional embedding. Números do paper: SOTA geral incl. PEMS (onde o PatchTST falha) e **+38,9% de promoção média de MSE** ao aplicar a inversão sobre Transformers padrão (Tab. 2):
$$\mathrm{Attention}(H,H,H),\quad H \in \mathbb{R}^{C \times d}$$

### 3.4 Contraponto eficiente: MLP/CNN hierárquicos

- **NHITS (Challú et al. 2023).** [AAAI](https://doi.org/10.1609/aaai.v37i6.25854) · *verified*. Estende o N-BEATS com **multi-rate sampling** (MaxPool por bloco) + **interpolação hierárquica** (cada stack especializa-se numa banda de frequência, baixa→alta). Números do paper: **~+20% de acurácia sobre os Transformers mais recentes**; −14% MAE / −16% MSE no long-horizon; só 26% dos parâmetros da 2ª melhor alternativa; 1,26× mais rápido e 54% dos parâmetros do N-BEATS original:
$$\hat{y} = \sum_b \mathrm{Interpolate}(\theta_b), \quad \theta_b = \mathrm{MLP}_b(\mathrm{Pool}(x, r_b))$$
- **TSMixer (Ekambaram et al. 2023).** [KDD'23 workshop](https://doi.org/10.1145/3580305.3599533) · [cópia OA arXiv](https://arxiv.org/abs/2306.09364) · *verified*. Backbone MLP-Mixer (mixing no eixo do tempo × eixo das features, sem atenção) em 3 variantes — vanilla, channel-independent e inter-channel — com **reconciliation heads online** (cross-channel + hierárquica por patches) e gated attention. Números do paper: **+8–60% sobre MLP/Transformers**, +1–2% sobre Patch-Transformers, com **2–3× menos memória/tempo**.
- **TimesNet (Wu et al. 2022).** [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) · *verified* (preprint). Detecta os top-$k$ períodos por FFT e dobra a série 1D em tensores 2D (colunas = intraperíodo, linhas = entre-períodos), processados por Inception 2D — SOTA consistente em **5 tarefas** (forecasting curto/longo, imputação, classificação, detecção de anomalia):
$$A = \mathrm{Avg}(\mathrm{Amp}(\mathrm{FFT}(X_{1D}))), \quad X^{2D}_i = \mathrm{Reshape}_{p_i,f_i}(\mathrm{Pad}(X_{1D}))$$
- **MSGNet (Cai et al. 2024).** [AAAI](https://doi.org/10.1609/aaai.v38i10.28991) · *verified*. Escalas via FFT + **grafo adaptativo MixHop por escala** (matriz de adjacência aprendida) para correlações inter-séries que mudam com a escala. Números do paper: no dataset Flight (COVID, OOD) **MSE 0,265→0,208 (−21,5%) e MAE −13,7% vs TimesNet**; melhor rank médio em 8 datasets (Flight, Weather, ETT×4, Exchange, Electricity).

### 3.5 Tráfego espaço-temporal (benchmarks honestos: METR-LA, PEMS-BAY, PEMS03/04/07/08)

- **STAEformer (Liu et al. 2023, CIKM).** [DOI](https://doi.org/10.1145/3583780.3615160) · *verified*. Um embedding adaptativo $E_a \in \mathbb{R}^{T \times N \times d_a}$ + **Transformer vanilla** (eixos temporal e espacial) basta para SOTA em 6 datasets de tráfego; a ablação sem $E_a$ degrada fortemente (ex.: PEMS04 MAPE 12,01%→14,26%):
$$Z = \mathrm{Transformer}_{espacial}(\mathrm{Transformer}_{temporal}([X \,\|\, E_{per} \,\|\, E_a]))$$
- **MegaCRN (Jiang et al. 2023, AAAI).** [DOI](https://doi.org/10.1609/aaai.v37i7.25976) · *verified*. **Meta-Node Bank**: protótipos de padrões de tráfego consultados por atenção geram embeddings aumentados por memória, de onde sai um meta-grafo adaptativo (treino: MAE + triplet loss nos protótipos). SOTA em METR-LA/PEMS-BAY + dataset novo EXPY-TKY, com só **133.597 parâmetros**.

### 3.6 Multi-horizonte interpretável: TFT

- **Lim et al. (2019/2021) — Temporal Fusion Transformer.** Publicado: [Int. J. Forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012) · [cópia OA arXiv](https://arxiv.org/abs/1912.09363) · *verified*. Entradas tripartidas $(s, x, y)$, **GRN** com gating, seleção de variáveis $v_t = \mathrm{softmax}(\mathrm{GRN}(\Xi_t))$, atenção temporal interpretável e saída por quantis $Q = \{0.1, 0.5, 0.9\}$ com a loss de §3.0. Avaliação em 4 datasets reais — **Electricity, Traffic, Retail (Favorita) e Volatility (OMI)** — com a Tab. 2 mostrando TFT à frente de DeepAR, MQRNN, ARIMA/ETS e TRMF (percentuais de q-Risk vs TFT por dataset e quantil):
$$\mathrm{GRN}(a,c) = \mathrm{LayerNorm}\big(a + \mathrm{GLU}(\eta_1)\big), \;\; \eta_2 = \mathrm{ELU}(W_2 a + W_3 c + b_2)$$

### 3.7 Comparativo clássico vs moderno (seus baselines)

ARIMA$(p,d,q)$ — o baseline a bater:

$$\phi(B)\,(1-B)^d\, y_t = \theta(B)\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

com $B$ operador de atraso, $d$ diferenciações até estacionariedade (teste ADF no roteiro §4).

- **Kontopoulou et al. (2023) — ARIMA vs ML em redes.** [Future Internet](https://doi.org/10.3390/fi15080255) · *verified* · texto integral indisponível (bot-wall no OA) — citação por metadados verificados.

### 3.8 Fronteira: representação e foundation

- **TS2Vec (Yue et al. 2022, AAAI).** [DOI](https://doi.org/10.1609/aaai.v36i8.20881) · 763 citações · *verified*. Contraste **hierárquico** (instance-wise + temporal) sobre views aumentadas por **timestamp masking + random cropping** ("contextual consistency" em vez de invariância a transformações de imagem) — representação universal para classificação, forecasting e anomalia, robusta a dados faltantes. Encoder: projeção + dilated CNN; [código](https://github.com/yuezhihan/ts2vec).
- **Liang et al. (2024) — Foundation Models for Time Series.** [DOI](https://doi.org/10.1145/3637528.3671451) · *verified*. Taxonomia metodologia-cêntrica: arquitetura (Transformer / não-Transformer / diffusion) × pré-treino (supervisionado / generativo / híbrido) × adaptação (**zero-shot**, fine-tuning, prompt). Cobre Lag-Llama e TimeGPT-1 (decoder-only), TimesFM, Moirai (+ LOTSA, 27B pontos), Chronos, TimeGrad/DiffSTG e a tese de escolha de framework ainda em disputa.

> Sem texto integral OA (citados por metadados verificados, regra paywall do relatório §10): Chen 2023 ([Inf. Fusion](https://doi.org/10.1016/j.inffus.2023.101819)), Torres 2020 ([Big Data](https://doi.org/10.1089/big.2020.0159)), Diagnosisformer 2023 ([Eng. Appl. AI](https://doi.org/10.1016/j.engappai.2023.106507), adjacente — diagnóstico, não forecasting), Pedestrian-2020 ([capítulo Springer](https://doi.org/10.1007/978-3-030-58610-2_30), adjacente — trajetórias) e Stock-2022 ([ESWA](https://doi.org/10.1016/j.eswa.2022.118128)).

### Domínios com evidência no corpus

Tráfego — o benchmark mais honesto do corpus, com tabelas comparáveis: METR-LA/PEMS-BAY/PEMS03/04/07/08 ([STAEformer §3.5](https://doi.org/10.1145/3583780.3615160) · [MegaCRN §3.5](https://doi.org/10.1609/aaai.v37i7.25976)); energia/cargas (Electricity, ETT) e varejo/volatilidade ([TFT §3.6](https://doi.org/10.1016/j.ijforecast.2021.03.012)); ações ([Stock index](https://doi.org/10.1016/j.eswa.2022.118128), metadados); trajetórias de pedestres (adjacente, [DOI 10.1007/978-3-030-58610-2_30](https://doi.org/10.1007/978-3-030-58610-2_30), metadados). Saúde/clima **não** têm âncora nominal nos incluídos — gap declarado no relatório.

## 4. Como criar seu modelo (roteiro)

1. **Defina:** univariada ou multivariada? Horizonte `H`? Precisa de intervalo de confiança?
2. **Dados:** dataframe longo `id | ds | y`. Cheque ADF (estacionariedade), sazonalidade, missing.
3. **Janelamento:** transforme em supervisionado. Ex: `lookback=30 → H=7`. Split temporal 70/15/15 sem shuffle. Validação por expanding window.
4. **Baseline 1 (1 dia de trabalho):** ARIMA/SARIMA (`statsmodels`) + Prophet. Métricas: MAE, RMSE, MAPE, SMAPE.
5. **Baseline 2:** LightGBM com lags (`mlforecast`, `Darts`) + LSTM/GRU (`PyTorch`).
6. **Avançado:** `pytorch-forecasting` TFT ou PatchTST (`Time-Series-Library`, `GluonTS`, `NeuralProphet`, `sktime`).
7. **Avaliação + deploy:** backtest, quantile loss se probabilístico, API `FastAPI /predict` + retreino.

Ordem sugerida de leitura (links na §3): surveys (§3.1, ex.: Wen → Lim/Zohren → Benidis) → debate Zeng (§3.2) → Kontopoulou + ARIMA (§3.7) → TFT (§3.6) → Informer → PatchTST/iTransformer (§3.3) → NHITS/TSMixer/TimesNet/MSGNet (§3.4) → tráfego STAEformer/MegaCRN (§3.5) → TS2Vec + foundation survey (§3.8).

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
- [x] Definir dataset — CETESB EF01 Mogi das Cruzes, univariado (pH e OD em `dados/`, ver `dados/README.md`)
- [x] Criar `notebooks/00-baseline-ph.ipynb` (regime anual: baselines no pH 2024, treino + val 4 fatias — executado em servidor remoto; régua: **sazonal-naive 0,0421**; Prophet colapsa 0,1703; artefatos em `resultados/00-baseline-ph/`)
- [x] Criar `notebooks/01-baseline-od.ipynb` (baselines no OD 2024 — executado em servidor remoto; régua: **sazonal-naive 0,1579**; Prophet colapsa 0,4091; artefatos em `resultados/01-baseline-od/`)
- [x] Criar `notebooks/02-lstnet-ph.ipynb` (LSTNet nativo no pH 2024 — executado em servidor remoto; **NOVA RÉGUA: 0,0373, −11,4% sobre o sazonal**; artefatos em `resultados/02-lstnet-ph/`)
- [x] Criar `notebooks/03-lstnet-od.ipynb` (LSTNet no OD 2024 — executado em servidor remoto; **NOVA RÉGUA (1ª vez no OD): 0,1380, −12,6%**; artefatos em `resultados/03-lstnet-od/`)
- [x] Criar `notebooks/04-patchtst-ph.ipynb` (PatchTST + DLinear no pH 2024 — executado em servidor remoto; régua segue LSTNet; dlinear 0,0394 > patchtst 0,0414 (overfita); artefatos em `resultados/04-patchtst-ph/`)
- [x] Criar `notebooks/05-patchtst-od.ipynb` (os três no OD 2024 — executado em servidor remoto; régua segue LSTNet; patchtst/dlinear ~0,143; artefatos em `resultados/05-patchtst-od/`)
- [x] Criar `notebooks/06-ensemble-ph.ipynb` (ensemble NNLS no pH 2024 — executado em servidor remoto; **NOVA RÉGUA: ens 0,0357, −4,3%**; LGBM zerado; artefatos em `resultados/06-ensemble-ph/`)
- [x] Criar `notebooks/07-ensemble-od.ipynb` (ensemble NNLS no OD 2024 — executado em servidor remoto; **NOVA RÉGUA: ens 0,1325, −4,0%**; dlres com peso 0,30; artefatos em `resultados/07-ensemble-od/`)
- [x] Criar `notebooks/08-benchmark-2025.ipynb` (todos os campeões × 2025 intocado + lag-365, só inferência em servidor remoto; **réguas finais: pH ens 0,0509 (−15%), OD ens 0,2107 (−22%)**; lag-365 inútil 0,46/0,93; Prophet explode; artefatos em `resultados/08-benchmark-2025/`)
- [x] Expor `app.py` FastAPI (ensembles 06/07 servidos localmente; `POST /prever` com CSV CETESB + `?horizonte_horas=1..24`; Swagger em `/docs`; golden test vs recomputação ±5e-5)
- [ ] Saída probabilística (quantis) e teste de transferência para 2026 quando houver dado validado

## 7b. API de previsão (local)

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
# Swagger UI: http://127.0.0.1:8000/docs  (botão "Try it out" no POST /prever)
curl -X POST "http://127.0.0.1:8000/prever?variavel=ph&horizonte_horas=24" \
  -F "arquivo=@teste_api_ph.csv"
```

Envie um CSV CETESB (pH ou OD, ≥ ~8 dias a cada 5 min) e receba 12–288 valores do
período seguinte com timestamps. O pipeline validado roda sempre 24 h e devolve o
prefixo pedido; gaps > 2 h no fim da série retornam `422` em vez de prever no escuro.
Modelos carregados no startup a partir de `resultados/` (o `.pkl` do LGBM é gitignored —
para deploy, copie `resultados/0{6,7}-*/modelos/lgbm_steps.pkl` junto).

## 8. Busca bibliográfica sistemática — modelos de predição temporal (2026-09-10)

Pipeline `deep-research-br`, template `rigorous`, framework Decomposição (Problema/Solução/Avaliação/Limitações), recorte 2019–2026. Questão: como criar modelos que recebem `y(1..t)` e geram `y(t+1..t+H)`, e quais trade-offs orientam a escolha entre estatísticos, ML, DL e foundation models.

- **Relatório:** [`busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) — síntese temática (não paper-a-paper), mapa de 5 tensões, roteiro de criação, gaps, níveis de confiança e auto-crítica 14pt. Toda afirmação factual tem âncora `[^rank]` ligada aos registros verificados (ver `evidence_table.csv`).
- **Números do funil (PRISMA):** 133 identificados após dedup (120 brutos federados em 5 queries × OpenAlex/arXiv/EPMC/Crossref + 9 PDFs locais da época + 30 achados de chasing forward) → 34 incluídos (24 top por score + 9 locais + TFT canônico) → 1 excluído com motivo → 98 pendentes (skim). Saturação atingida (taxa de novidade 0,0 na 2ª rodada).
- **Integridade:** `verify_citations.py` EXIT 0 — 124 `verified`, 9 `sem-id` (o núcleo local; o `dedupe_rank.py` descarta `arquivo_local` na fusão, documentado no relatório), 0 `not_found/mismatched/retracted` após remover 2 falsos-positivos de astrofísica. Chasing backward sem retorno (melhor esforço, declarado).
- **Achados federados verificados que estendem a seção 3** (todos `verified`, todos incluídos): Informer ([DOI 10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325)), "Are Transformers Effective for Time Series Forecasting?" ([DOI 10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317)), PatchTST "A Time Series is Worth 64 Words" ([arXiv 2211.14730](https://arxiv.org/abs/2211.14730)), iTransformer ([arXiv 2310.06625](https://arxiv.org/abs/2310.06625)), NHITS ([DOI 10.1609/aaai.v37i6.25854](https://doi.org/10.1609/aaai.v37i6.25854)), TSMixer, TimesNet ([arXiv 2210.02186](https://arxiv.org/abs/2210.02186)), MSGNet, TS2Vec, TFT publicado ([DOI 10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012)) e o survey de foundation models (2024). Tabela completa em [`busca_bibliografica/evidence_table.csv`](busca_bibliografica/evidence_table.csv); claims→fontes em [`busca_bibliografica/passport.json`](busca_bibliografica/passport.json).
- **Nota de honestidade:** os números de desempenho na §3 foram lidos nos textos integrais open-access linkados em cada entrada — exceto os 6 registros paywall/bot-wall, citados por metadados verificados. Regra permanente: número novo só entra com citação verificável (autor + tabela/página do artigo). As fórmulas da §3 são formulação canônica para estudo, não citação literal dos papers.

> Todos os links da §3 são open-access (editoras OA, AAAI/IJCAI, arXiv) para estudo pessoal — verifique a licença de cada um antes de redistribuir. Registros paywall (Chen, Torres, Kontopoulou, Diagnosisformer, Pedestrian, Stock) vivem só como metadados em `busca_bibliografica/`.
