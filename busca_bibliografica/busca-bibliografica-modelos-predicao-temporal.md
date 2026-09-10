# Modelos de Predição Temporal: como criar — busca bibliográfica sistemática (2019–2026)

**Questão (PICO):** modelos-predicao-temporal · **Tipo:** revisão sistemática PRISMA 2020 · **Protocolo:** reviews/modelos-predicao-temporal/protocol.md

## Resumo estruturado

Contexto / Objetivos / Métodos (bases, strings, critérios) / Resultados / Conclusões.

## Métodos

Bases, data da busca, strings exatas, critérios inclusão/exclusão, processo de triagem (duplo-pass), avaliação de qualidade.

## Resultados

Fluxo PRISMA (ver reviews/modelos-predicao-temporal/prisma.md) + tabela de extração + síntese por outcome.

## 1. Métodos da busca (reprodutível)

Questão (Decomposição — Problema/Solução/Avaliação/Limitações): como criar modelos de predição temporal que recebem `y(1..t)` e geram `y(t+1..t+H)`, e quais arquiteturas, dados, janelamento, avaliação e trade-offs orientam a escolha entre estatísticos, ML, DL e foundation models. Protocolo em `reviews/modelos-predicao-temporal/protocol.md`.

Cinco sub-queries mapeadas ao framework, todas logadas com data/fonte/retorno em `reviews/modelos-predicao-temporal/search_log.md`: survey geral de deep learning para forecasting [^4] [^12]; Transformers para long-term (PatchTST/Autoformer) [^6] [^14]; multi-horizonte probabilístico interpretável (TFT) [^40]; benchmarks e comparativos ARIMA/LSTM/Prophet [^13]; foundation/zero-shot (Chronos/Mamba) [^24]. Fontes: OpenAlex, arXiv, EuropePMC e Crossref (todas com API pública, sem key) mais o corpus local de 9 PDFs em `artigos/` [^92] [^93] [^106] [^107] [^108] [^121] [^125] [^129] [^133]. Recorte 2019–2026, com os canônicos TFT 2019/2021 [^40] [^129] e Autoformer 2021 [^125] mantidos por decisão de protocolo.

Fluxo PRISMA (`reviews/modelos-predicao-temporal/prisma.md`): 133 identificados após dedup → 133 triados → 1 excluído → 34 incluídos → 98 pendentes (skim não priorizado). Dedup por DOI > arXiv ID > título normalizado; score = 0,5·citação + 0,3·recência + 0,2·cobertura. Saturação atingida na 2ª rodada (21 novos em 135, taxa 0,156 < 0,20). Chasing forward via OpenAlex a partir de 3 sementes (Informer [^3], TFT publicado [^40], PatchTST [^6]) retornou 30 achados; backward não retornou registros (melhor esforço, registrado). Limpeza de Gate 7: 2 falsos-positivos do arXiv (astrofísica, "Deep Lens/Pencil Beam") removidos como wrong-population + `not_found`, com re-dedup e re-verify.

Gate de integridade: `verify_citations.py` EXIT 0 — 124 `verified`, 9 `sem-id` (os 9 locais, cujo `arquivo_local` é descartado pelo `dedupe_rank.py` na fusão e por isso caem em `sem-id`; avaliados por full-text local), 0 `not_found/mismatched/retracted` após a limpeza. Os 9 `sem-id` são exatamente o núcleo local [^92] [^93] [^106] [^107] [^108] [^121] [^125] [^129] [^133].

## 2. Panorama: o que o corpus incluído cobre

Surveys/revisões gerais e de Transformers (alta cobertura, múltiplas fontes): [^1] [^4] [^8] [^9] [^12] [^17] [^22], mais os locais [^93] [^106] [^107] [^121]. Debate "Transformers funcionam?" [^2]. Long-term eficiente: Informer [^3] [^25] (preprint+publicado, near-duplicata), PatchTST [^6], Autoformer local [^125], revisão sistemática de LTSF com Transformers [cobertura via busca; ver núcleo]. Multi-horizonte interpretável: TFT publicado [^40] e local [^129] (near-duplicata). Alternativas a atenção plena: NHITS hierárquico [^5], TSMixer MLP-Mixer [^10], TimesNet 2D-variation [^15], iTransformer invertido [^14], MSGNet multi-escala inter-séries [^23], representação universal TS2Vec [^11]. Comparativo clássico vs moderno: ARIMA vs ML [^13] e o local ARIMA/LSTM/Prophet [^92]. Foundation: tutorial/survey [^24]. Domínios observáveis nos títulos dos incluídos: tráfego [^16] [^19], índice de ações [^21], trajetórias de pedestres [^20] — este último e [^18] (diagnóstico de rolamentos, adjacente, não forecasting puro) vieram do snowball e são tratados como contexto adjacente, não núcleo. Near-duplicatas preprint↔publicado (Informer [^3]↔[^25]) e publicado↔local (TFT [^40]↔[^129]) foram mantidas e sinalizadas em vez de fundidas, por chaves DOI distintas.

## 3. Taxonomia para criar modelos (Problema → Solução)

**Problema.** Prever `y(t+1..t+H)` a partir de `y(1..t)` e covariáveis; o TFT formaliza a mistura de entradas que o projeto deve modelar desde o dia 1: covariáveis estáticas, entradas futuras conhecidas e séries exógenas observadas [^129] [^40] — confiança alta (abstract local recuperado + registro publicado cruzado). Multi-horizonte e probabilisticidade (quantis) são o regime do TFT [^40], não um acessório.

**Solução — famílias.** (a) Estatísticos/ML com lags: ARIMA e congêneres seguem como baseline interpretável e o comparativo ARIMA-vs-ML estrutura quando confrontá-los com DL [^13] [^92] — confiança alta quanto à existência do comparativo, média quanto ao "quando usar" (depende do regime; ver tensões). (b) Transformers long-term: Informer [^3], PatchTST [^6], Autoformer [^125], iTransformer [^14] — confiança alta de que são o núcleo LTSF do corpus. (c) MLP/CNN hierárquicos e leves: NHITS [^5], TSMixer [^10], TimesNet [^15] — confiança alta como contraponto eficiente à atenção plena. (d) Representação e fundação: TS2Vec [^11], MSGNet [^23], foundation survey [^24] — confiança média (emergente, menos replicação no corpus). (e) Interpretáveis multi-horizonte: TFT [^40] [^129] — confiança alta para a arquitetura e o regime de entradas.

**Solução — detalhe recuperado em texto (não só título).** Do cabeçalho/abstract extraído via `pdftotext` (2 primeiras páginas): TFT combina covariáveis estáticas + futuras conhecidas + exógenas para multi-horizonte [^129]; Autoformer propõe Auto-Correlation com decomposição progressiva para séries longas [^125]; os surveys locais organizam o campo por arquitetura e desafios abertos (dependência entre canais, shift distribucional, causalidade) [^106] [^107]; o comparativo local estrutura ARIMA vs LSTM vs Prophet com ensembles ponderados como caminho robusto [^92]; o survey 2026 cobre RNN/CNN/GNN/Transformer/LLM/MLP/Diffusion com tabela comparativa [^93]. Números de desempenho (ex.: MAPE, "38%", "O(L log L)") **não** são afirmados aqui porque os trechos recuperados nesta sessão não os contêm — ver Limitações.

## 4. Avaliação (como testar o seu modelo)

O corpus incluído legitima avaliação comparativa contra baselines clássicos e modernos [^13] [^92], benchmarking em domínios reais (tráfego [^16] [^19], ações [^21]) e a tradição de surveys que consolidam métodos por literatura [^4] [^9]. Recomendação derivada (síntese, confiança média): split temporal sem shuffle, backtest por janela expansiva, baseline 1 (estatístico) → baseline 2 (ML com lags/recorrente) → avançado (TFT/PatchTST/iTransformer), com saída por quantis quando incerteza importar [^40]. Métricas específicas e datasets nominais (ETTh/Electricity) **não** são afirmados por falta de âncora nesta sessão.

## 5. Mapa de tensões (o que está em disputa)

- **T1 — Transformers vs. alternativas simples.** [^2] questiona a efetividade de Transformers para forecasting; respondem com fixes arquiteturais o PatchTST por patches [^6], o iTransformer invertido [^14], o TSMixer sem atenção [^10], o NHITS hierárquico [^5] e o TimesNet 2D [^15] — confiança alta na existência do debate, média no veredito (depende de horizonte, canais e protocolo de avaliação).
- **T2 — Dependência entre canais.** iTransformer inverte a atenção para o canal [^14] e MSGNet modela correlações inter-séries multi-escala [^23], contra a tradição canal-independente do PatchTST [^6]; os surveys locais listam channel-dependency como desafio aberto [^107] — confiança média.
- **T3 — Interpretabilidade vs. acurácia.** TFT entrega seleção de variáveis e saídas por quantis ao custo de arquitetura mais complexa [^40] [^129] — confiança alta na arquitetura, média no trade-off quantitativo.
- **T4 — Foundation/zero-shot vs. especializado.** O survey de foundation models [^24] e a representação universal [^11] prometem transferência; evidência replicada no corpus é escassa — confiança baixa/média, gap explícito.
- **T5 — Clássico vs. profundo.** Os comparativos [^13] [^92] sustentam que não há vencedor universal — confiança média no princípio "baseline primeiro, DL quando há não-linearidade/volume".

## 6. Roteiro de criação (derivação prática, não citação literal)

1. Defina H, granularidade e se precisa de quantis; modele desde o início os três tipos de entrada do TFT [^40] [^129]. 2. Monte dataframe longo temporal e janelas lookback→H com split temporal e backtest (protocolo padrão dos comparativos [^13] [^92], confiança média). 3. Baselines: estatístico + ML com lags [^13] [^92]. 4. DL: recorrente/CNN leve, depois NHITS/TSMixer/TimesNet como intermediários eficientes [^5] [^10] [^15]. 5. Long-term multivariado: PatchTST/iTransformer/Autoformer/Informer [^6] [^14] [^125] [^3]. 6. Multi-horizonte interpretável/probabilístico: TFT [^40]. 7. Fronteira: representação [^11] e foundation [^24] como trilha separada, com confiança menor.

## 7. Gaps e próximos passos

Gap 1: veredito quantitativo Transformers-vs-lineares sem âncora numérica nesta sessão (só títulos) — exige extração de resultados dos full-texts locais [^106] [^107] [^133]. Gap 2: foundation/zero-shot com um survey e um modelo de representação apenas [^24] [^11] — precisa busca dedicada. Gap 3: metodologia de avaliação (métricas, testes de significância, protocolos de backtest) aparece como prática dos comparativos [^13] [^92] mas sem extração de tabela — gerar `evidence_table.csv` completa é o próximo passo (este kit entrega a v1 a partir de metadados). Gap 4: domínios energia/saúde/clima, comuns na literatura, não têm âncora nominal nos incluídos desta sessão — declarar como não coberto.

## 8. Níveis de confiança (resumo auditável)

Alta: taxonomia das famílias e núcleo LTSF (múltiplas fontes cruzadas: [^1] [^3] [^4] [^6] [^14] [^40]); regime de entradas do TFT [^40] [^129]; existência dos comparativos clássico-vs-moderno [^13] [^92]; integridade do corpus (verify EXIT 0). Média: trade-offs (T1/T2/T3/T5) e roteiro de avaliação; fronteira foundation [^24] [^11] [^23]. Baixa: qualquer número de desempenho; tese "encoder-only com patching vence" do local [^133] (só cabeçalho recuperado — exige leitura do full-text em `artigos/`); cobertura de domínios além de tráfego/ações/trajetórias.

## 9. Auto-crítica (checklist 14pt — resumo executivo; detalhe em `reviews/modelos-predicao-temporal/`)

1 SIM (âncoras só em incluídos; [^18] [^20] sinalizados como adjacentes). 2 SIM (preprints [^6] [^14] [^15] [^25] rotulados; locais sem DOI como `sem-id`/full-text local). 3 SIM (todos os IDs vieram de respostas de API ou do nome/conteúdo dos PDFs locais; nenhum digitado de memória). 4 SIM (`verification.json`: 124 verified, 9 sem-id locais, 0 problemas; EXIT 0). 5 SIM (canônicos Informer [^3], TFT [^40] [^129], Autoformer [^125], PatchTST [^6] incluídos). 6 SIM (5 queries × 3 fontes + local, `search_log.md` com datas/queries/retornos). 7 SIM (saturação 0,156→0,0, `dedupe_rank.py`). 8 PARCIAL-SIM (forward executado, 30 achados; backward sem retorno — declarado). 9 SIM (exclusão 1 com motivo; 2 ruídos documentados e removidos; pendentes declarados). 10 SIM (síntese temática + tensões, não paper-a-paper). 11 SIM (alta/média/baixa marcadas). 12 SIM (limitações declaradas abaixo). 13 Devil's Advocate: objeção A "Transformers são desnecessários" — réplica sustenta T1 com [^2] [^5] [^6] [^10] [^14] [^15], nota 4/5, concede-se o debate mas não o veredito; objeção B "Foundation resolve tudo" — réplica nega por evidência escassa [^24] [^11], nota 2/5, mantém-se a posição cética. 14 SIM (pós-processamento: 34 âncoras no corpo, 34 definições em Referências, 1:1; .bib/.ris com 133 registros do corpus em `reports/` + kit em `busca_bibliografica/`).

## 10. Limitações do pipeline (declaradas)

Bases sem API pública não consultadas (Scopus/WoS/Google Scholar); EPMC devolveu só contexto biomédico marginal; paywall avaliado só por metadados; extração profunda limitada ao cabeçalho de 2 páginas dos PDFs (números de tabelas/resultados não extraídos — por isso nenhum MAPE/percentual é afirmado); `dedupe_rank.py` descarta `arquivo_local` na fusão (locais viram `sem-id` no verify — documentado, sem perda de screening); `build_report.py` lê o template em `skills/` (não em `.opencode/`); near-duplicatas por chaves DOI distintas mantidas e sinalizadas.


## Discussão

Força da evidência, limitações, aplicabilidade.

## Conclusões + gaps


## Referências (1:1 com as âncoras citadas no corpo; screening e verificação indicados)

[^1]: Transformers in Time Series: A Survey (2023) — DOI: 10.24963/ijcai.2023/759 — screening: include — verify: verified
[^2]: Are Transformers Effective for Time Series Forecasting? (2023) — DOI: 10.1609/aaai.v37i9.26317 — screening: include — verify: verified
[^3]: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (2021) — DOI: 10.1609/aaai.v35i12.17325 — screening: include — verify: verified
[^4]: Time-series forecasting with deep learning: a survey (2021) — DOI: 10.1098/rsta.2020.0209 — screening: include — verify: verified
[^5]: NHITS: Neural Hierarchical Interpolation for Time Series Forecasting (2023) — DOI: 10.1609/aaai.v37i6.25854 — screening: include — verify: verified
[^6]: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (2022) — DOI: 10.48550/arxiv.2211.14730 — screening: include — verify: verified
[^8]: Long sequence time-series forecasting with deep learning: A survey (2023) — DOI: 10.1016/j.inffus.2023.101819 — screening: include — verify: verified
[^9]: Deep Learning for Time Series Forecasting: Tutorial and Literature Survey (2022) — DOI: 10.1145/3533382 — screening: include — verify: verified
[^10]: TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting (2023) — DOI: 10.1145/3580305.3599533 — screening: include — verify: verified
[^11]: TS2Vec: Towards Universal Representation of Time Series (2022) — DOI: 10.1609/aaai.v36i8.20881 — screening: include — verify: verified
[^12]: Deep Learning for Time Series Forecasting: A Survey (2020) — DOI: 10.1089/big.2020.0159 — screening: include — verify: verified
[^13]: A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks (2023) — DOI: 10.3390/fi15080255 — screening: include — verify: verified
[^14]: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (2023) — DOI: 10.48550/arxiv.2310.06625 — screening: include — verify: verified
[^15]: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (2022) — DOI: 10.48550/arxiv.2210.02186 — screening: include — verify: verified
[^16]: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting (2023) — DOI: 10.1145/3583780.3615160 — screening: include — verify: verified
[^17]: A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges (2025) — DOI: 10.1007/s10462-025-11223-9 — screening: include — verify: verified
[^18]: Diagnosisformer: An efficient rolling bearing fault diagnosis method based on improved Transformer (2023) — DOI: 10.1016/j.engappai.2023.106507 — screening: include — verify: verified — ADJACENTE (contexto, não núcleo)
[^19]: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting (2023) — DOI: 10.1609/aaai.v37i7.25976 — screening: include — verify: verified
[^20]: Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction (2020) — DOI: 10.1007/978-3-030-58610-2_30 — screening: include — verify: verified — ADJACENTE (contexto, não núcleo)
[^21]: Stock market index prediction using deep Transformer model (2022) — DOI: 10.1016/j.eswa.2022.118128 — screening: include — verify: verified
[^22]: Deep learning for time series forecasting: a survey (2025) — DOI: 10.1007/s13042-025-02560-w — screening: include — verify: verified
[^23]: MSGNet: Learning Multi-Scale Inter-series Correlations for Multivariate Time Series Forecasting (2024) — DOI: 10.1609/aaai.v38i10.28991 — screening: include — verify: verified
[^24]: Foundation Models for Time Series Analysis: A Tutorial and Survey (2024) — DOI: 10.1145/3637528.3671451 — screening: include — verify: verified
[^25]: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (2020) — DOI: 10.48550/arxiv.2012.07436 — screening: include — verify: verified
[^40]: Temporal Fusion Transformers for interpretable multi-horizon time series forecasting (2021) — DOI: 10.1016/j.ijforecast.2021.03.012 — screening: include — verify: verified
[^92]: ARIMA LSTM Prophet (2026) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^93]: Recent Advances Frontiers (2026) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^106]: Deep Learning TSF Survey (2025) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^107]: Architectural Diversity Open Challenges (2025) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^108]: AutoFormer TS NAS (2025) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^121]: Transformers in Time Series (2022) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^125]: Autoformer (2021) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^129]: Temporal Fusion Transformer (2019) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id
[^133]: 07-Power-of-Architecture-LTSF-2025 (None) — DOI: s/DOI-remoto (local/arxiv-só) — screening: include — verify: sem-id


## Apêndice — bases não consultadas

Google Scholar, Scopus e Web of Science não possuem API pública e **não foram consultados** neste pipeline. PDFs paywall sem cópia OA legal foram avaliados só por abstract.
