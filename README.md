# Modelo de Predição Temporal — CETESB EF01 Mogi das Cruzes

Projeto para criar um modelo de IA que **recebe uma série temporal e gera predição futura**,
aplicado a dados reais de qualidade da água (pH e oxigênio dissolvido, passo de 5 min).
Trilha univariada concluída (v1 + v2 + API); trilha multivariável concluída
(M1–M4 em `multivariavel/`, veredito: CI vence CD, solo multi não bate as réguas uni).

## Documentação

| Documento | Conteúdo |
|---|---|
| `README.md` (este) | Apresentação: dados, estrutura, resultados e status |
| [`COMO-RODAR.md`](COMO-RODAR.md) | Como rodar: ambiente, notebooks, API, checkpoints, probe |
| [`METODOLOGIA.md`](METODOLOGIA.md) | Metodologia: taxonomia, matemática das famílias, roteiro, busca bibliográfica |

Detalhes por área: [`univariavel/dados/`](univariavel/dados/README.md) · [`univariavel/notebooks/`](univariavel/notebooks/README.md) · [`univariavel/resultados/`](univariavel/resultados/README.md) · [`univariavel/benchmark-2025/`](univariavel/benchmark-2025/README.md) · [`busca_bibliografica/`](busca_bibliografica/busca-bibliografica-modelos-predicao-temporal.md) · [`CONCLUSAO.md`](CONCLUSAO.md)

## Estrutura

```
temporal-model/
├── README.md              <- este arquivo (apresentação)
├── COMO-RODAR.md          <- como rodar o projeto
├── METODOLOGIA.md         <- metodologias de predição + bibliografia
├── univariavel/             <- trilha univariada: dados/ (séries CETESB pH + OD, 5 min), app.py (API FastAPI, Swagger em /docs), notebooks/, resultados/ e benchmark-2025/
├── multivariavel/           <- experimentos multivariados: PLANO.md, notebooks/, resultados/ (M1–M4, treino 2022–2024 + benchmark 2025) (dados próprios em multivariavel/dados/)
├── requirements.txt         <- deps (instalar com `uv pip install -r requirements.txt`)
└── busca_bibliografica/   <- kit da busca sistemática (relatório + .bib/.ris + evidence_table.csv + passport.json + prisma.md)
```

## Dados

Séries **univariadas** da CETESB (estação EF01 – Mogi das Cruzes, dados públicos):
pH e oxigênio dissolvido a cada 5 min. Regime anual — **treino = 2024**
(`univariavel/dados/treino/`), **benchmark = 2025** (`univariavel/dados/benchmark/`, intocado até a
avaliação final); validação em 4 fatias de 10 dias dentro de 2024, uma por estação.
Formato (encoding `windows-1252`, `;`, vírgula decimal) e estatísticas em
[`univariavel/dados/README.md`](univariavel/dados/README.md).

## Resultados (réguas — MAE)

Benchmark 2025 (critério principal) e validação 2024. Tabelas completas por
experimento em [`univariavel/resultados/`](univariavel/resultados/README.md);
veredito em [`CONCLUSAO.md`](CONCLUSAO.md).

| Alvo | Benchmark 2025 (v1 → v2) | Val 2024 |
|---|---|---|
| pH | ens **0,0509** → patchtst **0,0465** (ens 0,0470) | ens **0,0357** |
| OD | ens **0,2107** → patchtst **0,2056** (ens 0,2133) | ens **0,1325** |

Leitura curta: ensembles NNLS vencem na v1; na v2 (L=2304, purge/embargo,
5 seeds) o PatchTST passa à frente nas duas variáveis. Sazonal-naive,
lag-365 e Prophet ficam para trás nos dois regimes.

## Status

- [x] Busca bibliográfica sistemática (2026-09-10, 133 → 34 incluídos, `verify` EXIT 0)
- [x] Dataset CETESB definido e validado (2024 treino + 2025 benchmark)
- [x] Cadeia v1: baselines → LSTNet → PatchTST/DLinear → ensembles → benchmark 2025 (notebooks 00–08)
- [x] Cadeia v2: mesmo protocolo em L=2304 com purge/embargo e 5 seeds (notebooks 10–18)
- [x] Probe versionado v1×v2 + API FastAPI (`POST /prever`, Swagger em `/docs`)
- [x] Conclusão do projeto em `CONCLUSAO.md`
- [x] Trilha multivariável M1–M4 (DLinear-multi → PatchTST CI×CD → benchmark 2025; ver `multivariavel/PLANO.md` e `multivariavel/resultados/`)
- [ ] Saída probabilística (quantis) e teste de transferência para 2026 quando houver dado validado

## Licença

MIT — ver [`LICENSE`](LICENSE). Dados CETESB citados conforme `univariavel/dados/README.md` § Licença.
