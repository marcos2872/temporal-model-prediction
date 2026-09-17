# Dados — CETESB EF01 Mogi das Cruzes (pH + Oxigênio Dissolvido)

Séries temporais **univariadas** de qualidade da água usadas para treinar o modelo de predição (uma variável por vez: `y(t)` → `y(t+1..t+H)`).

Regime anual: **treino = 2024** (`dados/treino/`, ano bissexto completo), **benchmark = 2025**
(`dados/benchmark/`, ano completo, intocado até a avaliação final). Validação dentro de 2024:
4 fatias de 10 dias, uma por estação (15–24 jan / 15–24 abr / 15–24 jul / 15–24 out,
deslocáveis se caírem em outage — ver cada experimento); janelas cujo alvo termina numa
fatia → val, o resto → treino.

## Origem

- **Fonte:** CETESB — SIMQUA, página de gráficos e tabelas: <https://simqua.cetesb.sp.gov.br/graficos_tabelas/>
- **Estação:** EF01 – Mogi das Cruzes (dados públicos)
- **Obtenção:** download direto do portal em setembro/2026 (arquivos renomeados a partir dos nomes originais `EF01 - Mogi das Cruzes - CETESB (01-01-20XX 00_00 a 31-12-20XX 00_00){PH,OD}.csv`; conteúdo inalterado)
- **Validação:** linha 1 de cada arquivo declara `Dados validados até 22/08/2026 09:00` — cobre 2024 e 2025 integralmente (100% validado, sem trecho provisório)

## Arquivos

| Arquivo | Variável | Unidade | Período | Passo | Registros | Faltantes | Outages > 2 h pós-interp (limite 24) |
|---|---|---|---|---|---|---|---|
| `treino/ef01-mogi-das-cruzes_ph_2024.csv` | pH | – | 01/01/2024 00:00 → 31/12/2024 00:00 | 5 min | 105.121 | 11.665 (11,1%) | 16/jan (~2,3 d) · 29/abr (~2,7 d) · **27/mai–12/jun (~17 d)** |
| `treino/ef01-mogi-das-cruzes_oxigenio-dissolvido_2024.csv` | Oxigênio Dissolvido | mg/L | 01/01/2024 00:00 → 31/12/2024 00:00 | 5 min | 105.121 | 594 (0,6%) | nenhum |
| `benchmark/ef01-mogi-das-cruzes_ph_2025.csv` | pH | – | 01/01/2025 00:00 → 31/12/2025 00:00 | 5 min | 104.833 | 6.093 (5,8%) | 6 blocos curtos (máx 230 slots ≈ 19 h, 02/jul) |
| `benchmark/ef01-mogi-das-cruzes_oxigenio-dissolvido_2025.csv` | Oxigênio Dissolvido | mg/L | 01/01/2025 00:00 → 31/12/2025 00:00 | 5 min | 104.833 | 387 (0,4%) | 2 blocos curtos (máx 55 slots) |

Estatísticas calculadas sobre os valores presentes em 13/09/2026. Os arquivos de 2025 têm
288 linhas a menos que a grade cheia (timestamps ausentes esparsos, não um dia inteiro) —
o `reindex` da limpeza os trata como faltantes normais.

## Formato (igual nos quatro arquivos)

- **Encoding:** Windows-1252 (Latin-1) — **não** é UTF-8
- **Separador:** `;` · **Decimal:** vírgula (ex.: `6,08`)
- **Linha 1:** cabeçalho da CETESB (`Entidade responsável...`)
- **Linha 2:** cabeçalho das colunas (`Data hora;<variável>`)
- **Coluna 1:** data-hora no formato `dd/mm/aaaa hh:mm` (ex.: `01/01/2024 00:00`)
- **Coluna 2:** valor medido; **célula vazia = dado faltante** (falha de transmissão/medição)
- A grade de 2024 tem 105.121 slots (ano bissexto); a de 2025, 105.121 teóricos com 288 timestamps ausentes como linhas

## Como ler (pandas)

```python
import pandas as pd
df = pd.read_csv(
    "dados/treino/ef01-mogi-das-cruzes_ph_2024.csv",
    sep=";", decimal=",", encoding="windows-1252",
    skiprows=1, parse_dates=["Data hora"], dayfirst=True,
    na_values=[""],
)
df = df.rename(columns={"Data hora": "ds", "pH": "y"}).sort_values("ds")
```

## Notas para a modelagem univariada

1. **Faltantes primeiro:** interpolação `limit=24` (2 h) + descarte de janelas com NaN, como no regime anterior — ver roteiro §4 da [METODOLOGIA](../METODOLOGIA.md).
2. **Corte temporal honesto:** treino/val dentro de 2024 por data de fim da janela (sem shuffle); 2025 só na avaliação final. Nada de 2025 no treino, na val, no early stopping ou no tuning.
3. **Escala:** pH varia pouco (σ pequeno, faixa ~1 unidade) — normalize (z-score) e avalie com MAE/RMSE além de MAPE, que é instável perto de zero relativo.
4. **Sazonalidade dupla:** passo de 5 min (288 pontos/dia) sugere ciclo diário forte — e com treino anual o baseline **sazonal lag-365** entra no jogo, mas só no benchmark 2025 (prevê 2025 copiando 2024; dentro de 2024 ele é incalculável, pois não há 2023). É a régua mais exigente do novo regime.
5. **Outage de maio/2024 no pH (~17 d):** parte o treino em dois segmentos para janelas `L+H`; cada experimento reporta a cobertura resultante.

## Licença

Dados públicos da CETESB. Ao usar, cite a fonte: CETESB / SIMQUA — Estação EF01 Mogi das Cruzes (<https://simqua.cetesb.sp.gov.br/graficos_tabelas/>).
