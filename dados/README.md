# Dados — CETESB EF01 Mogi das Cruzes (pH + Oxigênio Dissolvido)

Séries temporais **univariadas** de qualidade da água usadas para treinar o modelo de predição (uma variável por vez: `y(t)` → `y(t+1..t+H)`).

## Origem

- **Fonte:** CETESB — SIMQUA, página de gráficos e tabelas: <https://simqua.cetesb.sp.gov.br/graficos_tabelas/>
- **Estação:** EF01 – Mogi das Cruzes (dados públicos)
- **Obtenção:** download direto do portal em setembro/2026 (arquivos renomeados a partir dos nomes originais `EF01 - Mogi das Cruzes - CETESB (01-06-2026 00_00 a 31-08-2026 00_00) *.csv`; conteúdo inalterado)
- **Validação:** linha 1 de cada arquivo declara `Dados validados até 22/08/2026 09:00` — o trecho de 23/08 a 31/08 é **provisório** (não validado); cuidado redobrado com ele no treino/avaliação

## Arquivos

| Arquivo | Variável | Unidade | Período | Passo | Registros | Faltantes | Faixa (min–max) | Média |
|---|---|---|---|---|---|---|---|---|
| `ef01-mogi-das-cruzes_ph_2026-06-01_a_2026-08-31.csv` | pH | – | 01/06/2026 00:00 → 31/08/2026 00:00 | 5 min | 26.209 | 4.794 (18,3%) | 5,56–6,59 | 6,14 |
| `ef01-mogi-das-cruzes_oxigenio-dissolvido_2026-06-01_a_2026-08-31.csv` | Oxigênio Dissolvido | mg/L | 01/06/2026 00:00 → 31/08/2026 00:00 | 5 min | 26.209 | 4.767 (18,2%) | 4,84–8,91 | 6,67 |

Estatísticas calculadas sobre os valores presentes em 10/09/2026.

## Formato (igual nos dois arquivos)

- **Encoding:** Windows-1252 (Latin-1) — **não** é UTF-8
- **Separador:** `;` · **Decimal:** vírgula (ex.: `6,08`)
- **Linha 1:** cabeçalho da CETESB (`Entidade responsável...`)
- **Linha 2:** cabeçalho das colunas (`Data hora;<variável>`)
- **Coluna 1:** data-hora no formato `dd/mm/aaaa hh:mm` (ex.: `01/06/2026 00:00`)
- **Coluna 2:** valor medido; **célula vazia = dado faltante** (falha de transmissão/medição, frequente — ~18%, em gaps de até 24 passos = 2 h)
- A grade do período tem exatamente 26.209 slots (01/06 00:00 → 31/08 00:00 a cada 5 min) e todos estão presentes como linhas — não há timestamps ausentes, só células vazias

## Como ler (pandas)

```python
import pandas as pd
df = pd.read_csv(
    "dados/ef01-mogi-das-cruzes_ph_2026-06-01_a_2026-08-31.csv",
    sep=";", decimal=",", encoding="windows-1252",
    skiprows=1, parse_dates=["Data hora"], dayfirst=True,
    na_values=[""],
)
df = df.rename(columns={"Data hora": "ds", "pH": "y"}).sort_values("ds")
```

## Notas para a modelagem univariada

1. **Faltantes primeiro:** ~18% de gaps + 287 timestamps ausentes exigem estratégia explícita (interpolação limitada, máscara ou reamostragem) antes do janelamento — ver roteiro §4 do [README principal](../README.md).
2. **Corte temporal honesto:** split sem shuffle; considere separar o trecho provisório (pós-22/08/2026) do treino ou sinalizá-lo na avaliação.
3. **Escala:** pH varia pouco (σ pequeno, faixa ~1 unidade) — normalize (z-score) e avalie com MAE/RMSE além de MAPE, que é instável perto de zero relativo.
4. **Sazonalidade diária:** passo de 5 min (288 pontos/dia) sugere ciclo diário forte — verifique antes de escolher `lookback` (ex.: 288 = 1 dia).

## Licença

Dados públicos da CETESB. Ao usar, cite a fonte: CETESB / SIMQUA — Estação EF01 Mogi das Cruzes (<https://simqua.cetesb.sp.gov.br/graficos_tabelas/>).
