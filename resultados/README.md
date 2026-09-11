# Resultados — baseline univariado pH (EF01)

Artefatos gerados por `notebooks/00-baseline-arima-prophet.ipynb` (executado de ponta a ponta, 0 erros).
Reproduzir: `uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace notebooks/00-baseline-arima-prophet.ipynb`
(com o `.venv` ativo, ou `uv pip install -r requirements.txt` antes).

## Configuração do experimento

- **Série:** pH, estação EF01 Mogi das Cruzes, passo de 5 min (`dados/`, ver `dados/README.md`)
- **Lookback `L`:** 2016 (7 dias) · **Horizonte `H`:** 12 (1 h) · **Split temporal 70/15/15 sem shuffle** (teste: 3.628 origens, alvos 18/08 → 31/08/2026)
- Limpeza: grade completa de 5 min + interpolação temporal máx. 2 h (0 NaN restante; 0 janelas descartadas)

## Tabela principal — teste completo (3.628 origens)

| modelo | MAE | RMSE | MAPE | sMAPE |
|---|---|---|---|---|
| **persistência** | **0,0368** | 0,0542 | 0,5942 | 0,5942 |
| sazonal-naive (lag 288) | 0,0470 | 0,0654 | 0,7578 | 0,7577 |
| Prophet | 0,0471 | 0,0621 | 0,7577 | 0,7579 |
| média móvel 288 | 0,0618 | 0,0773 | 0,9943 | 0,9944 |

**Baseline a bater: persistência (MAE 0,0368).** O pH é tão estável que repetir o último valor supera os demais no teste cheio.

## Tabela no subconjunto ARIMA (38 origens, comparação justa)

O ARIMA(2,1,2) é reestimado por origem (custo), por isso roda numa subamostra (stride 96):

| modelo | MAE | RMSE | MAPE | sMAPE |
|---|---|---|---|---|
| persistência | 0,0298 | 0,0439 | 0,4817 | 0,4810 |
| arima_212 | 0,0298 | 0,0439 | 0,4817 | 0,4810 |
| sazonal-naive 288 | 0,0463 | 0,0668 | 0,7461 | 0,7465 |
| média móvel 288 | 0,0581 | 0,0703 | 0,9379 | 0,9369 |
| prophet | 0,0484 | 0,0626 | 0,7799 | 0,7800 |

Nota honesta: no subconjunto, ARIMA empata com a persistência (4 casas) — convergiu para perto dela, o esperado numa série quase-passeio-aleatório de curtíssimo horizonte. Não leia o 0,0298 como "vitória" fora do subconjunto.

## Arquivos

| Arquivo | O quê |
|---|---|
| `metricas_baseline.csv` | Tabela principal (teste completo) em CSV |
| `modelos/arima212_cauda_treino.pkl` (13 MB) | ARIMA(2,1,2) ajustado na cauda do treino (inspeção/reuso) |
| `modelos/prophet_ph.json` (2,7 MB) | Modelo Prophet serializado (sazonalidade diária + semanal) |
| `figs/01-eda.png` | Série completa (com marca do trecho provisório), distribuição e boxplot por hora |
| `figs/02-limpeza.png` | Cru × interpolado (semana 08–15/06) |
| `figs/03-stl.png` | Decomposição STL (tendência/sazonalidade/resíduo, período 288) |
| `figs/04-forecasts.png` | Forecast × real em 3 origens (início/meio/fim do teste) |
| `figs/05-mae.png` | Barras de MAE por modelo |

## Leitura dos resultados

1. **Horizonte curto + série estável ⇒ persistência é forte.** Modelos futuros (LSTM, PatchTST) precisam superá-la **neste split**.
2. **Sazonal-naive perde da persistência** — o ciclo diário existe (ver `01-eda.png`), mas em H=1 h o último valor informa mais que ontem-na-mesma-hora.
3. **Prophet ≈ sazonal-naive** — a sazonalidade diária/semanal que ele modela não paga neste horizonte; útil como referência, não como solução.
4. **Trecho provisório:** o teste inclui dados pós-22/08/2026 (não validados) — ver `dados/README.md`; refazer a avaliação só no trecho validado é um sanity check barato.
