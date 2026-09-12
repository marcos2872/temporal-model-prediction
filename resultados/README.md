# Resultados — índice de experimentos

Cada experimento tem sua pasta `NN-<modelo>-<variavel>/` com `README.md` próprio
(tabela de métricas, artefatos e leitura dos resultados), `metricas*.csv`,
`modelos/` e `figs/`. Não commitar fora de pastas de experimento.

| Pasta | Experimento | Baseline a bater |
|---|---|---|
| [`00-baseline-ph/`](00-baseline-ph/) | Baselines clássicos no pH, L=30d/H=1d + holdout de 10 dias (persistência, sazonal-naive, MM, ARIMA-h, Prophet) | sazonal-naive, MAE 0,0501 (rolante) / 0,0466 (holdout) |

Próximos experimentos previstos (§7 do [README principal](../README.md)):
`01-lstm-ph/`, `02-patchtst-ph/` — mesma estrutura, mesmo split, para comparação justa.
