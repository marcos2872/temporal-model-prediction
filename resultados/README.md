# Resultados — índice de experimentos

Cada experimento tem sua pasta `NN-<modelo>-<variavel>/` com `README.md` próprio
(tabela de métricas, artefatos e leitura dos resultados), `metricas*.csv`,
`modelos/` e `figs/`. Não commitar fora de pastas de experimento.

| Pasta | Experimento | Baseline a bater |
|---|---|---|
| [`00-baseline-ph/`](00-baseline-ph/) | Baselines clássicos no pH, L=30d/H=1d + holdout de 10 dias (persistência, sazonal-naive, MM, ARIMA-h, Prophet) | sazonal-naive, MAE 0,0501 (rolante) / 0,0466 (holdout) |
| [`00b-baseline-od/`](00b-baseline-od/) | Mesmo protocolo no OD, segmento limpo 01/06→21/07 (sensor morto 21/07–06/08) | sazonal-naive, MAE 0,1525 (rolante) / 0,1550 (holdout) |
| [`01-lstm-ph/`](01-lstm-ph/) | LSTM-h no pH, mesmo protocolo (grade horária Lh=720/Hh=24 + ×12) — **não bate a régua** (grade horária descartada p/ H=288) | sazonal-naive segue régua (0,0501 / 0,0466); lstm_h 0,1120 / 0,1081 |
| [`01b-lstm-od/`](01b-lstm-od/) | Mesmo método no OD, segmento limpo 01/06→21/07 — **não bate a régua** (colapso progressivo com a amplitude de julho) | sazonal-naive segue régua (0,1525 / 0,1550); lstm_h 0,2277 / 0,6220 |

Próximos experimentos previstos (§7 do [README principal](../README.md)):
`02-patchtst-ph/` (resolução nativa) — mesma estrutura, mesmo split, para comparação justa.
