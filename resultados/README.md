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
| [`02-lstnet-ph/`](02-lstnet-ph/) | LSTNet nativo 5 min no pH (conv + GRU + skip p=1d + AR-288 + RevIN) — **NOVA RÉGUA do pH** | **lstnet 0,0456 (rolante) / 0,0446 (holdout)**, −9% / −4% sobre o sazonal-naive |
| [`02b-lstnet-od/`](02b-lstnet-od/) | Mesmo método no OD, segmento limpo 01/06→21/07 — melhor neural no **teste** (−36%), régua do **holdout** segue sazonal | lstnet 0,0981 (rolante) / 0,2428 (holdout); régua 0,1525 / 0,1550 |
| [`03-patchtst-ph/`](03-patchtst-ph/) | PatchTST nativo + DLinear + régua LSTNet recarregada — **régua segue LSTNet**; DLinear bate o sazonal no rolante | lstnet 0,0456/0,0446; patchtst 0,0611/0,0457; dlinear 0,0498/0,0538 |
| [`03b-patchtst-od/`](03b-patchtst-od/) | Os três no OD limpo — **régua segue sazonal**; melhor neural no holdout é o linear | sazonal 0,1525/0,1550; dlinear 0,1376/0,2146; lstnet 0,0981/0,2428; patchtst 0,1327/0,2434 |
| [`04-ensemble-ph/`](04-ensemble-ph/) | Ensemble residual + LightGBM no pH (piso sazonal + 288 LGBM + DLinear-res + NNLS) — **régua segue LSTNet**; ens 2º nas duas tabelas | lstnet 0,0456/0,0446; ens 0,0475/0,0462; sazonal 0,0501/0,0466; lgbm 0,0609/0,0567; dlres 0,0508/0,0571 |
| [`04b-ensemble-od/`](04b-ensemble-od/) | Mesmo método no OD limpo — **régua segue sazonal**; ens 2º no teste, 3º no holdout; lgbm supera lstnet no holdout | sazonal 0,1525/0,1550; lstnet 0,0981/0,2428; ens 0,1024/0,2105; lgbm 0,1889/0,2090; dlres 0,1440/0,2591 |
| [`05-amplitude-od/`](05-amplitude-od/) | Correção de amplitude no OD limpo (saz reescalado + LGBM-mult + DLinear ponderado) — **régua segue sazonal**; saz_escalado 2º no holdout | sazonal 0,1525/0,1550; saz_esc 0,1506/0,1874; ens 0,1009/0,2126; lgbm_mult 0,1784/0,2190; dlw 0,1526/0,2371 |
| [`06-refit-od/`](06-refit-od/) | Refit honesto + amplitude-forward no OD (A comparável + B walk-forward) — **régua segue sazonal**; ensA novo melhor no teste, lstnet_ft a +6,6% no holdout | sazonal 0,1525/0,1550; ensA 0,0920/0,1994; lstnet_ft 0,1097/0,1653; lgbm_A 0,1662/0,1808 |

Quadro das réguas (holdout diário, critério principal): **pH → LSTNet 0,0446** · **OD → sazonal-naive 0,1550**.
