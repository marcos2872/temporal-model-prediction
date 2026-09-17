# Conclusão — previsão de pH e OD, estação EF01 (regime anual)

Veredito final do projeto em 13/09/2026: treino em 2024, val em 4 fatias sazonais,
benchmark no ano intocado de 2025 + auditoria de campo em set/2026.

## Réguas finais (benchmark 2025, MAE H=1 dia)

| Variável | Campeão | MAE | vs sazonal-naive |
|---|---|---|---|
| pH | ensemble NNLS (sazonal 0,27 + LSTNet 0,72) | **0,0509** | −15% (0,0597) |
| OD | ensemble NNLS (LSTNet 0,62 + DLinear-res 0,30) | **0,2107** | −22% (0,2714) |

Vice em ambas: LSTNet puro (0,0529 / 0,2127). DLinear é o modelo "seguro"
(barato, nunca envergonha). PatchTST overfita no pH. LGBM sozinho perde do
sazonal nas duas. lag-365 é inútil (deriva interanual: 0,46 / 0,93).
Prophet explode (1,9 / 5,0). ARIMA ≈ persistência (fallback).

Servidos em `univariavel/app.py` (`POST /prever`, Swagger em `/docs`).

## O que os experimentos provaram

1. **O gargalo era dado, não modelo.** Com 3 meses de treino o OD era imbátivel;
   com 1 ano (incluindo verão/outono de alta amplitude), todas as réguas caíram.
2. **Métodos convergem:** LSTNet, DLinear, PatchTST e ensemble chegam a números
   próximos — sinal de que o teto univariado está perto.
3. **Ensemble generaliza:** o medo de overfit da val (pesos fitados e avaliados no
   mesmo conjunto) não se confirmou no ano intocado.
4. **Sazonalidade anual climática ≠ repetição:** copiar o ano anterior falha porque
   o nível das séries deriva entre anos.

## Auditoria de campo (12–13/09/2026, via `univariavel/app.py` × dado real medido depois)

| Ponto | Observado | Referência | Regime |
|---|---|---|---|
| pH 24 h (280 slots) | 0,165 | 0,051 | evento (rio 5,8 → 5,5) |
| OD 1 h calma (9 slots) | 0,040 | 0,211 | estável |
| OD 24 h (285 slots) | 0,574 | 0,211 | evento (rio 4,9 → 4,2) |

pH e OD colapsaram **juntos** em 12→13/09 — duas variáveis independentes caindo ao
mesmo tempo indicam causa exógena (chuva, descarga, efluente). O modelo ainda previu
a subida do ciclo diário da manhã enquanto a realidade afundava: o retrato do limite
univariado. Nas horas estáveis antes da queda, ambas as variáveis pagaram o prometido
(pH 0,06 nas 6 primeiras horas; OD 0,04 na hora calma). Conclusão: a referência é média
sobre regimes mistos — **dia de evento custa ~3×, dia estável paga o prometido** (ou melhor).
Ver `univariavel/app.py` (resposta inclui `mae_referencia_24h` exatamente para calibrar a leitura).

## Próximo passo natural: multivariados — BLOQUEADO por falta de covariáveis

O erro de campo tem cara de causa externa (chuva, vazão, temperatura) — e o LSTNet
já nasceu multivariado, então a extensão seria natural (Fase 1 óbvia; TFT/
cross-channel depois). **Mas não há com o que treinar:** os downloads da CETESB
para a EF01 nesta pesquisa trouxeram só pH e OD; temperatura, turbidez,
condutividade, chuva e vazão não estão nos dados — e multivariado sem covariável
boa e sincronizável (disponível na hora da previsão, sem leakage) não sai do lugar.
O dia 13/09 seria o teste crítico perfeito, se um dia houver chuva para condicionar.

## Fila restante (sem dependência de dados novos)

- Acumular pontos de auditoria de campo (3 pontos em 13/09/2026; meta: ~30 dias → MAE real vs referência, por regime);
- Alerta de deriva na API (avisar quando a entrada sai da distribuição de treino);
- Saída probabilística (quantis) — agora com motivação concreta de campo.
