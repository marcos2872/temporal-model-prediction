# Conclusão — previsão de pH e OD, estação EF01 (regime anual)

Veredito final do projeto em 13/09/2026: treino em 2024, val em 4 fatias sazonais,
benchmark no ano intocado de 2025 + auditoria de campo em set/2026.

> **Adendo 17–18/09/2026:** cadeia v2 (L=2304, purge/embargo, 5 fatias, 5 seeds) e
> trilha multivariável M1–M4 concluídas — ver §§ Adendo v2 e Adendo multivariável
> ao final. O veredito v1 abaixo segue intacto como registro.

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

## Próximo passo natural: multivariados — BLOQUEADO por falta de covariáveis (v1, superado — ver Adendo multivariável)

> **Superado em 17–18/09/2026** pela trilha M1–M4 (`multivariavel/PLANO.md`,
> `multivariavel/resultados/`): com Temp+Turbidez como covariáveis observadas, o
> multivariado saiu do lugar. Texto original abaixo preservado como registro.

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

## Adendo v2 (17/09/2026 — protocolo L=2304, purge/embargo, 5 fatias, 5 seeds)

Cadeia 10–18 em `univariavel/resultados/` (`10-v2-baseline-*` → `12/13-v2-lstnet-*` →
`14/15-v2-patchtst-*` → `16/17-v2-ensemble-*` → `18-v2-benchmark-2025`, só inferência
em 2025). Detalhes e tabelas completas em
[`univariavel/resultados/`](univariavel/resultados/README.md) e
[`univariavel/resultados/18-v2-benchmark-2025/`](univariavel/resultados/18-v2-benchmark-2025/).

Benchmark 2025 rolante v2 (MAE H=1 dia): **pH → PatchTST 0,0465** (ens 0,0470,
sazonal 0,0565) · **OD → PatchTST 0,2056** (ens 0,2133, sazonal 0,2519).
O protocolo v2 curou o overfit do PatchTST no pH (v1: 0,0688, perdia do sazonal);
no OD manteve a ordem do v1. Exceções honestas: LSTNet piora no OD
(0,2127 → 0,2330) e o ensemble NNLS **não** vence em 2025 — os pesos ajustados
na val 2024 (lstnet-dominantes) não transferiram o ranking para o ano intocado.
lag-365 e Prophet seguem fora do jogo nos dois regimes.

> **Réguas servidas:** a API (`univariavel/app.py`, `/regras`) segue servindo os
> ensembles v1 (pH 0,0509 · OD 0,2107) até decisão do usuário — o benchmark v2
> coroa o PatchTST, mas a troca não foi ativada (ver
> `univariavel/resultados/18-v2-benchmark-2025/README.md` § Veredito, item 5).

## Adendo multivariável (18/09/2026 — M1–M4, treino 2022–2024 + benchmark 2025)

Trilha em `multivariavel/` (plano em `multivariavel/PLANO.md`, notebooks M1–M4,
índice em `multivariavel/resultados/README.md`): alvo pH+OD (2 heads),
Temp/Turbidez só como covariáveis observadas, `L=2304 → H∈{12,72,288}`, 3 seeds
por (config, H). Benchmark M4 = 27 checkpoints × 2025 intocado, só inferência.

Veredito M4 (rolante 2025, seed-mean): **CI vence CD nos 6/6 (H,var)** e o ranking
CI×CD×DLinear transfere da val para 2025 em 5/6 (exceção honesta: H288-pH troca
CI×DL dentro do dp — empate técnico). Em H=288, nenhum modelo multi solo bate as
réguas uni-v2 do mesmo ano (pH DL 0,0475 / CI 0,0491 / CD 0,0493 × régua 0,0465 ·
OD DL 0,2305 / CI 0,2410 / CD 0,2498 × régua 0,2056). A média-simples-diagnóstico
lidera 5/6 mas é só diagnóstico (PLANO §2): **sem régua nova declarada**.
Leitura: canal cruzado ajuda no curto (H=12/72, folga do CI sobre o DLinear no OD),
mas em 24 h o teto univariado segue intacto.
