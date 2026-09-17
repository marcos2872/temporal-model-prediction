# Resultados — índice de experimentos multivariáveis (EF01 Mogi das Cruzes)

Cada experimento tem sua pasta `M*/` com `README.md` próprio
(tabela de métricas, artefatos e leitura dos resultados), `metricas*.csv`,
`modelos/` e `figs/`. Não commitar fora de pastas de experimento.
Os checkpoints de `modelos/` (`.pt`) **não vão ao git** — são regeneráveis pelos
notebooks M1/M2/M3; só `normalizacao.json` fica versionado.

Protocolo (`multivariavel/PLANO.md`): janelas `L=2304 → H∈{12,72,288}` (8 d → 1 h / 6 h / 24 h),
um pipeline dedicado por H · treino = 2022–2024 menos val com purge/embargo ±288 (5 fatias
v2-2024 + 2 auxiliares interanuais) · benchmark = 2025 inteiro (M4, só inferência).
Números abaixo = MAE seed-mean (3 seeds), salvo indicação.

| Pasta | Experimento | Métricas |
|---|---|---|
| [`M1-dlinear-multi/`](M1-dlinear-multi/) | DLinear multicanal 4→2 heads (OD+pH+Temp+Turb+tempo), 3 H × 3 seeds — valida o pipeline, fecha os pisos | val pooled: H12 pH 0,0224 · OD 0,0590 / H72 pH 0,0291 · OD 0,1051 / H288 pH 0,0382 · OD 0,1477 (bate o sazonal nos 6) |
| [`M2-patchtst-multi-CI/`](M2-patchtst-multi-CI/) | PatchTST channel-independent (pesos compartilhados, forward por canal), 3 H × 3 seeds — braço controle da ablação CI vs CD | val pooled: H12 pH 0,0178 · OD 0,0263 / H72 pH 0,0257 · OD 0,0880 / H288 pH 0,0377 · OD 0,1499 (vence o M1 em 5/6) |
| [`M3-patchtst-multi-CD/`](M3-patchtst-multi-CD/) | PatchTST channel-dependent (joint-attention 4 canais), 3 H × 3 seeds — braço tratamento da ablação | val pooled: H12 pH 0,0181 · OD 0,0282 / H72 pH 0,0262 · OD 0,0934 / H288 pH 0,0390 · OD 0,1503 (ganho curto; H288 empata/perde p/ o linear) |
| [`M4-benchmark-2025/`](M4-benchmark-2025/) | 27 checkpoints M1/M2/M3 × 2025 intocado (só inferência) + média-simples-diagnóstico — **veredito final** | rolante 2025: H12 ph CI 0,0298 · od CI 0,0305 / H72 ph CI 0,0367 · od CI 0,1194 / H288 ph DL 0,0475 · od DL 0,2305; ms-diag. lidera 5/6; ranking transfere 5/6; multi solo não bate as réguas uni-v2 (pH 0,0465 · OD 0,2056) |

Quadro do benchmark 2025 (rolante, seed-mean): **pH → media-simples-diag. 0,0454
(DLinear 0,0475 melhor solo) · OD → media-simples-diag. 0,2160 (DLinear 0,2305 melhor
solo)** — sem régua nova declarada (diagnóstico + margens, ver M4).
