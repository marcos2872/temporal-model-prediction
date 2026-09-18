# API multivariável (`multivariavel/src/`)

Serve um dos modelos M (treino 2022–2024, `L=2304`, horizontes dedicados).
Checkpoints (`.pt`, gitignored) via Release `modelo-multivariavel-v1`:

```bash
bash scripts/baixar_modelos.sh --tag modelo-multivariavel-v1 --dir /tmp/multimodelos
```

## Rodar (a partir da raiz do repo)

```bash
.venv/bin/python -m multivariavel.src.app --modelo M2          # M1 | M2 | M3 (default M2)
.venv/bin/python -m multivariavel.src.app --modelo M1 --port 8001
# Swagger UI: http://127.0.0.1:8000/docs  (botão "Try it out" no POST /prever)
curl -X POST "http://127.0.0.1:8000/prever?variavel=ph&horizonte_horas=24" \
  -F "arquivo=@multivariavel/dados/benchmark/ef01-mogi-das-cruzes_multivariavel_2025.csv"
```

`MULTIV_MODELO` faz o mesmo que `--modelo`. Sem GPU, cai para CPU sozinho.

## Contrato

- **Input:** CSV CETESB multivariável (`Data hora;OD;pH;Precip;Temp;Turb`, `;`,
  vírgula decimal) com **os 4 canais** e ao menos `L+H` slots a cada 5 min
  (~8,1 dias p/ 1 h; ~8,5 p/ 6 h; 9 p/ 24 h). Precipitação é ignorada.
- **`variavel`**: `ph` | `od` (uma por vez). **`horizonte_horas`**: `1` (H=12),
  `6` (H=72) ou `24` (H=288) — qualquer outro valor dá `400`.
- **Resposta:** H pontos `{ds, y}` com timestamps de 5 min + `mae_referencia`
  (MAE seed-mean do modelo/H/var no benchmark 2025, M4) + `checkpoint_sha`.
- **Gaps > 2 h na cauda** `L+H` retornam `422` (não prevê no escuro), como na API uni.
- **Seed-mean fixa (3 seeds)** por (modelo, H) — sem pesos ajustáveis, sem
  ensemble entre famílias (veredito M4: média-simples só diagnóstico).

## Arquivos

| Arquivo | O quê |
|---|---|
| `app.py` | FastAPI (`/saude`, `/regras`, `/prever`) + carga no startup + `--modelo` |
| `models.py` | `DLinearMulti` / `PatchTSTCI` / `PatchTST_CD` — cópias fiéis dos notebooks M1/M2/M3 §8 |
| `features.py` | Parse CETESB 4-canais + grade 5 min + interp limite 24 + winsorize p99 + z-score + 11 séries (verbatim M1/M3; cauda LN p/ M2) |
| `teste.py` | Pipeline de teste: 9 partes × 3H × ph/od contra `dados.csv` (2026), via subprocesso uvicorn |
| `dados.csv` | Série 2026 unseen p/ o teste (01/01–17/09, 5 min, 4 canais) |
| `.logs/` | `M1.log`, `M2.log`, `M3.log` + `geral.log` (MAE/RMSE/MAPE/acertividade; `SEM_COBERTURA` = parte recusada com 422, fora do placar) |
