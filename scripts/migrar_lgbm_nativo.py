#!/usr/bin/env python3
"""Migração one-shot: LGBM pickle -> formato nativo (Fase 2, passo 1).

Carrega UMA vez resultados/07-ensemble-od/modelos/lgbm_steps.pkl.gz
(fallback .pkl) e ressalva os 288 boosters em formato nativo
(`booster.save_model`) sob resultados/07-ensemble-od/modelos/lgbm_nativo/
(um arquivo por horizonte: lgbm_h000.txt ... lgbm_h287.txt).

Uso:
    .venv/bin/python scripts/migrar_lgbm_nativo.py

O loader (app.py) prefere o nativo (`lgb.Booster(model_file=)`) e cai para
o .pkl.gz com warning logado se o subdir estiver ausente/incompleto.
O subdir é gitignored (ver .gitignore); publicação futura via Release
`modelos-v2`. Idempotente: sobrescreve os .txt.
"""

import gzip
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "resultados" / "07-ensemble-od" / "modelos"
DEST = BASE / "lgbm_nativo"
N = 288


def main() -> int:
    pk_gz, pk = BASE / "lgbm_steps.pkl.gz", BASE / "lgbm_steps.pkl"
    src = pk_gz if pk_gz.exists() else pk
    if not src.exists():
        print(f"ERRO: checkpoint ausente: {pk_gz} (ou {pk})", file=sys.stderr)
        return 1
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rb") as f:
        modelos = pickle.load(f)
    if len(modelos) != N:
        print(f"ERRO: esperado {N} modelos em {src}, achei {len(modelos)}", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    for j, mdl in enumerate(modelos):
        booster = mdl.booster_ if hasattr(mdl, "booster_") else mdl
        booster.save_model(str(DEST / f"lgbm_h{j:03d}.txt"))
    # Verificação rápida: recarrega o horizonte 0 via nativo e compara 1 predição.
    import lightgbm as lgb
    import numpy as np

    b0 = lgb.Booster(model_file=str(DEST / "lgbm_h000.txt"))
    rng = np.random.default_rng(0)
    X = rng.random((1, b0.num_feature()), dtype=np.float64)
    ref = float(modelos[0].predict(X)[0])
    novo = float(b0.predict(X)[0])
    diff = abs(ref - novo)
    print(f"OK: {N} boosters nativos em {DEST} (fonte: {src.name}; "
          f"spot-check h000 |pickle-nativo|={diff:.3e})")
    return 0 if diff <= 1e-9 else 2


if __name__ == "__main__":
    sys.exit(main())
