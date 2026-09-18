"""API FastAPI multivariável — serve um dos modelos M (M1/M2/M3, flag --modelo).

Uso local (a partir da raiz do repo):
    .venv/bin/python -m multivariavel.src.app --modelo M2
    .venv/bin/python -m multivariavel.src.app --modelo M1 --port 8001
Swagger UI: http://127.0.0.1:8000/docs

Fluxo: POST /prever com CSV CETESB multivariável (OD + pH + Temp + Turb,
>= L+H slots a cada 5 min) + variável (ph|od) + horizonte (1|6|24 h).
O pipeline dedicado do H pedido roda a seed-mean das 3 seeds e devolve
H valores. Modelos carregados uma vez no startup
(ver multivariavel/resultados/M*/). Lógica de inferência espelha
os notebooks M1/M2/M3 (+ M4 no modo só-inferência) — sem treino aqui.
"""

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from .features import L, prepara, ler_csv_multivariavel
from .models import LN, N_TF, DLinearMulti, PatchTST_CD, PatchTSTCI

logger = logging.getLogger(__name__)

CHECKPOINT_TAG = "modelo-multivariavel-v1"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ROOT = Path(__file__).resolve().parent.parent
H_POR_HORAS = {1: 12, 6: 72, 24: 288}
SEEDS = [42, 7, 123]
VAR_IDX = {"od": 0, "ph": 1}  # ordem dos canais [od, ph, temp, turb]

FAMILIA = os.environ.get("MULTIV_MODELO", "M2")
NOMES = {
    "M1": "DLinearMulti (M1)",
    "M2": "PatchTST-CI (M2)",
    "M3": "PatchTST-CD (M3)",
}
BASES = {
    "M1": ("M1-dlinear-multi", "dlinear_multi_H%d_s%d.pt", DLinearMulti),
    "M2": ("M2-patchtst-multi-CI", "patchtst_CI_H%d_s%d.pt", PatchTSTCI),
    "M3": ("M3-patchtst-multi-CD", "patchtst_CD_H%d_s%d.pt", PatchTST_CD),
}
# MAE seed-mean no benchmark 2025 (M4/metricas_benchmark.csv) por (família, H, var)
MAE_REF = {
    "M1": {12: {"ph": 0.0326, "od": 0.0773}, 72: {"ph": 0.0391, "od": 0.1436}, 288: {"ph": 0.0475, "od": 0.2305}},
    "M2": {12: {"ph": 0.0298, "od": 0.0305}, 72: {"ph": 0.0367, "od": 0.1194}, 288: {"ph": 0.0491, "od": 0.2410}},
    "M3": {12: {"ph": 0.0301, "od": 0.0343}, 72: {"ph": 0.0381, "od": 0.1365}, 288: {"ph": 0.0493, "od": 0.2498}},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELO = {}


def _sha12(caminhos) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in caminhos:
        with open(p, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 20), b""):
                h.update(bloco)
    return h.hexdigest()[:12]


def carrega(fam: str):
    """Carrega as 3 seeds × 3 Hs da família (idempotente, chamado no startup)."""
    if fam in MODELO:
        return
    if fam not in BASES:
        raise ValueError(f"família '{fam}' inválida (use M1, M2 ou M3)")
    pasta, pat, cls = BASES[fam]
    base = ROOT / "resultados" / pasta / "modelos"
    norm_candidates = [base / "normalizacao.json", ROOT / "resultados" / "M1-dlinear-multi" / "modelos" / "normalizacao.json"]
    norm_path = next((p for p in norm_candidates if p.exists()), None)
    if norm_path is None:
        raise FileNotFoundError(
            f"normalizacao.json ausente em {base} "
            "(checkpoints vivem no GitHub Release `modelo-multivariavel-v1` — "
            "rode: bash scripts/baixar_modelos.sh --tag modelo-multivariavel-v1)"
        )
    norm = json.load(open(norm_path))
    ckpts, redes = [], {}
    for H in (12, 72, 288):
        redes[H] = []
        for seed in SEEDS:
            p = base / (pat % (H, seed))
            if not p.exists():
                raise FileNotFoundError(
                    f"checkpoint ausente: {p} "
                    "(checkpoints vivem no GitHub Release `modelo-multivariavel-v1` — "
                    "rode: bash scripts/baixar_modelos.sh --tag modelo-multivariavel-v1)"
                )
            try:
                state = torch.load(p, map_location=DEVICE, weights_only=True)["state"]
            except Exception:
                state = torch.load(p, map_location=DEVICE, weights_only=False)["state"]
            m = cls(H=H).to(DEVICE)
            m.load_state_dict(state)
            redes[H].append(m.eval())
            ckpts.append(p)
    MODELO[fam] = {
        "redes": redes,
        "mu": {H: np.array(norm["per_H"][str(H)]["mu"], dtype=np.float64) for H in (12, 72, 288)},
        "sd": {H: np.array(norm["per_H"][str(H)]["sd"], dtype=np.float64) for H in (12, 72, 288)},
        "turb_p99": float(norm["turb_winsor_p99_train_only_H288"]),
        "checkpoint_tag": CHECKPOINT_TAG,
        "checkpoint_sha": _sha12(ckpts),
    }


def _stack4_ci(xv: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    B = xv.size(0)
    return torch.cat([xv.permute(1, 0, 2).reshape(B * 4, 1, LN), xt.repeat(4, 1, 1)], dim=1)


@torch.no_grad()
def inferencia(fam: str, H: int, X: np.ndarray) -> np.ndarray:
    """Seed-mean (H, 2) [ph, od] em unidade original. X = (11, L) z-score."""
    m = MODELO[fam]
    mu, sd = m["mu"][H], m["sd"][H]
    acc = None
    xb = torch.from_numpy(X[None]).to(DEVICE)
    for rede in m["redes"][H]:
        if fam == "M2":
            tail = xb[:, :, -LN:]  # cauda LN=2016, verbatim M2
            xv = tail[:, :4, :]
            xt = tail[:, 4:, :]
            z = rede(_stack4_ci(xv, xt)).cpu().numpy().reshape(4, H).T  # (H,4) [od,ph,temp,turb]
            out = np.stack([z[:, 1] * sd[1] + mu[1], z[:, 0] * sd[0] + mu[0]], axis=1)
        else:
            z = rede(xb).cpu().numpy()[0]  # (H, 2) [ph, od] em z-score
            out = np.stack([z[:, 0] * sd[1] + mu[1], z[:, 1] * sd[0] + mu[0]], axis=1)
        acc = out if acc is None else acc + out
    return acc / len(m["redes"][H])


@asynccontextmanager
async def lifespan(app: FastAPI):
    carrega(FAMILIA)
    yield


app = FastAPI(
    title="Temporal Model Multi — previsão pH/OD (EF01)",
    description=(
        "Serve um modelo multivariável M1/M2/M3 (treino 2022–2024, benchmark 2025). "
        "Envie um CSV CETESB com os 4 canais (OD + pH + Temperatura + Turbidez), "
        "escolha a variável (ph|od) e o horizonte (1|6|24 h). "
        "Teste interativo no botão **Try it out** do `POST /prever`. "
        "Selecione o modelo com a flag `--modelo M1|M2|M3` na inicialização."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class Ponto(BaseModel):
    ds: str = Field(examples=["2026-09-09T00:05:00"])
    y: float = Field(examples=[5.83])


class PrevisaoOut(BaseModel):
    variavel: Literal["ph", "od"]
    horizonte_horas: int
    modelo: str
    mae_referencia: float
    inicio_previsto: str
    fim_previsto: str
    cobertura_entrada: dict
    valores: list[Ponto]
    checkpoint_tag: str = Field(default=CHECKPOINT_TAG, examples=[CHECKPOINT_TAG])
    checkpoint_sha: str = Field(default="", examples=["a1b2c3d4e5f6"])


@app.get("/saude", summary="Saúde da API e modelo carregado")
def saude():
    return {
        "status": "ok",
        "modelo": NOMES[FAMILIA],
        "horizontes_suportados_h": [1, 6, 24],
        "device": str(DEVICE),
        "checkpoint_tag": CHECKPOINT_TAG,
        "checkpoint_sha": MODELO.get(FAMILIA, {}).get("checkpoint_sha", ""),
    }


@app.get("/regras", summary="Réguas (MAE seed-mean no benchmark 2025, por H)")
def regras():
    return {"modelo": NOMES[FAMILIA], "mae_benchmark": MAE_REF[FAMILIA]}


@app.post("/prever", response_model=PrevisaoOut, summary="Prevê 1|6|24 h de ph|od a partir de CSV CETESB 4-canais")
async def prever(
    request: Request,
    arquivo: UploadFile = File(description="CSV CETESB multivariável: `Data hora;Oxigênio Dissolvido;pH;Precipitação;Temperatura;Turbidez` com `;` e vírgula decimal (>= L+H slots a cada 5 min)"),
    variavel: Literal["ph", "od"] = Query("ph", description="Variável a prever (o input contém sempre os 4 canais)"),
    horizonte_horas: int = Query(24, ge=1, le=24, description="Horizonte dedicado: 1 h (H=12), 6 h (H=72) ou 24 h (H=288)"),
):
    if int(horizonte_horas) not in H_POR_HORAS:
        raise HTTPException(400, "horizonte_horas deve ser 1, 6 ou 24")
    if FAMILIA not in MODELO:
        raise HTTPException(500, f"modelo '{FAMILIA}' não carregado")
    H = H_POR_HORAS[int(horizonte_horas)]
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "arquivo excede o limite de 10 MB")
    conteudo = await arquivo.read()
    if len(conteudo) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "arquivo excede o limite de 10 MB")
    try:
        df = ler_csv_multivariavel(conteudo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    m = MODELO[FAMILIA]
    try:
        X, grade, cobertura = prepara(df, H, m["mu"][H], m["sd"][H], m["turb_p99"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    fim = grade.max()
    try:
        full = await asyncio.to_thread(inferencia, FAMILIA, H, X)
    except Exception:
        logger.exception("falha na inferência (%s H=%d)", FAMILIA, H)
        raise HTTPException(500, "falha interna — ver logs")
    if not np.isfinite(full).all():
        raise HTTPException(500, "falha interna — ver logs")
    col = 0 if variavel == "ph" else 1
    idx = pd.date_range(fim + pd.Timedelta(minutes=5), periods=H, freq="5min")
    valores = [Ponto(ds=str(t), y=round(float(v), 4)) for t, v in zip(idx, full[:, col])]
    return PrevisaoOut(
        variavel=variavel,
        horizonte_horas=int(horizonte_horas),
        modelo=NOMES[FAMILIA],
        mae_referencia=MAE_REF[FAMILIA][H][variavel],
        inicio_previsto=str(idx[0]),
        fim_previsto=str(idx[-1]),
        cobertura_entrada=cobertura,
        valores=valores,
        checkpoint_tag=m.get("checkpoint_tag", CHECKPOINT_TAG),
        checkpoint_sha=m.get("checkpoint_sha", ""),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="API multivariável pH/OD (flag --modelo M1|M2|M3)")
    p.add_argument("--modelo", choices=["M1", "M2", "M3"], default=os.environ.get("MULTIV_MODELO", "M2"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    os.environ["MULTIV_MODELO"] = a.modelo
    global FAMILIA
    FAMILIA = a.modelo
    import uvicorn

    uvicorn.run("multivariavel.src.app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
