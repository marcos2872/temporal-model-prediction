"""API FastAPI — serve os ensembles campeões (06 pH, 07 OD).

Uso local (a partir da raiz do repo):
    .venv/bin/uvicorn univariavel.app:app --host 127.0.0.1 --port 8000
Swagger UI: http://127.0.0.1:8000/docs

Fluxo: POST /prever com CSV CETESB (pH ou OD, >= ~8 dias a cada 5 min) +
horizonte de 1–24 h. A API roda o pipeline validado de H=288 e devolve o
prefixo pedido. Modelos carregados uma vez no startup (ver univariavel/resultados/06, 07).
Lógica de inferência espelha os notebooks 02/03 (LSTNet), 04/05 (DLinear-res),
06/07 (LGBM + NNLS) e 08 (janelamento) — sem treino aqui.

Nota de compatibilidade: L=8640 (30 d) é mantido por compatibilidade com o
janelamento validado (cobertura, features LGBM end-relativas), mas a
rede usa só a cauda ctx[-2016:] (LN=2016, ~7 d) — L é vestigial p/ torch/DL.
"""

from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Literal

import asyncio
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CHECKPOINT_TAG = "modelos-v1"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ROOT = Path(__file__).resolve().parent
L, H_FULL, SEASON, LN = 8640, 288, 288, 2016
INTERP_LIMIT, CTX_MIN = 24, 2304  # 2304 = 2016 (rede) + 288 (alvo)

REGRAS = {
    "ph": {"modelo": "ensemble-NNLS (06)", "mae_benchmark_24h": 0.0509},
    "od": {"modelo": "ensemble-NNLS (07)", "mae_benchmark_24h": 0.2107},
}


# ---------- arquiteturas (cópias fiéis dos notebooks) ----------
class LSTNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 32, kernel_size=12, stride=6)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.skipcell = nn.GRUCell(32, 32)
        self.head = nn.Linear(96, 288)
        self.ar = nn.Linear(288, 288)
        self.drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, xv, tod):
        mu = xv.mean(dim=1, keepdim=True); sg = xv.std(dim=1, keepdim=True).clamp_min(1e-3)
        vn = self.gamma * (xv - mu) / sg + self.beta
        f = self.drop(torch.relu(self.conv(torch.cat([vn.unsqueeze(1), tod.transpose(1, 2)], dim=1))))
        f = f.transpose(1, 2)
        _, h = self.gru(f)
        b, t, _ = f.shape
        hs = torch.zeros(b, 32)
        states = [hs]
        for i in range(t):
            prev = states[i - 48] if i - 48 >= 0 else states[0]
            hs = self.skipcell(f[:, i, :], prev)
            states.append(hs)
        g = self.gamma.clamp_min(1e-3)
        yn = self.head(self.drop(torch.cat([h.squeeze(0), hs], dim=1)))
        ya = self.ar(vn[:, -288:])
        return (yn + ya - self.beta) / g * sg + mu


class DLinearLite(nn.Module):
    def __init__(self, k=25, residual=False):
        super().__init__()
        self.pool = nn.AvgPool1d(k, stride=1, padding=k // 2)
        self.lin_t = nn.Linear(LN, H_FULL)
        self.lin_s = nn.Linear(LN, H_FULL)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.residual = residual

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True); sg = x.std(dim=1, keepdim=True).clamp_min(1e-3)
        xn = self.gamma * (x - mu) / sg + self.beta
        t = self.pool(xn.unsqueeze(1)).squeeze(1)
        y = self.lin_t(t) + self.lin_s(xn - t)
        yn = (y - self.beta) / self.gamma.clamp_min(1e-3) * sg
        return yn if self.residual else yn + mu


def _load_state(cls, path, **kw):
    m = cls(**kw)
    # weights_only=True: checkpoints contêm só tensores + cfg simples;
    # bloqueia deserialização de pickle arbitrário (hardening Fase 2).
    # Verificado: os 4 .pt (02/03 lstnet + 06/07 dlres) carregam com True.
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
    return m.eval()


def _sha12(caminhos) -> str:
    """sha256 (12 chars) sobre o conteúdo concatenado dos checkpoints usados."""
    import hashlib

    h = hashlib.sha256()
    for p in caminhos:
        with open(p, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 20), b""):
                h.update(bloco)
    return h.hexdigest()[:12]


MODELOS = {}


def carrega(var: str):
    """Carrega o ensemble de uma variável (idempotente, chamado no startup)."""
    import json
    import pickle

    if var == "ph":
        base = ROOT / "resultados" / "06-ensemble-ph" / "modelos"
        ck02 = ROOT / "resultados" / "02-lstnet-ph" / "modelos" / "lstnet_ph.pt"
    else:
        base = ROOT / "resultados" / "07-ensemble-od" / "modelos"
        ck02 = ROOT / "resultados" / "03-lstnet-od" / "modelos" / "lstnet_od.pt"
    dlres_nome = "dlinear_res_ph.pt" if var == "ph" else "dlinear_res_od.pt"
    for p in (base / "ensemble.json", base / dlres_nome, ck02):
        if not p.exists():
            raise FileNotFoundError(
                f"checkpoint ausente: {p} "
                "(checkpoints vivem no GitHub Release `modelos-v1` — "
                "rode: bash scripts/baixar_modelos.sh)"
            )
    ens = json.load(open(base / "ensemble.json"))["pesos"]
    modelos = {
        "pesos": [ens["sazonal"], ens["lstnet"], ens["lgbm"], ens["dlres"]],
        "lstnet": _load_state(LSTNet1D, ck02),
        "dlres": _load_state(DLinearLite, base / dlres_nome, residual=True),
        "lgbm": None,
    }
    if ens["lgbm"] != 0.0:  # pH tem peso 0 — nem carrega o LGBM
        import gzip

        import lightgbm as lgb

        # Formato nativo (preferido): 288 boosters salvos via
        # booster.save_model (scripts/migrar_lgbm_nativo.py), um .txt por
        # horizonte em modelos/lgbm_nativo/. Evita pickle no caminho quente.
        nat = base / "lgbm_nativo"
        nativos = sorted(nat.glob("lgbm_h*.txt")) if nat.is_dir() else []
        if len(nativos) == H_FULL:
            modelos["lgbm"] = [lgb.Booster(model_file=str(p)) for p in nativos]
            modelos["_lgbm_arquivos"] = nativos
        else:
            if nat.is_dir():
                logger.warning(
                    "lgbm_nativo incompleto (%d/288 .txt em %s) — "
                    "caindo para o pickle",
                    len(nativos), nat,
                )
            else:
                logger.warning(
                    "subdir nativo ausente (%s) — caindo para o pickle "
                    "(rode: .venv/bin/python scripts/migrar_lgbm_nativo.py)",
                    nat,
                )
            pk_gz, pk = base / "lgbm_steps.pkl.gz", base / "lgbm_steps.pkl"
            pk = pk_gz if pk_gz.exists() else pk  # .pkl local (gitignored) como fallback
            if not pk.exists():
                raise FileNotFoundError(
                    f"checkpoint ausente: {pk_gz} (ou {pk}) "
                    "(checkpoints vivem no GitHub Release `modelos-v1` — "
                    "rode: bash scripts/baixar_modelos.sh)"
                )
            opener = gzip.open if pk.suffix == ".gz" else open
            with opener(pk, "rb") as f:
                modelos["lgbm"] = pickle.load(f)
            modelos["_lgbm_arquivos"] = [pk]
    usados = [base / "ensemble.json", base / dlres_nome, ck02]
    usados += list(modelos.get("_lgbm_arquivos", []))
    modelos["checkpoint_tag"] = CHECKPOINT_TAG
    modelos["checkpoint_sha"] = _sha12(usados)
    MODELOS[var] = modelos


@asynccontextmanager
async def lifespan(app: FastAPI):
    carrega("ph")
    carrega("od")
    yield


app = FastAPI(
    title="Temporal Model — previsão pH/OD (EF01)",
    description=(
        "Serve os ensembles campeões do regime anual (treino 2024, benchmark 2025). "
        "Envie um CSV CETESB e receba o dia seguinte (ou 1–24 h). "
        "Teste interativo no botão **Try it out** do `POST /prever`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- schemas (Swagger) ----------
class Ponto(BaseModel):
    ds: str = Field(examples=["2026-09-09T00:05:00"])
    y: float = Field(examples=[5.83])


class PrevisaoOut(BaseModel):
    variavel: Literal["ph", "od"]
    horizonte_horas: int
    modelo: str
    mae_referencia_24h: float
    inicio_previsto: str
    fim_previsto: str
    cobertura_entrada: dict
    valores: list[Ponto]
    checkpoint_tag: str = Field(default="modelos-v1", examples=["modelos-v1"])
    checkpoint_sha: str = Field(default="", examples=["a1b2c3d4e5f6"])


# ---------- pipeline ----------
def ler_csv(conteudo: bytes) -> pd.DataFrame:
    try:
        texto = conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = conteudo.decode("windows-1252")
    linhas = texto.splitlines()
    if not linhas:
        raise ValueError("CSV vazio")
    skip = 1 if linhas[0].strip().lower().startswith("entidade") else 0
    try:
        df = pd.read_csv(
            StringIO(texto), sep=";", decimal=",", skiprows=skip,
            parse_dates=[0], dayfirst=True, na_values=[""],
        )
    except Exception as e:
        raise ValueError(f"não consegui ler o CSV como CETESB (sep=';', decimal=','): {e}")
    if df.shape[1] < 2:
        raise ValueError("CSV precisa de 2 colunas: Data hora;<variável>")
    df = df.iloc[:, :2]
    df.columns = ["ds", "y"]
    df = df.sort_values("ds").reset_index(drop=True)
    if df["ds"].isna().any():
        raise ValueError("há datas ilegíveis na 1ª coluna (formato esperado dd/mm/aaaa hh:mm)")
    return df


def prepara(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq="5min")
    if len(idx) < CTX_MIN + H_FULL:
        raise ValueError(
            f"série curta: {len(idx)} slots — mínimo {CTX_MIN + H_FULL} "
            f"(~8 dias de contexto + 1 dia de margem)"
        )
    s = df.set_index("ds")["y"].reindex(idx)
    falt_antes = int(s.isna().sum())
    s = s.interpolate(method="time", limit=INTERP_LIMIT)
    cauda = s.iloc[-CTX_MIN:]
    if int(cauda.isna().sum()):
        raise ValueError(
            "contexto insuficiente no fim da série: há falha maior que 2 h "
            "nos últimos 8 dias — o modelo não prevê no escuro"
        )
    ctx = s.iloc[-L:].to_numpy().astype(np.float32) if len(s) >= L else s.to_numpy().astype(np.float32)
    cobertura = {
        "slots": len(idx),
        "faltantes_originais": falt_antes,
        "pct_interpolado": round(100 * (falt_antes - int(s.isna().sum())) / len(idx), 2),
    }
    return ctx, s.index, cobertura


def inferencia(var: str, ctx: np.ndarray, fim: pd.Timestamp) -> np.ndarray:
    """Roda o ensemble validado (H=288) sobre a cauda de contexto. Retorna 288 valores."""
    m = MODELOS[var]
    x = ctx[-2016:].astype(np.float32)
    n = len(ctx)
    # sazonal-naive: copia ontem (lag-288)
    saz = ctx[n - SEASON:n - SEASON + H_FULL].copy()
    # hora-do-dia p/ LSTNet
    t0 = fim - pd.Timedelta(minutes=5 * (2015))
    horas = pd.date_range(t0, periods=2016, freq="5min")
    sin = np.sin(2 * np.pi * (horas.hour.to_numpy() * 60 + horas.minute.to_numpy()) / 1440.0).astype(np.float32)
    cos = np.cos(2 * np.pi * (horas.hour.to_numpy() * 60 + horas.minute.to_numpy()) / 1440.0).astype(np.float32)
    with torch.no_grad():
        pn = m["lstnet"](
            torch.from_numpy(x[None, :]),
            torch.from_numpy(np.stack([sin, cos], axis=1)[None, :, :]),
        ).numpy()[0]
        dr = saz + m["dlres"](torch.from_numpy(x[None, :])).numpy()[0]
    w = m["pesos"]
    if m["lgbm"] is None:
        return w[0] * saz + w[1] * pn + w[3] * dr
    # LGBM (só OD tem peso != 0)
    Xb = ctx[None, :]
    em = np.array([(fim.hour * 60 + fim.minute)], dtype=np.float32)
    F = np.stack(
        [Xb[:, -k] for k in [1, 2, 3, 6, 12, 24, 36, 72, 144, 287, 288, 289, 576, 2016]]
        + [np.stack([Xb[:, n - 288 * k] for k in range(1, 8)], axis=1).mean(1)]
        + [np.stack([Xb[:, n - 288 * k] for k in range(1, 8)], axis=1).std(1)]
        + [c for w_ in (12, 36, 144, 288)
           for c in (Xb[:, -w_:].mean(1), Xb[:, -w_:].std(1))]  # intercalado = base_feats (06/07/08)
        + [Xb[:, -2016:].mean(1)],
        axis=1,
    ).astype(np.float32)
    gb = np.empty(H_FULL, dtype=np.float32)
    for j, mdl in enumerate(m["lgbm"]):
        hh = ((em - (H_FULL - 1 - j) * 5) % 1440 // 60).astype(np.float32)
        sh = np.sin(2 * np.pi * hh / 24).astype(np.float32)
        ch = np.cos(2 * np.pi * hh / 24).astype(np.float32)
        gb[j] = saz[j] + mdl.predict(np.column_stack([F, sh, ch]))[0]
    return w[0] * saz + w[1] * pn + w[2] * gb + w[3] * dr


# ---------- endpoints ----------
@app.get("/saude", summary="Saúde da API e modelos carregados")
def saude():
    return {
        "status": "ok",
        "modelos": sorted(MODELOS),
        "horizonte_suportado_h": [1, 24],
        "checkpoint_tag": CHECKPOINT_TAG,
        "checkpoint_sha": {v: MODELOS[v].get("checkpoint_sha", "") for v in sorted(MODELOS)},
    }


@app.get("/regras", summary="Réguas atuais (MAE no benchmark 2025)")
def regras():
    return REGRAS


@app.post("/prever", response_model=PrevisaoOut, summary="Prevê as próximas horas a partir de um CSV CETESB")
async def prever(
    request: Request,
    arquivo: UploadFile = File(description="CSV CETESB: linha 1 cabeçalho da CETESB (opcional), depois `Data hora;<variável>` com `;` e vírgula decimal"),
    variavel: Literal["ph", "od"] = Query("ph", description="Variável a prever"),
    horizonte_horas: int = Query(24, ge=1, le=24, description="Horizonte: 1–24 h (o pipeline validado roda 24 h e devolve o prefixo)"),
):
    if variavel not in MODELOS:
        raise HTTPException(500, f"modelo '{variavel}' não carregado")
    # Limite de upload (10 MB): checa o header antes de ler e o tamanho
    # real após ler (chunked pode omitir o Content-Length).
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "arquivo excede o limite de 10 MB")
    conteudo = await arquivo.read()
    if len(conteudo) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "arquivo excede o limite de 10 MB")
    try:
        df = ler_csv(conteudo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        ctx, grade, cobertura = prepara(df)
    except ValueError as e:
        raise HTTPException(422, str(e))
    fim = grade.max()
    try:
        # torch + LGBM são CPU-bound e bloqueantes: roda em thread separada
        # para não travar o event loop do FastAPI (asyncio) enquanto a
        # inferência de H=288 ocupa a CPU.
        full = await asyncio.to_thread(inferencia, variavel, ctx, fim)
    except Exception:
        # Nunca vazar tipo/mensagem de exceção interna ao cliente (500
        # genérico); o traceback completo vai para os logs.
        logger.exception("falha na inferência (%s)", variavel)
        raise HTTPException(500, "falha interna — ver logs")
    n = horizonte_horas * 12
    idx = pd.date_range(fim + pd.Timedelta(minutes=5), periods=n, freq="5min")
    valores = [Ponto(ds=str(t), y=round(float(v), 4)) for t, v in zip(idx, full[:n])]
    return PrevisaoOut(
        variavel=variavel,
        horizonte_horas=horizonte_horas,
        modelo=REGRAS[variavel]["modelo"],
        mae_referencia_24h=REGRAS[variavel]["mae_benchmark_24h"],
        inicio_previsto=str(idx[0]),
        fim_previsto=str(idx[-1]),
        cobertura_entrada=cobertura,
        valores=valores,
        checkpoint_tag=MODELOS[variavel].get("checkpoint_tag", CHECKPOINT_TAG),
        checkpoint_sha=MODELOS[variavel].get("checkpoint_sha", ""),
    )
