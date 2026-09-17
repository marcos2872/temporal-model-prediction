"""Benchmark versionado com pipeline própria — probe em 4 períodos de 2025.

NÃO lê `univariavel/resultados/08-benchmark-2025/`. Carrega os checkpoints de
`univariavel/resultados/` (00–07) e roda inferência (só inferência, sem treino/tuning) sobre 4 períodos
de 10 dias em 2025, um por estação, só dias-âncora (23:55):

    P1 2025-03-08 → 2025-03-17   P2 2025-05-03 → 2025-05-12
    P3 2025-08-22 → 2025-08-31   P4 2025-10-01 → 2025-10-10

Períodos escolhidos por busca (max cobertura conjunta pH+OD de âncoras); com a
tolerância de 3 h todos os períodos têm ~10 origens (P4 pH preenche o furo de
11 slots do gap 30/set). Fixos daqui em diante p/ comparar v1×v2.

11 modelos por variável (fiel ao 08): persistencia, sazonal_naive_288,
media_movel_288, sazonal_lag365 (+fallback saz-288), lstnet, patchtst,
dlinear, lgbm (288), dlres, ens (pesos NNLS, sem refit) + prophet (opcional).

--tag v1 usa os checkpoints 00–07 (LSTNet 3 canais, LGBM pickle por passo).
--tag v2 usa os checkpoints 10–17 com o MESMO probe (mesmos períodos,
L/H/TOL_NAN): LSTNet 8 canais seed-mean ×5, PatchTST/DLinear seed-mean ×5,
DLinear-res seed-mean ×5, LGBM 288 nativos (lgbm_h*/model_j*), ensemble via
ensemble.json do 16/17 (sem refit), prophet do 10/11 (opcional).

Fontes de código (não duplicar à toa):
- LSTNet1D, DLinearLite: importadas de `univariavel/app.py` (fonte única com a API).
- PatchTST, base_feats, hour_sincos, lag365, prophet: cópia fiel de
  `univariavel/notebooks/08-benchmark-2025.ipynb` cels 7/9/11 (pin: commit 97bceb0).

Uso:
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --tag v1
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --tag v1 --refazer
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --tag v2
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --diff v1 v2
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --plot v1
    .venv/bin/python univariavel/benchmark-2025/benchmark_versoes.py --plot v1 v2

Saídas (tudo sob univariavel/benchmark-2025/resultados/):
    historico.csv                        # append-only: MAE pooled do probe
    <tag>/metricas_periodos_{ph,od}.csv  # modelo × (P1..P4 + n) — individual
    <tag>/metricas_media_{ph,od}.csv     # modelo, MAE, RMSE, n — média p/ gráfico
    <tag>/01-mae-benchmark-<tag>.png     # gráfico a partir da média
    <tag>/meta.json                      # tag, SHA, protocolo, períodos+n, fontes
    <tag>/vals/<exp>-metricas_val.csv    # val 2024 atual (referência, sem 2025)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    for p in (HERE, *HERE.parents):
        if (p / "univariavel" / "dados" / "treino").exists():
            return p
    raise SystemExit("raiz do repo não encontrada (univariavel/dados/treino ausente)")


ROOT = repo_root()
sys.path.insert(0, str(ROOT / "univariavel"))
from app import DLinearLite, LSTNet1D  # noqa: E402  (fonte única com a API)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402

SRC08 = ROOT / "univariavel" / "resultados" / "08-benchmark-2025"  # só p/ checar; nada é lido de lá
OUTROOT = HERE / "resultados"
HIST = OUTROOT / "historico.csv"

# Protocolo travado (AGENTS.md): L=8640 (30 d), H=288 (1 d), passo 5 min.
L, H, SEASON, INTERP_LIMIT, LN, HN = 8640, 288, 288, 24, 2016, 288

# Tolerância do PROBE (só aqui; notebooks seguem descartando qualquer NaN):
# janelas-âncora com até TOL_NAN slots faltantes (3 h) são aceitas e têm os
# furos preenchidos por interpolação linear local. Gap de horas não invalida
# 31 dias de contexto para fins de benchmark. Gaps maiores descartam a origem.
TOL_NAN = 36

PERIODOS = [
    ("P1", "2025-03-08", "2025-03-17"),
    ("P2", "2025-05-03", "2025-05-12"),
    ("P3", "2025-08-22", "2025-08-31"),
    ("P4", "2025-10-01", "2025-10-10"),
]

FILES = {
    "ph": ("ef01-mogi-das-cruzes_ph_2024.csv", "ef01-mogi-das-cruzes_ph_2025.csv", "pH"),
    "od": ("ef01-mogi-das-cruzes_oxigenio-dissolvido_2024.csv",
           "ef01-mogi-das-cruzes_oxigenio-dissolvido_2025.csv", "Oxigênio Dissolvido (mg/L)"),
}
CKPTS = {
    "ph": {"lstnet": "02-lstnet-ph/modelos/lstnet_ph.pt",
           "patch": "04-patchtst-ph/modelos/patchtst_ph.pt",
           "dlin": "04-patchtst-ph/modelos/dlinear_ph.pt",
           "lgbm": "06-ensemble-ph/modelos/lgbm_steps.pkl.gz",
           "lgbm_fb": "06-ensemble-ph/modelos/lgbm_steps.pkl",
           "dlres": "06-ensemble-ph/modelos/dlinear_res_ph.pt",
           "ens": "06-ensemble-ph/modelos/ensemble.json",
           "prophet": "00-baseline-ph/modelos/prophet_ph.json"},
    "od": {"lstnet": "03-lstnet-od/modelos/lstnet_od.pt",
           "patch": "05-patchtst-od/modelos/patchtst_od.pt",
           "dlin": "05-patchtst-od/modelos/dlinear_od.pt",
           "lgbm": "07-ensemble-od/modelos/lgbm_steps.pkl.gz",
           "lgbm_fb": "07-ensemble-od/modelos/lgbm_steps.pkl",
           "dlres": "07-ensemble-od/modelos/dlinear_res_od.pt",
           "ens": "07-ensemble-od/modelos/ensemble.json",
           "prophet": "01-baseline-od/modelos/prophet_od.json"},
}
VAL_EXPS = {
    "ph": ["00-baseline-ph", "02-lstnet-ph", "04-patchtst-ph", "06-ensemble-ph"],
    "od": ["01-baseline-od", "03-lstnet-od", "05-patchtst-od", "07-ensemble-od"],
}

# ---------- protocolo v2 (exps 10–17; o probe NÃO muda — só os checkpoints) ----------
# LSTNet 8 canais (valor + 7 cov), PatchTST/DLinear seed-mean ×5, DLinear-res
# seed-mean ×5, LGBM 288 nativos, ensemble via ensemble.json do 16/17 (sem refit).
# Redes v2 treinadas com L=2304 mas nativas em LN=2016: no probe (L=8640) elas
# leem Xt=X[:,-LN:] + covariáveis computadas dos timestamps (como o 18 fez).
SEEDS_V2 = [42, 7, 123, 2024, 999]
L_TREINO_V2 = 2304  # L de treino v2 (8 d); o probe segue L=8640 (comparável v1×v2)
CKPTS_V2 = {
    "ph": {"lstnet_dir": "12-v2-lstnet-ph/modelos", "lstnet_pat": "lstnet_ph_s{seed}.pt",
           "td_dir": "14-v2-patchtst-ph/modelos", "patch_pat": "patchtst_ph_s{seed}.pt",
           "dlin_pat": "dlinear_ph_s{seed}.pt",
           "lgbm_dir": "16-v2-ensemble-ph/modelos/lgbm_nativo",
           "dlres_pat": "dlinear_res_ph_s{seed}.pt",  # sob lgbm_dir.parent
           "ens": "16-v2-ensemble-ph/modelos/ensemble.json",
           "prophet": "10-v2-baseline-ph/modelos/prophet_ph.json"},
    "od": {"lstnet_dir": "13-v2-lstnet-od/modelos", "lstnet_pat": "lstnet_od_s{seed}.pt",
           "td_dir": "15-v2-patchtst-od/modelos", "patch_pat": "patchtst_od_s{seed}.pt",
           "dlin_pat": "dlinear_od_s{seed}.pt",
           "lgbm_dir": "17-v2-ensemble-od/modelos/lgbm_nativo",
           "dlres_pat": "dlinear_res_od_s{seed}.pt",  # sob lgbm_dir.parent
           "ens": "17-v2-ensemble-od/modelos/ensemble.json",
           "prophet": "11-v2-baseline-od/modelos/prophet_od.json"},
}
VAL_EXPS_V2 = {
    "ph": ["10-v2-baseline-ph", "12-v2-lstnet-ph", "14-v2-patchtst-ph", "16-v2-ensemble-ph"],
    "od": ["11-v2-baseline-od", "13-v2-lstnet-od", "15-v2-patchtst-od", "17-v2-ensemble-od"],
}

HIST_FIELDS = [
    "timestamp_utc", "git_sha", "tag", "variavel", "modelo",
    "MAE_val", "RMSE_val", "MAE_benchmark", "RMSE_benchmark", "MAE_diaria",
]


# ---------- cópia fiel do 08 (cel 9) — univariavel/app.py não tem PatchTST ----------
class PatchTST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = (LN - 48) // 24 + 1
        self.proj = torch.nn.Linear(48, 64)
        self.pos = torch.nn.Parameter(torch.randn(1, self.N, 64) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, 3)
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(self.N * 64, HN)
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True); sg = x.std(dim=1, keepdim=True).clamp_min(1e-3)
        xn = self.gamma * (x - mu) / sg + self.beta
        z = self.proj(xn.unfold(1, 48, 24)) + self.pos
        z = self.enc(self.drop(z))
        y = self.head(self.drop(z.flatten(1)))
        return (y - self.beta) / self.gamma.clamp_min(1e-3) * sg + mu


def load_state(cls, rel, **kw):
    m = cls(**kw).to("cpu")
    m.load_state_dict(torch.load(ROOT / "univariavel" / "resultados" / rel,
                                 map_location="cpu", weights_only=False)["state"])
    return m.eval()


def norm(nome: str) -> str:
    return nome.replace("(02)", "").strip()


def read_val_index0(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            out[norm(row[0])] = {header[i]: row[i] for i in range(1, len(header))}
    return out


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def mae(a, b) -> float:
    return float(mean_absolute_error(np.asarray(a).ravel(), np.asarray(b).ravel()))


def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(a).ravel(), np.asarray(b).ravel())))


# ---------- carga (fiel ao 08, cel 3) ----------
def ler(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="windows-1252",
                     skiprows=1, parse_dates=["Data hora"], dayfirst=True,
                     na_values=[""])
    df = df.rename(columns={"Data hora": "ds", col: "y"}).sort_values("ds").reset_index(drop=True)
    idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq="5min")
    s = df.set_index("ds")["y"].reindex(idx).interpolate(method="time", limit=INTERP_LIMIT)
    return s


# ---------- probe: origens-âncora por período (tolera furo ≤ TOL_NAN) ----------
def origens_probe(s: pd.Series) -> tuple[np.ndarray, np.ndarray, object, list[str], dict[str, int]]:
    v = s.to_numpy().astype(np.float32)
    grade = s.index
    Xs, Ys, pids, ends = [], [], [], []
    maxfill: dict[str, int] = {}
    for pid, a, b in PERIODOS:
        d0 = pd.Timestamp(a).date()
        mf = 0
        for k in range(10):
            e = pd.Timestamp(d0 + pd.Timedelta(days=k), hour=23, minute=55)
            if e not in grade:
                continue
            pos = grade.get_loc(e)
            if pos < L + H - 1:
                continue
            win = v[pos - (L + H) + 1: pos + 1]
            n = int(np.isnan(win).sum())
            if n > TOL_NAN:
                continue  # gap grande: descarta a origem (não prevê no escuro)
            if n:
                win = pd.Series(win).interpolate(
                    method="linear", limit_direction="both").to_numpy(dtype=np.float32)
                if bool(np.isnan(win).any()):
                    continue  # furo na borda sem vizinho: descarta
                mf = max(mf, n)
            Xs.append(win[:L]); Ys.append(win[L:]); pids.append(pid); ends.append(e)
        maxfill[pid] = mf
    X = np.stack(Xs).astype(np.float32) if Xs else np.empty((0, L), np.float32)
    Y = np.stack(Ys).astype(np.float32) if Ys else np.empty((0, H), np.float32)
    return X, Y, pd.DatetimeIndex(ends), pids, maxfill


# ---------- modelos baratos (fiel ao 08, cel 5) ----------
def cheap_preds(X_: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "persistencia": np.repeat(X_[:, -1:], H, axis=1),
        "sazonal_naive_288": np.stack([X_[:, L - SEASON + h] for h in range(H)], axis=1),
        "media_movel_288": np.repeat(X_[:, -SEASON:].mean(axis=1, keepdims=True), H, axis=1),
    }


# ---------- lag-365 (fiel ao 08, cel 7) ----------
def lag365_preds(s24: pd.Series, X: np.ndarray, Y: np.ndarray,
                 ends: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    s24v, s24i = s24.to_numpy(), s24.index
    P = np.empty_like(Y)
    fb = 0
    for k in range(len(Y)):
        e = ends[k]
        try:
            loc = s24i.get_loc(pd.Timestamp(year=2024, month=e.month, day=e.day,
                                            hour=e.hour, minute=e.minute))
            src = s24v[loc - 287: loc + 1]
        except KeyError:
            src = None
        if src is None or np.isnan(src).any():
            src = X[k][L - SEASON:L]  # fallback honesto: saz-288
            fb += 1
        P[k] = src
    return P, fb


# ---------- LGBM feats (fiel ao 08, cel 11) ----------
def base_feats(Xb: np.ndarray, E: pd.DatetimeIndex):
    cols = [Xb[:, -k] for k in [1, 2, 3, 6, 12, 24, 36, 72, 144, 287, 288, 289, 576, 2016]]
    phase = np.stack([Xb[:, L - 288 * k] for k in range(1, 8)], axis=1)
    cols += [phase.mean(1), phase.std(1)]
    for w in [12, 36, 144, 288]:
        cols += [Xb[:, -w:].mean(1), Xb[:, -w:].std(1)]
    cols += [Xb[:, -2016:].mean(1)]
    F = np.stack(cols, axis=1)
    em = (E.hour.to_numpy() * 60 + E.minute.to_numpy()).astype(np.float32)
    return F.astype(np.float32), em


def hour_sincos(em: np.ndarray, j: int):
    hh = ((em - (H - 1 - j) * 5) % 1440 // 60).astype(np.float32)
    return np.sin(2 * np.pi * hh / 24).astype(np.float32), np.cos(2 * np.pi * hh / 24).astype(np.float32)


def load_lgbm(var: str):
    p = ROOT / "univariavel" / "resultados" / CKPTS[var]["lgbm"]
    fb = ROOT / "univariavel" / "resultados" / CKPTS[var]["lgbm_fb"]
    p = p if p.exists() else fb  # .pkl local como fallback
    if not p.exists():
        raise SystemExit(f"LGBM ausente: {p}")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rb") as f:
        return pickle.load(f)


# ---------- v2: covariáveis + LSTNet 8 canais (verbatim 12 §cov / 18 §9) ----------
LAT, LON, TZ = -23.52, -46.19, -3  # Mogi das Cruzes; ts locais (UTC-3, sem DST)
N_COV, N_CH = 7, 8  # tod_sin/cos + solar + f1..f4 = 7; + valor = 8
CONV_CH, CONV_K, CONV_S = 32, 12, 6
GRU_H, SKIP_H, SKIP_P = 64, 32, 48
AR_Q = 288
DROPOUT = 0.1


def elevacao_solar(ts, lat=LAT, lon=LON, tz=TZ):
    ts = pd.DatetimeIndex(ts)
    doy = ts.dayofyear.to_numpy() + (ts.hour.to_numpy() + ts.minute.to_numpy() / 60) / 24
    g = 2 * np.pi / 365 * (doy - 1 + (ts.hour.to_numpy() - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (ts.hour.to_numpy() * 60 + ts.minute.to_numpy()) + eq + 4 * lon - 60 * tz
    ha = np.radians(tst / 4 - 180)
    cosz = np.sin(np.radians(lat)) * np.sin(decl) + np.cos(np.radians(lat)) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cosz, -1, 1)))


def fourier_doy(ts, n=366):
    d = pd.DatetimeIndex(ts).dayofyear.to_numpy()
    return (np.sin(2 * np.pi * d / n), np.cos(2 * np.pi * d / n),
            np.sin(4 * np.pi * d / n), np.cos(4 * np.pi * d / n))


class LSTNetV2(torch.nn.Module):  # = LSTNet1D do 12/18; nome difere p/ não colidir com app.LSTNet1D (v1)
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(N_CH, CONV_CH, kernel_size=CONV_K, stride=CONV_S)
        self.gru = torch.nn.GRU(CONV_CH, GRU_H, batch_first=True)
        self.skipcell = torch.nn.GRUCell(CONV_CH, SKIP_H)
        self.head = torch.nn.Linear(GRU_H + SKIP_H, HN)
        self.ar = torch.nn.Linear(AR_Q, HN)
        self.drop = torch.nn.Dropout(DROPOUT)
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xv, tod):
        mu = xv.mean(dim=1, keepdim=True); sg = xv.std(dim=1, keepdim=True).clamp_min(1e-3)
        vn = self.gamma * (xv - mu) / sg + self.beta
        f = self.drop(torch.relu(self.conv(torch.cat([vn.unsqueeze(1), tod.transpose(1, 2)], dim=1))))
        f = f.transpose(1, 2)
        _, h = self.gru(f)
        B, T, _ = f.shape
        hs = torch.zeros(B, SKIP_H, device=f.device)
        states = [hs]
        for t in range(T):
            prev = states[t - SKIP_P] if t - SKIP_P >= 0 else states[0]
            hs = self.skipcell(f[:, t, :], prev)
            states.append(hs)
        g = self.gamma.clamp_min(1e-3)
        yn = self.head(self.drop(torch.cat([h.squeeze(0), hs], dim=1)))
        ya = self.ar(vn[:, -AR_Q:])
        return (yn + ya - self.beta) / g * sg + mu


def solar_passo(E: pd.DatetimeIndex, j: int):  # verbatim 18 §11
    return (elevacao_solar(E - pd.to_timedelta((H - 1 - j) * 5, unit="min")) / 90.0).astype(np.float32)


def carrega_lgbm_v2(nat_dir: Path, ens_path: Path):
    """Loader tolerante verbatim do 18 §11: aceita lgbm_h*.txt (16) e
    model_j*.txt (17). Convenção Fourier decidida pela lista de nomes
    (normalizacao/ensemble) ou pelo glob; trava se nomes×arquivos divergirem."""
    import lightgbm as lgb

    h = sorted(nat_dir.glob("lgbm_h*.txt"))
    j = sorted(nat_dir.glob("model_j*.txt"))
    assert (len(h) == H) ^ (len(j) == H), (f"esperado 288 boosters de UM padrão em {nat_dir}: "
                                           f"lgbm_h*={len(h)} model_j*={len(j)}")
    paths, glob_flavor = (h, "fim") if len(h) == H else (j, "ctx")
    flavor = glob_flavor
    for key in ("lgbm_features", "features"):
        for src in (nat_dir.parent / "normalizacao.json", ens_path):
            try:
                meta = json.load(open(src))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            nomes = meta.get(key) or (meta.get("lgbm") or {}).get(key)
            if nomes:
                if "orig_f1sin" in nomes:
                    flavor = "fim"  # 16: fourier do fim do alvo (E)
                elif "f1_sin_orig" in nomes:
                    flavor = "ctx"  # 17: fourier da origem (E − H)
                break
    assert flavor == glob_flavor, f"nomes × arquivos divergem em {nat_dir}: {flavor} vs {glob_flavor}"
    boosters = [lgb.Booster(model_file=str(p)) for p in paths]
    print(f"  lgbm: {len(boosters)} boosters ({paths[0].name}…{paths[-1].name}, fourier={flavor})")
    return boosters, flavor


def preve_lgbm_v2(boosters, flavor: str, Xb: np.ndarray, E: pd.DatetimeIndex) -> np.ndarray:
    """Inferência LGBM nativa verbatim do 18 §11 (32 feats = base 25 + fourier 4 + hora 2 + solar 1)."""
    F, em = base_feats(Xb, E)
    if flavor == "fim":
        Ff = np.column_stack([a.astype(np.float32) for a in fourier_doy(E)])
    else:
        Ff = np.column_stack([a.astype(np.float32) for a in fourier_doy(E - pd.Timedelta(minutes=5 * H))])
    S = np.stack([Xb[:, L - SEASON + h] for h in range(H)], axis=1)
    P = np.empty((len(Xb), H), dtype=np.float32)
    for j, bst in enumerate(boosters):
        sh, ch = hour_sincos(em, j)
        P[:, j] = S[:, j] + bst.predict(np.column_stack([F, Ff, sh, ch, solar_passo(E, j)]))
    return P


@torch.no_grad()
def _fwd_seed_mean(modelos, Xt_t: torch.Tensor,
                   Tln_t: torch.Tensor | None = None, batch: int = 64) -> np.ndarray:
    acc = None
    for m in modelos:
        m.eval()
        outs = []
        for b in range(0, len(Xt_t), batch):
            xb = Xt_t[b:b + batch]
            if Tln_t is None:
                outs.append(m(xb).numpy())
            else:
                outs.append(m(xb, Tln_t[b:b + batch]).numpy())
        P = np.concatenate(outs)
        acc = P if acc is None else acc + P
    return (acc / len(modelos)).astype(np.float32)


def _load_seed(cls, pattern: str, seed: int, **kw):
    m = cls(**kw).to("cpu")
    m.load_state_dict(torch.load(ROOT / "univariavel" / "resultados" / pattern.format(seed=seed),
                                 map_location="cpu", weights_only=False)["state"])
    return m.eval()


def infer_v2(var: str, C: dict, X: np.ndarray, Xt: np.ndarray,
             ends: pd.DatetimeIndex, S: np.ndarray) -> dict[str, np.ndarray]:
    """Redes v2 seed-mean ×5 + LGBM nativo + ensemble NNLS (pesos do 16/17, sem refit)."""
    t0 = time.time()
    # Covariáveis dos slots de contexto (determinísticas; p/ mesmos timestamps == Tln do 18)
    e_min = ends.values.astype("datetime64[m]").astype(np.int64)
    ts = pd.DatetimeIndex(pd.to_datetime(
        ((e_min - H * 5)[:, None] - 5 * np.arange(LN - 1, -1, -1)[None, :]).ravel(), unit="m"))
    tod = (ts.hour.to_numpy() * 60 + ts.minute.to_numpy()).astype(np.float32)
    F1, F2, F3, F4 = [a.astype(np.float32) for a in fourier_doy(ts)]
    Tln = np.stack([np.sin(2 * np.pi * tod / 1440).astype(np.float32),
                    np.cos(2 * np.pi * tod / 1440).astype(np.float32),
                    (elevacao_solar(ts) / 90.0).astype(np.float32),
                    F1, F2, F3, F4], axis=1).reshape(len(X), LN, N_COV).astype(np.float32)
    Xt_t, Tln_t = torch.from_numpy(Xt), torch.from_numpy(Tln)
    Pn = _fwd_seed_mean([_load_seed(LSTNetV2, C["lstnet_dir"] + "/" + C["lstnet_pat"], sd)
                         for sd in SEEDS_V2], Xt_t, Tln_t)
    Pt = _fwd_seed_mean([_load_seed(PatchTST, C["td_dir"] + "/" + C["patch_pat"], sd)
                         for sd in SEEDS_V2], Xt_t)
    Dl = _fwd_seed_mean([_load_seed(DLinearLite, C["td_dir"] + "/" + C["dlin_pat"], sd)
                         for sd in SEEDS_V2], Xt_t)
    nat = ROOT / "univariavel" / "resultados" / C["lgbm_dir"]
    boosters, flavor = carrega_lgbm_v2(nat, ROOT / "univariavel" / "resultados" / C["ens"])
    Gb = preve_lgbm_v2(boosters, flavor, X, ends)
    del boosters
    Dr = S + _fwd_seed_mean([_load_seed(DLinearLite, str(Path(C["lgbm_dir"]).parent / C["dlres_pat"]), sd,
                                       residual=True) for sd in SEEDS_V2], Xt_t)
    ens = json.load(open(ROOT / "univariavel" / "resultados" / C["ens"]))["pesos"]
    En = (ens["sazonal"] * S + ens["lstnet"] * Pn
          + ens["lgbm"] * Gb + ens["dlres"] * Dr).astype(np.float32)
    print(f"  {var}: pesos ensemble {ens} | redes seed-mean em {time.time() - t0:.0f}s")
    return {"lstnet": Pn, "patchtst": Pt, "dlinear": Dl,
            "lgbm": Gb, "dlres": Dr, "ens": En}


def read_val_v2(exp: str) -> tuple[str, dict[str, dict[str, str]]]:
    """Referência val-2024 de um experimento v2 → (arquivo copiado p/ vals/, {modelo: {MAE, RMSE}}).

    Só o 10/11/17 têm metricas_val.csv no formato índice-0 (o 17 com linhas
    extras — read_val_index0/norm dão conta); os demais usam o melhor
    equivalente disponível: 12/13 pooled 5 seeds (métrica × media/dp),
    14/15 pooled por modelo (modelo × MAE_media/...) e 16 zona report (honesta, dez).
    """
    base = ROOT / "univariavel" / "resultados" / exp
    p = base / "metricas_val.csv"
    if p.exists():
        return p.name, read_val_index0(p)
    p = base / "metricas_val_media_dp.csv"
    if p.exists():
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
        header, body = rows[0], rows[1:]
        if header[1] == "media":  # 12/13: métrica × (media, dp)
            d = {r[0]: r[1] for r in body}
            return p.name, {"lstnet": {"MAE": d.get("MAE", ""), "RMSE": d.get("RMSE", "")}}
        out = {}  # 14/15: modelo × (MAE_media, RMSE_media, ...)
        for r in body:
            m = dict(zip(header[1:], r[1:]))
            out[norm(r[0])] = {"MAE": m.get("MAE_media", ""), "RMSE": m.get("RMSE_media", "")}
        return p.name, out
    p = base / "metricas_zonas.csv"  # 16: zona,modelo,MAE,... → report (honesta)
    if p.exists():
        out = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["zona"] == "report":
                    out[norm(r["modelo"])] = {"MAE": r["MAE"], "RMSE": r["RMSE"]}
        return p.name, out
    raise SystemExit(f"val ausente: {base}")


# ---------- inferência completa de uma variável ----------
def roda_variavel(var: str, s24: pd.Series, s: pd.Series,
                  X: np.ndarray, Y: np.ndarray,
                  ends: pd.DatetimeIndex, v2: bool = False) -> dict[str, np.ndarray]:
    t0 = time.time()
    preds = cheap_preds(X)
    preds["sazonal_lag365"], fb = lag365_preds(s24, X, Y, ends)
    # Janela LN a partir do PRÓPRIO X (já com furos ≤3h preenchidos) — nunca da
    # série crua, senão o NaN original vaza p/ as redes. Xt == Wln do 08 em
    # janelas limpas (mesmos valores, mesma ordem).
    Xt = X[:, -LN:].astype(np.float32)
    C = (CKPTS_V2 if v2 else CKPTS)[var]
    if v2:
        # Redes v2 (treino L=2304) leem a cauda LN + covariáveis dos timestamps.
        preds.update(infer_v2(var, C, X, Xt, ends, preds["sazonal_naive_288"]))
        Pn, Pt, Dl = (preds["lstnet"], preds["patchtst"], preds["dlinear"])
        Gb, Dr, En = preds["lgbm"], preds["dlres"], preds["ens"]
    else:
        e_min = ends.values.astype("datetime64[m]").astype(np.int64)
        mins = (e_min - H * 5)[:, None] - 5 * np.arange(LN - 1, -1, -1)[None, :]
        ang = (2 * np.pi * (mins % 1440) / 1440.0).astype(np.float32)
        T = np.stack([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)
        lstnet = load_state(LSTNet1D, C["lstnet"])
        patch = load_state(PatchTST, C["patch"])
        dlin = load_state(DLinearLite, C["dlin"])
        dlres = load_state(DLinearLite, C["dlres"], residual=True)
        lgbms = load_lgbm(var)
        ens = json.load(open(ROOT / "univariavel" / "resultados" / C["ens"]))["pesos"]
        w = [ens["sazonal"], ens["lstnet"], ens["lgbm"], ens["dlres"]]

        @torch.no_grad()
        def fwd1(model, tod=None, batch=64):
            outs = []
            for b in range(0, len(X), batch):
                xb = torch.from_numpy(Xt[b:b + batch])
                if tod is None:
                    outs.append(model(xb).numpy())
                else:
                    outs.append(model(xb, torch.from_numpy(tod[b:b + batch])).numpy())
            return np.concatenate(outs)

        Pn = fwd1(lstnet, T)
        Pt = fwd1(patch)
        Dl = fwd1(dlin)
        F, em = base_feats(X, ends)
        S = preds["sazonal_naive_288"]
        Gb = np.empty((len(X), H), dtype=np.float32)
        for j, m in enumerate(lgbms):
            sh, ch = hour_sincos(em, j)
            Gb[:, j] = S[:, j] + m.predict(np.column_stack([F, sh, ch]))
        Dr = S + fwd1(dlres)
        En = w[0] * S + w[1] * Pn + w[2] * Gb + w[3] * Dr
        preds.update({"lstnet": Pn, "patchtst": Pt, "dlinear": Dl,
                      "lgbm": Gb, "dlres": Dr, "ens": En})
    for m, P in (("lstnet", Pn), ("patchtst", Pt), ("dlinear", Dl),
                 ("dlres", Dr), ("ens", En)):
        n = int(np.isnan(np.asarray(P)).sum())
        if n:
            raise SystemExit(f"{var}/{m}: {n} NaN na predição — origem contaminada")
    preds.update({"lstnet": Pn, "patchtst": Pt, "dlinear": Dl,
                  "lgbm": Gb, "dlres": Dr, "ens": En})
    Pp = ROOT / "univariavel" / "resultados" / C["prophet"]
    if Pp.exists():
        from prophet.serialize import model_from_json
        m = model_from_json(Pp.read_text())
        fmap = m.predict(pd.DataFrame({"ds": s.index})).set_index("ds")["yhat"]
        preds["prophet"] = np.stack(
            [[fmap.loc[d - pd.Timedelta(minutes=5 * (H - 1 - h))] for h in range(H)]
             for d in ends])
        print(f"  {var}: prophet incluído")
    else:
        print(f"  {var}: prophet ausente — pulado")
    print(f"  {var}: {len(X)} origens, lag365 fallback {fb} | "
          f"inferência em {time.time() - t0:.0f}s | modelos: {sorted(preds)}")
    return preds


# ---------- agregação: individual por período + média ----------
def agrega(var: str, preds: dict[str, np.ndarray], Y: np.ndarray,
           pids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pids = np.array(pids)
    per_rows, med_rows = {}, {}
    for m, P in preds.items():
        per_rows[m] = {}
        for pid, _, _ in PERIODOS:
            sel = pids == pid
            n = int(sel.sum())
            per_rows[m][f"{pid}_n"] = n
            per_rows[m][f"{pid}_MAE"] = round(mae(Y[sel], P[sel]), 4) if n else ""
            per_rows[m][f"{pid}_RMSE"] = round(rmse(Y[sel], P[sel]), 4) if n else ""
        med_rows[m] = {"MAE": round(mae(Y, P), 4),
                       "RMSE": round(rmse(Y, P), 4), "n": len(Y)}
    cols = [c for pid, _, _ in PERIODOS for c in (f"{pid}_MAE", f"{pid}_RMSE", f"{pid}_n")]
    return (pd.DataFrame(per_rows).T[cols], pd.DataFrame(med_rows).T[["MAE", "RMSE", "n"]])


# ---------- snapshot versionado ----------
def cmd_snapshot(tag: str, refazer: bool = False) -> None:
    dest = OUTROOT / tag
    if dest.exists() and not refazer:
        raise SystemExit(f"tag '{tag}' já existe em {dest} — use --refazer p/ recongelar")
    if dest.exists():
        shutil.rmtree(dest)
        print(f"[{tag}] pasta anterior removida (--refazer)")
    if HIST.exists():
        with open(HIST, newline="") as f:
            if any(r.get("tag") == tag for r in csv.DictReader(f)) and not refazer:
                raise SystemExit(f"tag '{tag}' já consta em {HIST} — use --refazer")
    t00 = time.time()
    TR, BM = ROOT / "univariavel" / "dados" / "treino", ROOT / "univariavel" / "dados" / "benchmark"
    v2 = (tag == "v2")
    CKD = CKPTS_V2 if v2 else CKPTS
    if v2:
        for var, d in CKD.items():
            for sd in SEEDS_V2:
                pl = ROOT / "univariavel" / "resultados" / d["lstnet_dir"] / d["lstnet_pat"].format(seed=sd)
                pp = ROOT / "univariavel" / "resultados" / d["td_dir"] / d["patch_pat"].format(seed=sd)
                pd_ = ROOT / "univariavel" / "resultados" / d["td_dir"] / d["dlin_pat"].format(seed=sd)
                pr = ROOT / "univariavel" / "resultados" / Path(d["lgbm_dir"]).parent / d["dlres_pat"].format(seed=sd)
                for p in (pl, pp, pd_, pr):
                    if not p.exists():
                        raise SystemExit(f"checkpoint ausente: {p}")
            nat = ROOT / "univariavel" / "resultados" / d["lgbm_dir"]
            nh = len(list(nat.glob("lgbm_h*.txt")))
            nj = len(list(nat.glob("model_j*.txt")))
            if not ((nh == H) ^ (nj == H)):
                raise SystemExit(f"LGBM ausente p/ {var}: {nat} (lgbm_h*={nh} model_j*={nj})")
            if not (ROOT / "univariavel" / "resultados" / d["ens"]).exists():
                raise SystemExit(f"checkpoint ausente: {ROOT / 'univariavel' / 'resultados' / d['ens']}")
            # prophet do 10/11 é opcional (pulado na inferência se ausente)
    else:
        for var, d in CKD.items():
            for k in ("lstnet", "patch", "dlin", "dlres", "ens"):
                p = ROOT / "univariavel" / "resultados" / d[k]
                if not p.exists():
                    raise SystemExit(f"checkpoint ausente: {p}")
            if not ((ROOT / "univariavel" / "resultados" / d["lgbm"]).exists()
                    or (ROOT / "univariavel" / "resultados" / d["lgbm_fb"]).exists()):
                raise SystemExit(f"LGBM ausente p/ {var}")

    dest.mkdir(parents=True)
    (dest / "vals").mkdir()
    cobertura: dict[str, dict[str, int]] = {}
    preench: dict[str, dict[str, int]] = {}
    medias: dict[str, pd.DataFrame] = {}
    for var, (f24, f25, col) in FILES.items():
        print(f"[{tag}] {var}: carga 2024+2025...")
        s24, s25 = ler(TR / f24, col), ler(BM / f25, col)
        X, Y, ends, pids, maxfill = origens_probe(s25)
        if len(X) == 0:
            raise SystemExit(f"[{tag}] {var}: zero origens — revise PERIODOS")
        nper = {pid: int((np.array(pids) == pid).sum()) for pid, _, _ in PERIODOS}
        cobertura[var] = nper
        preench[var] = maxfill
        print(f"[{tag}] {var}: {len(X)} origens {nper} | máx preenchido {maxfill}")
        preds = roda_variavel(var, s24, s25, X, Y, ends, v2=v2)
        per, med = agrega(var, preds, Y, pids)
        per.to_csv(dest / f"metricas_periodos_{var}.csv")
        med.to_csv(dest / f"metricas_media_{var}.csv")
        medias[var] = med
        print(f"[{tag}] {var}: pooled ens MAE={med.loc['ens', 'MAE']} "
              f"(lstnet {med.loc['lstnet', 'MAE']})")

    # val 2024 (referência; sem 2025 aqui)
    val: dict[str, dict[str, dict[str, str]]] = {}
    for var, exps in (VAL_EXPS_V2 if v2 else VAL_EXPS).items():
        val[var] = {}
        for exp in exps:
            if v2:
                fname, d = read_val_v2(exp)
                shutil.copy2(ROOT / "univariavel" / "resultados" / exp / fname, dest / "vals" / f"{exp}-{fname}")
            else:
                p = ROOT / "univariavel" / "resultados" / exp / "metricas_val.csv"
                if not p.exists():
                    raise SystemExit(f"val ausente: {p}")
                shutil.copy2(p, dest / "vals" / f"{exp}-metricas_val.csv")
                d = read_val_index0(p)
            for modelo, m in d.items():
                val[var][modelo] = m  # última fonte canônica vence

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = git_sha()
    protocolo = {"L": L, "H": H, "nota": "só inferência em 2025, sem tuning",
                 "tolerancia_nan_slots": TOL_NAN,
                 "nota_tolerancia": "origens com furo ≤3h têm o furo interpolado localmente; gap maior descarta a origem"}
    if v2:
        protocolo.update({"L_treino": L_TREINO_V2, "modelos_LN": LN, "seeds": SEEDS_V2,
                          "covariaveis": ["tod_sin", "tod_cos", "solar Elev/90",
                                          "f1_sinA", "f1_cosA", "f2_sinS", "f2_cosS"],
                          "lgbm": "288 nativos/horizonte (lgbm_h* ph / model_j* od), 32 feats",
                          "ensemble": "pesos ensemble.json do 16/17 (sem refit)"})
    (dest / "meta.json").write_text(json.dumps({
        "tag": tag, "timestamp_utc": ts, "git_sha": sha,
        "metodologia": "probe-4-periodos-ancoras-23h55 tolerancia-3h (pipeline própria; 08 não lido)",
        "protocolo": protocolo,
        "periodos": {pid: {"inicio": a, "fim": b, "n_ph": cobertura["ph"][pid],
                           "n_od": cobertura["od"][pid],
                           "preenchido_ph": preench["ph"][pid],
                           "preenchido_od": preench["od"][pid]}
                     for pid, a, b in PERIODOS},
        "runtime_s": round(time.time() - t00),
    }, indent=2, sort_keys=True) + "\n")

    linhas = []
    for var in ("ph", "od"):
        for modelo in medias[var].index:
            v = val[var].get(norm(modelo), {})
            linhas.append({"timestamp_utc": ts, "git_sha": sha, "tag": tag,
                           "variavel": var, "modelo": norm(modelo),
                           "MAE_val": v.get("MAE", ""), "RMSE_val": v.get("RMSE", ""),
                           "MAE_benchmark": medias[var].loc[modelo, "MAE"],
                           "RMSE_benchmark": medias[var].loc[modelo, "RMSE"],
                           "MAE_diaria": ""})
    OUTROOT.mkdir(parents=True, exist_ok=True)
    if refazer and HIST.exists():  # remove linhas antigas da tag
        with open(HIST, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("tag") != tag]
        with open(HIST, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HIST_FIELDS)
            w.writeheader()
            w.writerows(rows)
    novo = not HIST.exists()
    with open(HIST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HIST_FIELDS)
        if novo:
            w.writeheader()
        w.writerows(linhas)
    print(f"[{tag}] {len(linhas)} linhas → {dest.relative_to(ROOT)} + "
          f"{HIST.relative_to(ROOT)} ({time.time() - t00:.0f}s total)")
    cmd_plot(tag)


def cmd_diff(a: str, b: str) -> None:
    if not HIST.exists():
        raise SystemExit(f"sem histórico: {HIST}")
    with open(HIST, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["tag"] in (a, b)]
    if not rows:
        raise SystemExit(f"tags '{a}'/'{b}' não encontradas em {HIST}")
    base = {(r["variavel"], r["modelo"]): r for r in rows if r["tag"] == a}
    novo = {(r["variavel"], r["modelo"]): r for r in rows if r["tag"] == b}
    if not base or not novo:
        raise SystemExit(f"tag ausente: {a} ({len(base)}) / {b} ({len(novo)})")
    print(f"{'var':<4} {'modelo':<18} {'MAE_probe ' + a:>12} {'MAE_probe ' + b:>12} "
          f"{'Δ':>10} {'Δ%':>8}")
    for key in sorted(novo):
        r0, r1 = base.get(key), novo[key]
        if r0 is None:
            print(f"{key[0]:<4} {key[1]:<18} {'—':>12} {r1['MAE_benchmark']:>12} "
                  f"{'novo':>10} {'':>8}")
            continue
        try:
            v0, v1 = float(r0["MAE_benchmark"]), float(r1["MAE_benchmark"])
            d, pct = v1 - v0, 100 * (v1 - v0) / v0
            print(f"{key[0]:<4} {key[1]:<18} {v0:>12.4f} {v1:>12.4f} {d:>+10.4f} {pct:>+7.2f}%")
        except ValueError:
            print(f"{key[0]:<4} {key[1]:<18} {r0['MAE_benchmark']:>12} {r1['MAE_benchmark']:>12}")


# ---------- gráficos (a partir da média do probe) ----------
ORDEM_MODELOS = [
    "persistencia", "sazonal_naive_288", "media_movel_288", "sazonal_lag365",
    "lstnet", "patchtst", "dlinear", "lgbm", "dlres", "ens", "prophet",
]
ROTULO = {
    "persistencia": "persist", "sazonal_naive_288": "saz-288",
    "media_movel_288": "mm-288", "sazonal_lag365": "lag-365",
}


def _linhas_tag(tag: str) -> list[dict[str, str]]:
    if not HIST.exists():
        raise SystemExit(f"sem histórico: {HIST}")
    with open(HIST, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["tag"] == tag]
    if not rows:
        raise SystemExit(f"tag '{tag}' não encontrada em {HIST}")
    return rows


def _prepara(tag: str) -> tuple[dict[str, dict[str, float]], str]:
    mae: dict[str, dict[str, float]] = {"ph": {}, "od": {}}
    meta = ""
    for r in _linhas_tag(tag):
        try:
            mae[r["variavel"]][r["modelo"]] = float(r["MAE_benchmark"])
        except ValueError:
            continue
        meta = f"{tag} · {r['git_sha']} · {r['timestamp_utc']}"
    return mae, meta


def _fig_base():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    return fig, ax, plt


def cmd_plot(tag: str) -> None:
    """Retrato de uma versão: MAE probe por modelo, pH e OD (escala log)."""
    mae, meta = _prepara(tag)
    fig, ax, plt = _fig_base()
    for i, var in enumerate(("ph", "od")):
        modelos = [m for m in ORDEM_MODELOS if m in mae[var]] or sorted(mae[var])
        valores = [mae[var][m] for m in modelos]
        cores = ["#c0392b" if m == "ens" else "#2980b9" for m in modelos]
        a = ax[i]
        barras = a.bar(range(len(modelos)), valores, color=cores)
        a.set_yscale("log")
        a.set_xticks(range(len(modelos)))
        a.set_xticklabels([ROTULO.get(m, m) for m in modelos], rotation=45, ha="right")
        a.set_title(f"{var.upper()} — MAE probe 2025 (4 períodos)")
        a.set_ylabel("MAE (log) — menor é melhor")
        for b, v in zip(barras, valores):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                   ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"Benchmark versionado — retrato {meta}")
    fig.tight_layout()
    out = OUTROOT / tag / f"01-mae-benchmark-{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[plot {tag}] → {out.relative_to(ROOT)}")


def cmd_plot_vs(a: str, b: str) -> None:
    """Evolução A → B: barras lado a lado por modelo."""
    import numpy as np
    ma, _ = _prepara(a)
    mb, _ = _prepara(b)
    fig, ax, plt = _fig_base()
    for i, var in enumerate(("ph", "od")):
        modelos = [m for m in ORDEM_MODELOS if m in ma[var] and m in mb[var]]
        va = [ma[var][m] for m in modelos]
        vb = [mb[var][m] for m in modelos]
        x = np.arange(len(modelos))
        p = ax[i]
        p.bar(x - 0.2, va, 0.4, label=a, color="#7f8c8d")
        p.bar(x + 0.2, vb, 0.4, label=b, color="#c0392b")
        p.set_yscale("log")
        p.set_xticks(x)
        p.set_xticklabels([ROTULO.get(m, m) for m in modelos], rotation=45, ha="right")
        p.set_title(f"{var.upper()} — MAE probe ({a} cinza × {b} vermelho)")
        p.set_ylabel("MAE (log) — menor é melhor")
        p.legend(fontsize=8)
    fig.suptitle(f"Benchmark versionado — evolução {a} → {b}")
    fig.tight_layout()
    out = OUTROOT / b / f"01-mae-benchmark-{a}-vs-{b}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[plot {a} vs {b}] → {out.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark versionado (probe 4 períodos).")
    ap.add_argument("--tag", help="roda o probe e congela a versão (ex.: v1)")
    ap.add_argument("--refazer", action="store_true",
                    help="com --tag: apaga a tag existente e recongela")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="compara duas tags")
    ap.add_argument("--plot", nargs="+", metavar="TAG",
                    help="gráfico da média: 1 tag (retrato) ou 2 tags (evolução)")
    args = ap.parse_args(argv)
    if args.diff:
        cmd_diff(*args.diff)
    elif args.plot:
        if len(args.plot) == 1:
            cmd_plot(args.plot[0])
        elif len(args.plot) == 2:
            cmd_plot_vs(*args.plot)
        else:
            raise SystemExit("--plot aceita 1 tag (retrato) ou 2 tags (evolução)")
    elif args.tag:
        cmd_snapshot(args.tag, refazer=args.refazer)
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
