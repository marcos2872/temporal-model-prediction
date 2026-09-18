"""Pré-processamento multivariável — espelha os notebooks M1/M2/M3.

Grade 5 min → interp `time` limite 24 → winsorize da Turbidez (p99 do treino)
→ z-score por canal (stats do `normalizacao.json` da família) → 11 séries:
[od, ph, temp, turb] + tod_sin/cos + solar + 4 Fourier do dia-da-origem.
"""

from io import StringIO

import numpy as np
import pandas as pd

LAT, LON, TZ = -23.52, -46.19, -3  # Mogi das Cruzes (verbatim 12/16)
L = 2304
INTERP_LIMIT = 24

# mapeamento coluna CETESB -> canal (match por substring, case-insensitive)
CANAIS = (
    ("od", ("oxig",)),
    ("ph", ("ph",)),
    ("temp", ("temperatura",)),
    ("turb", ("turbidez",)),
)


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


def ler_csv_multivariavel(conteudo: bytes) -> pd.DataFrame:
    """Lê CSV CETESB multivariável; exige os 4 canais (OD, pH, Temp, Turb)."""
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
    cols = [str(c).strip().lower() for c in df.columns]
    mapa = {}
    for canal, chaves in CANAIS:
        hit = next((df.columns[i] for i, c in enumerate(cols)
                    if all(k in c for k in chaves) and not (canal == "ph" and "oxig" in c)),
                   None)
        if hit is None:
            raise ValueError(f"coluna de {canal} não encontrada no CSV (esperado OD, pH, Temperatura, Turbidez)")
        mapa[canal] = hit
    out = pd.DataFrame({"ds": df.iloc[:, 0]})
    for canal in ("od", "ph", "temp", "turb"):
        out[canal] = pd.to_numeric(df[mapa[canal]], errors="coerce")
    out = out.sort_values("ds").reset_index(drop=True)
    if out["ds"].isna().any():
        raise ValueError("há datas ilegíveis na 1ª coluna (formato esperado dd/mm/aaaa hh:mm)")
    return out


def prepara(df: pd.DataFrame, H: int, mu: np.ndarray, sd: np.ndarray,
            turb_p99: float) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    """Contexto (11, L) em z-score + grade + cobertura. Erro 422 se a cauda L+H não fecha."""
    idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq="5min")
    if len(idx) < L + H:
        raise ValueError(
            f"série curta: {len(idx)} slots — mínimo {L + H} "
            f"(~8 dias de contexto + {H * 5 // 60 or 1} h de margem)"
        )
    g = df.set_index("ds").reindex(idx)
    falt_antes = int(g.isna().sum().sum())
    g = g.interpolate(method="time", limit=INTERP_LIMIT)
    cauda = g.iloc[-(L + H):]
    if int(cauda.isna().sum().sum()):
        raise ValueError(
            "contexto insuficiente no fim da série: há falha maior que 2 h "
            f"nos últimos {L + H} slots — o modelo não prevê no escuro"
        )
    v = g[["od", "ph", "temp", "turb"]].to_numpy(dtype=np.float64)
    v[:, 3] = np.minimum(v[:, 3], turb_p99)
    vz = ((v - mu) / sd).astype(np.float32)
    wl = vz[-L:]  # (L, 4)
    t_idx = idx[-L:]
    tod_sin = np.sin(2 * np.pi * (t_idx.hour.to_numpy() * 60 + t_idx.minute.to_numpy()) / 1440.0)
    tod_cos = np.cos(2 * np.pi * (t_idx.hour.to_numpy() * 60 + t_idx.minute.to_numpy()) / 1440.0)
    solar = elevacao_solar(t_idx) / 90.0
    f1s, f1c, f2s, f2c = fourier_doy([idx.max()])
    forg = np.column_stack([np.full(L, a, dtype=np.float32) for a in (f1s[0], f1c[0], f2s[0], f2c[0])])
    X = np.concatenate([
        wl.T,  # (4, L) [od, ph, temp, turb]
        np.stack([tod_sin, tod_cos, solar], axis=0).astype(np.float32),
        forg.T,
    ], axis=0).astype(np.float32)  # (11, L)
    cobertura = {
        "slots": len(idx),
        "faltantes_originais": falt_antes,
        "pct_interpolado": round(100 * (falt_antes - int(g.isna().sum().sum())) / len(idx), 2),
    }
    return X, idx, cobertura
