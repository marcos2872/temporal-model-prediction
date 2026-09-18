"""Pipeline de teste da API multivariável (M1/M2/M3) sobre `dados.csv` (2026, unseen).

Sobe um app por modelo (subprocesso uvicorn, flag --modelo), divide a série em
até 9 partes contíguas e, por parte × horizonte (1h/6h/24h) × variável (ph/od),
envia o contexto (últimos L+H slots da parte) ao POST /prever e compara com o
holdout (primeiros H slots da parte seguinte — verdade nunca enviada).
Só inferência + comparação: nenhum peso é ajustado, 2025 não é tocado.

Uso (a partir da raiz do repo):
    .venv/bin/python multivariavel/src/teste.py [--partes 9] [--modelos M1 M2 M3]
    .venv/bin/python multivariavel/src/teste.py --so-validados   # corta em 14/09/2026

Logs em multivariavel/src/.logs/: M1.log, M2.log, M3.log + geral.log
(métricas por previsão: MAE, RMSE, MAPE e acertividade = 100 - MAPE).
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from multivariavel.src.features import L, ler_csv_multivariavel  # noqa: E402

HS = {1: 12, 6: 72, 24: 288}
VARS = ("ph", "od")
LOGDIR = Path(__file__).resolve().parent / ".logs"
HEADER_CETESB = "Entidade responsável: CETESB | Teste automatizado\n"
HEADER_COLS = "Data hora;Oxigênio Dissolvido (mg/L);pH;Temperatura (°C);Turbidez (NTU)\n"


def serie_para_csv(df: pd.DataFrame) -> bytes:
    """Re-serializa um trecho no formato CETESB (exercita o parser real)."""
    linhas = [HEADER_CETESB, HEADER_COLS]
    for _, r in df.iterrows():
        vals = []
        for c in ("od", "ph", "temp", "turb"):
            v = r[c]
            vals.append("" if pd.isna(v) else f"{float(v):.2f}".replace(".", ","))
        linhas.append(f"{r['ds'].strftime('%d/%m/%Y %H:%M')};{';'.join(vals)}\n")
    return "".join(linhas).encode("windows-1252")


def metricas(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = np.abs(y - yhat)
    mae = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err / np.abs(y)).mean() * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape,
            "acertividade": max(0.0, 100.0 - mape), "n": len(y)}


def espera_saude(base: str, limite_s: int = 240) -> None:
    t0 = time.time()
    while time.time() - t0 < limite_s:
        try:
            r = httpx.get(f"{base}/saude", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"app não subiu em {base} após {limite_s}s")


def fmt_linha(cols, larguras):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, larguras))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Testa M1/M2/M3 sobre dados.csv (9 partes × 3H × ph/od)")
    ap.add_argument("--partes", type=int, default=9)
    ap.add_argument("--modelos", nargs="+", default=["M1", "M2", "M3"])
    ap.add_argument("--port-base", type=int, default=8001)
    ap.add_argument("--so-validados", action="store_true",
                    help="corta a série em 14/09/2026 (só trecho validado CETESB)")
    a = ap.parse_args(argv)

    raw = (ROOT / "multivariavel" / "src" / "dados.csv").read_bytes()
    df = ler_csv_multivariavel(raw)
    if a.so_validados:
        df = df[df["ds"] < pd.Timestamp("2026-09-14")].reset_index(drop=True)
    print(f"série: {len(df)} linhas, {df['ds'].min()} → {df['ds'].max()}")

    n = len(df) // a.partes
    partes = [df.iloc[i * n:(i + 1) * n if i < a.partes - 1 else len(df)].reset_index(drop=True)
              for i in range(a.partes)]
    print(f"{len(partes)} partes de ~{n} linhas; pontuáveis: {len(partes) - 1} (última = smoke)")
    por_fatia = df.set_index("ds")

    LOGDIR.mkdir(parents=True, exist_ok=True)
    geral = {}  # modelo -> lista de (parte, H, var, dict|None)

    for mi, fam in enumerate(a.modelos):
        porta = a.port_base + mi
        base = f"http://127.0.0.1:{porta}"
        proc = subprocess.Popen(
            [sys.executable, "-m", "multivariavel.src.app", "--modelo", fam, "--port", str(porta)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        linhas_log = [f"# {fam} — {datetime.now():%Y-%m-%d %H:%M} — {len(partes) - 1} partes pontuáveis",
                       f"# dados.csv: {df['ds'].min()} → {df['ds'].max()}"]
        try:
            espera_saude(base)
            with httpx.Client(timeout=300) as c:
                # self-checks do contrato de erro
                r = c.post(f"{base}/prever", files={"arquivo": ("x.csv", b"foo,bar", "text/csv")},
                           params={"variavel": "ph", "horizonte_horas": 24})
                linhas_log.append(f"# self-check 400 csv-lixo: {r.status_code} (esperado 400)")
                for pi in range(len(partes) - 1):
                    parte, prox = partes[pi], partes[pi + 1]
                    for horas, H in HS.items():
                        ctx = parte.iloc[-(L + H):]
                        payload = serie_para_csv(ctx)
                        for var in VARS:
                            r = c.post(f"{base}/prever",
                                       files={"arquivo": (f"p{pi}.csv", payload, "text/csv")},
                                       params={"variavel": var, "horizonte_horas": horas})
                            chave = (pi, horas, var)
                            if r.status_code == 422:
                                geral.setdefault(fam, []).append((pi, horas, var, None))
                                linhas_log.append(f"parte {pi} H={horas}h {var}: SEM_COBERTURA (422)")
                                continue
                            if r.status_code != 200:
                                geral.setdefault(fam, []).append((pi, horas, var, None))
                                linhas_log.append(f"parte {pi} H={horas}h {var}: ERRO_HTTP {r.status_code}")
                                continue
                            vals = r.json()["valores"]
                            ts = pd.DatetimeIndex([p["ds"] for p in vals])
                            try:
                                y = por_fatia.loc[ts, var].to_numpy(dtype=float)
                            except KeyError:
                                geral.setdefault(fam, []).append((pi, horas, var, None))
                                linhas_log.append(f"parte {pi} H={horas}h {var}: SEM_HOLDOUT")
                                continue
                            if np.isnan(y).any():
                                geral.setdefault(fam, []).append((pi, horas, var, None))
                                linhas_log.append(f"parte {pi} H={horas}h {var}: HOLDOUT_COM_FALTA")
                                continue
                            met = metricas(y, np.array([p["y"] for p in vals]))
                            geral.setdefault(fam, []).append((pi, horas, var, met))
                            linhas_log.append(
                                f"parte {pi} H={horas}h {var}: "
                                f"MAE={met['MAE']:.4f} RMSE={met['RMSE']:.4f} "
                                f"MAPE={met['MAPE']:.2f}% acert={met['acertividade']:.2f}%")
                # smoke da última parte (sem holdout)
                ctx = partes[-1].iloc[-(L + 288):]
                r = c.post(f"{base}/prever",
                           files={"arquivo": ("smoke.csv", serie_para_csv(ctx), "text/csv")},
                           params={"variavel": "ph", "horizonte_horas": 24})
                linhas_log.append(f"# smoke parte final H=24h ph: {r.status_code} (esperado 200)")
        finally:
            proc.terminate()
            proc.wait(timeout=30)

        # totais do modelo (média macro das partes pontuáveis, por H/var)
        linhas_log.append("# ---- totais (média macro, só pontuáveis) ----")
        for horas in HS:
            for var in VARS:
                ms = [m for (p, h, v, m) in geral.get(fam, []) if h == horas and v == var and m]
                if ms:
                    linhas_log.append(
                        f"H={horas}h {var}: n={len(ms)} "
                        f"MAE={np.mean([m['MAE'] for m in ms]):.4f} "
                        f"MAPE={np.mean([m['MAPE'] for m in ms]):.2f}% "
                        f"acert={np.mean([m['acertividade'] for m in ms]):.2f}%")
        (LOGDIR / f"{fam}.log").write_text("\n".join(linhas_log) + "\n")
        print(f"{fam}: log em {LOGDIR / f'{fam}.log'}")

    # geral.log — matriz modelo × período + ranking
    gl = ["# GERAL — acertividade (%) por modelo × (parte, H, var)", ""]
    w = [16, 10, 10, 10]
    gl.append(fmt_linha(["(parte,H,var)", "M1", "M2", "M3"], w))
    chaves = sorted({(p, h, v) for fam in geral for (p, h, v, _) in geral[fam]})
    for chave in chaves:
        row = [f"p{chave[0]} H={chave[1]}h {chave[2]}"]
        for fam in ("M1", "M2", "M3"):
            m = next((m for (p, h, v, m) in geral.get(fam, []) if (p, h, v) == chave), None)
            row.append(f"{m['acertividade']:.2f}" if m else "-")
        gl.append(fmt_linha(row, w))
    gl += ["", "# ---- ranking (média macro de acertividade, por H/var) ----"]
    for horas in HS:
        for var in VARS:
            medias = {}
            for fam in ("M1", "M2", "M3"):
                ms = [m["acertividade"] for (p, h, v, m) in geral.get(fam, [])
                      if h == horas and v == var and m]
                if ms:
                    medias[fam] = float(np.mean(ms))
            if medias:
                rank = " > ".join(f"{f} {medias[f]:.2f}%" for f in sorted(medias, key=medias.get, reverse=True))
                gl.append(f"H={horas}h {var}: {rank}")
    (LOGDIR / "geral.log").write_text("\n".join(gl) + "\n")
    print(f"geral: {LOGDIR / 'geral.log'}")


if __name__ == "__main__":
    main()
