#!/usr/bin/env python3
"""Teste de paridade notebook <-> app.py (Fase 1.4).

Roda com (a partir da raiz do repo, sem rede, sem escrita no repo):
    .venv/bin/python scripts/test_paridade.py

Exit 0 se tudo passa; exit 1 se qualquer item (a-f) falhar.

O que cada item verifica:
  (a) app.carrega("ph") / carrega("od") executam sem erro;
  (b) paridade numerica: recomputa a previsao de forma INDEPENDENTE (funcoes
      proprias deste arquivo que leem os state_dicts via torch.load + aplicam
      as formulas transcritas dos notebooks 02/03/06/07 — NUNCA chama
      app.inferencia nem reutiliza app.LSTNet1D/app.DLinearLite) em 2 janelas
      de 2024 por variavel; compara com app.inferencia (tol 1e-5). Para o LGBM,
      checagem dedicada do caminho completo em 1 janela de OD (288 horizontes);
  (c) pesos usados em app.inferencia conferem com ensemble.json
      (ordem: sazonal, lstnet, lgbm, dlres) e com os READMEs 06/07;
  (d) consistencia lgbm_steps.pkl.gz x lgbm_steps.pkl (mesmas predicoes);
  (e) smoke test do endpoint POST /prever (TestClient; fallback direto
      prepara()+inferencia()+schema se TestClient indisponivel);
  (f) edge cases ler_csv/prepara: CSV vazio, serie curta, gap >2h na cauda.
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app as api  # noqa: E402  (app.py da raiz; so usamos carrega/ler_csv/prepara/inferencia/schemas)

TOL_PARIDADE = 1e-5
TOL_LGBM_COMP = 1e-4  # checagem isolada do termo LGBM amplia erro fp32 (~1/w_lgbm); ver (b)
L, H, LN, SEASON = 8640, 288, 2016, 288

FALHAS = []
MAXDIV = {}  # rotulo -> maior |app - independente|


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- janelas 2024
COL = {"ph": "pH", "od": "Oxigênio Dissolvido (mg/L)"}
CSV_2024 = {
    "ph": ROOT / "dados/treino/ef01-mogi-das-cruzes_ph_2024.csv",
    "od": ROOT / "dados/treino/ef01-mogi-das-cruzes_oxigenio-dissolvido_2024.csv",
}


def serie_2024(var):
    df = pd.read_csv(CSV_2024[var], sep=";", decimal=",", encoding="windows-1252",
                     skiprows=1, parse_dates=["Data hora"], dayfirst=True, na_values=[""])
    return (df.rename(columns={"Data hora": "ds", COL[var]: "y"})[["ds", "y"]]
              .sort_values("ds").reset_index(drop=True))


def janelas_limpas(var, bases=("2024-08-15 12:00", "2024-10-10 12:00"), n=2):
    """2 janelas (ctx float32 len 8640 sem NaN, fim) — datas fixas, avanca dias se sujo."""
    df = serie_2024(var)
    achadas = []
    for b in bases:  # uma janela por base (estacoes distintas), avanca dias se sujo
        for k in range(7):
            c = pd.Timestamp(b) + pd.Timedelta(days=k)
            sub = df[df["ds"] <= c].tail(40 * 288)
            try:
                ctx, grade, _ = api.prepara(sub)
            except ValueError:
                continue
            fim = grade.max()
            if abs((fim - c).total_seconds()) > 6 * 3600:
                continue  # fim caiu longe do alvo (buraco no dado) — tenta proximo dia
            if len(ctx) != L or bool(np.isnan(ctx).any()):
                continue
            achadas.append((ctx, fim))
            break
    check(len(achadas) == n,
          f"({var}) nao achei {n} janelas limpas de 2024 perto de {bases}")
    return achadas


# ------------------------------------------------- reimplementacao independente
def _sd(var, nome):
    base = {"ph": ("02-lstnet-ph", "lstnet_ph.pt", "06-ensemble-ph", "dlinear_res_ph.pt"),
            "od": ("03-lstnet-od", "lstnet_od.pt", "07-ensemble-od", "dlinear_res_od.pt")}[var]
    if nome == "lstnet":
        p = ROOT / "univariavel" / "resultados" / base[0] / "modelos" / base[1]
    else:
        p = ROOT / "univariavel" / "resultados" / base[2] / "modelos" / base[3]
    return torch.load(p, map_location="cpu", weights_only=False)["state"]


def tod_indep(fim):
    """sin/cos da hora-do-dia p/ 2016 passos terminando em `fim` (via aritmetica, sem pandas)."""
    fim_min = fim.hour * 60 + fim.minute
    mins = (fim_min - (LN - 1) * 5 + 5 * np.arange(LN, dtype=np.float64)) % 1440.0
    return (np.sin(2 * np.pi * mins / 1440.0).astype(np.float32),
            np.cos(2 * np.pi * mins / 1440.0).astype(np.float32))


def lstnet_indep(state, x, tod):
    """Forward LSTNet com ops manuais sobre os tensores do state_dict (GRU/GRUCell
    desenrolados na mao; sem nn.Module, sem app.LSTNet1D). Tudo float32, sem dropout."""
    xv = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, :]
    mu = xv.mean(dim=1, keepdim=True)
    sg = xv.std(dim=1, keepdim=True).clamp_min(1e-3)
    g, b = state["gamma"], state["beta"]
    vn = g * (xv - mu) / sg + b
    tt = torch.from_numpy(np.ascontiguousarray(tod, dtype=np.float32))[None, :, :]
    xc = torch.cat([vn.unsqueeze(1), tt.transpose(1, 2)], dim=1)  # (1,3,2016)
    Wc, bc = state["conv.weight"], state["conv.bias"]
    unf = xc.unfold(2, 12, 6)  # (1,3,335,12)
    f = torch.relu(torch.einsum("bctk,ock->bot", unf, Wc) + bc[None, :, None]).transpose(1, 2)

    def gru_manual(seq, Wi, Wh, bi, bh):
        hs = torch.zeros(1, Wi.shape[0] // 3)
        for t in range(seq.shape[1]):
            gi = seq[:, t, :] @ Wi.T + bi
            gh = hs @ Wh.T + bh
            ri, zi, ni = gi.chunk(3, dim=1)
            rh, zh, nh = gh.chunk(3, dim=1)
            r = torch.sigmoid(ri + rh)
            z = torch.sigmoid(zi + zh)
            n = torch.tanh(ni + r * nh)
            hs = (1 - z) * n + z * hs
        return hs

    h = gru_manual(f, state["gru.weight_ih_l0"], state["gru.weight_hh_l0"],
                   state["gru.bias_ih_l0"], state["gru.bias_hh_l0"])
    hs, states = torch.zeros(1, 32), [torch.zeros(1, 32)]
    for t in range(f.shape[1]):  # skip-lag 48, cf. notebooks 02/03 (SKIP_P=48)
        prev = states[t - 48] if t - 48 >= 0 else states[0]
        # passo GRUCell manual (input f[:,t,:], hidden prev)
        gi = f[:, t, :] @ state["skipcell.weight_ih"].T + state["skipcell.bias_ih"]
        gh = prev @ state["skipcell.weight_hh"].T + state["skipcell.bias_hh"]
        ri, zi, ni = gi.chunk(3, dim=1)
        rh, zh, nh = gh.chunk(3, dim=1)
        r = torch.sigmoid(ri + rh)
        z = torch.sigmoid(zi + zh)
        n = torch.tanh(ni + r * nh)
        hs = (1 - z) * n + z * prev
        states.append(hs)
    yn = torch.cat([h, hs], dim=1) @ state["head.weight"].T + state["head.bias"]
    ya = vn[:, -288:] @ state["ar.weight"].T + state["ar.bias"]
    with torch.no_grad():
        return ((yn + ya - b) / g.clamp_min(1e-3) * sg + mu).numpy()[0]


def dlinear_indep(state, x):
    """Forward DLinear-residual em numpy float32 (media movel k=25 c/ zero-pad,
    como AvgPool1d(padding=12) com count_include_pad=True)."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    mu = np.float32(x32.mean())
    sg = np.float32(max(float(x32.std(ddof=1)), 1e-3))  # torch.std padrao: unbiased
    g, b = np.float32(state["gamma"].numpy()), np.float32(state["beta"].numpy())
    xn = g * (x32 - mu) / sg + b
    xp = np.concatenate([np.zeros(12, np.float32), xn, np.zeros(12, np.float32)])
    cs = np.cumsum(np.concatenate([[0.0], xp.astype(np.float64)]))
    trend = ((cs[25:] - cs[:-25]) / 25.0).astype(np.float32)
    assert trend.shape == (LN,), trend.shape
    Wt, bt = state["lin_t.weight"].numpy(), state["lin_t.bias"].numpy()
    Ws, bs = state["lin_s.weight"].numpy(), state["lin_s.bias"].numpy()
    y = (trend @ Wt.T + bt + (xn - trend) @ Ws.T + bs).astype(np.float32)
    return ((y - b) / max(g, np.float32(1e-3)) * sg)  # modo residual (alvo = Y - sazonal)


def lgbm_feats_indep(ctx, fim):
    """Features transcritas do base_feats dos notebooks 06/07 (end-relativas)."""
    n = len(ctx)
    Xb = np.ascontiguousarray(ctx, dtype=np.float32)[None, :]
    cols = [Xb[:, -k] for k in [1, 2, 3, 6, 12, 24, 36, 72, 144, 287, 288, 289, 576, 2016]]
    phase = np.stack([Xb[:, n - 288 * k] for k in range(1, 8)], axis=1)
    cols += [phase.mean(1), phase.std(1)]
    for w in [12, 36, 144, 288]:
        cols += [Xb[:, -w:].mean(1), Xb[:, -w:].std(1)]
    cols += [Xb[:, -2016:].mean(1)]
    F = np.stack(cols, axis=1).astype(np.float32)
    em = np.float32(fim.hour * 60 + fim.minute)
    return F, em


def ensemble_indep(var, ctx, fim, states, pesos_json, modelos_lgbm):
    """Ensemble completo independente: sazonal + lstnet + lgbm + dlres (ordem do NNLS)."""
    n = len(ctx)
    saz = np.ascontiguousarray(ctx, dtype=np.float32)[n - SEASON:n - SEASON + H].copy()
    x = np.ascontiguousarray(ctx, dtype=np.float32)[-LN:]
    sn, cs = tod_indep(fim)
    pn = lstnet_indep(states["lstnet"], x, np.stack([sn, cs], axis=1)).astype(np.float32)
    dr = (saz + dlinear_indep(states["dlres"], x)).astype(np.float32)
    w = [np.float64(pesos_json[k]) for k in ("sazonal", "lstnet", "lgbm", "dlres")]
    if modelos_lgbm is None:
        return (w[0] * saz + w[1] * pn + w[3] * dr).astype(np.float32), saz, pn, None, dr
    F, em = lgbm_feats_indep(np.ascontiguousarray(ctx, dtype=np.float32), fim)
    gb = np.empty(H, dtype=np.float32)
    for j, mdl in enumerate(modelos_lgbm):
        hh = ((em - (H - 1 - j) * 5) % 1440 // 60).astype(np.float32)
        sh = np.sin(2 * np.pi * hh / 24).astype(np.float32)
        ch = np.cos(2 * np.pi * hh / 24).astype(np.float32)
        gb[j] = np.float32(saz[j] + mdl.predict(np.column_stack([F, sh, ch]))[0])
    return (w[0] * saz + w[1] * pn + w[2] * gb + w[3] * dr).astype(np.float32), saz, pn, gb, dr


def carrega_pkl_direto(var, compactado):
    base = ROOT / "univariavel" / "resultados" / ("06-ensemble-ph" if var == "ph" else "07-ensemble-od") / "modelos"
    p = base / ("lgbm_steps.pkl.gz" if compactado else "lgbm_steps.pkl")
    opener = gzip.open if compactado else open
    with opener(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ itens a-f
def item_a():
    api.carrega("ph")
    api.carrega("od")
    check(set(api.MODELOS) >= {"ph", "od"}, "(a) MODELOS sem 'ph'/'od' apos carrega()")
    for v in ("ph", "od"):
        check(set(api.MODELOS[v]) >= {"pesos", "lstnet", "dlres", "lgbm"},
              f"(a) MODELOS['{v}'] incompleto: {sorted(api.MODELOS[v])}")
    check(api.MODELOS["ph"]["lgbm"] is None, "(a) pH deveria pular o LGBM (peso 0)")
    check(len(api.MODELOS["od"]["lgbm"]) == 288,
          f"(a) OD deveria ter 288 modelos LGBM, tem {len(api.MODELOS['od']['lgbm'])}")


def item_b():
    pesos_json, states = {}, {}
    for var in ("ph", "od"):
        base = ROOT / "univariavel" / "resultados" / ("06-ensemble-ph" if var == "ph" else "07-ensemble-od")
        pesos_json[var] = json.load(open(base / "modelos/ensemble.json"))["pesos"]
        states[var] = {"lstnet": _sd(var, "lstnet"), "dlres": _sd(var, "dlres")}
    lgbm_od = carrega_pkl_direto("od", compactado=True)  # via pickle direto, sem app.MODELOS
    check(len(lgbm_od) == 288, f"(b) lgbm_steps.pkl.gz de OD tem {len(lgbm_od)} modelos, esperado 288")
    for var in ("ph", "od"):
        for k, (ctx, fim) in enumerate(janelas_limpas(var)):
            glb = lgbm_od if (var == "od") else None
            with torch.no_grad():
                out_app = np.asarray(api.inferencia(var, ctx, fim), dtype=np.float64)
            full, saz, pn, gb, dr = ensemble_indep(var, ctx, fim, states[var], pesos_json[var], glb)
            d = float(np.max(np.abs(out_app - full.astype(np.float64))))
            MAXDIV[f"(b) {var} janela{k + 1} fim={fim}"] = d
            check(out_app.shape == (288,), f"(b) {var} janela{k + 1}: shape {out_app.shape} != (288,)")
            check(np.isfinite(out_app).all(), f"(b) {var} janela{k + 1}: app retornou NaN/inf")
            check(d <= TOL_PARIDADE,
                  f"(b) {var} janela{k + 1} (fim={fim}): divergencia {d:.3e} > tol {TOL_PARIDADE:.0e}")
            if var == "od" and k == 0:
                # isola o termo LGBM: gb_impl = (app - resto)/w2  (tol maior: amplificacao ~1/w2)
                w = pesos_json["od"]
                resto = w["sazonal"] * saz + w["lstnet"] * pn + w["dlres"] * dr
                gb_impl = (out_app - resto) / w["lgbm"]
                dg = float(np.max(np.abs(gb_impl - gb.astype(np.float64))))
                MAXDIV["(b) od termo-LGBM isolado (288 horizontes)"] = dg
                check(dg <= TOL_LGBM_COMP,
                      f"(b) OD termo LGBM isolado: divergencia {dg:.3e} > tol {TOL_LGBM_COMP:.0e}")


def item_c():
    esperados = {"ph": {"sazonal": 0.2701, "lstnet": 0.7247, "lgbm": 0.0, "dlres": 0.0045},
                 "od": {"sazonal": 0.0543, "lstnet": 0.6223, "lgbm": 0.0255, "dlres": 0.303}}
    for var in ("ph", "od"):
        base = ROOT / "univariavel" / "resultados" / ("06-ensemble-ph" if var == "ph" else "07-ensemble-od")
        ens = json.load(open(base / "modelos/ensemble.json"))["pesos"]
        w_app = list(api.MODELOS[var]["pesos"])
        check(w_app == [ens["sazonal"], ens["lstnet"], ens["lgbm"], ens["dlres"]],
              f"(c) {var}: app.MODELOS pesos {w_app} != ensemble.json na ordem "
              f"[sazonal, lstnet, lgbm, dlres] = {[ens[k] for k in ('sazonal', 'lstnet', 'lgbm', 'dlres')]}")
        for k, v in esperados[var].items():
            check(abs(ens[k] - v) < 1e-9,
                  f"(c) {var}: ensemble.json['{k}']={ens[k]} != README 06/07 ({v})")


def item_d():
    a = carrega_pkl_direto("od", compactado=True)
    b = carrega_pkl_direto("od", compactado=False)
    check(len(a) == len(b) == 288, f"(d) tamanhos gz={len(a)} pkl={len(b)}, esperado 288/288")
    ctx, fim = janelas_limpas("od")[0][:2]
    F, em = lgbm_feats_indep(ctx, fim)
    pa, pb = np.empty(H, np.float32), np.empty(H, np.float32)
    for j, (ma, mb) in enumerate(zip(a, b)):
        hh = ((em - (H - 1 - j) * 5) % 1440 // 60).astype(np.float32)
        sh = np.sin(2 * np.pi * hh / 24).astype(np.float32)
        ch = np.cos(2 * np.pi * hh / 24).astype(np.float32)
        X = np.column_stack([F, sh, ch])
        pa[j], pb[j] = ma.predict(X)[0], mb.predict(X)[0]
    d = float(np.max(np.abs(pa.astype(np.float64) - pb.astype(np.float64))))
    MAXDIV["(d) gz x pkl (288 horizontes)"] = d
    check(d <= 1e-9, f"(d) lgbm_steps.pkl.gz x .pkl divergem em {d:.3e} (esperado ~0)")


def csv_sintetico(var, dias=10, inicio="09/09/2026 00:00"):
    t0 = pd.Timestamp("2026-09-09 00:00")
    idx = pd.date_range(t0, periods=dias * 288, freq="5min")
    mins = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    base = 6.0 if var == "ph" else 7.0
    y = base + 0.3 * np.sin(2 * np.pi * mins / 1440.0) + 0.05 * np.sin(2 * np.pi * mins / 288.0)
    linhas = ["Entidade responsavel CETESB - sintetico", f"Data hora;{COL[var]}"]
    linhas += [f"{t.strftime('%d/%m/%Y %H:%M')};{v:.2f}".replace(".", ",")
               for t, v in zip(idx, y)]
    return ("\n".join(linhas) + "\n").encode("windows-1252"), idx


def checa_resposta_prever(body, var, horizonte_horas, fim_entrada):
    out = api.PrevisaoOut.model_validate(body)  # valida o schema
    check(len(out.valores) == horizonte_horas * 12,
          f"(e) {var}/{horizonte_horas}h: {len(out.valores)} valores, esperado {horizonte_horas * 12}")
    ts = pd.DatetimeIndex([pd.Timestamp(p.ds) for p in out.valores])
    check(bool((ts[1:] - ts[:-1] == pd.Timedelta(minutes=5)).all()),
          f"(e) {var}/{horizonte_horas}h: timestamps nao contiguos de 5min")
    check(ts[0] == fim_entrada + pd.Timedelta(minutes=5),
          f"(e) {var}/{horizonte_horas}h: inicio {ts[0]} != fim_entrada+5min ({fim_entrada})")
    check(out.modelo == api.REGRAS[var]["modelo"] and
          out.mae_referencia_24h == api.REGRAS[var]["mae_benchmark_24h"],
          f"(e) {var}: modelo/mae de referencia inconsistentes com REGRAS")
    return out


def item_e():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        TestClient = None
    if TestClient is None:  # fallback: exercita prepara()+inferencia()+schema direto
        for var in ("ph", "od"):
            raw, idx = csv_sintetico(var)
            ctx, grade, _ = api.prepara(api.ler_csv(raw))
            full = api.inferencia(var, ctx, grade.max())
            check(full.shape == (288,), f"(e-fallback) {var}: shape {full.shape}")
            n, fidx = 288, pd.date_range(grade.max() + pd.Timedelta(minutes=5), periods=288, freq="5min")
            checa_resposta_prever({"variavel": var, "horizonte_horas": 24,
                                   "modelo": api.REGRAS[var]["modelo"],
                                   "mae_referencia_24h": api.REGRAS[var]["mae_benchmark_24h"],
                                   "inicio_previsto": str(fidx[0]), "fim_previsto": str(fidx[-1]),
                                   "cobertura_entrada": {"slots": 1, "faltantes_originais": 0,
                                                         "pct_interpolado": 0.0},
                                   "valores": [{"ds": str(t), "y": round(float(v), 4)}
                                               for t, v in zip(fidx, full[:n])]},
                                  var, 24, idx.max())
        print("(e) PASS via fallback direto (TestClient indisponivel no .venv)")
        return
    with TestClient(api.app) as client:
        for var in ("ph", "od"):
            raw, idx = csv_sintetico(var)
            r = client.post("/prever", files={"arquivo": ("sint.csv", raw, "text/csv")},
                            params={"variavel": var, "horizonte_horas": 24})
            check(r.status_code == 200, f"(e) {var}/24h: status {r.status_code}: {r.text[:300]}")
            checa_resposta_prever(r.json(), var, 24, idx.max())
        raw, idx = csv_sintetico("ph")
        r = client.post("/prever", files={"arquivo": ("sint.csv", raw, "text/csv")},
                        params={"variavel": "ph", "horizonte_horas": 1})
        check(r.status_code == 200, f"(e) ph/1h: status {r.status_code}: {r.text[:300]}")
        checa_resposta_prever(r.json(), "ph", 1, idx.max())


def item_f():
    try:
        api.ler_csv(b"")
        raise SystemExit("(f) FALHA INTERNA: ler_csv(b'') nao levantou")
    except ValueError:
        pass
    raw, _ = csv_sintetico("ph")
    linhas = raw.decode("windows-1252").splitlines()
    curta = ("\n".join(linhas[:2 + 100]) + "\n").encode("windows-1252")
    try:
        api.prepara(api.ler_csv(curta))
        raise SystemExit("(f) FALHA INTERNA: serie de 100 linhas nao levantou")
    except ValueError:
        pass
    idx = pd.date_range("2026-01-01", periods=10 * 288, freq="5min")
    df = pd.DataFrame({"ds": idx, "y": 6.0 + 0.1 * np.sin(np.arange(len(idx)) / 50.0)})
    df.loc[df.index[-30:], "y"] = np.nan  # gap de 2,5h na cauda (> limite de 24)
    try:
        api.prepara(df)
        raise SystemExit("(f) FALHA INTERNA: gap >2h na cauda nao levantou")
    except ValueError:
        pass


def main():
    itens = [("a", item_a), ("b", item_b), ("c", item_c),
             ("d", item_d), ("e", item_e), ("f", item_f)]
    for nome, fn in itens:
        try:
            fn()
            print(f"({nome}) PASS")
        except (AssertionError, SystemExit) as e:
            print(f"({nome}) FAIL: {e}")
            FALHAS.append(nome)
        except Exception as e:  # erro inesperado tb reprova o item
            print(f"({nome}) FAIL (erro inesperado {type(e).__name__}): {e}")
            FALHAS.append(nome)
    print("--- divergencias maximas |app - independente| ---")
    for k, v in MAXDIV.items():
        print(f"  {k}: {v:.3e}")
    if FALHAS:
        print(f"RESULTADO: FAIL itens {FALHAS}")
        return 1
    print("RESULTADO: PASS (a-f)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
