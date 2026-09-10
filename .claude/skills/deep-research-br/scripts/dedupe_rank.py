#!/usr/bin/env python3
"""Dedup + rank + saturação → corpus.json. Preserva screening humano em re-runs.

Lê reviews/{slug}/raw_*.json + local_*.json, deduplica por DOI > arxiv_id > título
normalizado, calcula score = 0.5*cit_norm + 0.3*recencia + 0.2*cobertura_fontes,
define tier deep/skim e checa saturação (novidade < limiar = saturado).

Uso: dedupe_rank.py --slug X --top 25 --limiar-saturacao 0.2
"""
import argparse
import json
import math
import pathlib
import re
import sys


def norm_title(t):
    return re.sub(r"\s+", " ", (t or "").lower().strip().strip("."))


def chave(r):
    if r.get("doi"):
        return "doi:" + r["doi"].lower().strip()
    if r.get("arxiv_id"):
        return "arxiv:" + r["arxiv_id"].lower().strip()
    t = norm_title(r.get("titulo"))
    if t:
        aut = (r.get("autores") or [None])[0] or ""
        return f"titulo:{t[:120]}|{str(aut).lower()[:40]}|{r.get('ano')}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--limiar-saturacao", type=float, default=0.2)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    d = root / "reviews" / args.slug
    if not (d / "state.json").exists():
        print(json.dumps({"status": "error", "message": "slug não inicializado", "hint": "rode state.py init"}, ensure_ascii=False))
        return 1
    corpus_p = d / "corpus.json"
    anterior = {}
    if corpus_p.exists():
        try:
            for r in json.loads(corpus_p.read_text(encoding="utf-8")).get("registros", []):
                k = chave(r)
                if k:
                    anterior[k] = r
        except Exception:
            pass
    brutos = []
    for raw in sorted(d.glob("raw_*.json")) + sorted(d.glob("local_*.json")):
        try:
            recs = json.loads(raw.read_text(encoding="utf-8"))
            if isinstance(recs, list):
                for r in recs:
                    r["_fonte"] = raw.stem
                brutos.extend(recs)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"falha lendo {raw.name}: {e}"}, ensure_ascii=False))
            return 1
    merged = {}
    for r in brutos:
        k = chave(r)
        if not k:
            continue
        if k not in merged:
            merged[k] = {"titulo": r.get("titulo"), "ano": r.get("ano"), "doi": r.get("doi"),
                         "arxiv_id": r.get("arxiv_id"), "pmid": r.get("pmid"), "pmcid": r.get("pmcid"),
                         "openalex_id": r.get("openalex_id"), "venue": r.get("venue"),
                         "autores": r.get("autores") or [], "citacoes": int(r.get("citacoes") or 0),
                         "found_via": [r.get("_fonte", "?")], "screening": {"status": None, "stage": None, "reason": None}}
        else:
            m = merged[k]
            m["citacoes"] = max(m["citacoes"], int(r.get("citacoes") or 0))
            if r.get("_fonte") not in m["found_via"]:
                m["found_via"].append(r["_fonte"])
            for campo in ("doi", "arxiv_id", "pmid", "pmcid", "openalex_id", "venue"):
                if not m.get(campo) and r.get(campo):
                    m[campo] = r[campo]
    # preserva screening humano
    for k, old in anterior.items():
        if k in merged and old.get("screening", {}).get("status"):
            merged[k]["screening"] = old["screening"]
            if old.get("role"):
                merged[k]["role"] = old["role"]
    regs = list(merged.values())
    novos = sum(1 for k in merged if k not in anterior)
    taxa_novidade = (novos / max(1, len(merged)))
    # score
    max_cit = max([r["citacoes"] for r in regs] + [1])
    for r in regs:
        cit_n = math.log1p(r["citacoes"]) / math.log1p(max_cit)
        ano = r.get("ano") or 2000
        rec = max(0.0, min(1.0, (ano - 2000) / 26))
        cob = min(1.0, len(r["found_via"]) / 3)
        r["score"] = round(0.5 * cit_n + 0.3 * rec + 0.2 * cob, 4)
        r["score_partes"] = {"citacao": round(cit_n, 3), "recencia": round(rec, 3), "cobertura": round(cob, 3)}
    regs.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(regs):
        r["rank"] = i + 1
        r["tier"] = "deep" if i < args.top else "skim"
    corpus_p.write_text(json.dumps({"registros": regs}, ensure_ascii=False, indent=2), encoding="utf-8")
    saturado = taxa_novidade < args.limiar_saturacao and len(anterior) > 0
    print(json.dumps({"status": "ok", "data": {"slug": args.slug, "total": len(regs),
                      "novos": novos, "taxa_novidade": round(taxa_novidade, 3),
                      "saturado": saturado, "top_deep": args.top}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
