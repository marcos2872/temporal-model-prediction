#!/usr/bin/env python3
"""Busca federada OpenAlex + arXiv + EuropePMC + Crossref — stdlib, sem key.

Exemplo:
  python3 search_all.py --slug meu-tema --query "time series forecasting transformer" --fontes openalex,arxiv,epmc --max 30 --desde 2019

Saída: envelope JSON + grava brutos em reviews/{slug}/raw_{fonte}.json e registra no search_log.md
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "deep-research-br/0.1.0 (mailto: pesquisador@exemplo.br)"}


def http_get_json(url: str, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def search_openalex(query, max_n, desde=None, mailto=None):
    params = {"search": query, "per-page": min(max_n, 50)}
    if desde:
        params["filter"] = f"from_publication_date:{desde}-01-01"
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = http_get_json(url)
    out = []
    for w in data.get("results", [])[:max_n]:
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        try:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            venue = src.get("display_name")
        except Exception:
            venue = None
        try:
            oa = w.get("open_access") or {}
            oa_url = oa.get("oa_url")
        except Exception:
            oa_url = None
        try:
            autores = [(a.get("author") or {}).get("display_name") for a in (w.get("authorships") or [])][:6]
        except Exception:
            autores = []
        out.append({
            "titulo": w.get("display_name"), "ano": w.get("publication_year"),
            "doi": doi or None, "openalex_id": w.get("id"),
            "citacoes": w.get("cited_by_count", 0),
            "venue": venue,
            "autores": autores,
            "tipo": w.get("type"), "oa_url": oa_url,
            "abstract_inverted": bool(w.get("abstract_inverted_index")),
        })
    return out


def search_arxiv(query, max_n):
    q = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": min(max_n, 30), "sortBy": "relevance"})
    url = "http://export.arxiv.org/api/query?" + q
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read().decode("utf-8", "replace")
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(xml)
        for e in root.findall("a:entry", ns)[:max_n]:
            title = (e.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
            arxiv_id = (e.findtext("a:id", default="", namespaces=ns) or "").rsplit("/abs/", 1)[-1]
            year = (e.findtext("a:published", default="", namespaces=ns) or "")[:4]
            out.append({"titulo": title, "ano": int(year) if year.isdigit() else None, "arxiv_id": arxiv_id, "doi": None, "citacoes": 0})
    except Exception:
        pass
    return out


def search_epmc(query, max_n):
    params = urllib.parse.urlencode({"query": query, "format": "json", "pageSize": min(max_n, 50), "resultType": "core"})
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + params
    try:
        data = http_get_json(url)
    except Exception:
        return []
    out = []
    for d in (data.get("resultList", {}).get("result", []) or [])[:max_n]:
        out.append({"titulo": d.get("title"), "ano": int(d["pubYear"]) if str(d.get("pubYear", "")).isdigit() else None,
                    "doi": d.get("doi") or None, "pmid": d.get("pmid"), "pmcid": d.get("pmcid"), "citacoes": int(d.get("citedByCount") or 0)})
    return out


def search_crossref(query, max_n):
    params = urllib.parse.urlencode({"query": query, "rows": min(max_n, 20), "select": "DOI,title,author,published,URL"})
    url = "https://api.crossref.org/works?" + params
    try:
        data = http_get_json(url)
    except Exception:
        return []
    out = []
    for it in (data.get("message", {}).get("items", []) or [])[:max_n]:
        title = (it.get("title") or [""])[0]
        ano = None
        try:
            ano = (it.get("published", {}).get("date-parts") or [[None]])[0][0]
        except Exception:
            pass
        out.append({"titulo": title, "ano": ano, "doi": it.get("DOI")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--fontes", default="openalex,arxiv,epmc")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--desde", type=int, default=None)
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    slug = args.slug.strip().lower().replace(" ", "-")[:60]
    d = root / "reviews" / slug
    if not (d / "state.json").exists():
        print(json.dumps({"status": "error", "message": f"slug não inicializado: {slug}", "hint": "rode state.py init primeiro"}, ensure_ascii=False))
        return 1
    fontes = [f.strip() for f in args.fontes.split(",") if f.strip()]
    per = max(5, args.max // max(1, len(fontes)))
    bruto, warnings = {}, []
    for f in fontes:
        try:
            if f == "openalex":
                bruto[f] = search_openalex(args.query, per, args.desde, args.mailto)
            elif f == "arxiv":
                bruto[f] = search_arxiv(args.query, per)
            elif f == "epmc":
                bruto[f] = search_epmc(args.query, per)
            elif f == "crossref":
                bruto[f] = search_crossref(args.query, per)
            else:
                warnings.append(f"fonte desconhecida ignorada: {f}")
                continue
            time.sleep(1)  # etiqueta 1 req/s
        except Exception as e:
            warnings.append(f"{f} falhou: {e}")
            bruto[f] = []
    for f, recs in bruto.items():
        (d / f"raw_{f}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(d / "search_log.md", "a", encoding="utf-8") as lf:
            lf.write(f"| {now} | {f} | `{args.query}` | {len(recs)} |\n")
    total = sum(len(v) for v in bruto.values())
    print(json.dumps({"status": "ok", "data": {"slug": slug, "fontes": fontes, "total_bruto": total,
                      "por_fonte": {k: len(v) for k, v in bruto.items()}}, "warnings": warnings}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
