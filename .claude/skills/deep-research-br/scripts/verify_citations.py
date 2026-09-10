#!/usr/bin/env python3
"""Gate anti-alucinação: verifica cada registro com DOI/arXiv contra Crossref/arXiv.

Classifica: verified | not_found | mismatched (título diverge muito) | local (corpus local, pula verificação remota).
Retração: checa Crossref `update-to` com intuito de retração quando disponível (melhor esforço).

Uso: verify_citations.py --slug X [--estrito]
Exit 0 se tudo verified/local; exit 2 se houver not_found/mismatched (gate fecha).
"""
import argparse
import difflib
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "deep-research-br/0.1.0 (mailto: pesquisador@exemplo.br)"}


def crossref_por_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace")).get("message", {})


def arxiv_por_id(arxiv_id):
    """Verifica existência no arXiv via API Atom. Retorna título ou None."""
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": f"id:{arxiv_id}", "max_results": 1})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read().decode("utf-8", "replace")
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    return (entry.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")


def arxiv_id_de_registro(r, doi):
    if r.get("arxiv_id"):
        return r["arxiv_id"]
    if doi and doi.startswith("10.48550/arxiv."):
        return doi.split("10.48550/arxiv.", 1)[1]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--estrito", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    d = root / "reviews" / args.slug
    corp_p = d / "corpus.json"
    if not corp_p.exists():
        print(json.dumps({"status": "error", "message": "corpus.json não encontrado", "hint": "rode dedupe_rank.py"}, ensure_ascii=False))
        return 1
    corpus = json.loads(corp_p.read_text(encoding="utf-8"))
    relatorio = []
    for r in corpus.get("registros", []):
        if r.get("arquivo_local") and not r.get("doi") and not r.get("arxiv_id"):
            r["verificacao"] = "local"
            relatorio.append({"titulo": r.get("titulo"), "status": "local"})
            continue
        doi = r.get("doi")
        if not doi:
            # sem DOI mas com arXiv ID → verifica no arXiv
            aid = r.get("arxiv_id")
            if aid:
                try:
                    t = arxiv_por_id(aid)
                    r["verificacao"] = "verified" if t else "not_found"
                    relatorio.append({"titulo": r.get("titulo"), "arxiv": aid, "status": r["verificacao"]})
                except Exception as e:
                    r["verificacao"] = "erro_rede"
                    relatorio.append({"titulo": r.get("titulo"), "arxiv": aid, "status": "erro_rede"})
                time.sleep(1)
                continue
            r["verificacao"] = "sem-id"
            relatorio.append({"titulo": r.get("titulo"), "status": "sem-id"})
            continue
        # DOIs arXiv (10.48550) vivem no DataCite/arXiv, não no Crossref → valida via arXiv
        aid = arxiv_id_de_registro(r, doi)
        if aid:
            try:
                t_oficial = arxiv_por_id(aid)
                if not t_oficial:
                    r["verificacao"] = "not_found"
                    relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": "not_found"})
                else:
                    ratio = difflib.SequenceMatcher(None, (r.get("titulo") or "").lower(), t_oficial.lower()).ratio()
                    r["verificacao"] = "mismatched" if (ratio < 0.4 and args.estrito) else "verified"
                    r["titulo_oficial"] = t_oficial
                    relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": r["verificacao"], "similaridade": round(ratio, 2)})
            except Exception:
                r["verificacao"] = "erro_rede"
                relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": "erro_rede"})
            time.sleep(1)
            continue
        try:
            m = crossref_por_doi(doi)
            titulo_oficial = ((m.get("title") or [""])[0] or "").strip()
            ratio = difflib.SequenceMatcher(None, (r.get("titulo") or "").lower(), titulo_oficial.lower()).ratio()
            updates = m.get("update-to") or []
            retratado = any("retract" in str(u).lower() for u in updates)
            if retratado:
                r["verificacao"] = "retracted"
            elif ratio < 0.4 and args.estrito:
                r["verificacao"] = "mismatched"
            else:
                r["verificacao"] = "verified"
            r["titulo_oficial"] = titulo_oficial
            relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": r["verificacao"], "similaridade": round(ratio, 2)})
            time.sleep(1)
        except Exception as e:
            if "404" in str(e):
                r["verificacao"] = "not_found"
                relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": "not_found"})
            else:
                r.setdefault("verificacao", "erro_rede")
                relatorio.append({"titulo": r.get("titulo"), "doi": doi, "status": "erro_rede", "detalhe": str(e)[:120]})
            time.sleep(1)
    corp_p.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "verification.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    ruins = [x for x in relatorio if x["status"] in ("not_found", "mismatched", "retracted")]
    print(json.dumps({"status": "ok" if not ruins else "gate-fechado",
                      "data": {"slug": args.slug, "verificados": sum(1 for x in relatorio if x["status"] in ("verified", "local")),
                               "problemas": ruins}}, ensure_ascii=False))
    return 2 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
