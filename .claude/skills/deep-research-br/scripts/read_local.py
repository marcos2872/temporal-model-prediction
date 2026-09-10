#!/usr/bin/env python3
"""Indexa corpus local (PDFs) — sem dependência pesada.

Lê --dir (padrão: artigos/), extrai metadados do nome do arquivo e tenta
extrair as 2 primeiras páginas de texto via pdftotext (se instalado) ou
pypdf (se instalado). Sem nenhum dos dois, registra só metadados do nome.

Grava reviews/{slug}/local_corpus.json (lista) para o dedupe_rank.py absorver.

Uso: read_local.py --slug X --dir artigos
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys


def meta_do_nome(nome):
    # ex: 04-Lim2019-Temporal-Fusion-Transformer.pdf
    m = re.match(r"(\d+)-([A-Za-z]+)(\d{4})-(.+)\.pdf$", nome)
    if m:
        return {"ordem": m.group(1), "primeiro_autor": m.group(2), "ano": int(m.group(3)),
                "titulo_slug": m.group(4).replace("-", " ")}
    return {}


def extrai_texto(pdf: pathlib.Path, max_paginas=2):
    if shutil.which("pdftotext"):
        try:
            out = str(pdf) + ".tmp.txt"
            subprocess.run(["pdftotext", "-l", str(max_paginas), "-layout", str(pdf), out],
                           check=True, timeout=30, capture_output=True)
            txt = pathlib.Path(out).read_text(encoding="utf-8", errors="replace")[:6000]
            pathlib.Path(out).unlink(missing_ok=True)
            return txt
        except Exception:
            pass
    try:
        from pypdf import PdfReader  # opcional
        reader = PdfReader(str(pdf))
        txt = ""
        for p in reader.pages[:max_paginas]:
            txt += (p.extract_text() or "") + "\n"
        return txt[:6000]
    except Exception:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--dir", default="artigos")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    d = root / "reviews" / args.slug
    if not (d / "state.json").exists():
        print(json.dumps({"status": "error", "message": "slug não inicializado", "hint": "rode state.py init"}, ensure_ascii=False))
        return 1
    pasta = root / args.dir
    if not pasta.exists():
        print(json.dumps({"status": "error", "message": f"pasta não encontrada: {args.dir}"}, ensure_ascii=False))
        return 1
    recs = []
    for pdf in sorted(pasta.glob("*.pdf")):
        meta = meta_do_nome(pdf.name)
        recs.append({
            "titulo": meta.get("titulo_slug") or pdf.stem,
            "ano": meta.get("ano"),
            "autores": [meta.get("primeiro_autor")] if meta.get("primeiro_autor") else [],
            "arquivo_local": f"{args.dir}/{pdf.name}",
            "doi": None, "citacoes": 0,
            "resumo_local": extrai_texto(pdf)[:1500],
            "fulltext": "local",
        })
    (d / "local_corpus.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "data": {"slug": args.slug, "pdfs_indexados": len(recs),
                      "arquivo": "local_corpus.json"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
