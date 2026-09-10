#!/usr/bin/env python3
"""Renderiza reports/{slug}.md + .bib + .ris a partir do corpus + arquétipo.

Uso: build_report.py --slug X --arquetipo systematic_review --titulo "Título"
O agente LLM preenche a síntese chamando este script com --corpo-markdown arquivo,
ou deixa o esqueleto para completar. Toda afirmação deve citar [^id] (rank).
"""
import argparse
import json
import pathlib
import sys


def bib_entry(r, i):
    key = f"ref{i:02d}"
    autores = " and ".join(r.get("autores") or ["Anônimo"])
    ano = r.get("ano") or "s.d."
    titulo = (r.get("titulo") or "Sem título").replace("{", "").replace("}", "")
    venue = r.get("venue") or "s.l."
    doi = r.get("doi")
    url = f"https://doi.org/{doi}" if doi else (r.get("openalex_id") or r.get("arquivo_local") or "")
    return key, (f"@{('article' if doi else 'misc')}{{{key},\n  author = {{{autores}}},\n"
                 f"  title = {{{titulo}}},\n  year = {{{ano}}},\n  journal = {{{venue}}},\n"
                 f"  doi = {{{doi or ''}}},\n  url = {{{url}}}\n}}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--arquetipo", default="literature_review")
    ap.add_argument("--titulo", default="Revisão bibliográfica")
    ap.add_argument("--corpo-markdown", default=None)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    d = root / "reviews" / args.slug
    corpus = json.loads((d / "corpus.json").read_text(encoding="utf-8")).get("registros", [])
    tpl_p = root / "skills" / "deep-research-br" / "references" / "archetypes" / f"{args.arquetipo}.md"
    tpl = tpl_p.read_text(encoding="utf-8") if tpl_p.exists() else "# {titulo}\n\n{corpo}\n"
    corpo = ""
    if args.corpo_markdown:
        corpo = pathlib.Path(args.corpo_markdown).read_text(encoding="utf-8")
    else:
        linhas = ["## Registros priorizados (complete a síntese temática; não liste paper-a-paper)\n"]
        for r in corpus[:15]:
            linhas.append(f"- **{r.get('titulo')}** ({r.get('ano')}) — score {r.get('score')} "
                           f"[^{r.get('rank')}] — {', '.join(r.get('found_via', []))}")
        corpo = "\n".join(linhas)
    md = tpl.replace("{titulo}", args.titulo).replace("{slug}", args.slug).replace("{corpo}", corpo)
    bibs, ris = [], []
    for i, r in enumerate(corpus, 1):
        key, b = bib_entry(r, i)
        bibs.append(b)
        ris.append(f"TY  - JOUR\nTI  - {r.get('titulo')}\nAU  - {'; '.join(r.get('autores') or [])}\n"
                   f"PY  - {r.get('ano') or ''}\nDO  - {r.get('doi') or ''}\nER  - \n")
    md += "\n\n## Referências\n\n" + "\n".join(f"[^{r.get('rank')}]: {r.get('titulo')} ({r.get('ano')}) — DOI: {r.get('doi') or r.get('arquivo_local') or 'n/d'}" for r in corpus[:30])
    md += ("\n\n## Apêndice — bases não consultadas\n\nGoogle Scholar, Scopus e Web of Science não possuem API pública e **não foram consultados** "
           "neste pipeline. PDFs paywall sem cópia OA legal foram avaliados só por abstract.\n")
    rep = root / "reports"
    rep.mkdir(exist_ok=True)
    (rep / f"{args.slug}.md").write_text(md, encoding="utf-8")
    (rep / f"{args.slug}.bib").write_text("\n\n".join(bibs), encoding="utf-8")
    (rep / f"{args.slug}.ris").write_text("\n".join(ris), encoding="utf-8")
    print(json.dumps({"status": "ok", "data": {"md": f"reports/{args.slug}.md", "bib": f"reports/{args.slug}.bib",
                      "n_refs": len(corpus)}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
