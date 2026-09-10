#!/usr/bin/env python3
"""Estado unificado reviews/{slug}/ — camada pura de dados, sem LLM.

Comandos:
  init --slug X --questao Q --arquetipo A --template T [--questao...]
  status --slug X
  set-phase --slug X --fase N
  log-search --slug X --fonte F --query Q --retornados N

Envelope JSON no stdout: {"status": "ok", ...} / {"status":"error",...}
"""
import argparse
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
# quando instalada em .opencode/skills ou .claude/skills, raiz do projeto = cwd
# permite --root override para testes
ARQUETIPOS = {"literature_review", "systematic_review", "scoping_review", "comparative_analysis", "grant_background"}
TEMPLATES = {"quick", "rigorous", "comprehensive"}


def out_ok(data=None, warnings=None):
    print(json.dumps({"status": "ok", "data": data or {}, "warnings": warnings or []}, ensure_ascii=False))
    return 0


def out_err(message, hint=""):
    print(json.dumps({"status": "error", "message": message, "hint": hint}, ensure_ascii=False))
    return 1


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "revisao"


def reviews_dir(root: pathlib.Path) -> pathlib.Path:
    d = root / "reviews"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_init(args) -> int:
    root = pathlib.Path(args.root)
    slug = slugify(args.slug)
    if args.arquetipo not in ARQUETIPOS:
        return out_err(f"arquetipo inválido: {args.arquetipo}", f"use um de: {sorted(ARQUETIPOS)}")
    if args.template not in TEMPLATES:
        return out_err(f"template inválido: {args.template}", f"use um de: {sorted(TEMPLATES)}")
    d = reviews_dir(root) / slug
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = {
        "slug": slug,
        "questao": args.questao,
        "arquetipo": args.arquetipo,
        "template": args.template,
        "fase": 1,
        "criado_em": now,
        "atualizado_em": now,
        "framework": args.framework,
    }
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if not (d / "corpus.json").exists():
        (d / "corpus.json").write_text(json.dumps({"registros": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not (d / "passport.json").exists():
        (d / "passport.json").write_text(json.dumps({
            "slug": slug, "fontes": [], "claims": [],
            "experimentos_externos": [], "declaracao": "sem_experimentos_declarados",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "protocol.md").write_text(
        f"# Protocolo — {slug}\n\n- **Questão:** {args.questao}\n- **Framework:** {args.framework}\n"
        f"- **Arquétipo:** {args.arquetipo}\n- **Template:** {args.template}\n- **Criado:** {now}\n\n"
        "## Critérios (editar antes do Gate 1)\n\n- Inclusão: (preencher)\n- Exclusão: (preencher)\n"
        "- Recorte temporal: (preencher)\n- Idiomas: (preencher)\n\n## Bases consultadas\n\n(nenhuma ainda — log em search_log.md)\n",
        encoding="utf-8")
    if not (d / "search_log.md").exists():
        (d / "search_log.md").write_text(f"# Search log — {slug}\n\n| Data | Fonte | Query | Retornados |\n|---|---|---|---|\n", encoding="utf-8")
    return out_ok({"slug": slug, "dir": str(d), "fase": 1})


def cmd_status(args) -> int:
    root = pathlib.Path(args.root)
    d = root / "reviews" / slugify(args.slug)
    if not (d / "state.json").exists():
        return out_err(f"revisão não encontrada: {args.slug}", "rode state.py init primeiro")
    state = json.loads((d / "state.json").read_text(encoding="utf-8"))
    corpus = json.loads((d / "corpus.json").read_text(encoding="utf-8")) if (d / "corpus.json").exists() else {"registros": []}
    state["n_registros"] = len(corpus.get("registros", []))
    return out_ok(state)


def cmd_set_phase(args) -> int:
    root = pathlib.Path(args.root)
    d = root / "reviews" / slugify(args.slug)
    p = d / "state.json"
    if not p.exists():
        return out_err("state.json não encontrado", "rode init primeiro")
    state = json.loads(p.read_text(encoding="utf-8"))
    state["fase"] = int(args.fase)
    state["atualizado_em"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_ok({"slug": state["slug"], "fase": state["fase"]})


def cmd_log_search(args) -> int:
    root = pathlib.Path(args.root)
    d = root / "reviews" / slugify(args.slug)
    if not d.exists():
        return out_err("slug não encontrado", "rode init primeiro")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"| {now} | {args.fonte} | `{args.query}` | {args.retornados} |\n"
    with open(d / "search_log.md", "a", encoding="utf-8") as f:
        f.write(line)
    return out_ok({"logged": True})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--slug", required=True)
    p.add_argument("--questao", required=True)
    p.add_argument("--arquetipo", default="literature_review")
    p.add_argument("--template", default="rigorous")
    p.add_argument("--framework", default="PICO")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("status")
    p.add_argument("--slug", required=True)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("set-phase")
    p.add_argument("--slug", required=True)
    p.add_argument("--fase", required=True)
    p.set_defaults(fn=cmd_set_phase)
    p = sub.add_parser("log-search")
    p.add_argument("--slug", required=True)
    p.add_argument("--fonte", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--retornados", default="0")
    p.set_defaults(fn=cmd_log_search)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
