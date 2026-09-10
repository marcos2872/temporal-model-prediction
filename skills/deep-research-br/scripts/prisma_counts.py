#!/usr/bin/env python3
"""Contagens PRISMA 2020 a partir do corpus.json → reviews/{slug}/prisma.md + JSON no stdout."""
import argparse
import json
import pathlib
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    d = pathlib.Path(args.root) / "reviews" / args.slug
    p = d / "corpus.json"
    if not p.exists():
        print(json.dumps({"status": "error", "message": "corpus.json não encontrado"}, ensure_ascii=False))
        return 1
    regs = json.loads(p.read_text(encoding="utf-8")).get("registros", [])
    identificados = len(regs)
    sem_duplicadas = len(regs)  # dedup já aplicada no dedupe_rank
    triados = sum(1 for r in regs if (r.get("screening", {}) or {}).get("stage") in ("title", "abstract", "fulltext", None))
    incluidos = sum(1 for r in regs if (r.get("screening", {}) or {}).get("status") == "include")
    excluidos = sum(1 for r in regs if (r.get("screening", {}) or {}).get("status") == "exclude")
    pendentes = identificados - incluidos - excluidos
    md = (f"# PRISMA 2020 — {args.slug}\n\n- Identificados (após dedup): **{identificados}**\n"
          f"- Triados (título/resumo): **{triados}**\n- Excluídos: **{excluidos}**\n"
          f"- Incluídos: **{incluidos}**\n- Pendentes de screening: **{pendentes}**\n\n"
          "```\nIdentificação → Triagem → Elegibilidade → Incluídos\n"
          f"   {identificados}    →   {triados}   →   {incluidos + excluidos}   →   {incluidos}\n```\n")
    (d / "prisma.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": "ok", "data": {"identificados": identificados, "triados": triados,
                      "excluidos": excluidos, "incluidos": incluidos, "pendentes": pendentes}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
