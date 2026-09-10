#!/usr/bin/env python3
"""Smoke test offline da skill unificada (sem rede): init → read_local → dedupe → prisma → build.

Uso: python3 evals/smoke.py [--root .]
Exit 0 = ok. Valida o piloto temporal-model/artigos/.
"""
import argparse
import json
import pathlib
import subprocess
import sys


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        payload = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"raw": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    return r.returncode, payload, r.stderr[-500:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    sk = root / "skills" / "deep-research-br" / "scripts"
    slug = "smoke-temporal"
    py = sys.executable
    checks = []
    rc, out, _ = run([py, str(sk / "state.py"), "--root", str(root), "init", "--slug", slug,
                      "--questao", "Smoke: TSF com transformers?", "--arquetipo", "scoping_review", "--template", "quick"])
    checks.append(("init", rc == 0))
    rc, out, _ = run([py, str(sk / "read_local.py"), "--root", str(root), "--slug", slug, "--dir", "artigos"])
    n = out.get("data", {}).get("pdfs_indexados", 0) if isinstance(out, dict) else 0
    checks.append((f"read_local ({n} pdfs)", rc == 0 and n >= 5))
    rc, out, _ = run([py, str(sk / "dedupe_rank.py"), "--root", str(root), "--slug", slug, "--top", "5"])
    checks.append(("dedupe_rank", rc == 0))
    rc, out, _ = run([py, str(sk / "prisma_counts.py"), "--root", str(root), "--slug", slug])
    checks.append(("prisma", rc == 0))
    rc, out, _ = run([py, str(sk / "build_report.py"), "--root", str(root), "--slug", slug,
                      "--arquetipo", "scoping_review", "--titulo", "Smoke temporal"])
    md_ok = (root / "reports" / f"{slug}.md").exists()
    checks.append(("build_report", rc == 0 and md_ok))
    print("SMOKE deep-research-br")
    ok = True
    for nome, passed in checks:
        print(f"  [{'OK' if passed else 'FALHOU'}] {nome}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
