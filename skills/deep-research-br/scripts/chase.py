#!/usr/bin/env python3
"""Citation chasing backward+forward via OpenAlex — stdlib.

Uso:
  chase.py --slug X --ids 10.xxxx/yyyy,W123456 --direcao both --max 20

--ids aceita DOIs e/ou OpenAlex IDs (W...). Backward = referenced_works do seed;
forward = works que citam o seed (filter=cites:W...). Grava raw_snowball.json.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "deep-research-br/0.1.0 (mailto: pesquisador@exemplo.br)"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def resolve_openalex(seed):
    seed = seed.strip()
    if seed.startswith("W"):
        return f"https://api.openalex.org/works/{seed}"
    if seed.startswith("10."):
        return "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + seed, safe="")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--direcao", default="both", choices=["backward", "forward", "both"])
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root)
    d = root / "reviews" / args.slug
    if not (d / "state.json").exists():
        print(json.dumps({"status": "error", "message": "slug não inicializado"}, ensure_ascii=False))
        return 1
    seeds = [s.strip() for s in args.ids.split(",") if s.strip()]
    achados, warnings = [], []
    for seed in seeds:
        url = resolve_openalex(seed)
        if not url:
            warnings.append(f"seed ignorado (use DOI ou W-ID): {seed}")
            continue
        try:
            w = get_json(url)
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            if args.direcao in ("backward", "both"):
                for ref_url in (w.get("referenced_works") or [])[:args.max]:
                    try:
                        rw = get_json(ref_url)
                        achados.append({"titulo": rw.get("display_name"), "ano": rw.get("publication_year"),
                                        "doi": (rw.get("doi") or "").replace("https://doi.org/", "") or None,
                                        "openalex_id": rw.get("id"), "citacoes": rw.get("cited_by_count", 0),
                                        "snowball": f"backward:{seed}"})
                    except Exception as e:
                        warnings.append(f"ref falhou: {e}")
                        continue
                    time.sleep(0.5)
            if args.direcao in ("forward", "both") and wid.startswith("W"):
                fwd = get_json(f"https://api.openalex.org/works?filter=cites:{wid}&per-page={min(args.max, 50)}")
                for it in fwd.get("results", [])[:args.max]:
                    achados.append({"titulo": it.get("display_name"), "ano": it.get("publication_year"),
                                    "doi": (it.get("doi") or "").replace("https://doi.org/", "") or None,
                                    "openalex_id": it.get("id"), "citacoes": it.get("cited_by_count", 0),
                                    "snowball": f"forward:{seed}"})
            time.sleep(1)
        except Exception as e:
            warnings.append(f"seed {seed} falhou: {e}")
    (d / "raw_snowball.json").write_text(json.dumps(achados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "data": {"slug": args.slug, "achados": len(achados),
                      "arquivo": "raw_snowball.json"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
