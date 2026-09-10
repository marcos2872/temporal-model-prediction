# Instalação PT-BR — deep-research-br

Requisito: Python 3.10+, sem dependências (stdlib). Rede para OpenAlex/arXiv/EPMC/Crossref.

```bash
# local ao projeto (recomendado)
bash skills/deep-research-br/install.sh
# global (OpenCode + Claude)
bash skills/deep-research-br/install.sh --global
# teste offline (usa artigos/)
python3 skills/deep-research-br/evals/smoke.py
```

## Uso mínimo

```bash
python3 skills/deep-research-br/scripts/state.py init --slug meu-tema --questao "..." --arquetipo systematic_review --template rigorous
python3 skills/deep-research-br/scripts/read_local.py --slug meu-tema --dir artigos
python3 skills/deep-research-br/scripts/search_all.py --slug meu-tema --query "time series forecasting transformer" --max 30 --desde 2019
python3 skills/deep-research-br/scripts/dedupe_rank.py --slug meu-tema --top 25
# edite reviews/meu-tema/corpus.json marcando screening.status=include/exclude, depois:
python3 skills/deep-research-br/scripts/verify_citations.py --slug meu-tema
python3 skills/deep-research-br/scripts/prisma_counts.py --slug meu-tema
python3 skills/deep-research-br/scripts/build_report.py --slug meu-tema --arquetipo systematic_review --titulo "Minha revisão"
```

Dica OpenAlex: adicione `--mailto seu@email` para polite pool (mais rápido).
