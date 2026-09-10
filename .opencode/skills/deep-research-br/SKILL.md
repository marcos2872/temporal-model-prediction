---
name: deep-research-br
description: "Revisão bibliográfica rigorosa PT-BR: busca federada (OpenAlex/arXiv/EuropePMC/Crossref) + corpus local, triagem com saturação, chasing de citações, verificação anti-alucinação, síntese temática e relatório por arquétipo. Use quando o usuário pedir literature review, revisão sistemática, scoping, estado da arte, levantamento bibliográfico ou mapa de lacunas."
license: MIT
metadata:
  version: "0.1.0"
  language: pt-BR
  platforms: [opencode, claude-code, cursor, codex]
  uniao-de: [scholar-deep-research(fases/auditoria), co-researcher(toolchains/verificação), academic-research-skills(passport/gates)]
---

# Deep Research BR — skill unificada (1+2+4)

Pipeline de **10 fases** com rigor máximo e checkpoints humanos. Toda afirmação factual no relatório final precisa de âncora `[^id]` ligada a um registro em `corpus.json`. Sem âncora → gate reprova.

## Templates

| Template | Quando usar | O que pula |
|---|---|---|
| `quick` | Scan rápido, orientação inicial | Fases 6 (chasing) parcial, sem gate 2 |
| `rigorous` (padrão) | Revisão publicável, TCC/dissertação | Nada — pipeline completo |
| `comprehensive` | Deep multi-método, grant background | Adiciona análise comparativa + mapa de tensões expandido |

Modos: interativo (padrão, pede aprovação do plano), `--auto` (executa sem pausar, exceto gates de integridade), `--plan-only` (só gera plano).

## As 10 fases

```
0 Intake PT-BR       → pergunta socrática curta: tema, questão, PICO/SPIDER/Decomposição, recorte temporal, critérios inclusão/exclusão
1 Escopo + protocolo → decompõe questão, escolhe arquétipo, cria reviews/{slug}/ {state.json, protocol.md, passport.json}
2 Descoberta         → search_all.py (OpenAlex + arXiv + EuropePMC + Crossref) + read_local.py (artigos/ locais)
3 Dedup + triagem    → dedupe_rank.py → corpus.json + saturation check (novidade <20% = saturado) + tier deep/skim
4 GATE HUMANO 1      → usuário aprova protocolo + N incluídos. Sem aprovação, não avança (exceto --auto p/ quick)
5 Leitura profunda   → deep tier: texto completo OA quando houver; skim tier: abstract. Extrai PICO/método/achado/limite
6 Chasing            → chase.py backward (referências) + forward (citantes) via OpenAlex; merge com preservação de screening
7 GATE INTEGRIDADE   → verify_citations.py (existência + retratação) + prisma_counts.py. Falha fecha o gate
8 Síntese            → clusters temáticos, tension map, tabela extração, gaps. Separa confiança alta/média/baixa
9 Auto-crítica       → checklist 14 pontos (references/adversarial-14pt.md) + Devil's Advocate com threshold: só concede com evidência ≥4/5
10 Relatório         → build_report.py renderiza archetypes/*.md → reports/{slug}.md + .bib + .ris + passport final
```

## Contratos de script (envelope JSON no stdout)

Todos os scripts em `scripts/` são **camada pura de dados, sem chamadas LLM**, e imprimem um único JSON:

```json
{"status": "ok", "data": {...}, "warnings": [...]}
{"status": "error", "message": "...", "hint": "..."}
```

Isso garante reprodutibilidade e auditoria (ideia da skill 1). O agente LLM orquestra de fora.

### Uso rápido

```bash
# 0-1. estado + protocolo
python3 skills/deep-research-br/scripts/state.py init --slug meu-tema --questao "PatchTST vs TFT p/ LTSF multivariado?" --arquetipo systematic_review --template rigorous
# 2. busca federada (grátis, sem key; polite pool via --mailto)
python3 skills/deep-research-br/scripts/search_all.py --slug meu-tema --query "time series forecasting transformer" --fontes openalex,arxiv,epmc --max 30 --desde 2019
# 2b. corpus local (seu projeto temporal-model)
python3 skills/deep-research-br/scripts/read_local.py --slug meu-tema --dir artigos
# 3. dedup + rank + saturação
python3 skills/deep-research-br/scripts/dedupe_rank.py --slug meu-tema --top 25
# 6. chasing (após screening manual do corpus.json)
python3 skills/deep-research-br/scripts/chase.py --slug meu-tema --ids W123456,10.xxxx/xxxxx --direcao both --max 20
# 7. integridade + PRISMA
python3 skills/deep-research-br/scripts/verify_citations.py --slug meu-tema
python3 skills/deep-research-br/scripts/prisma_counts.py --slug meu-tema
# 10. relatório
python3 skills/deep-research-br/scripts/build_report.py --slug meu-tema --arquetipo systematic_review --titulo "Minha revisão"
```

Estado vive em `reviews/{slug}/`: `state.json` (autoritativo), `corpus.json` (registros + screening), `passport.json` (claims→fontes), `search_log.md`, `protocol.md`.

## Regras de honestidade (resumo; detalhe em references/honesty.md)

1. Só afirme o que foi recuperado nesta sessão. Sem fonte → sem afirmação.
2. Nunca invente DOI/PMID/arXiv ID. Todo ID vem de resposta de API.
3. Marque preprints como preprints; reviews como reviews; OA vs paywall.
4. Toda citação passa por `verify_citations.py` antes do relatório.
5. Declare bases **não consultadas** (ex: Scopus/WoS sem API pública) no apêndice.
6. Sem bypass de paywall. OA legal apenas (Unpaywall/OpenAlex/EPMC/arXiv).
7. Screening re-executado nunca apaga decisão humana (merge preserva `screening.status`).

## Gates

- **Gate 1 (Fase 4):** protocolo + PICO + critérios + N identificados. Pausa segura (pode fechar sessão e retomar via `state.py status --slug X`).
- **Gate 2 (Fase 7):** integridade. `verify` com qualquer `not_found/retracted/mismatched` → corrige ou remove antes de sintetizar.
- **Gate 3 (Fase 9):** auto-crítica 14pt + Devil's Advocate. Não concede sem evidência.

## Saídas

`reports/{slug}.md`, `reports/{slug}.bib`, `reports/{slug}.ris`, `reviews/{slug}/passport.json`, `reviews/{slug}/prisma.md`, `reviews/{slug}/evidence_table.csv` (o agente gera a partir do corpus).
