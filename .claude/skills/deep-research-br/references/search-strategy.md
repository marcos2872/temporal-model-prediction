# Estratégia de busca (PICO padrão; SPIDER/Decomposição como fallback)

- **PICO** (clínico): Population / Intervention / Comparison / Outcome.
- **SPIDER** (qualitativo/social): Sample / Phenomenon / Design / Evaluation / Research type.
- **Decomposição** (tecnologia): Problema / Solução / Avaliação / Limitações.
- Gere 4-5 sub-queries mapeadas ao framework + 2 buscas de reviews (`systematic review`, `meta-analysis`) + 2 com recorte temporal (histórico + recente) + 1 follow-up no paper mais citado.
- Disciplina: sequencial, ~1 req/s, `mailto` no OpenAlex (polite pool), ≤3 req/s no PubMed/EPMC.
- Registre tudo em `search_log.md` (base, data, string exata, filtros, N).
