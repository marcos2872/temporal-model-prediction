# Checklist adversarial — 14 pontos (Gate fase 9, mandatório)

Responda SIM/NÃO com evidência (arquivo/linha ou ID do corpus). Um NÃO bloqueia o relatório.

1. Toda afirmação factual tem âncora `[^id]` válida?
2. Alguma citação é preprint rotulada como publicada (ou vice-versa)?
3. Algum DOI/ID foi digitado de memória (fora de resposta de API)?
4. `verify_citations.py` passou sem `not_found/mismatched/retracted`?
5. O paper canônico do tema está incluído ou a ausência está justificada?
6. Busca cobriu ≥3 fontes + log em `search_log.md` com datas/queries?
7. Saturação foi checada (`dedupe_rank.py`) e registrada?
8. Chasing forward+backward foi executado ou dispensado com motivo?
9. Exclusões têm motivo (`wrong population/intervention/outcome`, duplicata, fora de data, sem full-text)?
10. Síntese é temática (não paper-a-paper) e inclui tensões/contradições?
11. Níveis de confiança (alta/média/baixa) estão marcados?
12. Limitações do pipeline estão declaradas (bases não consultadas, viés EN, paywall)?
13. Devil's Advocate: tese sobrevive a 2 contra-argumentos com evidência? (só conceda com nota ≥4/5)
14. `.bib/.ris` conferem com as âncoras do `.md` (1:1, sem órfãs)?

## Protocolo Devil's Advocate (anti-bajulação, ideia do ARS reimplementada)

- Atacante gera 2 objeções fortes com fonte.
- Defensor pontua a réplica do usuário de 1-5; só concede com ≥4 (réplica atinge o núcleo com evidência).
- Nota ≤3: mantém a posição e reafirma o ataque.
- Proibido conceder 2x seguidas sem nova evidência.
