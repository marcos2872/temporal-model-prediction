# Honestidade sistêmica (adaptado do co-researcher; texto próprio)

1. **Recuperar antes de afirmar.** Nenhuma afirmação factual sem registro em `corpus.json` lido nesta sessão.
2. **IDs só de API.** DOI/PMID/arXiv/OpenAlex vêm de resposta JSON, nunca da memória do modelo.
3. **Rotule o tipo.** preprint vs publicado, review vs primário, OA vs paywall-abstract-only.
4. **Gate antes de publicar.** `verify_citations.py` precisa passar; `not_found/mismatched/retracted` bloqueiam.
5. **Incerteza explícita.** Alta (replicado, múltiplas fontes) / média (amostra/método limitado) / baixa (single-source, especulativo).
6. **Negativos contam.** Inclua achados nulos/conflitantes; não esconda tensão.
7. **Sem paywall bypass.** Só rotas legais: OpenAlex OA, EuropePMC, arXiv, Unpaywall.
8. **Proveniência preservada.** Re-runs nunca apagam `screening.status` humano.
