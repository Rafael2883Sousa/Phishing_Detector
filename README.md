# TCC Phishing NLP — Skeleton
Ambiente alvo: WSL2 Ubuntu 22.04 LTS · Python 3.11 · CPU.

## Estrutura
- data/: dados públicos sem PII. `raw/` e `processed/`.
- data/schema/: esquemas e templates.
- src/features/: extração de sinais (headers, urls, html).
- src/models/: pipelines (clássico, embeddings) [a preencher].
- src/api/: API local [a preencher].
- src/rules/: regras explicáveis e agregação de razões.
- tests/: testes unitários e smoke.
- configs/: listas branca/negra locais e parâmetros.
- notebooks/: exploração e gráficos (PR/ROC).

