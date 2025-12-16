# 🛡️ Phishing Detector — Projeto de Fim de Curso

Deteção automática de phishing utilizando técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning

## 📌 Visão Geral

Este repositório contém o projeto final de licenciatura cujo objetivo é o desenvolvimento de um sistema de deteção de phishing baseado em features linguísticas, estruturais e técnicas, aplicado a URLs, cabeçalhos e conteúdos HTML de mensagens.

O projeto foi concebido com foco em:

🎯 Precisão e explicabilidade (modelos clássicos)

🔁 Reprodutibilidade científica

🧪 Aplicabilidade real em contexto de cibersegurança

🧠 Abordagem Técnica

A solução combina engenharia de features com Machine Learning supervisionado, evitando dependência excessiva de modelos opacos.

## 🔍 Tipos de Features

🌐 URL-based: comprimento, entropia, presença de IP, TLD suspeitos, etc.

🧾 HTML-based: discrepâncias entre links visíveis e reais (anchor mismatch)

📩 Headers: análise de campos relevantes (quando disponíveis)

🔗 Regras heurísticas: motor de regras complementar

## 🤖 Modelo

Algoritmos clássicos (ex.: Logistic Regression / SVM / Random Forest)

Vetorização TF-IDF (baseline)

Classificação binária: phishing vs legítimo

## 🗂️ Estrutura do Projeto

Phishing_Detector/
│
├── src/
│ ├── api/ # API FastAPI
│ │ └── main.py
│ ├── features/ # Extração de features
│ │ ├── url_signals.py
│ │ ├── html_url.py
│ │ └── headers.py
│ ├── rules/ # Motor de regras heurísticas
│ │ └── engine.py
│ └── models/ # Definições e schemas
│
├── data/
│ ├── raw/ # Dados originais
│ ├── processed/ # Dados tratados
│ └── samples/ # Amostras de teste
│
├── tools/ # Scripts auxiliares (debug/experimentos)
├── outputs/ # Resultados, gráficos, métricas
├── requirements.txt
├── README.md
└── .gitignore

## 🚀 Instalação
### 🧩 Pré-requisitos

🐍 Python 3.11 (obrigatório — alinhado com o relatório)

🐧 Linux / WSL (recomendado)

📦 pip atualizado


## 🧪 Criar Ambiente Virtual

'python3.11 -m venv .venv
source .venv/bin/activate'

## 📦 Instalar Dependências

```pip install --upgrade pip```
pip install -r requirements.txt

## ▶️ Execução
🔧 Iniciar a API
```uvicorn src.api.main:app --reload --port 8000

A API ficará disponível em:

🌍 http://127.0.0.1:8000

📘 Swagger UI: http://127.0.0.1:8000/docs

## 🧪 Exemplo de Utilização
📤 Pedido de Classificação

{
"url": "http://secure-login-update.example.com"
}

📥 Resposta

{
"label": "phishing",
"score": 0.87,
"rules_triggered": ["suspicious_tld", "long_url"]
}

## 📊 Avaliação

Métricas utilizadas:

Accuracy

Precision / Recall

F1-score

Avaliação realizada sobre dataset rotulado

Resultados documentados em /outputs

⚠️ Limitação conhecida: datasets públicos podem introduzir enviesamento temporal.

## 🧩 Limitações

Dataset limitado em diversidade temporal

Ausência de modelos deep learning (decisão consciente)

Não cobre engenharia social puramente semântica

## 🔮 Trabalho Futuro

🔍 Integração com feeds OSINT

🧠 Comparação com modelos Transformer

📈 Expansão do dataset

🛠️ Integração com gateways de e-mail

## 🎓 Contexto Académico

Projeto desenvolvido no âmbito de Trabalho de Fim de Curso (Licenciatura)

Área: Cibersegurança / Inteligência Artificial

## ⚖️ Licença

Este projeto é disponibilizado apenas para fins académicos e educativos.


✅ Estado do projeto: congelado para defesa
