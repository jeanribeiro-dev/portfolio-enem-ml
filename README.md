# 🧠 Portfólio: O Custo da Evasão — Machine Learning aplicado ao ENEM (2023)

Modelo preditivo de abstenção treinado sobre **3,9 milhões de registros reais** do INEP, com cálculo do impacto financeiro estimado em **R$ 161,5 milhões** desperdiçados em 2023.

> Este projeto é a continuação analítica do [Dashboard de Abstenção](https://github.com/jeanribeiro-dev/portfolio-enem-abstencao). Enquanto aquele responde *o quê*, este responde o ***por quê*** e o ***quanto custa***.

## 🔗 Acesse o Relatório Online

👉 **[Clique aqui para acessar o Relatório Executivo](https://jeanribeiro-dev.github.io/portfolio-enem-ml/)**

## 💡 Motivação

Depois de construir o painel de abstenção e ver que estados do Norte e famílias de baixa renda lideravam as faltas, surgiu uma pergunta mais funda: **o que exatamente faz um candidato desistir no dia da prova?**

A análise descritiva (gráficos de barra, mapas) consegue mostrar *onde* e *quanto*. Mas não consegue dizer *por quê* com precisão estatística. Aí entra o Machine Learning: ao treinar um algoritmo para prever quem vai faltar, ele nos diz quais variáveis têm mais peso nessa decisão — e isso é informação de política pública.

A segunda pergunta era mais pragmática: **quanto dinheiro público vai pelo ralo?** Com o custo médio estimado do INEP de R$ 146 por candidato, a conta é direta.

## 🎯 O que este projeto responde

1. Quais fatores socioeconômicos têm mais poder preditivo sobre a abstenção?
2. Qual o perfil de risco máximo — quem tem maior probabilidade de faltar?
3. Quanto custa financeiramente cada estado, cada faixa de renda?

## 📊 Principais Resultados

| Análise | Resultado |
|:---|:---|
| **Feature Importance** | Renda Familiar e Faixa Etária dominam o critério de decisão do modelo |
| **ROC-AUC (Out-of-Sample)** | `0.6636` — com apenas 5 variáveis brutas |
| **Custo estimado desperdiçado** | **R$ 161.580.244** em provas não realizadas |
| **Perfil de maior risco** | Adultos acima de 26 anos com renda familiar até R$ 1.500 |
| **Curva da desigualdade** | A taxa de falta cai em degraus conforme a renda sobe nos primeiros R$ 3.000 |

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|:---|:---|
| Big Data Processing | Python, Pandas (`chunksize` — Out-of-Core) |
| Machine Learning | Scikit-Learn (`RandomForestClassifier`, `warm_start=True`) |
| Visualização | Matplotlib, Seaborn |
| Web Report | HTML5, CSS3 (Glassmorphism), Lucide Icons |
| Deploy | GitHub Pages |

## ⚙️ Decisão Técnica: Por que Out-of-Core?

O arquivo de microdados do ENEM 2023 tem **3,9 milhões de linhas** e pesa ~1,7 GB. Carregar tudo na RAM de uma só vez é inviável na maioria das máquinas.

A solução foi combinar duas técnicas:
- **`pd.read_csv(chunksize=200000)`** — leitura em blocos de 200 mil linhas, processando um de cada vez.
- **`RandomForestClassifier(warm_start=True)`** — permite adicionar novas árvores ao modelo (5 por bloco) sem descartar as anteriores. O modelo "aprende" incrementalmente sem precisar ver todos os dados de uma vez.

O resultado: 20 blocos lidos, 100 árvores construídas, 0 estouros de memória.

## 📂 Estrutura do Projeto

```
portfolio-enem-ml/
├── src/
│   ├── ml_evasao.py            # Treinamento Out-of-Core + Feature Importance
│   └── business_insights.py    # Análise financeira: custo por UF, curva de renda, heatmap
├── docs/
│   ├── index.html              # Relatório executivo web (GitHub Pages)
│   ├── style.css               # Design Dark Mode + Glassmorphism
│   ├── feature_importance.png  # Gráfico 1: importância das variáveis
│   ├── waste_by_state.png      # Gráfico 2: prejuízo por estado (Top 10)
│   ├── decay_by_income.png     # Gráfico 3: curva de decaimento por renda
│   └── heatmap_risk.png        # Gráfico 4: matriz de risco Renda × Idade
├── requirements.txt
└── README.md
```

## ▶️ Como Executar

1. Clone este repositório.
2. Certifique-se de ter os microdados do ENEM 2023 no caminho `../portfolio-enem-abstencao/data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv` (ou ajuste o `CSV_PATH` nos scripts).
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
4. Rode os scripts (nessa ordem):
```bash
python src/ml_evasao.py          # Treina o modelo e gera feature_importance.png
python src/business_insights.py  # Gera os 3 gráficos de negócio
```
5. Para visualizar o relatório localmente:
```bash
cd docs
python -m http.server 8080
# Acesse http://localhost:8080
```

## 📋 Fonte dos Dados

- **Microdados Oficiais:** INEP — Microdados do ENEM 2023. Disponíveis em [inep.gov.br](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).
- **Custo por Candidato:** Estimativa baseada em estudos publicados sobre o custo operacional do ENEM (R$ 146/candidato).

## 👤 Autor

**Jean Ribeiro** — Analista de Dados

---

*Parte de um portfólio de projetos de Ciência de Dados aplicada a dados públicos brasileiros.*
