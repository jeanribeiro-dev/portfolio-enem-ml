# 🧠 Portfólio: O Custo da Evasão — Machine Learning aplicado ao ENEM (2023)

<!-- CI/CD Badges -->
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)

![Report Preview](docs/report.png)

Modelo preditivo de abstenção treinado sobre **3,9 milhões de registros reais** do INEP, com cálculo do impacto financeiro estimado em **R$ 161,5 milhões** desperdiçados em 2023.

> Este projeto é a continuação analítica do [Dashboard de Abstenção](https://github.com/jeanribeiro-dev/portfolio-enem-abstencao). Enquanto aquele responde *o quê*, este responde o ***por quê*** e o ***quanto custa***.

## 🔗 Acesse o Relatório Online

👉 **[Clique aqui para acessar o Relatório Executivo](https://jeanribeiro-dev.github.io/portfolio-enem-ml/)**

## 💡 Motivação

Depois de construir o painel de abstenção e ver que estados do Norte e famílias de baixa renda lideravam as faltas, surgiu uma pergunta mais funda: **o que exatamente faz um candidato desistir no dia da prova?**

A análise descritiva (gráficos de barra, mapas) consegue mostrar *onde* e *quanto*. Mas não consegue dizer *por quê* com precisão estatística. Aí entra o Machine Learning: ao treinar um algoritmo para prever quem vai faltar, ele nos diz quais variáveis têm mais peso nessa decisão — e isso é informação de política pública.

A segunda pergunta era mais pragmática: **quanto dinheiro público vai pelo ralo?** Com o custo médio estimado do INEP de R$ 146 por candidato, a conta é direta.

## 🎯 Business Impact & Key Questions
Este projeto vai além da predição pura. O objetivo é mensurar o impacto logístico e financeiro do INEP para informar políticas de realocação.

1. Quais fatores socioeconômicos têm mais poder preditivo sobre a abstenção?
2. Qual o perfil de risco máximo — quem tem maior probabilidade de faltar?
3. Quanto custa financeiramente cada estado, cada faixa de renda?

## 📊 Principais Resultados

| Métrica Out-of-Sample | Valor | Contexto / Significado |
|:---|:---|:---|
| **ROC-AUC** | `0.6636` | Poder de separação geral do modelo com apenas 5 variáveis brutas |
| **PR-AUC** | `0.5781` | Mostra a real capacidade preditiva lidando com a classe minoritária |
| **Precision** | `0.6270` | Quando o modelo aponta "Falta", ele acerta 63% das vezes |
| **Recall** | `0.1406` | Modelo hiper-conservador: captura apenas os casos de altíssimo risco |
| **Acurácia vs Baseline**| `0.5792` vs `0.5538` | Vence o chute ingênuo (baseline) consistentemente |
| **Custo desperdiçado** | **R$ 161.580.244** | Baseado na média de custo operacional (R$ 146 por ausente) |

*Nota Científica: Em problemas complexos de comportamento humano (como evasão), o recall costuma ser baixo pois o algoritmo não possui dados de imprevistos do dia-a-dia (chuva, doenças súbitas, trânsito). A prioridade do modelo foi treinar uma Precisão alta para focar as políticas públicas apenas nos alvos certeiros.*

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|:---|:---|
| Big Data Processing | Python, Pandas (`chunksize` — Out-of-Core) |
| Machine Learning | Scikit-Learn (`RandomForestClassifier`, `warm_start=True`) |
| Visualização | Matplotlib, Seaborn |
| Web Report | HTML5, CSS3 (Glassmorphism), Lucide Icons |
| Deploy | GitHub Pages |

## ⚙️ Arquitetura e Decisão Técnica: Out-of-Core

O arquivo de microdados do ENEM 2023 tem **3,9 milhões de linhas** e pesa ~1,7 GB. Carregar tudo na RAM de uma só vez é inviável na maioria das máquinas.

```mermaid
graph LR
    A["Microdados INEP (1.7GB)"] -->|"Chunk 1: 200k"| B("RandomForest - Warm Start")
    A -->|"Chunk 2: 200k"| B
    A -->|"Chunk N"| B
    B --> C["Feature Importance Extract"]
    C --> D["Business Insights Gen"]
```

A solução foi combinar duas técnicas:
- **`pd.read_csv(chunksize=200000)`** — leitura em blocos de 200 mil linhas, processando um de cada vez.
- **`RandomForestClassifier(warm_start=True)`** — permite adicionar novas árvores ao modelo (5 por bloco) sem descartar as anteriores. O modelo "aprende" incrementalmente.

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

## ▶️ Reprodutibilidade e Setup Local

1. Clone este repositório.
2. Certifique-se de ter os microdados do ENEM 2023 no caminho `../portfolio-enem-abstencao/data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv`.
3. Instale as dependências via ambiente virtual rigoroso:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
4. Rode os scripts (nessa ordem):
```bash
python src/ml_evasao.py          # Treina o modelo incremental
python src/business_insights.py  # Gera os gráficos de negócio
```
5. Para visualizar o relatório localmente:
```bash
cd docs
python -m http.server 8080
```

## 📋 Proveniência de Dados e Limitações do Modelo

- **Fonte Oficial:** Os dados brutos vêm do INEP. [Link de acesso público](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem).
- **Custo por Candidato:** Estimativa baseada em estudos publicados sobre o custo operacional do ENEM (R$ 146/candidato).

### Limitações do Modelo (Honestidade Intelectual)
O ROC-AUC do modelo é restrito por conta da natureza do problema. A evasão escolar/abandono de exame carrega um peso enorme de variáveis não-observáveis (ex: o candidato acordou doente, o ônibus quebrou, problemas familiares súbitos). Nenhuma das 5 features socioeconômicas consegue mapear o imponderável. A precisão atual é o limite estatístico do que se pode prever apenas observando condições estruturais de vida.

## 👤 Autor

**Jean Ribeiro** — Analista de Dados

---

*Parte de um portfólio de projetos de Ciência de Dados aplicada a dados públicos brasileiros.*
