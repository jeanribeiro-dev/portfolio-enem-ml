# 🧠 Portfólio: Machine Learning & Prejuízo Financeiro na Abstenção do ENEM (2023)

Análise preditiva utilizando modelo **Random Forest (Out-of-Core)** para investigar os fatores determinantes da evasão no ENEM e calcular o impacto financeiro (desperdício) aos cofres públicos.

## 🔗 Acesse o Web Report Online

👉 [Clique aqui para acessar o Relatório Executivo Interativo](https://jeancribeiro1982-creator.github.io/portfolio-enem-ml/)

## 🚀 Visão Geral do Projeto

Este projeto é a segunda parte de uma análise de dados *End-to-End*, focado na extração de **Business Insights** e predições usando a base de microdados real do INEP (3.9 milhões de registros). Todo o pipeline cobre:

1. **Big Data Processing (Out-of-Core):** Leitura de uma base gigantesca em blocos de 200 mil a 300 mil linhas (chunks) utilizando `Pandas`, viabilizando o processamento em máquinas com memória RAM limitada.
2. **Machine Learning:** Treinamento iterativo de um modelo `RandomForestClassifier` (com `warm_start=True`) para prever o risco de abstenção de cada candidato com base em apenas 5 variáveis socioeconômicas.
3. **Business Analytics:** Tradução do modelo matemático para métricas de negócio reais, como o cálculo do **Rombo Financeiro de R$ 161,5 Milhões**.
4. **Web Report:** Um dashboard estático (HTML/CSS) com design premium (Glassmorphism) para apresentação executiva dos resultados (hospedado no GitHub Pages).

## 📊 Principais Descobertas e Insights

| Dimensão Analisada | Insight Principal |
| :--- | :--- |
| **Importância das Variáveis** | A **Renda Familiar** e a **Faixa Etária** são os dois maiores determinantes matemáticos para a evasão de um candidato. |
| **Custo Financeiro** | Apenas no ano de 2023, estimou-se um desperdício de **R$ 161.580.244,00** em provas impressas para candidatos que não compareceram. |
| **Curva da Desigualdade** | Ocorre uma queda drástica na taxa de abstenção nos primeiros degraus de aumento de renda, provando o impacto direto da vulnerabilidade extrema. |
| **Matriz de Risco (Idade x Renda)** | O perfil com risco quase absoluto de falta é composto por **adultos acima de 26 anos com baixíssima renda**, refletindo o conflito do trabalho informal aos domingos. |

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Manipulação de Dados:** Pandas, Numpy
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Visualização de Dados:** Matplotlib, Seaborn
- **Front-End (Web Report):** HTML5, CSS3, Lucide Icons

## ⚙️ Como Executar Localmente

1. Clone este repositório.
2. Certifique-se de que os microdados do ENEM 2023 estão no caminho correto (`data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv`).
3. Instale as dependências:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
4. Rode os scripts para gerar os gráficos na pasta `docs/`:
```bash
python src/ml_evasao.py
python src/business_insights.py
```
5. Para ver o Web Report, inicie um servidor local na pasta `docs`:
```bash
cd docs
python -m http.server 8080
```
Acesse `http://localhost:8080` no seu navegador.
