import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# Configurações do matplotlib
plt.style.use('ggplot')
sns.set_palette("husl")

# Caminho do CSV real (na pasta do outro projeto)
CSV_PATH = '../portfolio-enem-abstencao/data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv'

# Features escolhidas
FEATURES = ['Q006', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO', 'TP_FAIXA_ETARIA']
TARGET = 'ausente'

def preprocess_chunk(chunk):
    # Criar variável alvo: 1 se faltou em CH, 0 se foi
    chunk['ausente'] = (chunk['TP_PRESENCA_CH'] == 0).astype(int)
    
    # Preencher NAs caso existam
    chunk['Q006'] = chunk['Q006'].fillna('C') # Moda
    chunk['TP_SEXO'] = chunk['TP_SEXO'].fillna('F')
    
    # Encoding básico numérico
    # Q006 vai de A a Q (subtrair 65 da tabela ascii dá A=0, B=1, etc)
    chunk['Q006_num'] = chunk['Q006'].apply(lambda x: ord(str(x).upper()[0]) - 65 if type(x) == str else 2)
    chunk['TP_SEXO_num'] = chunk['TP_SEXO'].map({'M': 0, 'F': 1}).fillna(1)
    
    # Substituir colunas e tratar NAs numéricos
    X = chunk[['Q006_num', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO_num', 'TP_FAIXA_ETARIA']].fillna(0)
    y = chunk['ausente']
    
    # Renomear para ficar legível no gráfico
    X.columns = ['Renda_Familiar', 'Tipo_Escola', 'Cor_Raca', 'Sexo', 'Faixa_Etaria']
    return X, y

print("Iniciando treinamento Out-of-Core (Random Forest)...")

# Modelo Random Forest preparado para receber novos lotes (warm_start=True)
rf_model = RandomForestClassifier(
    n_estimators=0, 
    warm_start=True, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1
)

chunk_size = 200000
total_processed = 0
last_X_test, last_y_test = None, None

# Lemos em chunks iterativos
chunks = pd.read_csv(CSV_PATH, sep=';', encoding='latin1', chunksize=chunk_size, 
                     usecols=['TP_PRESENCA_CH', 'Q006', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO', 'TP_FAIXA_ETARIA'])

for i, chunk in enumerate(chunks):
    X, y = preprocess_chunk(chunk)
    
    # Adiciona 5 novas árvores para este chunk
    rf_model.n_estimators += 5
    rf_model.fit(X, y)
    
    total_processed += len(chunk)
    print(f"Bloco {i+1} processado. Total de candidatos vistos: {total_processed:,}. Árvores na floresta: {rf_model.n_estimators}")
    
    # Guardamos o último chunk para usar como validação de teste
    last_X_test, last_y_test = X, y

print("\nTreinamento concluído com sucesso!")
print(f"Avaliando modelo no último bloco Out-of-Sample ({len(last_X_test)} registros)...")

# Previsões
y_pred = rf_model.predict(last_X_test)
y_prob = rf_model.predict_proba(last_X_test)[:, 1]

acc = accuracy_score(last_y_test, y_pred)
roc = roc_auc_score(last_y_test, y_prob)

print(f"Acurácia: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")

# Gráfico de Feature Importance
importances = rf_model.feature_importances_
features_names = last_X_test.columns

importance_df = pd.DataFrame({
    'Feature': features_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Importância das Variáveis para Abstenção no ENEM (Random Forest)', pad=20, fontsize=14)
plt.xlabel('Grau de Importância (Peso)', fontsize=12)
plt.ylabel('Variáveis Socioeconômicas', fontsize=12)
plt.tight_layout()

# Salvar gráfico
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/feature_importance.png', dpi=300)
print("\nGráfico gerado e salvo em 'docs/feature_importance.png'")
