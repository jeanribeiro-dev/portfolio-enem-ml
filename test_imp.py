import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = '../portfolio-enem-abstencao/data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv'

def preprocess_chunk(chunk):
    chunk['ausente'] = (chunk['TP_PRESENCA_CH'] == 0).astype(int)
    chunk['Q006'] = chunk['Q006'].fillna('C')
    chunk['TP_SEXO'] = chunk['TP_SEXO'].fillna('F')
    chunk['Q006_num'] = chunk['Q006'].apply(lambda x: ord(str(x).upper()[0]) - 65 if type(x) == str else 2)
    chunk['TP_SEXO_num'] = chunk['TP_SEXO'].map({'M': 0, 'F': 1}).fillna(1)
    X = chunk[['Q006_num', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO_num', 'TP_FAIXA_ETARIA']].fillna(0)
    y = chunk['ausente']
    X.columns = ['Renda_Familiar', 'Tipo_Escola', 'Cor_Raca', 'Sexo', 'Faixa_Etaria']
    return X, y

chunks = pd.read_csv(CSV_PATH, sep=';', encoding='latin1', chunksize=200000, 
                     usecols=['TP_PRESENCA_CH', 'Q006', 'TP_ESCOLA', 'TP_COR_RACA', 'TP_SEXO', 'TP_FAIXA_ETARIA'])
chunk = next(chunks)
X, y = preprocess_chunk(chunk)

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_ * 100
for name, imp in zip(X.columns, importances):
    print(f"{name}: {imp:.1f}%")
