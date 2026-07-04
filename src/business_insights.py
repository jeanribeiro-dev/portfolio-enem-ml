import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import seaborn as sns
import os

# Configurações estéticas Premium
plt.style.use('ggplot')
sns.set_palette("Blues_r")
sns.set_style("whitegrid")

CSV_PATH = '../portfolio-enem-abstencao/data/raw_inep/DADOS/MICRODADOS_ENEM_2023.csv'
CUSTO_POR_CANDIDATO = 146.00  # Custo médio estimado do Inep por candidato (em Reais)

# Estruturas para acumular os dados
uf_agg = []
renda_agg = []
heatmap_agg = []

chunk_size = 300000
print("Iniciando varredura analítica de negócios (Big Data em Chunks)...")

chunks = pd.read_csv(CSV_PATH, sep=';', encoding='latin1', chunksize=chunk_size, 
                     usecols=['TP_PRESENCA_CH', 'Q006', 'TP_FAIXA_ETARIA', 'SG_UF_PROVA'])

total_lido = 0
for i, chunk in enumerate(chunks):
    # Tratar presença
    chunk['ausente'] = (chunk['TP_PRESENCA_CH'] == 0).astype(int)
    chunk['inscrito'] = 1
    
    # Tratar NAs
    chunk['Q006'] = chunk['Q006'].fillna('Sem Resposta')
    chunk['TP_FAIXA_ETARIA'] = chunk['TP_FAIXA_ETARIA'].fillna(1).astype(int)
    
    # Agregação por UF
    grp_uf = chunk.groupby('SG_UF_PROVA')[['ausente', 'inscrito']].sum().reset_index()
    uf_agg.append(grp_uf)
    
    # Agregação por Renda
    grp_renda = chunk.groupby('Q006')[['ausente', 'inscrito']].sum().reset_index()
    renda_agg.append(grp_renda)
    
    # Agregação Renda x Idade
    grp_heat = chunk.groupby(['Q006', 'TP_FAIXA_ETARIA'])[['ausente', 'inscrito']].sum().reset_index()
    heatmap_agg.append(grp_heat)
    
    total_lido += len(chunk)
    print(f"[{total_lido:,}] linhas computadas...")

print("\nFase Map finalizada. Iniciando Reduce...")

# Consolidar UF
df_uf = pd.concat(uf_agg).groupby('SG_UF_PROVA')[['ausente', 'inscrito']].sum().reset_index()
df_uf['custo_desperdicio'] = df_uf['ausente'] * CUSTO_POR_CANDIDATO

# Consolidar Renda
df_renda = pd.concat(renda_agg).groupby('Q006')[['ausente', 'inscrito']].sum().reset_index()
df_renda['taxa_abstencao'] = (df_renda['ausente'] / df_renda['inscrito']) * 100
# Filtrar só as letras reais (A a Q)
df_renda = df_renda[df_renda['Q006'].isin([chr(i) for i in range(65, 82)])].sort_values('Q006')

# Consolidar Heatmap
df_heat = pd.concat(heatmap_agg).groupby(['Q006', 'TP_FAIXA_ETARIA'])[['ausente', 'inscrito']].sum().reset_index()
df_heat['taxa_abstencao'] = (df_heat['ausente'] / df_heat['inscrito']) * 100
df_heat = df_heat[df_heat['Q006'].isin([chr(i) for i in range(65, 82)])]

os.makedirs('docs', exist_ok=True)

# ---------------------------------------------------------
# GRÁFICO 1: Prejuízo Financeiro por UF (Top 10)
# ---------------------------------------------------------
df_uf_top = df_uf.sort_values('custo_desperdicio', ascending=False).head(10)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='custo_desperdicio', y='SG_UF_PROVA', data=df_uf_top, palette='Reds_r', edgecolor='.2')
plt.title('Top 10 Estados com Maior Desperdício Financeiro (Abstenção)', pad=20, fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Prejuízo em Milhões (R$)', fontsize=12, fontweight='bold', color='#555555')
plt.ylabel('Estado', fontsize=12)
sns.despine(left=True, bottom=True)

for i in ax.containers:
    # Formatar para Milhões de Reais
    labels = [f"R$ {val/1000000:,.1f}M".replace(',', 'X').replace('.', ',').replace('X', '.') for val in i.datavalues]
    ax.bar_label(i, labels=labels, padding=5, fontsize=11, fontweight='bold', color='#333333')

plt.xlim(0, df_uf_top['custo_desperdicio'].max() * 1.20)
plt.xticks([]) # Esconder eixo X para deixar limpo
plt.tight_layout()
plt.savefig('docs/waste_by_state.png', dpi=300, bbox_inches='tight')

# ---------------------------------------------------------
# GRÁFICO 2: Decaimento de Abstenção por Renda
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
# A = Sem Renda ... Q = Mais de 20k
labels_renda = ['Nenhuma', 'Até 1k', '1k-1.5k', '1.5k-2k', '2k-2.5k', '2.5k-3k', '3k-4k', '4k-5k', '5k-6k', '6k-7k', '7k-8k', '8k-9k', '9k-10k', '10k-12k', '12k-15k', '15k-20k', '+20k']
df_renda['Renda_Label'] = labels_renda

ax = sns.lineplot(x='Renda_Label', y='taxa_abstencao', data=df_renda, marker='o', color='#1f77b4', linewidth=3, markersize=10)
plt.title('A Curva da Desigualdade: Taxa de Abstenção vs Renda', pad=20, fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Renda Familiar Declarada', fontsize=12, fontweight='bold', color='#555555')
plt.ylabel('Taxa de Abstenção (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
sns.despine(left=True, bottom=True)
plt.ylim(0, df_renda['taxa_abstencao'].max() * 1.1)
plt.fill_between(df_renda['Renda_Label'], df_renda['taxa_abstencao'], alpha=0.1, color='#1f77b4')
plt.tight_layout()
plt.savefig('docs/decay_by_income.png', dpi=300, bbox_inches='tight')

# ---------------------------------------------------------
# GRÁFICO 3: Mapa de Calor (Risco Evasão) Renda x Idade
# ---------------------------------------------------------
# Pivoteando o Heatmap
# Para Idade, vamos juntar algumas faixas para não ficar gigante
# 1-2 = <18, 3-5 = 18-20, 6-10 = 21-25, 11-15 = 26-40, 16-20 = >40
def map_idade_group(faixa):
    if faixa <= 2: return 'Até 17 anos'
    if faixa <= 5: return '18 a 20 anos'
    if faixa <= 10: return '21 a 25 anos'
    if faixa <= 15: return '26 a 40 anos'
    return '+40 anos'

df_heat['Idade_Group'] = df_heat['TP_FAIXA_ETARIA'].apply(map_idade_group)
# Re-agrupar com o novo nome
df_heat_grouped = df_heat.groupby(['Q006', 'Idade_Group'])[['ausente', 'inscrito']].sum().reset_index()
df_heat_grouped['taxa'] = (df_heat_grouped['ausente'] / df_heat_grouped['inscrito']) * 100

pivot_heat = df_heat_grouped.pivot(index='Idade_Group', columns='Q006', values='taxa')
# Reordenar Idade
ordem_idade = ['Até 17 anos', '18 a 20 anos', '21 a 25 anos', '26 a 40 anos', '+40 anos']
pivot_heat = pivot_heat.reindex(ordem_idade)
# Renomear colunas de Renda para algo mais simples (Baixa, Média, Alta)
renda_map_heat = {
    'A': 'Sem Renda', 'B': 'Até 1k', 'C': 'Até 1.5k', 'D': 'Até 2k', 'E': 'Até 2.5k', 
    'F': 'Até 3k', 'G': 'Até 4k', 'H': 'Até 5k', 'I': 'Até 6k', 'J': 'Até 7k',
    'K': 'Até 8k', 'L': 'Até 9k', 'M': 'Até 10k', 'N': 'Até 12k', 'O': 'Até 15k', 'P': 'Até 20k', 'Q': '+20k'
}
pivot_heat.columns = [renda_map_heat[c] for c in pivot_heat.columns]

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_heat, annot=True, fmt=".1f", cmap='YlOrRd', cbar_kws={'label': 'Risco de Falta (%)'}, linewidths=.5)
plt.title('Matriz de Risco: Fator Idade + Fator Renda', pad=20, fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Renda Familiar', fontsize=12, fontweight='bold')
plt.ylabel('Faixa Etária', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/heatmap_risk.png', dpi=300, bbox_inches='tight')

print("\nPipeline de Business Insights gerado com sucesso!")
print(f"Custo Financeiro Total Estimado Desperdiçado (BR): R$ {df_uf['custo_desperdicio'].sum():,.2f}")
