import pandas as pd
import numpy as np
import os

# Definindo coeficientes da equação de Schumacher-Hall (valores genéricos para Amazônia)
a, b, c = 0.000057, 2.0, 0.84  # Nota: Usar coeficientes por espécie para maior precisão

# Função para calcular o volume de uma árvore (em m³)
def calculate_tree_volume(dap, ht):
    if not isinstance(dap, (int, float)) or not isinstance(ht, (int, float)):
        return 0.0
    if pd.isna(dap) or pd.isna(ht) or dap <= 0 or ht <= 0:
        return 0.0
    volume = a * (dap ** b) * (ht ** c)
    return round(volume, 4)

# Função para carregar e limpar dados
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', sep=';')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1', sep=';')
    
    # Normalizar colunas
    df['Especie_campo'] = df['Especie_campo'].str.lower().str.strip()
    df['MB'] = df['MB'].str.strip()
    
    # Filtrar árvores vivas e DAP fora do padrão
    df = df[(df['Especie_campo'] != 'morto') & (df['MB'] != 'S')]
    df = df[~df['Obs'].str.contains('Dap -', na=False)]
    
    # Verificar valores nulos em colunas críticas
    if df[['UA', 'Subunidade', 'Subparcela', 'DAP', 'HT']].isnull().any().any():
        print("Warning: Valores nulos encontrados em colunas críticas.")
        df = df.dropna(subset=['UA', 'Subunidade', 'Subparcela', 'DAP', 'HT'])
    
    return df

# Função para calcular e resumir resultados
def calculate_and_summarize(df, area_subparcela_ha=0.025):
    df['Volume_m3'] = df.apply(lambda row: calculate_tree_volume(row['DAP'], row['HT']), axis=1)
    volume_by_subparcela = df.groupby(['UA', 'Subunidade', 'Subparcela'])['Volume_m3'].sum().reset_index()
    trees_by_subparcela = df.groupby(['UA', 'Subunidade', 'Subparcela'])['Narv'].count().reset_index()
    trees_by_subparcela.rename(columns={'Narv': 'Num_arvores'}, inplace=True)
    result = pd.merge(volume_by_subparcela, trees_by_subparcela, on=['UA', 'Subunidade', 'Subparcela'])
    result['Volume_m3_ha'] = (result['Volume_m3'] / area_subparcela_ha).round(2)
    result['Arvores_ha'] = (result['Num_arvores'] / area_subparcela_ha).round(2)
    return result

# Função para exibir resumo
def print_summary(result):
    print("Resumo do Inventário Florestal no Acre:")
    print(f"Volume total estimado (m³): {result['Volume_m3'].sum():.2f}")
    print(f"Número total de árvores: {result['Num_arvores'].sum()}")
    print(f"Volume médio por hectare (m³/ha): {result['Volume_m3_ha'].mean():.2f}")
    print(f"Densidade média de árvores por hectare: {result['Arvores_ha'].mean():.2f}")

# Carregar e processar dados
file_path = "dadosabertos_snif_dap_10_por_uf_ac_2024_sfb_29052025.csv"
df = load_and_clean_data(file_path)
result = calculate_and_summarize(df)

# Exibir resultados
print_summary(result)