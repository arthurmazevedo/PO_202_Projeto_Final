import pandas as pd
import numpy as np

# Definindo coeficientes da equacao de Schumacher-Hall
a = 0.000057  # Coeficiente 'a'
b = 2.0       # Exponente para DAP
c = 0.84      # Exponente para HT

# Dicionario de precos por especie em R$/m3 (baseado em IBGE PEVS 2023)
precos_por_especie = {
    'abacaba': 600.00,
    'abiorana branca': 650.00,
    'abiorana casca fina': 650.00,
    'abiorana vermelha': 650.00,
    'angelca amarela': 700.00,
    'apui amarelo': 750.00,
    'apui miudo': 750.00,
    'araca bravo': 600.00,
    'bordao de velho': 650.00,
    'breu mescla': 700.00,
    'burra leiteira': 650.00,
    'cacaui': 600.00,
    'cabelo de cutia': 650.00,
    'cajarana': 700.00,
    'canela de velho': 650.00,
    'carapanauba amarela': 750.00,
    'caripe roxo': 700.00,
    'caripe vermelho': 700.00,
    'castainha': 650.00,
    'cipo farinha seca': 600.00,
    'coacu': 650.00,
    'copaiba amarela': 800.00,
    'envira iodo branca': 650.00,
    'envira piaca': 650.00,
    'espinheiro': 700.00,
    'falso pau brasil': 750.00,
    'gameleira': 650.00,
    'gogo de guariba': 600.00,
    'guariuba roxa': 700.00,
    'imbirindiba amarela': 750.00,
    'inga vermelha': 650.00,
    'jaca brava': 650.00,
    'jenipapo': 700.00,
    'jito': 650.00,
    'jito branco': 650.00,
    'jito preto': 650.00,
    'joao mole caule flora': 600.00,
    'limaozinho': 650.00,
    'louro amarelo': 700.00,
    'louro preto': 700.00,
    'mamui': 650.00,
    'maparajuba vermelha': 650.00,
    'mororo vermelho': 650.00,
    'morototo': 650.00,
    'muiratinga grande': 650.00,
    'mulateirana': 700.00,
    'mulungu': 650.00,
    'murmuru': 600.00,
    'mutamba': 650.00,
    'ninhare': 650.00,
    'olacacea': 600.00,
    'pama preta': 650.00,
    'pau de remo': 700.00,
    'pau sangue': 650.00,
    'paxiubao': 600.00,
    'paxiubinha': 600.00,
    'pequi': 650.00,
    'priquiteira': 650.00,
    'sapota': 750.00,
    'sapotinha': 750.00,
    'seringarana': 650.00,
    'ucuuba preta': 650.00,
    'ucuuba vermelha': 650.00,
    'ucuuba vermelha folha grande': 650.00,
    'uricuri': 600.00,
    'urucurana vermelha': 650.00,
    'urtigao': 600.00,
    'xixa amarelo': 700.00
}

# Funcao para calcular o volume de uma arvore (em m3)
def calculate_tree_volume(dap, ht):
    if pd.isna(dap) or pd.isna(ht) or dap <= 0 or ht <= 0:
        return 0.0
    volume = a * (dap ** b) * (ht ** c)
    return volume

# Carregar o arquivo CSV
file_path = "dadosabertos_snif_dap_10_por_uf_ac_2024_sfb_29052025.csv"
try:
    df = pd.read_csv(file_path, encoding='utf-8', sep=';')
except FileNotFoundError:
    print(f"Erro: O arquivo {file_path} nao foi encontrado. Verifique o caminho do arquivo.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Filtrar arvores vivas (excluir "morto" e MB = "S")
df = df[(df['Especie_campo'] != 'morto') & (df['MB'] != 'S')]

# Calcular o volume para cada arvore
df['Volume_m3'] = df.apply(lambda row: calculate_tree_volume(row['DAP'], row['HT']), axis=1)

# Agrupar por UA (UP) e Especie_campo para calcular volume total e numero de arvores
grouped_volume = df.groupby(['UA', 'Especie_campo'])['Volume_m3'].sum().reset_index()
grouped_trees = df.groupby(['UA', 'Especie_campo']).size().reset_index(name='Numero de Arvores')

# Mesclar os grupos
bulletin = pd.merge(grouped_volume, grouped_trees, on=['UA', 'Especie_campo'])

# Renomear colunas para o boletim
bulletin.rename(columns={
    'UA': 'Unidade de Producao',
    'Especie_campo': 'Especie',
    'Volume_m3': 'Volume Total (m3)'
}, inplace=True)

# Adicionar preco por especie (usando o dicionario; preco padrao 600.00 se nao encontrado)
bulletin['Preco (R$/m3)'] = bulletin['Especie'].map(precos_por_especie).fillna(600.00)

# Calcular o valor estimado por especie em cada UP
bulletin['Valor Estimado (R$)'] = bulletin['Volume Total (m3)'] * bulletin['Preco (R$/m3)']

# Calcular o valor total por UP (somando os valores das especies)
total_por_up = bulletin.groupby('Unidade de Producao')['Valor Estimado (R$)'].sum().reset_index()
total_por_up.rename(columns={'Valor Estimado (R$)': 'Valor Total por UP (R$)'}, inplace=True)

# Calcular o numero total de arvores por UP
total_trees_por_up = bulletin.groupby('Unidade de Producao')['Numero de Arvores'].sum().reset_index()
total_trees_por_up.rename(columns={'Numero de Arvores': 'Numero Total de Arvores'}, inplace=True)

# Mesclar o numero total de arvores na tabela de totais por UP
total_por_up = pd.merge(total_por_up, total_trees_por_up, on='Unidade de Producao')

# Exportar o boletim completo para CSV
bulletin.to_csv('wood_volume_value_bulletin_acre.csv', index=False, encoding='utf-8')

# Exportar a nova tabela com os totais por UP para um CSV separado
total_por_up.to_csv('total_value_per_up_acre.csv', index=False, encoding='utf-8')

# Exibir mensagem de confirmacao e resultados
print("Boletim completo gerado e exportado para 'wood_volume_value_bulletin_acre.csv'")
print("Nova tabela com valor total e numero de arvores por UP exportada para 'total_value_per_up_acre.csv'")