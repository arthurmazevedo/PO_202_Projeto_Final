import matplotlib.pyplot as plt
import pandas as pd

# Carregar os dados do CSV
df = pd.read_csv('resultados_manejo_deterministic.csv')

# Configurar o gráfico
plt.figure(figsize=(10, 6))
plt.scatter(df['Volume (m³)'], df['Receita (R$)'], color='blue', label='Dados anuais')

# Adicionar linhas de referência para as faixas de tolerância (±35%)
meta_volume = df['Meta Volume (m³)'].iloc[0]
meta_receita = df['Meta Receita (R$)'].iloc[0]
plt.axhline(y=meta_receita * 1.35, color='r', linestyle='--', label='Limite superior receita')
plt.axhline(y=meta_receita * 0.65, color='r', linestyle='--', label='Limite inferior receita')
plt.axvline(x=meta_volume * 1.35, color='g', linestyle='--', label='Limite superior volume')
plt.axvline(x=meta_volume * 0.65, color='g', linestyle='--', label='Limite inferior volume')

# Configurações de legenda, títulos e eixos
plt.xlabel('Volume (m³)')
plt.ylabel('Receita (R$)')
plt.title('Relação Volume vs Receita por Ano')
plt.legend()
plt.grid(True)

# Salvar e exibir (substitua 'volume_receita_2D.png' por um caminho desejado)
plt.savefig('volume_receita_2D.png')
plt.show()