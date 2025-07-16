import matplotlib.pyplot as plt
import pandas as pd

# Carregar os dados do CSV
df = pd.read_csv('resultados_manejo_deterministic.csv')

# Configurar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(df['Ano'], df['Lucro (R$)'], marker='o', color='green', label='Lucro anual')

# Adicionar linha de referência para a média de lucro
plt.axhline(y=df['Lucro (R$)'].mean(), color='r', linestyle='--', label='Média de lucro')

# Configurações de legenda, títulos e eixos
plt.xlabel('Ano')
plt.ylabel('Lucro (R$)')
plt.title('Evolução do Lucro Anual')
plt.legend()
plt.grid(True)

# Salvar e exibir (substitua 'lucro_evolucao.png' por um caminho desejado)
plt.savefig('lucro_evolucao.png')
plt.show()