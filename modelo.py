import pandas as pd
import pulp

# Carregar os dados do CSV (substitua pelo caminho real se necessário)
df = pd.read_csv('boletim_madeira_acre_com_precos.csv')

# Agregar por UP: somar volumes e valores
up_group = df.groupby('Unidade de Producao')
V = up_group['Volume Total (m3)'].sum()  # Volume total por UP
R = up_group['Valor Estimado (R$)'].sum()  # Valor total (receita) por UP

# Conjuntos
U = list(V.index)  # Lista de UPs
H = 30  # Horizonte realista: ciclo de 30 anos
T = list(range(1, H + 1))  # Anos: 1 a 30

# Parâmetros ajustados para realismo
sum_V = V.sum()
intensidade_sustentavel = 20 / 101  # ~20 m³/ha por ciclo / estoque médio ~101 m³/ha
Vmeta = (sum_V * intensidade_sustentavel) / H  # ~416 m³/ano ajustado
sum_R = R.sum()
Rmeta = sum_R / H * intensidade_sustentavel  # Ajuste proporcional
alpha = 0.3  # Relaxado para viabilidade (piso de receita =70% da meta)

# Criar o problema de otimização
prob = pulp.LpProblem("Alocacao_Producao_Florestal_Realista", pulp.LpMinimize)

# Variáveis de decisão
x = pulp.LpVariable.dicts("x", (U, T), lowBound=0, upBound=1, cat='Continuous')
d = pulp.LpVariable.dicts("d", T, lowBound=0, cat='Continuous')

# Função-objetivo: minimizar soma dos desvios
prob += pulp.lpSum([d[t] for t in T]), "Min_Desvios_Volume"

# Restrições
for t in T:
    volume_t = pulp.lpSum([V[u] * x[u][t] for u in U])
    prob += volume_t <= Vmeta + d[t], f"Volume_Upper_{t}"
    prob += volume_t >= Vmeta - d[t], f"Volume_Lower_{t}"
    receita_t = pulp.lpSum([R[u] * x[u][t] for u in U])
    prob += receita_t >= (1 - alpha) * Rmeta, f"Receita_Lower_{t}"
    prob += receita_t <= (1 + alpha) * Rmeta, f"Receita_Upper_{t}"
    prob += volume_t >= 0, f"Volume_NonNeg_{t}"  # Explícito para evitar artefatos
    prob += receita_t >= 0, f"Receita_NonNeg_{t}"

for u in U:
    prob += pulp.lpSum([x[u][t] for t in T]) <= 1, f"Exaustao_{u}"  # 100% max total
    # Removido limite 70%; agora controlado por Vmeta ajustada
    for t in T:
        prob += x[u][t] <= 0.2, f"Intensidade_Anual_{u}_{t}"  # Max 20% por ano por UP (seletivo)

# Resolver o problema
prob.solve()

# Exibir status e resultados (agora factível e realista)
print("Status:", pulp.LpStatus[prob.status])
print("Valor ótimo da função-objetivo (soma dos desvios):", pulp.value(prob.objective))

print("\nResumo anual:")
for t in T:
    volume_t = sum(pulp.value(V[u] * x[u][t]) for u in U)
    receita_t = sum(pulp.value(R[u] * x[u][t]) for u in U)
    print(f"Ano {t}: Volume = {volume_t:.2f} m³ (meta: {Vmeta:.2f}), Receita = {receita_t:.2f} R$ (meta: {Rmeta:.2f})")