import pandas as pd
import pulp
from uuid import uuid4

# Definindo parâmetros
H = 10  # Horizonte de planejamento aumentado
ALPHA = 0.2  # Tolerância de receita aumentada
PRICE_PER_M3 = 824.60  # Preço por m³

# Lendo e processando o arquivo CSV
df = pd.read_csv('wood_volume_summary_acre.csv', sep=';')

# Agregando volume por UP
volume_by_up = df.groupby('UP')['Volume_m3'].sum().to_dict()
UPs = list(volume_by_up.keys())
T = range(1, H + 1)

# Calculando parâmetros
V_total = sum(volume_by_up.values())
V_meta = V_total / H
R_meta = V_total * PRICE_PER_M3 / H

# Criando o modelo
model = pulp.LpProblem("Forest_Allocation", pulp.LpMinimize)

# Variáveis de decisão (binárias)
x = pulp.LpVariable.dicts("x", [(u, t) for u in UPs for t in T], cat='Binary')
d = pulp.LpVariable.dicts("d", T, lowBound=0, cat='Continuous')

# Função-objetivo: Minimizar a soma dos desvios absolutos
model += pulp.lpSum(d[t] for t in T), "Minimize_Total_Deviation"

# Restrições
# 1. Balanço de volume por ano
for t in T:
    model += pulp.lpSum(volume_by_up[u] * x[(u, t)] for u in UPs) + d[t] >= V_meta, f"Volume_Lower_{t}"
    model += pulp.lpSum(volume_by_up[u] * x[(u, t)] for u in UPs) - d[t] <= V_meta, f"Volume_Upper_{t}"

# 2. Limites de receita por ano
for t in T:
    model += pulp.lpSum(volume_by_up[u] * PRICE_PER_M3 * x[(u, t)] for u in UPs) >= (1 - ALPHA) * R_meta, f"Revenue_Lower_{t}"
    model += pulp.lpSum(volume_by_up[u] * PRICE_PER_M3 * x[(u, t)] for u in UPs) <= (1 + ALPHA) * R_meta, f"Revenue_Upper_{t}"

# 3. Exaustão de UPs
for u in UPs:
    model += pulp.lpSum(x[(u, t)] for t in T) <= 1, f"Exhaustion_{u}"

# 4. Restrição de sustentabilidade (máximo 70% do volume por UP)
for u in UPs:
    model += pulp.lpSum(x[(u, t)] for t in T) <= 0.7, f"Sustainability_{u}"

# Resolvendo o modelo
model.solve()

# Verificando o status da solução
status = pulp.LpStatus[model.status]
print(f"Status da solução: {status}")

# Salvando resultados
results = []
for u in UPs:
    for t in T:
        if x[(u, t)].varValue > 0:
            results.append({
                'UP': u,
                'Ano': t,
                'Fração_Colhida': x[(u, t)].varValue,
                'Volume_Colhido_m3': volume_by_up[u] * x[(u, t)].varValue,
                'Receita_R$': volume_by_up[u] * x[(u, t)].varValue * PRICE_PER_M3
            })

# Resultados anuais
annual_results = []
for t in T:
    volume_t = sum(volume_by_up[u] * x[(u, t)].varValue for u in UPs)
    revenue_t = sum(volume_by_up[u] * x[(u, t)].varValue * PRICE_PER_M3 for u in UPs)
    deviation_t = d[t].varValue
    annual_results.append({
        'Ano': t,
        'Volume_Colhido_m3': volume_t,
        'Receita_R$': revenue_t,
        'Desvio_Volume_m3': deviation_t
    })

# Salvando em CSV
results_df = pd.DataFrame(results)
results_df.to_csv('allocation_results_adjusted.csv', index=False)
annual_results_df = pd.DataFrame(annual_results)
annual_results_df.to_csv('annual_results_adjusted.csv', index=False)

# Exibindo o valor da função-objetivo
objective_value = pulp.value(model.objective)
print(f"Valor da função-objetivo (soma dos desvios, m³): {objective_value}")

# Exibindo resumo anual
print("\nResumo Anual:")
for t in T:
    volume_t = sum(volume_by_up[u] * x[(u, t)].varValue for u in UPs)
    revenue_t = sum(volume_by_up[u] * x[(u, t)].varValue * PRICE_PER_M3 for u in UPs)
    print(f"Ano {t}: Volume = {volume_t:.2f} m³, Receita = R$ {revenue_t:.2f}, Desvio = {d[t].varValue:.2f} m³")

# Análise de sensibilidade
print("\nAnálise de Sensibilidade:")
for name, c in model.constraints.items():
    print(f"Restrição {name}: Shadow Price = {c.pi}, Slack = {c.slack}")