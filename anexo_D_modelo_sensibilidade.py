import pandas as pd
import pulp
import logging
import os

# Configurar logging profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Carrega os dados do CSV e realiza validações básicas."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")
    df = pd.read_csv(file_path)
    required_columns = ['Unidade de Producao', 'Volume Total (m3)', 'Valor Estimado (R$)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")
    logger.info(f"Dados carregados com sucesso: {len(df)} linhas")
    return df

def aggregate_data(df):
    """Agrega volumes e valores por Unidade de Producao (UP)."""
    up_group = df.groupby('Unidade de Producao')
    V = up_group['Volume Total (m3)'].sum()  # Volume total por UP
    R = up_group['Valor Estimado (R$)'].sum()  # Valor total por UP
    # Estimar custo variavel por UP baseado no numero de arvores
    num_arvores = up_group['Numero de Arvores'].sum()
    custo_base_por_m3 = 150.0  # Custo base fixo por m3
    custo_por_up = pd.Series(custo_base_por_m3 * (1 + 0.1 * (num_arvores / num_arvores.mean())), index=up_group.groups.keys())
    logger.info(f"{len(V)} Unidades de Producao agregadas")
    return V, R, custo_por_up

def setup_parameters(V, R, area_total_ha=175.35, ciclo_anos=30, intensidade_sustentavel_ha=25):
    """Define parametros baseados em praticas de manejo sustentavel na Amazonia."""
    sum_V = V.sum()
    sum_R = R.sum()

    # Calculo de metas sustentaveis sem variacao estocastica
    volume_sustentavel_total = area_total_ha * intensidade_sustentavel_ha
    frac_sustentavel = min(1.0, volume_sustentavel_total / sum_V)  # Nao exceder estoque

    H = ciclo_anos
    T = list(range(1, H + 1))

    Vmeta = (sum_V * frac_sustentavel) / H  # Volume anual sustentavel
    Rmeta = (sum_R * frac_sustentavel) / H  # Receita anual sustentavel

    alpha = 0.35  # Tolerancia de 35% para volume e receita
    frac_max_anual_por_up = 0.25  # Maximo 25% por ano por UP
    taxa_desconto = 0.08  # Taxa de desconto fixa
    variacao_minima = 0.08  # Variacao minima ano a ano
    penalidade_fator = 0.05  # Penalidade por desvios

    logger.info(f"Parametros configurados: H={H} anos, Vmeta={Vmeta:.2f} m3/ano, Rmeta={Rmeta:.2f} R$/ano, "
                f"alpha={alpha}, frac_max_anual={frac_max_anual_por_up}, taxa_desconto={taxa_desconto:.2f}, "
                f"variacao_minima={variacao_minima}, penalidade_fator={penalidade_fator}")
    return T, Vmeta, Rmeta, alpha, frac_max_anual_por_up, taxa_desconto, variacao_minima, penalidade_fator

def build_model(U, T, V, R, custo_por_up, Vmeta, Rmeta, alpha, frac_max_anual_por_up, taxa_desconto, variacao_minima, penalidade_fator):
    """Constrói e retorna o modelo de Programacao Linear usando PuLP."""
    prob = pulp.LpProblem("Manejo_Florestal_Sustentavel", pulp.LpMaximize)

    # Variaveis de decisao
    x = pulp.LpVariable.dicts("frac_colhida", (U, T), lowBound=0, upBound=1, cat='Continuous')

    # Variaveis auxiliares para volume, receita, custo por ano
    volume = pulp.LpVariable.dicts("volume_ano", T, lowBound=0, cat='Continuous')
    receita = pulp.LpVariable.dicts("receita_ano", T, lowBound=0, cat='Continuous')
    custo = pulp.LpVariable.dicts("custo_ano", T, lowBound=0, cat='Continuous')

    # Linearizacao de desvios absolutos para volume
    d_pos = pulp.LpVariable.dicts("desvio_pos", T, lowBound=0, cat='Continuous')
    d_neg = pulp.LpVariable.dicts("desvio_neg", T, lowBound=0, cat='Continuous')

    # Variaveis para forcar variacao
    delta_volume = pulp.LpVariable.dicts("delta_volume", [t for t in T if t > 1], lowBound=0, cat='Continuous')

    # Funcao-objetivo: Maximizar Valor Presente Liquido (VPL) com penalidade por desvios
    vpl = pulp.lpSum([(receita[t] - custo[t] - 0.01 * (d_pos[t] + d_neg[t])) / ((1 + taxa_desconto) ** (t-1)) for t in T])
    penalidade_desvio = penalidade_fator * pulp.lpSum([d_pos[t] + d_neg[t] for t in T])
    penalidade_variacao = 0.02 * pulp.lpSum([delta_volume[t] for t in [tt for tt in T if tt > 1]])
    prob += vpl - penalidade_desvio - penalidade_variacao, "Max_VPL_Com_Equilibrio"

    # Restricoes
    for t in T:
        prob += volume[t] == pulp.lpSum([V[u] * x[u][t] for u in U]), f"Def_Volume_{t}"
        prob += receita[t] == pulp.lpSum([R[u] * x[u][t] for u in U]), f"Def_Receita_{t}"
        prob += custo[t] == pulp.lpSum([custo_por_up[u] * V[u] * x[u][t] for u in U]), f"Def_Custo_{t}"

        # Linearizacao do desvio absoluto
        prob += volume[t] - Vmeta == d_pos[t] - d_neg[t], f"Desvio_Lin_{t}"

        # Faixas para volume e receita
        prob += volume[t] >= Vmeta * (1 - alpha * 1.2), f"Volume_Min_{t}"
        prob += volume[t] <= Vmeta * (1 + alpha), f"Volume_Max_{t}"
        prob += receita[t] >= Rmeta * (1 - alpha * 1.2), f"Receita_Min_{t}"
        prob += receita[t] <= Rmeta * (1 + alpha), f"Receita_Max_{t}"

        # Restricao de lucratividade minima
        prob += receita[t] - custo[t] >= 0.6 * receita[t], f"Lucratividade_Min_{t}"

    # Forcar variacao minima entre anos
    for t in T:
        if t > 1:
            prob += delta_volume[t] >= volume[t] - volume[t-1], f"Delta_Pos_{t}"
            prob += delta_volume[t] >= volume[t-1] - volume[t], f"Delta_Neg_{t}"
            prob += delta_volume[t] >= variacao_minima * Vmeta, f"Variacao_Min_{t}"
            prob += delta_volume[t] <= variacao_minima * Vmeta * 3, f"Variacao_Max_{t}"

    # Restricoes por UP
    for u in U:
        prob += pulp.lpSum([x[u][t] for t in T]) <= 1, f"Exaustao_Total_{u}"

        for t in T:
            prob += x[u][t] <= frac_max_anual_por_up, f"Intensidade_Anual_{u}_{t}"
            for k in range(1, 5):
                if t + k <= len(T):
                    prob += x[u][t] + x[u][t+k] <= frac_max_anual_por_up + 0.1, f"Regeneracao_{u}_{t}_{k}"

    logger.info(f"Modelo construido: {len(prob.variables())} variaveis, {len(prob.constraints)} restricoes")
    return prob, x, volume, receita, custo, d_pos, d_neg

def solve_and_report(prob, U, T, V, R, Vmeta, Rmeta, alpha, frac_max_anual_por_up, taxa_desconto, variacao_minima, penalidade_fator, output_file='resultados_manejo_deterministic.csv'):
    """Resolve o modelo, diagnostica problemas e gera relatório."""
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=1200, gapRel=0.03)
    status = prob.solve(solver)
    logger.info(f"Status da solucao: {pulp.LpStatus[status]}")

    if status != 1:
        logger.warning("Solucao nao otima. Relaxando restricoes de regeneracao e variacao maxima, resolvendo novamente.")
        for constraint in list(prob.constraints.values()):
            if 'Regeneracao' in constraint.name or 'Variacao_Max' in constraint.name:
                prob -= constraint
        status = prob.solve(solver)
        logger.info(f"Status apos relax: {pulp.LpStatus[status]}")

    if status != 1:
        logger.error("Modelo ainda inviavel. Sugestoes: Reduza variacao_minima, aumente alpha ou frac_max_anual_por_up.")
        return None

    obj_value = pulp.value(prob.objective)
    logger.info(f"Valor otimo (VPL ajustado): {obj_value:.2f} R$")

    # Coletar resultados
    results = []
    for t in T:
        volume_t = pulp.value(volume[t])
        receita_t = pulp.value(receita[t])
        custo_t = pulp.value(custo[t])
        lucro_t = receita_t - custo_t if receita_t is not None else 0
        desvio_volume = pulp.value(d_pos[t] + d_neg[t]) if d_pos[t] is not None else 0
        results.append({
            'Ano': t,
            'Volume (m3)': round(volume_t or 0, 2),
            'Meta Volume (m3)': round(Vmeta, 2),
            'Receita (R$)': round(receita_t or 0, 2),
            'Meta Receita (R$)': round(Rmeta, 2),
            'Custo (R$)': round(custo_t or 0, 2),
            'Lucro (R$)': round(lucro_t, 2),
            'Desvio Volume (m3)': round(desvio_volume, 2)
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Resultados salvos em: {output_file}")

    # Exibir resumo
    print(f"\nResumo Anual de Manejo Florestal Sustentavel (taxa_desconto={taxa_desconto:.2f}, "
          f"frac_max_anual={frac_max_anual_por_up}, alpha={alpha}):")
    print(df_results.to_string(index=False))

    return df_results

def run_sensitivity_analysis(file_path):
    """Executa análise de sensibilidade variando taxa de desconto, intensidade máxima e tolerância."""
    df = load_data(file_path)
    V, R, custo_por_up = aggregate_data(df)
    U = list(V.index)
    T, Vmeta, Rmeta, alpha, frac_max_anual_por_up, taxa_desconto, variacao_minima, penalidade_fator = setup_parameters(V, R)

    # Cenários de sensibilidade
    scenarios = [
        {'taxa_desconto': 0.10, 'frac_max_anual_por_up': frac_max_anual_por_up, 'alpha': alpha, 'output_file': 'resultados_delta_10.csv'},
        {'taxa_desconto': 0.12, 'frac_max_anual_por_up': frac_max_anual_por_up, 'alpha': alpha, 'output_file': 'resultados_delta_12.csv'},
        {'taxa_desconto': taxa_desconto, 'frac_max_anual_por_up': 0.20, 'alpha': alpha, 'output_file': 'resultados_fmax_20.csv'},
        {'taxa_desconto': taxa_desconto, 'frac_max_anual_por_up': 0.30, 'alpha': alpha, 'output_file': 'resultados_fmax_30.csv'},
        {'taxa_desconto': taxa_desconto, 'frac_max_anual_por_up': frac_max_anual_por_up, 'alpha': 0.30, 'output_file': 'resultados_alpha_30.csv'},
        {'taxa_desconto': taxa_desconto, 'frac_max_anual_por_up': frac_max_anual_por_up, 'alpha': 0.40, 'output_file': 'resultados_alpha_40.csv'}
    ]

    results_summary = []
    for scenario in scenarios:
        prob, x, volume, receita, custo, d_pos, d_neg = build_model(
            U, T, V, R, custo_por_up, Vmeta, Rmeta,
            scenario['alpha'], scenario['frac_max_anual_por_up'], scenario['taxa_desconto'], variacao_minima, penalidade_fator
        )
        result_df = solve_and_report(prob, U, T, V, R, Vmeta, Rmeta,
                                    scenario['alpha'], scenario['frac_max_anual_por_up'],
                                    scenario['taxa_desconto'], variacao_minima, penalidade_fator,
                                    scenario['output_file'])
        if result_df is not None:
            results_summary.append({
                'Taxa Desconto': scenario['taxa_desconto'],
                'Intensidade Maxima': scenario['frac_max_anual_por_up'],
                'Tolerancia': scenario['alpha'],
                'VPL (R$)': pulp.value(prob.objective),
                'Volume Medio (m3)': result_df['Volume (m3)'].mean(),
                'Receita Media (R$)': result_df['Receita (R$)'].mean(),
                'Custo Medio (R$)': result_df['Custo (R$)'].mean(),
                'Lucro Medio (R$)': result_df['Lucro (R$)'].mean(),
                'Desvio Medio (m3)': result_df['Desvio Volume (m3)'].mean()
            })

    # Exibir resumo da analise de sensibilidade
    df_summary = pd.DataFrame(results_summary)
    print("\nResumo da Analise de Sensibilidade:")
    print(df_summary.to_string(index=False))
    df_summary.to_csv('resumo_sensibilidade.csv', index=False)

if __name__ == "__main__":
    file_path = 'boletim_madeira_acre_com_precos.csv'
    try:
        run_sensitivity_analysis(file_path)
    except Exception as e:
        logger.error(f"Erro na execucao do programa: {str(e)}")