

import numpy as np
import pandas as pd
import pulp

from src.configuracion import DISCOUNT_GRID


def solve_discount_knapsack(
    df_work: pd.DataFrame,
    discount_grid: list = DISCOUNT_GRID,
    budget: float = None,
    top_n: int = 500,
) -> pd.DataFrame:
   
    df_sample = df_work.nlargest(top_n, 'revenue').copy()

    if budget is None:
        budget = 0.05 * df_work['original_price'].sum()

    prob = pulp.LpProblem('discount_alloc', pulp.LpMaximize)
    x    = {}
    rev  = {}

    for i, (_, row) in enumerate(df_sample.iterrows()):
        P0   = row['original_price']
        Q0   = max(row['purchased_last_month'], 1)
        eps  = row['elasticity']

        d_base = row.get('discount_percentage', 0)

        for d in discount_grid:
            price      = P0 * (1 - d / 100)
            q          = Q0 * np.exp(eps * (-(d - d_base) / 100))
            rev[(i, d)] = price * max(q, 0)
            x[(i, d)]   = pulp.LpVariable(f'x_{i}_{d}', cat='Binary')


    prob += pulp.lpSum(rev[(i, d)] * x[(i, d)] for (i, d) in x)

    for i in range(len(df_sample)):
        prob += pulp.lpSum(x[(i, d)] for d in discount_grid) == 1


    prob += pulp.lpSum(
        (d / 100 * df_sample.iloc[i]['original_price']) * x[(i, d)]
        for i in range(len(df_sample))
        for d in discount_grid
    ) <= budget

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] not in ('Optimal', 'Feasible'):
        print(f"  [knapsack] Advertencia: solver terminó con status '{pulp.LpStatus[status]}'. "
              "Se usará el descuento predicho para todos los productos.")
        df_out = df_work.copy()
        df_out['discount_opt_knapsack'] = df_out['discount_predicted']
        return df_out

    chosen = {}
    for i, idx in enumerate(df_sample.index):
        for d in discount_grid:
            val = x[(i, d)].value()
            if val is not None and val > 0.5:
                chosen[idx] = d
                break

    df_out = df_work.copy()

    df_out['discount_opt_knapsack'] = df_out.index.map(
        lambda idx: chosen.get(idx, df_out.loc[idx, 'discount_predicted'])
    )
    return df_out



def baseline_policy(catalog: pd.DataFrame, t: int) -> dict:
    """Política baseline: usa el descuento real observado de cada producto."""
    return dict(zip(catalog['product_id'], catalog['discount_percentage']))


def knapsack_policy(catalog: pd.DataFrame, t: int) -> dict:
    
    if 'discount_opt_knapsack' in catalog.columns:
        col = 'discount_opt_knapsack'
    elif 'optimal_discount' in catalog.columns:
        col = 'optimal_discount'
    else:
        col = 'discount_percentage'

    return dict(zip(catalog['product_id'], catalog[col]))