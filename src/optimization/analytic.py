

import numpy as np
import pandas as pd

from src.configuracion import DISCOUNT_GRID, GLOBAL_ELASTICITY, MIN_PRICE_RATIO


def estimate_demand(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    d_pred = df['discount_predicted'].clip(0, 70)
    d_real = df['discount_percentage'].clip(0, 70)
    delta_p = (d_pred - d_real) / 100
    df['demand_estimated'] = (
        df['purchased_last_month'] * np.exp(df['elasticity'] * (-delta_p))
    ).clip(lower=0, upper=df['purchased_last_month'] * 10)  
    return df


def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:

    df    = df.copy()
    price = (df['original_price'] * (1 - df['discount_predicted'] / 100)).clip(
        lower=df['original_price'] * MIN_PRICE_RATIO
    )
    df['revenue'] = price * df['demand_estimated']
    return df


def find_optimal_discount(row: pd.Series, discount_grid: list = DISCOUNT_GRID) -> pd.Series:

    P0         = row['original_price']
    Q0         = max(row['purchased_last_month'], 1)
    elasticity = row['elasticity']
    d_base     = row.get('discount_percentage', 0)

    if elasticity < -1.5:
        max_discount = 50
    elif elasticity < -1.0:
        max_discount = 40
    else:
        max_discount = 25

    best_rev, best_d = -np.inf, 0

    for d in discount_grid:
        if d > max_discount:
            continue
        price = P0 * (1 - d / 100)
        if price < P0 * MIN_PRICE_RATIO:
            continue

        q   = Q0 * np.exp(elasticity * (-(d - d_base) / 100))
        rev = price * q
        if rev > best_rev:
            best_rev, best_d = rev, d

    rev_actual = row.get('revenue', P0 * Q0)

    return pd.Series({
        'optimal_discount':    best_d,
        'optimal_revenue':     best_rev,
        'optimal_price':       P0 * (1 - best_d / 100),
        'revenue_gain':        best_rev - rev_actual,
        'revenue_gain_pct':   (best_rev - rev_actual) / (rev_actual + 1e-9) * 100,
    })