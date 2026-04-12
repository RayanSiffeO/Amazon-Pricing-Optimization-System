

from typing import Callable

import numpy as np
import pandas as pd
from src.configuracion import GLOBAL_ELASTICITY


def market_simulator(
    df_catalog: pd.DataFrame,
    policy_fn: Callable[[pd.DataFrame, int], dict],
    periods: int = 90,
    competitors: int = 2,
    seed: int = 42,
) -> pd.DataFrame:

    rng     = np.random.RandomState(seed)
    records = []

    for t in range(periods):
        season    = 1.0 + 0.2 * np.sin(2 * np.pi * (t / 30))
        discounts = policy_fn(df_catalog, t)

        df_t = df_catalog.copy()
        df_t['_d']       = df_t['product_id'].map(discounts).fillna(df_t.get('discount_percentage', 0))
        df_t['_price']   = df_t['original_price'] * (1 - df_t['_d'] / 100)
        df_t['_eps']     = df_t.get('elasticity', GLOBAL_ELASTICITY) if 'elasticity' in df_t.columns \
                           else GLOBAL_ELASTICITY
        df_t['_Q0']      = (df_t['purchased_last_month'].clip(lower=1)) * season

        df_t['_d_base']  = 0
        df_t['_expected'] = df_t['_Q0'] * np.exp(df_t['_eps'] * (-(df_t['_d'] - df_t['_d_base']) / 100))


        comp_pressure = np.ones(len(df_t))
        for _ in range(competitors):
            noise = rng.normal(0, 0.05, size=len(df_t))
            comp_price = df_t['_price'].values * (1 + noise)
            comp_pressure *= np.where(comp_price < df_t['_price'].values, 0.9, 1.0)

        expected = (df_t['_expected'] * comp_pressure).clip(lower=0.01)
        sold     = rng.poisson(expected)

        period_df = pd.DataFrame({
            'period':     t,
            'product_id': df_t['product_id'].values,
            'price':      df_t['_price'].values,
            'discount':   df_t['_d'].values,
            'units_sold': sold,
            'revenue':    df_t['_price'].values * sold,
        })
        records.append(period_df)

    return pd.concat(records, ignore_index=True)