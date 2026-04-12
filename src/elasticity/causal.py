from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.configuracion import GLOBAL_ELASTICITY, MIN_OBS_PER_CATEGORY, SHRINKAGE_STRENGTH, MIN_R2_FOR_ELASTICITY


def filter_valid_categories(
    ab_results: pd.DataFrame,
    min_obs: int = MIN_OBS_PER_CATEGORY,
) -> List[str]:
    """
    Devuelve categorías que tienen al menos `min_obs` observaciones
    en cada nivel de descuento del A/B test.
    """
    min_per_cat = (
        ab_results
        .groupby(['product_category', 'discount_applied'])
        .size()
        .reset_index(name='n')
        .groupby('product_category')['n']
        .min()
    )
    return min_per_cat[min_per_cat >= min_obs].index.tolist()


def compute_elasticity_regression(
    ab_results: pd.DataFrame,
    valid_cats: List[str],
) -> Dict[str, float]:

    elasticity_raw: Dict[str, float] = {}
    n_obs: Dict[str, int] = {}

    for cat in valid_cats:
        g = ab_results[ab_results['product_category'] == cat].copy()

        # Excluimos discount_applied == 0 (grupo control) de la regresión log-log
        g = g[(g['discount_applied'] > 0) & (g['discount_applied'] < 100)].copy()

        if len(g) < 20:
            elasticity_raw[cat] = GLOBAL_ELASTICITY
            n_obs[cat] = len(g)
            continue

        agg = (
            g.groupby('discount_applied')
            .agg(log_q=('units_sold', lambda x: np.log1p(x).median()))
            .reset_index()
        )
        agg['log_price_ratio'] = np.log(1 - agg['discount_applied'] / 100 + 1e-12)

        if len(agg) < 2:
            elasticity_raw[cat] = GLOBAL_ELASTICITY
            n_obs[cat] = len(g)
            continue

        X = agg[['log_price_ratio']].values
        y = agg['log_q'].values

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        # Si el R² es demasiado bajo, no hay señal real → usar global como fallback
        if r2 < MIN_R2_FOR_ELASTICITY:
            elasticity_raw[cat] = GLOBAL_ELASTICITY
            n_obs[cat] = SHRINKAGE_STRENGTH  # forzar shrinkage total
            continue

        # El coeficiente en log-log YA es la elasticidad
        eps_raw = float(model.coef_[0])
        elasticity_raw[cat] = float(np.clip(eps_raw, -5.0, -0.05))
        n_obs[cat] = len(g)

    # ── Shrinkage bayesiano hacia la media global ponderada ───────────────────
    total_obs  = sum(n_obs.values())
    global_avg = (
        sum(elasticity_raw[c] * n_obs[c] for c in valid_cats) / total_obs
        if total_obs > 0
        else GLOBAL_ELASTICITY
    )

    shrinkage_map: Dict[str, float] = {}
    for cat in valid_cats:
        n     = n_obs[cat]
        e_raw = elasticity_raw[cat]
        # Fórmula: ε_shrunk = (n·ε_raw + λ·ε_global) / (n + λ)
        e_shrunk = (n * e_raw + SHRINKAGE_STRENGTH * global_avg) / (n + SHRINKAGE_STRENGTH)
        shrinkage_map[cat] = float(np.clip(e_shrunk, -4.0, -0.05))

    return shrinkage_map