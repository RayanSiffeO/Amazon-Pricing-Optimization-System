# experiments/ab_test.py — Simulación de A/B test y validación de balance

import random
from typing import Callable, List

import numpy as np
import pandas as pd

from src.configuracion import AB_TEST_SIZE, DISCOUNT_LEVELS, REAL_ELASTICITY_BY_CATEGORY, GLOBAL_ELASTICITY




def registrar_ventas_real(
    product_id: int,
    descuento: int,
    catalogo: pd.DataFrame,
    np_rng: np.random.RandomState = None,
) -> float:
    """
    Simula unidades vendidas usando elasticidad real por categoría.
    `catalogo` debe tener product_id como índice.
    `np_rng` permite reproducibilidad; si es None usa el estado global.
    """
    producto       = catalogo.loc[product_id]
    cat            = producto['product_category']
    elasticidad    = REAL_ELASTICITY_BY_CATEGORY.get(cat, GLOBAL_ELASTICITY)
    demanda_base   = producto['purchased_last_month']
    efecto         = np.exp(elasticidad * (-descuento / 100))
    demanda_esp    = demanda_base * efecto
    rng = np_rng if np_rng is not None else np.random
    return max(0, rng.poisson(max(demanda_esp, 0)))




def run_ab_test(
    df_train: pd.DataFrame,
    registrar_ventas_fn: Callable,
    catalogo: pd.DataFrame,
    test_size: float = AB_TEST_SIZE,
    discount_levels: List[int] = DISCOUNT_LEVELS,
    random_state: int = 42,
) -> pd.DataFrame:

    rng    = random.Random(random_state)
    np_rng = np.random.RandomState(random_state)  
    muestra = df_train.sample(frac=test_size, random_state=random_state).copy()

    registros = []
    for _, row in muestra.iterrows():
        descuento  = rng.choice(discount_levels)
        units_sold = registrar_ventas_fn(row['product_id'], descuento, catalogo, np_rng)
        registros.append({
            'product_id':       row['product_id'],
            'product_category': row['product_category'],
            'original_price':   row['original_price'],
            'discount_applied': descuento,
            'units_sold':       units_sold,
        })

    return pd.DataFrame(registros)




def validate_ab_balance(ab_results: pd.DataFrame, min_obs: int = 10) -> None:
    """Imprime tabla de balance y alerta sobre celdas con pocas observaciones."""
    print("\n=== BALANCE DEL A/B TEST ===")
    pivot = (
        ab_results
        .groupby(['product_category', 'discount_applied'])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot)

    thin = (
        ab_results
        .groupby(['product_category', 'discount_applied'])
        .size()
        .reset_index(name='n')
    )
    problemas = thin[thin['n'] < min_obs]

    if not problemas.empty:
        print(f"\n Categorías con menos de {min_obs} observaciones:")
        print(problemas.to_string(index=False))
    else:
        print(f"\n✓ Balance correcto (≥{min_obs} obs por nivel)")