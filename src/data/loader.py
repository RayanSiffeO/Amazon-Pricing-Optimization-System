import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from src.configuracion import NUM_FEATURES, CAT_FEATURES, TOP_BRANDS
 
 
def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Lee el CSV, genera features derivadas y elimina filas incompletas.
    Retorna un DataFrame listo para train/test split.
    """
    df = pd.read_csv(path)
 
    # Garantizar columna product_id
    if 'product_id' not in df.columns:
        df = df.reset_index(drop=False).rename(columns={'index': 'product_id'})
 
    # Limpieza de texto
    df['title_clean'] = (
        df.get('product_title', pd.Series(['unknown product'] * len(df)))
        .fillna('unknown product')
        .str.lower()
    )
 
    # Features derivadas
    df['popularity']       = df['product_rating'] * np.log1p(df['total_reviews'])
    df['urgencia_de_venta'] = df['original_price'] / (df['purchased_last_month'] + 1)
    df['reviews_log']      = np.log1p(df['total_reviews'])
    # NOTA: price_vs_category se calcula en main.py DESPUÉS del train/test split
    # para evitar leakage (la mediana debe calcularse solo sobre df_train).
 
    # Temporalidad
    df['delivery_month'] = df['delivery_date'].astype(str).str[5:7]
    df['is_peak_season'] = df['delivery_month'].isin(['11', '12']).astype(int)
 
    # Brand group
    if 'brand_group' not in df.columns:
        df['brand_group'] = df['product_title'].apply(_detect_brand)
 
    # Eliminar filas con NaN en features o target.
    # price_vs_category se excluye aquí porque se calcula en main.py
    # después del split (para evitar leakage).
    cols_available = [c for c in NUM_FEATURES + CAT_FEATURES + ['discount_percentage']
                      if c != 'price_vs_category']
    df = df.dropna(subset=cols_available)
 
    return df
 
 
def cap_rare_categories(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list,
    min_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reemplaza categorías raras (< min_count en train) por 'Other'
    tanto en train como en test, para evitar categorías desconocidas en OHE.
    """
    for col in cols:
        counts = df_train[col].value_counts()
        rare   = set(counts[counts < min_count].index)
        df_train[col] = df_train[col].where(~df_train[col].isin(rare), 'Other')
        df_test[col]  = df_test[col].where(~df_test[col].isin(rare),  'Other')
    return df_train, df_test
 
 
# ── helpers privados ──────────────────────────────────────────────────────────
 
def _detect_brand(title: str) -> str:
    t = str(title).lower()
    for brand in TOP_BRANDS:
        if brand in t:
            return 'Top Global'
    return 'Other Brands'