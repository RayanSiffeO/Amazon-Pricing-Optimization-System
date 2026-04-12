

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.configuracion import (
    CAT_FEATURES,
    GLOBAL_ELASTICITY,
    NUM_FEATURES,
)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_elasticity_model(
    ab_results: pd.DataFrame,
    catalog: pd.DataFrame,
) -> Tuple[xgb.Booster, List[str], List[str]]:
 
    catalog_reset = catalog.reset_index()


    conflict = [
        c for c in catalog_reset.columns
        if c in ab_results.columns and c != 'product_id'
    ]
    catalog_safe = catalog_reset.drop(columns=conflict, errors='ignore')

    df = ab_results.merge(catalog_safe, on='product_id', how='left')


    for f in NUM_FEATURES:
        df[f'disc_x_{f}'] = df['discount_applied'] * df[f]

    df['y'] = np.log1p(df['units_sold'])


    cat_dummies = pd.get_dummies(
        df[CAT_FEATURES].astype(str),
        prefix=CAT_FEATURES,
        drop_first=True,
    )
    ohe_columns = cat_dummies.columns.tolist()  

    base_cols    = ['discount_applied'] + NUM_FEATURES + [f'disc_x_{f}' for f in NUM_FEATURES]
    X            = pd.concat([df[base_cols].reset_index(drop=True),
                               cat_dummies.reset_index(drop=True)], axis=1)
    y            = df['y'].values
    feature_names = X.columns.tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective':       'reg:squarederror',
        'eta':             0.05,
        'max_depth':       6,
        'subsample':       0.8,
        'colsample_bytree': 0.6,
        'seed':            42,
        'nthread': 1,         
        'tree_method': 'hist',
    }
    booster = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster, feature_names, ohe_columns



def predict_logq_batch(
    model: xgb.Booster,
    feature_names: List[str],
    ohe_columns: List[str],
    catalog_df: pd.DataFrame,
    discount_values: List[float],
) -> Dict[float, np.ndarray]:

    n_products = len(catalog_df)
    rows: List[dict] = []

    for d in discount_values:
        for idx in range(n_products):
            prod = catalog_df.iloc[idx]
            row  = {'discount_applied': d}
            for f in NUM_FEATURES:
                val        = prod[f]
                row[f]     = val
                row[f'disc_x_{f}'] = d * val
            for c in CAT_FEATURES:
                row[c] = str(prod[c])
            rows.append(row)

    X_df = pd.DataFrame(rows)


    cat_dummies = pd.get_dummies(
        X_df[CAT_FEATURES].astype(str),
        prefix=CAT_FEATURES,
        drop_first=True,
    )
    base_cols = ['discount_applied'] + NUM_FEATURES + [f'disc_x_{f}' for f in NUM_FEATURES]
    X_full    = pd.concat([X_df[base_cols].reset_index(drop=True),
                            cat_dummies.reset_index(drop=True)], axis=1)


    X_full = X_full.reindex(columns=feature_names, fill_value=0)
    preds  = model.predict(xgb.DMatrix(X_full))

    result: Dict[float, np.ndarray] = {}
    for i, d in enumerate(discount_values):
        result[d] = preds[i * n_products: (i + 1) * n_products]

    return result




def estimate_elasticities_batch(
    df_catalog: pd.DataFrame,
    model: xgb.Booster,
    feature_names: List[str],
    ohe_columns: List[str],
    delta: float = 1.0,
) -> np.ndarray:

    df = df_catalog.reset_index(drop=True).copy()
    base_discounts = df['discount_percentage'].fillna(0).clip(0, 69).values

    # Descuentos únicos que necesitamos predecir
    unique_d0 = np.unique(base_discounts)
    to_predict = sorted({d for d0 in unique_d0 for d in (d0, min(d0 + delta, 70.0))})

    preds_map = predict_logq_batch(model, feature_names, ohe_columns, df, to_predict)

    eps_list: List[float] = []
    for idx, row in df.iterrows():
        d0 = float(row['discount_percentage']) if not pd.isna(row['discount_percentage']) else 0.0
        d0 = min(d0, 69.0)
        d1 = min(d0 + delta, 70.0)
        
        
        logq0 = preds_map[d0][idx]
        logq1 = preds_map[d1][idx]

        log_p0 = np.log(1 - d0 / 100 + 1e-12)
        log_p1 = np.log(1 - d1 / 100 + 1e-12)
        delta_log_p = log_p1 - log_p0

        if abs(delta_log_p) < 1e-8:
            eps = GLOBAL_ELASTICITY
        else:
            eps = (logq1 - logq0) / delta_log_p

        eps_list.append(float(np.clip(eps, -5.0, -0.05)))

    return np.array(eps_list)


def assign_elasticity_from_model(
    df: pd.DataFrame,
    elasticity_array: np.ndarray,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df['elasticity'] = elasticity_array
    df['elasticity'] = df['elasticity'].fillna(GLOBAL_ELASTICITY)
    return df
