# models/discount_model.py — Pipeline de predicción de descuento óptimo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

from src.configuracion import CAT_FEATURES, NUM_FEATURES


def build_pipeline() -> Pipeline:

    preprocessor = ColumnTransformer([
        (
            'num',
            SimpleImputer(strategy='median'),
            NUM_FEATURES,
        ),
        (
            'cat',
            Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(
                    handle_unknown='infrequent_if_exist',
                    min_frequency=20,
                    sparse_output=False,
                )),
            ]),
            CAT_FEATURES,
        ),
        (
            'txt',
            TfidfVectorizer(
                max_features=50,
                ngram_range=(1, 2),
                stop_words='english',
            ),
            'title_clean',
        ),
    ])

    lgbm = LGBMRegressor(
        objective='tweedie',
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        min_child_samples=10,
        reg_lambda=0.1,
    )
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        max_samples=0.8,
    )
    stack = StackingRegressor(
        estimators=[('lgbm', lgbm), ('rf', rf)],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
    )
    return Pipeline([('prep', preprocessor), ('model', stack)])


def train_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:

    feat_cols = NUM_FEATURES + CAT_FEATURES + ['title_clean']

    X_train = df_train[feat_cols]
    y_train = np.log1p(df_train['discount_percentage'])
    X_test  = df_test[feat_cols]
    y_test  = np.log1p(df_test['discount_percentage'])

    pipe = build_pipeline()

    cv     = TimeSeriesSplit(n_splits=5)
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring='r2',
        return_train_score=True,
    )

    pipe.fit(X_train, y_train)

    cv_results = {
        'train_r2': float(scores['train_score'].mean()),
        'val_r2':   float(scores['test_score'].mean()),
        'gap':      float(scores['train_score'].mean() - scores['test_score'].mean()),
    }

    return pipe, X_test, y_test, cv_results
