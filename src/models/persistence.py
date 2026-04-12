

import json
import os
from datetime import datetime
from typing import List, Tuple

import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline


def save_models(
    pipe: Pipeline,
    booster: xgb.Booster,
    feature_names: List[str],
    ohe_columns: List[str],
    output_dir: str = 'models/saved',
) -> str:

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    joblib.dump(pipe, os.path.join(output_dir, f'pipeline_discount_{ts}.pkl'))
    booster.save_model(os.path.join(output_dir, f'xgb_elasticity_{ts}.json'))

    meta = {'feature_names': feature_names, 'ohe_columns': ohe_columns}
    with open(os.path.join(output_dir, f'feature_meta_{ts}.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n Modelos guardados en "{output_dir}" con timestamp: {ts}')
    return ts


def load_models(
    ts: str,
    output_dir: str = 'models/saved',
) -> Tuple[Pipeline, xgb.Booster, List[str], List[str]]:

    pipe = joblib.load(os.path.join(output_dir, f'pipeline_discount_{ts}.pkl'))

    booster = xgb.Booster()
    booster.load_model(os.path.join(output_dir, f'xgb_elasticity_{ts}.json'))

    with open(os.path.join(output_dir, f'feature_meta_{ts}.json')) as f:
        meta = json.load(f)

    return pipe, booster, meta['feature_names'], meta['ohe_columns']
