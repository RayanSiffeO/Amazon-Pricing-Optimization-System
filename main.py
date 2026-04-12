import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.configuracion import AB_TEST_SIZE, DISCOUNT_GRID, DISCOUNT_LEVELS, DATA_PATH
from src.data.loader import cap_rare_categories, load_and_prepare
from src.elasticity.causal import compute_elasticity_regression, filter_valid_categories
from src.elasticity.ml_model import (
    assign_elasticity_from_model,
    estimate_elasticities_batch,
    train_elasticity_model,
)
from src.experiments.ab_test import registrar_ventas_real, run_ab_test, validate_ab_balance
from src.models.discount_model import train_model
from src.models.persistence import save_models
from src.optimization.analytic import compute_revenue, estimate_demand, find_optimal_discount
from src.optimization.knapsack import baseline_policy, knapsack_policy, solve_discount_knapsack
from src.reporting.plots import (
    plot_ab_response,
    plot_elasticity_bar,
    plot_results,
    plot_simulation_comparison,
    print_report,
)
from src.simulation.market import market_simulator


def main():
    # ── 1. Datos ──────────────────────────────────────────────────────────────
    print(">>> CARGANDO DATOS <<<")
    df = load_and_prepare(DATA_PATH)

    has_temporal = 'delivery_date' in df.columns
    if has_temporal:
        df = df.sort_values('delivery_date').reset_index(drop=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    df_train, df_test = cap_rare_categories(
        df_train, df_test,
        cols=['product_category', 'brand_group'],
        min_count=50,
    )


    cat_median = df_train.groupby('product_category')['original_price'].median()
    df_train['price_vs_category'] = df_train['original_price'] / df_train['product_category'].map(cat_median)
    df_test['price_vs_category']  = df_test['original_price']  / df_test['product_category'].map(cat_median)
    df_train['price_vs_category'] = df_train['price_vs_category'].fillna(1.0)
    df_test['price_vs_category']  = df_test['price_vs_category'].fillna(1.0)

    catalogo = df_train.set_index('product_id')

    # ── 2. A/B test → elasticidades causales ─────────────────────────────────
    print("\n>>> A/B TEST <<<")
    ab_results = run_ab_test(
        df_train, registrar_ventas_real, catalogo,
        test_size=AB_TEST_SIZE, discount_levels=DISCOUNT_LEVELS, random_state=42,
    )
    validate_ab_balance(ab_results)
    plot_ab_response(ab_results)
    results_dir = Path(__file__).parent / 'src' / 'data' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    ab_results.to_csv(results_dir / 'ab_results.csv', index=False)

    valid_cats        = filter_valid_categories(ab_results)
    causal_elasticity = compute_elasticity_regression(ab_results, valid_cats) if valid_cats else {}
    print("\n>>> ELASTICIDADES CAUSALES (con shrinkage) <<<")
    for cat, e in causal_elasticity.items():
        print(f"  {cat}: {e:.3f}")
    if causal_elasticity:
        plot_elasticity_bar(causal_elasticity)

    # ── 3. Meta-modelo XGBoost de elasticidad ─────────────────────────────────
    print("\n>>> META-MODELO DE ELASTICIDAD (XGBoost) <<<")
    booster, feature_names, ohe_columns = train_elasticity_model(ab_results, df_train)

    df_test = df_test.reset_index(drop=True)
    elasticity_array = estimate_elasticities_batch(
        df_test, booster, feature_names, ohe_columns, delta=1.0,
    )
    df_test = assign_elasticity_from_model(df_test, elasticity_array)

    # ── 4. Modelo de predicción de descuento ──────────────────────────────────
    print("\n>>> MODELO DE DESCUENTO (Stacking LGBM + RF) <<<")
    pipe, X_test, y_test_log, cv_results = train_model(df_train, df_test)
    y_pred_log = pipe.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_test     = np.expm1(y_test_log)

    # ── 5. Demanda y revenue ──────────────────────────────────────────────────
    df_work = df_test.copy()
    df_work['discount_percentage'] = y_test
    df_work['discount_predicted']  = y_pred
    df_work = estimate_demand(df_work)
    df_work = compute_revenue(df_work)

    # ── 6. Optimización analítica por producto ────────────────────────────────
    opt_results = df_work.apply(find_optimal_discount, axis=1)
    df_work     = df_work.reset_index(drop=True)
    df_work     = df_work.join(opt_results)

    # ── 7. Knapsack con restricción de presupuesto ────────────────────────────
    print("\n>>> OPTIMIZACIÓN KNAPSACK (presupuesto = 5 % valor catálogo) <<<")
    budget  = 0.05 * df_work['original_price'].sum()
    df_work = solve_discount_knapsack(df_work, DISCOUNT_GRID, budget, top_n=500)

    # ── 8. Reporte ────────────────────────────────────────────────────────────
    print_report(y_test, y_pred, cv_results, df_work)
    plot_results(y_test, y_pred, df_work)

    # ── 9. Backtest simulado ──────────────────────────────────────────────────
    print("\n>>> BACKTEST: Baseline vs Knapsack (60 periodos) <<<")
    sim_catalog = df_work.copy()

    sim_base = market_simulator(sim_catalog, baseline_policy, periods=60, competitors=2)
    sim_knap = market_simulator(sim_catalog, knapsack_policy,  periods=60, competitors=2)

    print(f"  Revenue baseline : ${sim_base['revenue'].sum():>15,.0f}")
    print(f"  Revenue knapsack : ${sim_knap['revenue'].sum():>15,.0f}")
    plot_simulation_comparison(sim_base, sim_knap)

    # ── 10. Persistencia ──────────────────────────────────────────────────────
    save_models(pipe, booster, feature_names, ohe_columns)


if __name__ == '__main__':
    main()
