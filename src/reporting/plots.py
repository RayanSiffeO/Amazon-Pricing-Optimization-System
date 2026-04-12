

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score


# ── Plots A/B test ────────────────────────────────────────────────────────────

def plot_ab_response(ab_results: pd.DataFrame) -> None:

    agg = (
        ab_results
        .groupby(['product_category', 'discount_applied'])['units_sold']
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=agg, x='discount_applied', y='units_sold',
                 hue='product_category', marker='o')
    plt.title('Respuesta de ventas al descuento (A/B test)')
    plt.xlabel('Descuento aplicado (%)')
    plt.ylabel('Unidades vendidas (media)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── Plots elasticidad ─────────────────────────────────────────────────────────

def plot_elasticity_bar(elasticity_map: Dict[str, float]) -> None:

    cats = list(elasticity_map.keys())
    vals = list(elasticity_map.values())
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cats, y=vals, palette='coolwarm')
    plt.axhline(-1, linestyle='--', color='red', label='Elasticidad unitaria (−1)')
    plt.title('Elasticidad precio-demanda por categoría (con shrinkage)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Elasticidad')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Plots modelo de descuento ─────────────────────────────────────────────────

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_work: pd.DataFrame,
) -> None:


    # Residuals
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, y_true - y_pred, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals del modelo de descuento')
    plt.xlabel('Descuento predicho (%)')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.show()


    bins   = [0, 10, 20, 30, 40, 50, 100]
    labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
    y_true_bin = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    y_pred_bin = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (rangos de descuento)')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    # Distribución de revenue
    plt.figure(figsize=(10, 5))
    sns.histplot(df_work['revenue'], bins=50, kde=True)
    plt.title('Distribución de Revenue estimado (Test set)')
    plt.tight_layout()
    plt.show()


# ── Plots simulación ──────────────────────────────────────────────────────────

def plot_simulation_comparison(
    sim_baseline: pd.DataFrame,
    sim_knap: pd.DataFrame,
) -> None:
    """Compara revenue por periodo entre política baseline y knapsack."""
    rev_summary = pd.DataFrame({
        'baseline': sim_baseline.groupby('period')['revenue'].sum(),
        'knapsack': sim_knap.groupby('period')['revenue'].sum(),
    }).reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rev_summary, x='period', y='baseline', label='Baseline')
    sns.lineplot(data=rev_summary, x='period', y='knapsack',  label='Knapsack')
    plt.title('Revenue por periodo: Baseline vs Knapsack')
    plt.xlabel('Periodo')
    plt.ylabel('Revenue')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Reporte textual ───────────────────────────────────────────────────────────

def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_results: dict,
    df_work: pd.DataFrame,
) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*55}")
    print("  RENDIMIENTO DEL MODELO (Test set)")
    print(f"{'='*55}")
    print(f"  R² Score      : {r2:.4f}")
    print(f"  RMSE          : {rmse:.2f} %")
    print(f"  Train R² (CV) : {cv_results['train_r2']:.4f}")
    print(f"  Val R²  (CV)  : {cv_results['val_r2']:.4f}")
    print(f"  Overfitting   : {cv_results['gap']:.4f}")

    rev_pred = df_work['revenue'].sum()
    print(f"\n  Revenue con descuentos predichos : ${rev_pred:>15,.0f}")

    if 'optimal_revenue' in df_work.columns:
        rev_opt = df_work['optimal_revenue'].sum()
        print(f"  Revenue óptimo (por producto)   : ${rev_opt:>15,.0f}")
        print(f"  Upside potencial                : ${rev_opt - rev_pred:>15,.0f}")
    print(f"{'='*55}\n")
