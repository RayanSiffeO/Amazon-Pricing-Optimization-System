# Pricing Optimization System — Causal Inference + ML + ILP

End-to-end system to estimate price elasticity and optimize discount allocation under budget constraints, applied to an Amazon product catalogue.

---

## In 30 Seconds

1. Simulate a multi-arm A/B pricing experiment (5 discount levels)
2. Estimate price elasticity per category using causal regression + Bayesian shrinkage
3. Generalize elasticity to individual products via an XGBoost meta-model
4. Predict each product's discount using a stacking ensemble (LGBM + RF → Ridge)
5. Allocate discounts under a 5%-of-catalogue budget using Integer Linear Programming
6. Backtest the strategy in a simulated market with seasonality and competitors

**Result: +61% revenue vs. baseline over 60 simulated periods.**

---

## Key Results

| Metric | Value |
|---|---|
| Discount model R² (test) | 0.681 |
| Discount model RMSE | 7.42 pp |
| CV R² (validation) | 0.730 |
| Overfitting gap | 0.164 |
| Revenue — baseline policy (60 periods) | $14,573,446,965 |
| Revenue — knapsack policy (60 periods) | $23,497,257,245 |
| **Revenue uplift (backtest)** | **+61.2%** |
| Analytic upside per-product (test set) | +$115,776,137 |

> ⚠️ The 61% uplift is measured in a simulated market using elasticities derived from a simulated A/B test. It represents an upper bound. Realistic uplift in production, accounting for demand noise, competitor adaptation, and cross-product substitution, is estimated at **+12% to +18%**.

---

## Problem

E-commerce pricing teams must decide which products to discount and by how much, under strict budget constraints. Doing this manually across thousands of SKUs is infeasible:

- Demand is uncertain and non-linearly price-sensitive
- Naive rules ("10% discount everywhere") ignore elasticity differences across categories
- Overspending on low-elasticity products sacrifices margin with little volume gain
- Underdiscounting high-elasticity products leaves revenue on the table

---

## Solution Architecture

```
Load & engineer features
        │
        ▼
Simulate A/B experiment ──► Causal elasticity (OLS log-log + shrinkage)
        │                             │
        ▼                             ▼
XGBoost elasticity meta-model ◄───────┘
        │
        ▼
Stacking discount predictor (LGBM + RF → Ridge)
        │
        ▼
Analytic per-product optimization (grid search)
        │
        ▼
Budget-constrained knapsack (ILP / PuLP)
        │
        ▼
Market simulation backtest (60 periods, 2 competitors)
```

---

## Project Structure

```
pricing_clean/
├── main.py                        # Pipeline orchestrator
└── src/
    ├── configuracion.py           # All hyperparameters and constants
    ├── data/
    │   ├── loader.py              # Feature engineering and cleaning
    │   └── results/               # Auto-generated CSVs and plots
    ├── elasticity/
    │   ├── causal.py              # OLS log-log + Bayesian shrinkage
    │   └── ml_model.py            # XGBoost elasticity meta-model
    ├── experiments/
    │   └── ab_test.py             # A/B test simulation
    ├── models/
    │   ├── discount_model.py      # Stacking pipeline (LGBM + RF + Ridge)
    │   └── persistence.py         # Model serialization
    ├── optimization/
    │   ├── analytic.py            # Per-product grid search
    │   └── knapsack.py            # ILP budget-constrained allocation
    ├── simulation/
    │   └── market.py              # Market backtest with seasonality
    ├── reporting/
    │   └── plots.py               # Matplotlib/Seaborn visualizations
    └── analysis/
        └── stadistical_analysis.R # Independent R elasticity analysis (3 routes)
```

---

## Pipeline — Step by Step

### 1. Feature Engineering (`src/data/loader.py`)

Derived features added on top of raw product metadata:

| Feature | Formula | Rationale |
|---|---|---|
| `popularity` | `rating × log(1 + reviews)` | Combines rating signal with review volume |
| `urgencia_de_venta` | `price / (sales_last_month + 1)` | High price + low sales = urgent to move |
| `reviews_log` | `log(1 + reviews)` | Reduces skew in review counts |
| `price_vs_category` | `price / category_median_price` | Relative positioning within category |
| `is_peak_season` | `delivery_month ∈ {11, 12}` | Holiday demand flag |
| `brand_group` | Rule-based on title | Top Global vs. Other Brands |

**Anti-leakage note:** `price_vs_category` median is computed on `df_train` only, then applied to `df_test`. Rare categories (< 50 obs in train) are collapsed to `"Other"` in both splits.

---

### 2. A/B Test Simulation (`src/experiments/ab_test.py`)

A multi-arm experiment is simulated on 20% of the training data. Each product is randomly assigned one of five discount levels: **[0%, 5%, 10%, 15%, 20%]**. Units sold are drawn from a Poisson distribution using category-specific true elasticities.

**Balance test result:** χ² = 61.96, df = 56, p = 0.272 ✓

Categories with < 10 observations per arm (Gaming, Smart Home, Wearables) are flagged and handled via shrinkage rather than excluded.

A/B balance by category:

```
discount_applied      0    5   10   15   20
Cameras              67   71   70   77   76
Laptops             140  130  125  132  145
Other Electronics   125   97  135  129  124
Phones              121  140  128  110  117
...
Gaming                2    5    3    4    4   ← thin
Smart Home            5    5    7    6    5   ← thin
Wearables             6   13    6    7    2   ← thin
```

---

### 3. Causal Elasticity Estimation (`src/elasticity/causal.py`)

**Model:** log-log OLS — `median(log(Q)) ~ log(1 − d/100)` — aggregated by discount level to reduce Poisson noise.

**Filters applied:**
- Minimum 30 observations per category
- R² ≥ 0.05 required to use category-specific estimate (else falls back to global)
- Elasticity clipped to [−5.0, −0.05]

**Bayesian shrinkage** toward the precision-weighted global prior (λ = 150 pseudo-observations):

```
ε_shrunk = (n × ε_raw + λ × ε_global) / (n + λ)
```

**Causal elasticities after shrinkage (Python pipeline):**

| Category | Elasticity |
|---|---|
| Phones | −1.533 |
| Power & Batteries | −1.216 |
| Other Electronics | −1.151 |
| Cameras | −0.935 |
| Laptops | −0.707 |

Only categories with sufficient sample and signal pass the R² filter. The rest use the global fallback.

---

### 4. XGBoost Elasticity Meta-model (`src/elasticity/ml_model.py`)

To propagate elasticity to every individual product (not just per category), an XGBoost model is trained on the A/B results:

- **Target:** `log(1 + units_sold)`
- **Features:** product metadata + discount level + interaction terms (`discount × feature` for all numeric features)
- **Elasticity extraction:** finite differences in log-log space — `Δlog(Q) / Δlog(P)` between `d` and `d + 1`
- **OHE vocabulary** fixed at training time to prevent unseen categories at inference

---

### 5. Discount Prediction Model (`src/models/discount_model.py`)

**Architecture:** stacking ensemble

```
Base learners:
  ├── LightGBM  (objective: tweedie, 600 trees, lr=0.03)
  └── Random Forest  (150 trees, max_depth=10)
Meta-learner:
  └── Ridge  (α=1.0)
```

**Preprocessing pipeline:**
- Numeric features → median imputation
- Categorical features → OHE (min_frequency=20, infrequent_if_exist)
- Product title → TF-IDF (top 50 unigrams + bigrams)

**Target:** `log1p(discount_percentage)` — log-transforms the right-skewed discount distribution.

**Validation:** TimeSeriesSplit with 5 folds.

**Results:**

```
R² (test)       : 0.681
RMSE (test)     : 7.42 pp
R² (CV train)   : 0.895
R² (CV val)     : 0.730
Overfitting gap : 0.164
```

---

### 6. Analytic Optimization (`src/optimization/analytic.py`)

Per-product grid search over `DISCOUNT_GRID = [0, 5, 10, ..., 70]%` with elasticity-aware caps:

| Elasticity | Max discount allowed |
|---|---|
| ε < −1.5 | 50% |
| −1.5 ≤ ε < −1.0 | 40% |
| ε ≥ −1.0 | 25% |

**Demand model:** `Q(d) = Q₀ × exp(ε × (−(d − d_base) / 100))`

**Revenue model:** `R(d) = P₀ × (1 − d/100) × Q(d)` with floor at 50% of original price.

**Analytic upside on test set:** +$115,776,137 vs. predicted-discount revenue of $271,732,885.

---

### 7. Knapsack ILP (`src/optimization/knapsack.py`)

**Problem formulation:**

```
Maximize:   Σᵢ Σd  revenue(i, d) × x(i,d)
Subject to:
  Σd  x(i,d) = 1                          ∀i   (one discount per product)
  Σᵢ Σd  (d/100 × P₀ᵢ) × x(i,d) ≤ B           (budget constraint)
  x(i,d) ∈ {0, 1}
```

- **Solver:** PuLP / CBC
- **Budget:** 5% of total catalogue value
- **Scope:** top 500 products by revenue (tail uses predicted discount)

---

### 8. Market Simulation Backtest (`src/simulation/market.py`)

60-period simulation with:
- **Seasonality:** `season(t) = 1 + 0.2 × sin(2π × t/30)` (monthly cycle)
- **Competitor pressure:** 2 competitors apply random ±5% price noise; each competitor priced below yours reduces your demand by 10%
- **Demand:** Poisson draws from `expected = Q₀ × season × exp(ε × (−d/100)) × competitor_factor`

**Backtest results:**

| Policy | Revenue (60 periods) | vs Baseline |
|---|---|---|
| Baseline (historical discounts) | $14,573,446,965 | — |
| Knapsack (ILP optimized) | $23,497,257,245 | **+61.2%** |

---

## Elasticity Analysis in R (`src/analysis/stadistical_analysis.R`)

A parallel elasticity study was conducted in R using three independent methods to cross-validate the Python estimates:

| Route | Method | Key feature |
|---|---|---|
| A | OLS log-log (transaction level) | Direct, fast |
| B | Jitter simulation + bootstrap CI | Continuous price approximation |
| C | WLS on aggregated bins | Noise-robust, few points |

Results are combined via Bayesian shrinkage. Categories with anomalous or sign-unstable estimates (TV & Display, Other Electronics) are flagged and heavily shrunk toward the prior.

**Global elasticity (HC3 robust errors):** ε = −1.543 (SE = 0.353), 95% CI: [−2.235, −0.851]

---

## Validation

- **Anti-leakage:** category medians computed on train only; OHE vocabulary fixed at train time
- **Cross-validation:** TimeSeriesSplit (5 folds) for the stacking model
- **Elasticity triangulation:** 3 independent estimation routes in R
- **GAM test:** non-linearity check per category — edf ≈ 1 for all valid categories → log-log linearity justified
- **A/B balance:** Chi-squared test confirms balanced assignment (p = 0.272)

---

## Limitations

**Statistical:**
- The A/B test is simulated, not a real randomized experiment. Elasticities are derived from synthetic data with pre-specified true values — they cannot be validated against real consumer behavior.
- Most categories show R² < 0.10 in the elasticity regressions. Discrete pricing levels (only 5 arms) severely limit regression signal.
- Gaming, Smart Home, and Wearables have insufficient sample sizes and are dominated by the prior after shrinkage.

**System:**
- No cross-product demand effects. Products are treated as independent.
- Knapsack covers only top 500 products; the tail is left on the predicted discount.
- Market simulator assumes a fixed competitor model (2 competitors, 10% demand penalty each). Real competitive dynamics are more complex.
- No inventory, margin, or MAP (Minimum Advertised Price) constraints.
- `DATA_PATH` in `configuracion.py` is hardcoded to a local path — must be updated before running.

---

## Reproducing Results

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm pulp matplotlib seaborn
```

For the R elasticity analysis:

```r
install.packages(c("tidyverse", "mgcv", "sandwich", "lmtest", "broom"))
```

### Configuration

All hyperparameters are centralized in `src/configuracion.py`. The only required change before running:

```python
# src/configuracion.py
DATA_PATH = "path/to/amazon_products_sales_data_cleaned.csv"
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `AB_TEST_SIZE` | 0.20 | Fraction of train used in A/B simulation |
| `DISCOUNT_LEVELS` | [0,5,10,15,20] | Discount arms (%) |
| `DISCOUNT_GRID` | [0,5,...,70] | Grid for optimization |
| `SHRINKAGE_STRENGTH` | 150 | Bayesian prior weight (pseudo-obs) |
| `MIN_R2_FOR_ELASTICITY` | 0.05 | Minimum R² to accept category estimate |
| `GLOBAL_ELASTICITY` | −1.2 | Python fallback elasticity |
| `MIN_PRICE_RATIO` | 0.50 | Price floor (50% of original) |

### Run

```bash
# Python pipeline (full system)
python main.py

# R elasticity analysis (optional, independent)
Rscript src/analysis/stadistical_analysis.R
```

All outputs are written to `src/data/results/`: `ab_results.csv`, `elasticidades_finales_v2.csv`, and six diagnostic plots.

All random seeds are fixed (`random_state=42`, `np.random.RandomState(42)`, market simulator `seed=42`). Results are fully deterministic.

---

## Tech Stack

Python · XGBoost · LightGBM · scikit-learn · PuLP/CBC · R (mgcv · sandwich · broom)

---

## Future Work

- Replace simulated A/B test with real randomized pricing experiment on a product subset
- Model cross-product elasticity (substitution and complementarity between categories)
- Extend knapsack to full catalogue using Lagrangian relaxation or greedy heuristic
- Bayesian hierarchical elasticity model (Stan/PyMC) to replace manual shrinkage formula
- Real-time inference endpoint: given product list + budget → return optimal discounts
