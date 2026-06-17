

import os
from pathlib import Path
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "src", "data", "raw", "amazon_products_sales_data_cleaned.csv")

# ── Features ─────────────────────────────────────────────────────────────────
NUM_FEATURES = [
    'product_rating', 'total_reviews', 'purchased_last_month',
    'original_price', 'popularity', 'urgencia_de_venta' , 'reviews_log',
]
CAT_FEATURES = [
    'is_sponsored', 'has_coupon', 'buy_box_availability',
    'brand_group', 'product_category', 'delivery_month', 'is_peak_season',
]


MIN_PRICE_RATIO = 0.5       
DISCOUNT_GRID   = list(range(0, 71, 5)) 


DISCOUNT_LEVELS = [0, 5, 10, 15, 20, 30, 40]
AB_TEST_SIZE    = 0.2

MIN_OBS_PER_CATEGORY = 30
GLOBAL_ELASTICITY    = -1.2
SHRINKAGE_STRENGTH      = 50  
MIN_R2_FOR_ELASTICITY   = 0.15 

REAL_ELASTICITY_BY_CATEGORY = {
    'Electronics': -1.8,
    'Clothing':    -2.5,
    'Books':       -0.9,
    'Home':        -1.2,
    'Toys':        -2.0,
    'Other':       -1.0,
}


TOP_BRANDS = [
    'apple', 'samsung', 'sony', 'dji', 'seagate', 'hp', 'razer', 'logitech',
    'anker', 'canon', 'epson', 'fujifilm', 'bose', 'garmin', 'nintendo',
    'acer', 'asus', 'microsoft', 'xiaomi', 'oneplus', 'nike', 'adidas',
    'panasonic', 'gopro', 'intel', 'amd', 'lg',
]
