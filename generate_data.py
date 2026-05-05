"""
generate_data.py  –  Creates retail_sales_data.csv
Features: Date, Product_ID, Store_ID, Units_Sold, Revenue, Promotion, Holiday
Realistic patterns: seasonality, promotions lift, holiday spikes, store/product variance
"""
import numpy as np
import pandas as pd
import os

def generate_retail_data(n_days=730, seed=42):
    rng = np.random.default_rng(seed)

    # ── Calendar ────────────────────────────────────────────────────────────
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    # ── Products & Stores ───────────────────────────────────────────────────
    products = {
        "P001": {"name": "Electronics",  "base_price": 4500, "base_units": 18},
        "P002": {"name": "Clothing",      "base_price": 1200, "base_units": 45},
        "P003": {"name": "Grocery",       "base_price":  350, "base_units":180},
        "P004": {"name": "Furniture",     "base_price": 8500, "base_units":  8},
        "P005": {"name": "Sports",        "base_price": 2200, "base_units": 30},
    }
    stores = {
        "S01": {"city": "Mumbai",    "multiplier": 1.35},
        "S02": {"city": "Delhi",     "multiplier": 1.28},
        "S03": {"city": "Bangalore", "multiplier": 1.20},
        "S04": {"city": "Chennai",   "multiplier": 1.05},
        "S05": {"city": "Hyderabad", "multiplier": 1.10},
        "S06": {"city": "Pune",      "multiplier": 0.95},
    }

    # ── Indian holidays 2022–2023 ────────────────────────────────────────────
    holiday_dates = {
        "2022-01-01","2022-01-14","2022-01-26","2022-03-18","2022-04-14",
        "2022-04-15","2022-05-03","2022-08-15","2022-09-05","2022-10-02",
        "2022-10-05","2022-10-24","2022-11-08","2022-12-25","2022-12-31",
        "2023-01-01","2023-01-14","2023-01-26","2023-03-08","2023-04-04",
        "2023-04-14","2023-04-22","2023-05-05","2023-08-15","2023-09-19",
        "2023-10-02","2023-10-24","2023-11-13","2023-11-27","2023-12-25",
    }

    rows = []
    for date in dates:
        date_str = str(date.date())
        is_holiday = int(date_str in holiday_dates)
        is_weekend = int(date.dayofweek >= 5)
        month = date.month

        # Seasonal index (peak: Oct–Dec, dip: Jan–Feb)
        seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * (month - 3) / 12)

        # Annual trend growth
        t_idx = (date - dates[0]).days
        trend = 1.0 + 0.0004 * t_idx  # ~15% over 2 years

        for pid, pinfo in products.items():
            for sid, sinfo in stores.items():

                # Promotion: ~20% chance on weekdays, 35% on weekends/holidays
                promo_prob = 0.35 if (is_weekend or is_holiday) else 0.20
                promotion = int(rng.random() < promo_prob)
                discount  = rng.uniform(5, 30) * promotion  # 0 if no promo

                # Units sold
                base_u  = pinfo["base_units"]
                promo_lift   = 1.0 + (0.60 * promotion)     # +60% with promo
                holiday_lift = 1.0 + (0.45 * is_holiday)    # +45% on holidays
                weekend_lift = 1.0 + (0.15 * is_weekend)

                # Noise per product–store–day
                noise = rng.normal(1.0, 0.12)

                units_sold = int(max(1, round(
                    base_u
                    * seasonal * trend
                    * sinfo["multiplier"]
                    * promo_lift * holiday_lift * weekend_lift
                    * noise
                )))

                # Revenue (price adjusted for discount)
                effective_price = pinfo["base_price"] * (1 - discount / 100)
                revenue = round(units_sold * effective_price, 2)

                rows.append({
                    "Date":        date_str,
                    "Product_ID":  pid,
                    "Store_ID":    sid,
                    "Units_Sold":  units_sold,
                    "Revenue":     revenue,
                    "Promotion":   promotion,
                    "Holiday":     is_holiday,
                })

    df = pd.DataFrame(rows)
    print(f"Generated: {len(df):,} rows  ×  {df.shape[1]} columns")
    print(df.dtypes)
    print(df.describe())
    return df


if __name__ == "__main__":
    df = generate_retail_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/retail_sales_data.csv", index=False)
    print("\nSaved → data/retail_sales_data.csv")
    print(df.head(8).to_string())
