"""
train_model.py  –  Retail Sales Ridge Regression (v2)
Targets: Revenue per product-store-day
Features: temporal + lag/rolling + promo/holiday + product/store encodings
Run: python train_model.py
"""

import os, warnings, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = "data/retail_sales_data.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("RETAIL SALES MODEL TRAINING  (v2)")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
print(f"Loaded  {len(df):,} rows  ×  {df.shape[1]} columns")
print(df.dtypes, "\n")

# ── 2. Aggregate daily total revenue (all products × all stores) ──────────────
# Also preserve daily averages of features for the daily model
daily = (df.groupby("Date").agg(
    Revenue   = ("Revenue",   "sum"),
    Units_Sold= ("Units_Sold","sum"),
    Promotion = ("Promotion", "mean"),  # promo intensity 0-1
    Holiday   = ("Holiday",   "max"),
).reset_index().sort_values("Date").reset_index(drop=True))

print(f"Daily aggregated: {len(daily)} rows")
print(daily.describe(), "\n")

# ── 3. Feature engineering ────────────────────────────────────────────────────
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    # Temporal
    d["year"]           = d["Date"].dt.year
    d["month"]          = d["Date"].dt.month
    d["day"]            = d["Date"].dt.day
    d["dayofweek"]      = d["Date"].dt.dayofweek
    d["weekofyear"]     = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"]        = d["Date"].dt.quarter
    d["is_weekend"]     = (d["dayofweek"] >= 5).astype(int)
    d["is_month_end"]   = d["Date"].dt.is_month_end.astype(int)
    d["is_month_start"] = d["Date"].dt.is_month_start.astype(int)

    # Cyclical encoding (no ordinal gap)
    d["month_sin"]  = np.sin(2 * np.pi * d["month"]     / 12)
    d["month_cos"]  = np.cos(2 * np.pi * d["month"]     / 12)
    d["dow_sin"]    = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"]    = np.cos(2 * np.pi * d["dayofweek"] / 7)
    d["woy_sin"]    = np.sin(2 * np.pi * d["weekofyear"]/ 52)
    d["woy_cos"]    = np.cos(2 * np.pi * d["weekofyear"]/ 52)

    # Promo & Holiday interaction
    d["promo_x_holiday"]   = d["Promotion"] * d["Holiday"]
    d["promo_x_weekend"]   = d["Promotion"] * d["is_weekend"]
    d["holiday_x_weekend"] = d["Holiday"]   * d["is_weekend"]

    # Lag features on Revenue
    for lag in [1, 3, 7, 14, 21, 28]:
        d[f"lag_{lag}"] = d["Revenue"].shift(lag)

    # Rolling averages (use shift(1) to avoid leakage)
    for w in [7, 14, 30]:
        d[f"rolling_mean_{w}"] = d["Revenue"].shift(1).rolling(w, min_periods=1).mean()
        d[f"rolling_std_{w}"]  = d["Revenue"].shift(1).rolling(w, min_periods=1).std().fillna(0)

    # Rolling max/min
    d["rolling_max_7"]  = d["Revenue"].shift(1).rolling(7,  min_periods=1).max()
    d["rolling_min_7"]  = d["Revenue"].shift(1).rolling(7,  min_periods=1).min()
    d["rolling_max_30"] = d["Revenue"].shift(1).rolling(30, min_periods=1).max()

    # Exponential weighted mean
    d["ewm_14"] = d["Revenue"].shift(1).ewm(span=14, min_periods=1).mean()
    d["ewm_30"] = d["Revenue"].shift(1).ewm(span=30, min_periods=1).mean()

    # Units lag
    d["units_lag_7"]      = d["Units_Sold"].shift(7)
    d["units_rolling_7"]  = d["Units_Sold"].shift(1).rolling(7,  min_periods=1).mean()

    # Linear time index
    d["t"] = np.arange(len(d))

    return d

daily = engineer_features(daily)
daily = daily.dropna().reset_index(drop=True)
print(f"After feature engineering + dropna: {len(daily)} rows")

# ── 4. Feature columns ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Temporal
    "year","month","day","dayofweek","weekofyear","quarter",
    "is_weekend","is_month_end","is_month_start",
    # Cyclical
    "month_sin","month_cos","dow_sin","dow_cos","woy_sin","woy_cos",
    # Business flags
    "Promotion","Holiday","promo_x_holiday","promo_x_weekend","holiday_x_weekend",
    # Revenue lags
    "lag_1","lag_3","lag_7","lag_14","lag_21","lag_28",
    # Rolling revenue
    "rolling_mean_7","rolling_mean_14","rolling_mean_30",
    "rolling_std_7","rolling_std_14","rolling_std_30",
    "rolling_max_7","rolling_min_7","rolling_max_30",
    # EWM
    "ewm_14","ewm_30",
    # Units
    "units_lag_7","units_rolling_7",
    # Trend
    "t",
]

TARGET = "Revenue"

X = daily[FEATURE_COLS]
y = daily[TARGET]

print(f"\nFeatures: {len(FEATURE_COLS)}  |  Samples: {len(X)}")
print(f"Target range: ₹{y.min():,.0f} – ₹{y.max():,.0f}  |  Mean: ₹{y.mean():,.0f}")

# ── 5. Train / Test split (temporal 80/20) ────────────────────────────────────
split = int(len(daily) * 0.80)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]
print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

# ── 6. Pipeline: StandardScaler + Ridge with GridSearch ──────────────────────
tscv   = TimeSeriesSplit(n_splits=5)
alphas = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 200, 500, 1000]
pipe   = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])

gs = GridSearchCV(
    pipe,
    {"ridge__alpha": alphas},
    cv=tscv, scoring="r2", n_jobs=-1, refit=True
)
gs.fit(X_tr, y_tr)

best_model = gs.best_estimator_
best_alpha = gs.best_params_["ridge__alpha"]
best_cv_r2 = gs.best_score_
print(f"\nGridSearch best alpha : {best_alpha}")
print(f"GridSearch CV R²      : {best_cv_r2:.4f}")

# ── 7. Metrics ────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    safe = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / safe)) * 100
    print(f"  {label:<8} MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

print("\nMetrics:")
tr_m = metrics(y_tr, best_model.predict(X_tr), "Train")
te_m = metrics(y_te, best_model.predict(X_te), "Test")

gap = tr_m["r2"] - te_m["r2"]
flag = "✅ Healthy (no overfit)" if gap < 0.08 else "⚠ Overfit — increase alpha"
print(f"\n  R² gap (train–test) : {gap:.4f}  →  {flag}")

# ── 8. Cross-validation stability ─────────────────────────────────────────────
cv_r2 = cross_val_score(best_model, X, y, cv=tscv, scoring="r2")
print(f"\n  CV R² splits : {np.round(cv_r2,4)}")
print(f"  Mean ± Std   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ── 9. Generate & save charts ─────────────────────────────────────────────────
import io, base64
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=115)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

charts = {}

# 9a. Actual vs Predicted
y_pred_all = best_model.predict(X)
y_pred_te  = best_model.predict(X_te)
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(daily["Date"].iloc[:split], y_tr,        color="#1565C0", lw=1.4, label="Actual (train)")
ax.plot(daily["Date"].iloc[split:], y_te,        color="#1565C0", lw=1.4, ls="-", alpha=.5, label="Actual (test)")
ax.plot(daily["Date"].iloc[split:], y_pred_te,   color="#E53935", lw=1.6, ls="--", label="Predicted")
ax.axvline(daily["Date"].iloc[split], color="#999", ls=":", lw=1.2, label="Train/Test split")
ax.set(title="Actual vs Predicted Daily Revenue", xlabel="Date", ylabel="Revenue (₹)")
ax.legend(fontsize=9); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
fig.tight_layout(); charts["actual_vs_pred"] = fig_b64(fig)

# 9b. Residuals
residuals = y_te.values - y_pred_te
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_te, residuals, alpha=.4, color="#7B1FA2", s=18, edgecolors="none")
axes[0].axhline(0, color="red", ls="--", lw=1.4)
axes[0].set(title="Residuals vs Predicted", xlabel="Predicted (₹)", ylabel="Residual (₹)")
axes[1].hist(residuals, bins=40, color="#7B1FA2", edgecolor="white", alpha=.85)
axes[1].axvline(0, color="red", ls="--", lw=1.4)
axes[1].set(title="Residual Distribution", xlabel="Residual (₹)", ylabel="Frequency")
fig.tight_layout(); charts["residuals"] = fig_b64(fig)

# 9c. Feature importance
coefs = pd.Series(
    np.abs(best_model.named_steps["ridge"].coef_), index=FEATURE_COLS
).sort_values(ascending=True).tail(18)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#1565C0" if "lag" in c or "rolling" in c or "ewm" in c
          else "#2E7D32" if c in ["Promotion","Holiday","promo_x_holiday","promo_x_weekend"]
          else "#E65100"
          for c in coefs.index]
coefs.plot(kind="barh", ax=ax, color=colors)
ax.set(title="Top 18 Feature Importances  (|Ridge Coeff|)", xlabel="|Coefficient|")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#1565C0", label="Lag / Rolling"),
    Patch(color="#2E7D32", label="Promo / Holiday"),
    Patch(color="#E65100", label="Temporal"),
], fontsize=9, loc="lower right")
fig.tight_layout(); charts["feature_importance"] = fig_b64(fig)

# 9d. Monthly seasonality
daily["month_label"] = daily["Date"].dt.to_period("M").astype(str)
daily["month_num"]   = daily["Date"].dt.month
mo = daily.groupby("month_num")["Revenue"].mean()
fig, ax = plt.subplots(figsize=(9, 3.8))
bars = ax.bar(range(1,13), mo.values, color=plt.cm.Blues(np.linspace(.4,.9,12)), edgecolor="white")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set(title="Average Daily Revenue by Month (Seasonality)", ylabel="Avg Revenue (₹)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.2f}M'))
fig.tight_layout(); charts["monthly_seasonality"] = fig_b64(fig)

# 9e. Promo intensity vs Revenue & Holiday effect
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Promo: bin into low/high
daily["promo_group"] = pd.cut(daily["Promotion"], bins=[0,.15,.35,1.1],
    labels=["Low Promo\n(0-15%)","Med Promo\n(15-35%)","High Promo\n(35%+)"])
g1 = daily.groupby("promo_group", observed=True)["Revenue"].mean()
axes[0].bar(range(len(g1)), g1.values, color=["#90A4AE","#42A5F5","#1565C0"], edgecolor="white", width=.5)
axes[0].set_xticks(range(len(g1))); axes[0].set_xticklabels(g1.index, fontsize=9)
axes[0].set(title="Promotion Intensity vs Revenue", ylabel="Avg Revenue (₹)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.2f}M'))
for i, v in enumerate(g1.values):
    axes[0].text(i, v*1.01, f'₹{v/1e6:.2f}M', ha="center", fontsize=9, fontweight="bold")
# Holiday
g2 = daily.groupby("Holiday")["Revenue"].mean()
labels2 = ["Regular Day", "Holiday"]
axes[1].bar(range(len(g2)), g2.values, color=["#90A4AE","#E53935"], edgecolor="white", width=.5)
axes[1].set_xticks(range(len(g2))); axes[1].set_xticklabels(labels2)
axes[1].set(title="Holiday Effect on Revenue", ylabel="Avg Revenue (₹)")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.2f}M'))
for i, v in enumerate(g2.values):
    axes[1].text(i, v*1.01, f'₹{v/1e6:.2f}M', ha="center", fontsize=10, fontweight="bold")
fig.tight_layout(); charts["promo_holiday"] = fig_b64(fig)

# 9f. Revenue distribution
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(daily["Revenue"], bins=50, color="#F57F17", edgecolor="white", alpha=.9)
ax.axvline(daily["Revenue"].mean(), color="red", ls="--", lw=1.5, label=f'Mean ₹{daily["Revenue"].mean()/1e6:.2f}M')
ax.set(title="Daily Revenue Distribution", xlabel="Revenue (₹)", ylabel="Frequency")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
ax.legend(fontsize=9); fig.tight_layout(); charts["distribution"] = fig_b64(fig)

print(f"\nGenerated {len(charts)} charts ✓")

# ── 10. Save everything ───────────────────────────────────────────────────────
joblib.dump(best_model, f"{MODEL_DIR}/retail_sales_model.pkl")

meta = {
    "feature_cols":  FEATURE_COLS,
    "target_col":    TARGET,
    "date_col":      "Date",
    "best_alpha":    best_alpha,
    "cv_r2_mean":    round(float(cv_r2.mean()), 4),
    "cv_r2_std":     round(float(cv_r2.std()),  4),
    "train_r2":      round(tr_m["r2"],   4),
    "test_r2":       round(te_m["r2"],   4),
    "test_mae":      round(te_m["mae"],  2),
    "test_rmse":     round(te_m["rmse"], 2),
    "test_mape":     round(te_m["mape"], 2),
    "n_features":    len(FEATURE_COLS),
    "n_train":       len(X_tr),
    "n_test":        len(X_te),
    "revenue_mean":  round(float(y.mean()), 2),
    "revenue_std":   round(float(y.std()),  2),
    "daily_df":      daily[["Date", TARGET, "Units_Sold", "Promotion", "Holiday"] + FEATURE_COLS],
    "charts":        charts,
}

joblib.dump(meta, f"{MODEL_DIR}/model_meta.pkl")

print(f"\n{'='*60}")
print(f"  MODEL SAVED  →  {MODEL_DIR}/retail_sales_model.pkl")
print(f"  META  SAVED  →  {MODEL_DIR}/model_meta.pkl")
print(f"{'='*60}")
print(f"  Test  R²   : {te_m['r2']:.4f}")
print(f"  Test  MAE  : ₹{te_m['mae']:>12,.0f}")
print(f"  Test  RMSE : ₹{te_m['rmse']:>12,.0f}")
print(f"  Test  MAPE : {te_m['mape']:.2f}%")
print(f"  CV R²      : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"{'='*60}\n")
