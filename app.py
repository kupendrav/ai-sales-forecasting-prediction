"""
app.py  –  Retail Sales Prediction Flask App  (v2)
"""
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib, os, traceback, warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)

MODEL_PATH = "models/retail_sales_model.pkl"
META_PATH  = "models/model_meta.pkl"
model = meta = None

def load_artifacts():
    global model, meta
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta  = joblib.load(META_PATH)
        print(f"✓ Model loaded  R²={meta['test_r2']}  MAPE={meta['test_mape']}%")
    else:
        print("⚠  Run  python train_model.py  first")

load_artifacts()

# ── Feature builder ───────────────────────────────────────────────────────────
def build_row(date_str, promotion, holiday, units_sold,
              lag_vals, rolling_vals, ewm_vals, t_idx=0):
    """
    Build one feature row matching FEATURE_COLS exactly.
    lag_vals      = {1,3,7,14,21,28}
    rolling_vals  = {mean_7,mean_14,mean_30, std_7,std_14,std_30, max_7,min_7,max_30}
    ewm_vals      = {14, 30}
    """
    dt = pd.Timestamp(date_str)
    dow = dt.dayofweek
    mo  = dt.month
    woy = dt.isocalendar()[1]
    is_we = int(dow >= 5)
    is_me = int(dt.is_month_end)
    is_ms = int(dt.is_month_start)

    promo_x_holiday  = promotion * holiday
    promo_x_weekend  = promotion * is_we
    holiday_x_weekend= holiday   * is_we

    row = {
        "year":           dt.year,
        "month":          mo,
        "day":            dt.day,
        "dayofweek":      dow,
        "weekofyear":     woy,
        "quarter":        dt.quarter,
        "is_weekend":     is_we,
        "is_month_end":   is_me,
        "is_month_start": is_ms,
        "month_sin":  np.sin(2*np.pi*mo/12),
        "month_cos":  np.cos(2*np.pi*mo/12),
        "dow_sin":    np.sin(2*np.pi*dow/7),
        "dow_cos":    np.cos(2*np.pi*dow/7),
        "woy_sin":    np.sin(2*np.pi*woy/52),
        "woy_cos":    np.cos(2*np.pi*woy/52),
        "Promotion":      promotion,
        "Holiday":        holiday,
        "promo_x_holiday":promo_x_holiday,
        "promo_x_weekend":promo_x_weekend,
        "holiday_x_weekend": holiday_x_weekend,
        "lag_1":  lag_vals.get("lag_1",  lag_vals.get("lag_7", 3e6)),
        "lag_3":  lag_vals.get("lag_3",  lag_vals.get("lag_7", 3e6)),
        "lag_7":  lag_vals.get("lag_7",  3e6),
        "lag_14": lag_vals.get("lag_14", lag_vals.get("lag_7", 3e6)),
        "lag_21": lag_vals.get("lag_21", lag_vals.get("lag_14", 3e6)),
        "lag_28": lag_vals.get("lag_28", lag_vals.get("lag_14", 3e6)),
        "rolling_mean_7":  rolling_vals.get("rolling_mean_7",  3e6),
        "rolling_mean_14": rolling_vals.get("rolling_mean_14", 3e6),
        "rolling_mean_30": rolling_vals.get("rolling_mean_30", 3e6),
        "rolling_std_7":   rolling_vals.get("rolling_std_7",   2e5),
        "rolling_std_14":  rolling_vals.get("rolling_std_14",  2e5),
        "rolling_std_30":  rolling_vals.get("rolling_std_30",  2e5),
        "rolling_max_7":   rolling_vals.get("rolling_max_7",   4e6),
        "rolling_min_7":   rolling_vals.get("rolling_min_7",   2e6),
        "rolling_max_30":  rolling_vals.get("rolling_max_30",  5e6),
        "ewm_14":          ewm_vals.get("ewm_14", 3e6),
        "ewm_30":          ewm_vals.get("ewm_30", 3e6),
        "units_lag_7":     units_sold,
        "units_rolling_7": units_sold,
        "t":               t_idx,
    }
    fc = meta["feature_cols"]
    return pd.DataFrame([[row[f] for f in fc]], columns=fc)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model_info")
def model_info():
    if not meta:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "r2":         meta["test_r2"],
        "mae":        meta["test_mae"],
        "rmse":       meta["test_rmse"],
        "mape":       meta["test_mape"],
        "cv_r2_mean": meta["cv_r2_mean"],
        "cv_r2_std":  meta["cv_r2_std"],
        "train_r2":   meta["train_r2"],
        "n_features": meta["n_features"],
        "n_train":    meta["n_train"],
        "n_test":     meta["n_test"],
        "algorithm":  "Ridge Regression (GridSearchCV)",
        "revenue_mean": meta["revenue_mean"],
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not meta:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500
    try:
        d = request.get_json(force=True)
        date_str  = d.get("date", "")
        promotion = float(d.get("promotion", 0.0))
        holiday   = int(d.get("holiday", 0))
        units     = float(d.get("units_sold", meta["revenue_mean"] / 1200))
        lag7      = float(d.get("lag_7",  meta["revenue_mean"]))
        lag14     = float(d.get("lag_14", meta["revenue_mean"]))
        lag28     = float(d.get("lag_28", meta["revenue_mean"]))

        if not date_str:
            return jsonify({"error": "date is required"}), 400

        lag_vals = {
            "lag_1": lag7, "lag_3": lag7,
            "lag_7": lag7, "lag_14": lag14,
            "lag_21": lag14, "lag_28": lag28,
        }
        rolling_vals = {
            "rolling_mean_7": lag7, "rolling_mean_14": (lag7+lag14)/2,
            "rolling_mean_30": (lag7+lag14+lag28)/3,
            "rolling_std_7": abs(lag7 - lag14) * 0.5,
            "rolling_std_14": abs(lag7 - lag28) * 0.4,
            "rolling_std_30": abs(lag14 - lag28) * 0.3,
            "rolling_max_7": max(lag7, lag14),
            "rolling_min_7": min(lag7, lag14),
            "rolling_max_30": max(lag7, lag14, lag28),
        }
        ewm_vals = {
            "ewm_14": (lag7 * 0.6 + lag14 * 0.4),
            "ewm_30": (lag7 * 0.4 + lag14 * 0.35 + lag28 * 0.25),
        }

        X_row = build_row(date_str, promotion, holiday, units,
                          lag_vals, rolling_vals, ewm_vals)
        pred = float(model.predict(X_row)[0])
        pred = max(0, pred)

        mae  = meta["test_mae"]
        lo   = max(0, pred - 1.5 * mae)
        hi   = pred + 1.5 * mae

        return jsonify({
            "prediction": round(pred, 2),
            "lower_ci":   round(lo, 2),
            "upper_ci":   round(hi, 2),
            "date":       date_str,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/forecast", methods=["POST"])
def forecast():
    if not model or not meta:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        d         = request.get_json(force=True)
        start     = d.get("start_date", "")
        days      = min(int(d.get("days", 14)), 90)
        promotion = float(d.get("promotion", 0.2))
        holiday   = int(d.get("holiday", 0))
        base_rev  = float(d.get("base_revenue", meta["revenue_mean"]))
        base_units= float(d.get("base_units", 2700))

        if not start:
            return jsonify({"error": "start_date required"}), 400

        history = [base_rev] * 30
        preds, dates = [], []

        for i in range(days):
            dt = (pd.Timestamp(start) + timedelta(days=i)).strftime("%Y-%m-%d")
            is_hol = int(pd.Timestamp(dt).dayofweek in [6])  # Sunday approx

            lag_vals = {
                "lag_1":  history[-1],  "lag_3":  history[-3],
                "lag_7":  history[-7],  "lag_14": history[-14],
                "lag_21": history[-21] if len(history)>=21 else history[-14],
                "lag_28": history[-28] if len(history)>=28 else history[-14],
            }
            rolling_vals = {
                "rolling_mean_7":  np.mean(history[-7:]),
                "rolling_mean_14": np.mean(history[-14:]),
                "rolling_mean_30": np.mean(history[-30:]),
                "rolling_std_7":   np.std(history[-7:])  + 1e-6,
                "rolling_std_14":  np.std(history[-14:]) + 1e-6,
                "rolling_std_30":  np.std(history[-30:]) + 1e-6,
                "rolling_max_7":   max(history[-7:]),
                "rolling_min_7":   min(history[-7:]),
                "rolling_max_30":  max(history[-30:]),
            }
            ewm_vals = {
                "ewm_14": float(pd.Series(history[-14:]).ewm(span=14, min_periods=1).mean().iloc[-1]),
                "ewm_30": float(pd.Series(history[-30:]).ewm(span=30, min_periods=1).mean().iloc[-1]),
            }

            X_row = build_row(dt, promotion, is_hol, base_units,
                              lag_vals, rolling_vals, ewm_vals, t_idx=700+i)
            p = max(0, float(model.predict(X_row)[0]))
            history.append(p)
            preds.append(round(p, 2))
            dates.append(dt)

        return jsonify({"dates": dates, "forecasts": preds})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/charts")
def charts():
    if not meta:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify(meta.get("charts", {}))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
