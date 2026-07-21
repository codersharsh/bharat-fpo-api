"""
Bharat FPO - Core Logic (MongoDB Version)
RAM mein poora data load nahi karta — sirf jo chahiye MongoDB se query karta hai.
Isse free-tier (512MB) hosting par bhi crash nahi hoga.
"""

import os
import io
import warnings
import zipfile
import joblib
import certifi
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID   = "dodsa/bharat-fpo-mandi-models"
CACHE_DIR    = "/tmp/bharat_fpo_models"
MONGO_URI    = os.getenv("MONGO_URI", "mongodb+srv://gautamharsh031_db_user:3peAhFDqhutF7Jfw@bharat-fpo.6rm1gax.mongodb.net/?appName=bharat-fpo")

os.makedirs(CACHE_DIR, exist_ok=True)

# ── MongoDB Connection ────────────────────────────────────────────────────────
print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["bharat_fpo_db"]
mandi_collection = db["mandi_prices"]
print("Connected to MongoDB!")

# ── Model metrics (small file, load fully — this is OK, it's tiny) ───────────
try:
    METRICS_FILE = hf_hub_download(repo_id=HF_REPO_ID, filename="model_metrics.jsonl", repo_type="model", cache_dir=CACHE_DIR)
    metrics_df = pd.read_json(METRICS_FILE, lines=True)
except Exception:
    metrics_df = pd.DataFrame()

# ── MSP 2024-25 ───────────────────────────────────────────────────────────────
MSP = {
    "Wheat": 2275, "Rice": 2300, "Maize": 2090, "Bajra": 2500,
    "Jowar": 3180, "Ragi": 4290, "Arhar": 7000, "Moong": 8558,
    "Urad": 7400, "Groundnut": 6783, "Sunflower": 7280,
    "Soyabean": 4892, "Sesamum": 9267, "Cotton": 7121,
    "Sugarcane": 340, "Onion": 0, "Potato": 0, "Tomato": 0
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(s):
    return (str(s).replace(" ", "_").replace("/", "_")
            .replace("&", "and").replace("(", "")
            .replace(")", "").replace(",", ""))

def assign_tier(r2, mae):
    if r2 > 0.9 and mae < 100:   return "A"
    elif r2 > 0.7 and mae < 300: return "B"
    elif r2 > 0.5 and mae < 500: return "C"
    elif r2 > 0 and mae < 500:   return "D"
    else:                         return "F"

STATE_MAP = {
    "keralam": "Kerala", "uttarakhand": "Uttrakhand",
    "jammu & kashmir": "Jammu And Kashmir", "delhi": "Nct Of Delhi",
    "chhattisgarh": "Chattisgarh", "orissa": "Odisha",
    "j&k": "Jammu And Kashmir", "up": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "hp": "Himachal Pradesh",
}

def normalize_state(s):
    return STATE_MAP.get(s.strip().lower(), s.strip().title())

def fuzzy_match(query, options):
    from difflib import SequenceMatcher
    if not query: return None
    query = query.lower().strip()
    for opt in options:
        if opt.lower() == query: return opt
    matches = [opt for opt in options if query in opt.lower()]
    if matches: return matches[0]
    for opt in options:
        if any(w in opt.lower() for w in query.split()): return opt
    best, best_score = None, 0
    for opt in options:
        score = SequenceMatcher(None, query, opt.lower()).ratio()
        if score > best_score:
            best, best_score = opt, score
    return best if best_score >= 0.6 else None

# ── Fake "master" wrapper for compatibility ──────────────────────────────────
class MasterProxy:
    """
    Purane code mein `master["commodity"].unique()` jaisa syntax use hota tha.
    Yeh class MongoDB queries ko woh interface deti hai bina RAM mein data load kiye.
    """
    def __getitem__(self, col):
        return ColumnProxy(col)

class ColumnProxy:
    def __init__(self, col):
        self.col = col
    def unique(self):
        return mandi_collection.distinct(self.col)

master = MasterProxy()  # sirf backward-compat ke liye kahin use ho raha ho toh

# ── Core data fetch functions (query MongoDB directly) ────────────────────────
def get_group_data(state, market, commodity):
    """Ek state+market+commodity ka poora price history MongoDB se fetch karo"""
    cursor = mandi_collection.find(
        {"state": state, "market": market, "commodity": commodity},
        {"_id": 0, "arrival_date": 1, "modal_price": 1, "min_price": 1, "max_price": 1, "month": 1, "year": 1}
    ).sort("arrival_date", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    return df

def get_group_data_state_commodity(state, commodity):
    """Poore state ka ek commodity ka data (market-wise comparison ke liye)"""
    cursor = mandi_collection.find(
        {"state": state, "commodity": commodity},
        {"_id": 0, "market": 1, "arrival_date": 1, "modal_price": 1, "month": 1, "year": 1}
    )
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    return df

# ── Lazy Model Loading from HF (same as before) ───────────────────────────────
_state_zip_cache = {}

def _get_model_from_hf(state, fname):
    global _state_zip_cache
    if len(_state_zip_cache) >= 5 and state not in _state_zip_cache:
        oldest = next(iter(_state_zip_cache))
        del _state_zip_cache[oldest]

    if state not in _state_zip_cache:
        zip_filename = f"{clean(state)}.zip"
        zip_path = None
        try:
            zip_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=zip_filename,
                repo_type="model",
                cache_dir=CACHE_DIR
            )
        except Exception:
            return None

        if not zip_path:
            return None

        _state_zip_cache[state] = {}
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.pkl'):
                    _state_zip_cache[state][name] = zf.read(name)

    model_bytes = _state_zip_cache.get(state, {}).get(fname)
    if not model_bytes:
        return None
    return joblib.load(io.BytesIO(model_bytes))

def load_model_and_meta(state, market, commodity):
    fname = f"{clean(state)}__{clean(commodity)}__{clean(market)}.pkl"
    model = _get_model_from_hf(state, fname)
    if model is None:
        return None, None, "F"
    if metrics_df.empty:
        return model, None, "F"
    meta = metrics_df[
        (metrics_df["state"]     == state) &
        (metrics_df["market"]    == market) &
        (metrics_df["commodity"] == commodity)
    ]
    if meta.empty:
        return model, None, "F"
    row = meta.iloc[-1]
    return model, row, assign_tier(row["r2"], row["mae"])

def historical_fallback(state, market, commodity, month, last_known_price=None, blend_weight=0.5):
    grp = get_group_data(state, market, commodity)
    if grp.empty:
        grp = get_group_data_state_commodity(state, commodity)
    if grp.empty: return None
    m = grp[grp["month"] == month]["modal_price"] if "month" in grp.columns else pd.Series(dtype=float)
    if m.empty and "arrival_date" in grp.columns:
        m = grp[grp["arrival_date"].dt.month == month]["modal_price"]
    seasonal_avg = round(m.mean(), 2) if not m.empty else round(grp["modal_price"].mean(), 2)
    if last_known_price is None:
        return seasonal_avg
    return round((blend_weight * seasonal_avg) + ((1 - blend_weight) * last_known_price), 2)

# ── Core Functions (same logic, MongoDB-backed) ───────────────────────────────
def get_latest_price(state, market, commodity):
    grp = get_group_data(state, market, commodity)
    if grp.empty: return None

    today    = pd.Timestamp.now().normalize()
    grp_curr = grp[grp["arrival_date"] <= today]
    if grp_curr.empty: grp_curr = grp

    latest    = grp_curr.sort_values("arrival_date").iloc[-1]
    last_date = latest["arrival_date"]
    days_old  = (today - last_date).days
    _, meta, tier = load_model_and_meta(state, market, commodity)

    ai_price = None
    if days_old > 7:
        month_start = today.replace(day=1).strftime("%Y-%m-%d")
        res, _ = get_future_prediction(state, market, commodity,
                                       from_date=month_start, to_date=month_start)
        if res is not None and not res.empty:
            ai_price = res.iloc[0]["predicted_price"]

    if days_old <= 7:      freshness = "Live"
    elif days_old <= 30:   freshness = f"{days_old} days old"
    elif days_old <= 90:   freshness = f"{days_old} days old — seasonal gap"
    else:                  freshness = f"{days_old} days old — mandi not reporting"

    return {
        "date"       : last_date.strftime("%Y-%m-%d"),
        "modal_price": round(float(latest["modal_price"]), 2),
        "min_price"  : round(float(latest.get("min_price", latest["modal_price"])), 2),
        "max_price"  : round(float(latest.get("max_price", latest["modal_price"])), 2),
        "ai_price"   : round(float(ai_price), 2) if ai_price is not None else None,
        "tier"       : tier,
        "data_points": len(grp_curr),
        "days_old"   : days_old,
        "freshness"  : freshness
    }

def get_historical_price(state, market, commodity, year=None):
    grp = get_group_data(state, market, commodity)
    if grp.empty: return None
    if year: grp = grp[grp["year"] == year]
    monthly = grp.groupby(["year", "month"])["modal_price"].agg(
        avg_price="mean", min_price="min", max_price="max", records="count"
    ).reset_index()
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )
    return monthly.sort_values("date")[["date", "avg_price", "min_price", "max_price", "records"]]

def get_future_prediction(state, market, commodity, n_months=None, from_date=None, to_date=None):
    model, meta, tier = load_model_and_meta(state, market, commodity)
    grp               = get_group_data(state, market, commodity)
    use_fallback      = (model is None or tier == "F")

    if from_date and to_date:
        dates = pd.date_range(from_date, to_date, freq="MS")
    else:
        start = datetime.now().replace(day=1) + relativedelta(months=1)
        dates = [start + relativedelta(months=i) for i in range(n_months or 3)]

    try:
        feature_cols = list(model.feature_names_in_) if model else ["year", "month"]
    except Exception:
        feature_cols = ["year", "month"]

    recent = grp.sort_values("arrival_date").tail(3)["modal_price"].values.tolist() if not grp.empty else []
    last_known_price = recent[-1] if recent else None
    results = []

    for d in dates:
        yr, mo = d.year, d.month
        if use_fallback:
            price = historical_fallback(state, market, commodity, mo,
                                        last_known_price=last_known_price)
            conf  = "Historical Avg"
            last_known_price = price
        else:
            feat = {"year": yr, "month": mo}
            if "price_lag1" in feature_cols:
                feat["price_lag1"]     = recent[-1] if recent else (grp["modal_price"].mean() if not grp.empty else 0)
                feat["price_lag2"]     = recent[-2] if len(recent) >= 2 else (grp["modal_price"].mean() if not grp.empty else 0)
                feat["rolling_mean_3"] = float(np.mean(recent[-3:])) if recent else (grp["modal_price"].mean() if not grp.empty else 0)
                feat["rolling_std_3"]  = float(np.std(recent[-3:])) if len(recent) >= 2 else 0
            X     = pd.DataFrame([[feat.get(c, 0) for c in feature_cols]], columns=feature_cols)
            price = round(float(model.predict(X)[0]), 2)
            conf  = "AI Model"
            recent.append(price)
            if len(recent) > 3: recent.pop(0)

        results.append({
            "date"           : d.strftime("%Y-%m-%d"),
            "predicted_price": price,
            "source"         : conf,
            "mae"            : round(float(meta["mae"]), 2) if meta is not None else None
        })

    return pd.DataFrame(results), tier

def get_best_mandi(state, commodity, top_n=5):
    today = pd.Timestamp.now().normalize()
    month = today.month
    cursor = mandi_collection.find(
        {"state": state, "commodity": commodity, "month": month},
        {"_id": 0, "market": 1, "modal_price": 1}
    )
    state_data = pd.DataFrame(list(cursor))

    if state_data.empty:
        cursor = mandi_collection.find(
            {"state": state, "commodity": commodity},
            {"_id": 0, "market": 1, "modal_price": 1}
        )
        state_data = pd.DataFrame(list(cursor))

    if state_data.empty: return None
    mandi_avg         = state_data.groupby("market")["modal_price"].mean().reset_index()
    mandi_avg.columns = ["market", "avg_price"]
    mandi_avg["avg_price"] = mandi_avg["avg_price"].round(2)
    return mandi_avg.sort_values("avg_price", ascending=False).head(top_n)

def get_nearest_reporting_mandi(state, commodity, exclude_market=None, max_days_old=30):
    today = pd.Timestamp.now().normalize()
    query = {"state": state, "commodity": commodity}
    if exclude_market:
        query["market"] = {"$ne": exclude_market}

    cursor = mandi_collection.find(query, {"_id": 0, "market": 1, "arrival_date": 1, "modal_price": 1})
    df = pd.DataFrame(list(cursor))
    if df.empty: return None

    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    latest_per_market = df.sort_values("arrival_date").groupby("market").tail(1).copy()
    latest_per_market["days_old"] = (today - latest_per_market["arrival_date"]).dt.days
    fresh = latest_per_market[latest_per_market["days_old"] <= max_days_old]
    if fresh.empty: return None
    return fresh.sort_values("days_old")[["market", "arrival_date", "modal_price", "days_old"]].head(3)

def get_seasonal_advice(commodity, state, market):
    grp = get_group_data(state, market, commodity)
    if grp.empty:
        grp = get_group_data_state_commodity(state, commodity)
    if grp.empty: return None
    monthly_avg = grp.groupby("month")["modal_price"].mean()
    best_month  = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    return {
        "best_month" : month_names[best_month],
        "best_price" : round(monthly_avg[best_month], 2),
        "worst_month": month_names[worst_month],
        "worst_price": round(monthly_avg[worst_month], 2),
        "monthly_avg": monthly_avg.to_dict()
    }
