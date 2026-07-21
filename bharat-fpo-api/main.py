"""
Bharat FPO - Mandi Price Intelligence API
FastAPI backend — MongoDB-backed (no full dataset in RAM)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from core_logic_mongo import (
    mandi_collection, fuzzy_match, normalize_state,
    get_latest_price, get_historical_price,
    get_future_prediction, get_best_mandi,
    get_nearest_reporting_mandi, get_seasonal_advice,
    MSP
)

app = FastAPI(
    title="Bharat FPO - Mandi Price Intelligence",
    description="AI-powered mandi price prediction for Indian farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper ────────────────────────────────────────────────────────────────────
def resolve_params(state: str, market: str, commodity: str):
    """Fuzzy match state/market/commodity using MongoDB distinct queries"""
    all_commodities = sorted(mandi_collection.distinct("commodity"))
    resolved_commodity = fuzzy_match(commodity, all_commodities)
    if not resolved_commodity:
        raise HTTPException(status_code=404, detail=f"Commodity '{commodity}' not found")

    avail_states = sorted(mandi_collection.distinct("state", {"commodity": resolved_commodity}))
    resolved_state = fuzzy_match(normalize_state(state), avail_states) or fuzzy_match(state, avail_states)
    if not resolved_state:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found for {resolved_commodity}")

    avail_markets = sorted(mandi_collection.distinct(
        "market", {"commodity": resolved_commodity, "state": resolved_state}
    ))
    resolved_market = fuzzy_match(market, avail_markets)
    if not resolved_market:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found in {resolved_state}")

    return resolved_state, resolved_market, resolved_commodity

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Bharat FPO Mandi Price Intelligence API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    count = mandi_collection.estimated_document_count()
    return {"status": "ok", "records": count}

# ── 1. Search / Autocomplete ──────────────────────────────────────────────────
@app.get("/search/commodities")
def list_commodities(q: Optional[str] = None):
    all_c = sorted(mandi_collection.distinct("commodity"))
    if q:
        all_c = [c for c in all_c if q.lower() in c.lower()]
    return {"commodities": all_c}

@app.get("/search/states")
def list_states(commodity: Optional[str] = None):
    query = {}
    if commodity:
        all_c = sorted(mandi_collection.distinct("commodity"))
        resolved = fuzzy_match(commodity, all_c)
        if resolved:
            query["commodity"] = resolved
    return {"states": sorted(mandi_collection.distinct("state", query))}

@app.get("/search/markets")
def list_markets(state: str, commodity: str):
    all_c = sorted(mandi_collection.distinct("commodity"))
    resolved_commodity = fuzzy_match(commodity, all_c)
    if not resolved_commodity:
        raise HTTPException(status_code=404, detail="Commodity not found")

    avail_states = sorted(mandi_collection.distinct("state", {"commodity": resolved_commodity}))
    resolved_state = fuzzy_match(normalize_state(state), avail_states) or fuzzy_match(state, avail_states)
    if not resolved_state:
        raise HTTPException(status_code=404, detail="State not found")

    markets = sorted(mandi_collection.distinct(
        "market", {"commodity": resolved_commodity, "state": resolved_state}
    ))
    return {"state": resolved_state, "commodity": resolved_commodity, "markets": markets}

# ── 2. Latest Price ───────────────────────────────────────────────────────────
@app.get("/price/latest")
def latest_price(state: str, market: str, commodity: str):
    s, m, c = resolve_params(state, market, commodity)
    result = get_latest_price(s, m, c)
    if not result:
        raise HTTPException(status_code=404, detail="No price data found")
    return {"state": s, "market": m, "commodity": c, **result}

# ── 3. Historical Prices ──────────────────────────────────────────────────────
@app.get("/price/history")
def price_history(state: str, market: str, commodity: str, year: Optional[int] = None):
    s, m, c = resolve_params(state, market, commodity)
    df = get_historical_price(s, m, c, year=year)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data found")
    df["date"] = df["date"].astype(str)
    return {"state": s, "market": m, "commodity": c, "history": df.to_dict("records")}

# ── 4. Price Forecast ─────────────────────────────────────────────────────────
@app.get("/price/forecast")
def price_forecast(
    state: str, market: str, commodity: str,
    months: int = Query(default=6, ge=1, le=24)
):
    s, m, c = resolve_params(state, market, commodity)
    df, tier = get_future_prediction(s, m, c, n_months=months)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Could not generate forecast")
    return {
        "state": s, "market": m, "commodity": c,
        "model_tier": tier,
        "forecast": df.to_dict("records")
    }

# ── 5. Graph Data ─────────────────────────────────────────────────────────────
@app.get("/price/graph-data")
def graph_data(
    state: str, market: str, commodity: str,
    future_months: int = Query(default=6, ge=1, le=12)
):
    import pandas as pd
    from dateutil.relativedelta import relativedelta

    s, m, c = resolve_params(state, market, commodity)

    hist_df = get_historical_price(s, m, c)
    if hist_df is None or hist_df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    hist_df = hist_df.tail(24)
    hist_df["date"] = hist_df["date"].astype(str)

    last_hist_date  = pd.to_datetime(hist_df["date"].iloc[-1])
    today           = pd.Timestamp.now().normalize()
    days_gap        = (today - last_hist_date).days

    gap_data = None
    if days_gap > 30:
        gap_start = (last_hist_date + relativedelta(months=1)).replace(day=1)
        gap_end   = today.replace(day=1)
        if gap_start <= gap_end:
            gap_df, _ = get_future_prediction(
                s, m, c,
                from_date=gap_start.strftime("%Y-%m-%d"),
                to_date=gap_end.strftime("%Y-%m-%d")
            )
            if gap_df is not None and not gap_df.empty:
                gap_data = gap_df.to_dict("records")

    forecast_df, tier = get_future_prediction(s, m, c, n_months=future_months)

    return {
        "state": s, "market": m, "commodity": c,
        "model_tier": tier,
        "days_gap": days_gap,
        "msp": MSP.get(c.title(), 0),
        "historical": hist_df.to_dict("records"),
        "gap_fill": gap_data,
        "forecast": forecast_df.to_dict("records") if forecast_df is not None else []
    }

# ── 6. Best Mandi ─────────────────────────────────────────────────────────────
@app.get("/mandi/best")
def best_mandi(state: str, commodity: str, top_n: int = Query(default=5, ge=1, le=20)):
    all_c = sorted(mandi_collection.distinct("commodity"))
    resolved_c = fuzzy_match(commodity, all_c)
    if not resolved_c:
        raise HTTPException(status_code=404, detail="Commodity not found")

    avail_states = sorted(mandi_collection.distinct("state", {"commodity": resolved_c}))
    resolved_s = fuzzy_match(normalize_state(state), avail_states) or fuzzy_match(state, avail_states)
    if not resolved_s:
        raise HTTPException(status_code=404, detail="State not found")

    df = get_best_mandi(resolved_s, resolved_c, top_n=top_n)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No mandi data found")
    return {"state": resolved_s, "commodity": resolved_c, "top_mandis": df.to_dict("records")}

# ── 7. Compare Markets ────────────────────────────────────────────────────────
@app.get("/mandi/compare")
def compare_markets(state: str, market1: str, market2: str, commodity: str):
    s, m1, c = resolve_params(state, market1, commodity)
    _, m2, _ = resolve_params(state, market2, commodity)

    r1 = get_latest_price(s, m1, c)
    r2 = get_latest_price(s, m2, c)

    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="Price data not found for one or both markets")

    diff = round(r1["modal_price"] - r2["modal_price"], 2)
    better = m1 if diff > 0 else m2

    return {
        "state": s, "commodity": c,
        "market1": {"name": m1, **r1},
        "market2": {"name": m2, **r2},
        "price_difference": abs(diff),
        "better_market": better
    }

# ── 8. MSP Comparison ─────────────────────────────────────────────────────────
@app.get("/msp/compare")
def msp_compare(state: str, market: str, commodity: str):
    s, m, c = resolve_params(state, market, commodity)
    msp_val = MSP.get(c.title(), 0)
    if msp_val == 0:
        raise HTTPException(status_code=404, detail=f"MSP not available for {c}")

    result = get_latest_price(s, m, c)
    if not result:
        raise HTTPException(status_code=404, detail="No price data found")

    price      = result["modal_price"]
    diff       = round(price - msp_val, 2)
    above_msp  = diff > 0

    return {
        "state": s, "market": m, "commodity": c,
        "current_price": price,
        "msp": msp_val,
        "difference": diff,
        "above_msp": above_msp,
        "advice": f"Price is Rs.{abs(diff):,.0f} {'above' if above_msp else 'below'} MSP"
    }

# ── 9. Seasonal Advice ────────────────────────────────────────────────────────
@app.get("/advice/seasonal")
def seasonal_advice(state: str, market: str, commodity: str):
    s, m, c = resolve_params(state, market, commodity)
    result = get_seasonal_advice(c, s, m)
    if not result:
        raise HTTPException(status_code=404, detail="Not enough data for seasonal analysis")
    result.pop("monthly_avg", None)
    return {"state": s, "market": m, "commodity": c, **result}

# ── 10. Price Alert Check ─────────────────────────────────────────────────────
@app.get("/alert/check")
def price_alert(state: str, market: str, commodity: str, target_price: float):
    s, m, c = resolve_params(state, market, commodity)
    result = get_latest_price(s, m, c)
    if not result:
        raise HTTPException(status_code=404, detail="No price data found")

    current = result["modal_price"]
    reached = current >= target_price

    response = {
        "state": s, "market": m, "commodity": c,
        "current_price": current,
        "target_price": target_price,
        "target_reached": reached
    }

    if reached:
        response["advice"] = f"SELL NOW! Price Rs.{current:,.0f} is above target Rs.{target_price:,.0f}"
        response["extra_per_quintal"] = round(current - target_price, 2)
    else:
        forecast_df, _ = get_future_prediction(s, m, c, n_months=12)
        if forecast_df is not None:
            above = forecast_df[forecast_df["predicted_price"] >= target_price]
            if not above.empty:
                response["forecast_date"] = above.iloc[0]["date"]
                response["advice"] = f"Price may reach target around {above.iloc[0]['date'][:7]}"
            else:
                response["advice"] = f"Price may NOT reach Rs.{target_price:,.0f} in next 12 months"
        response["below_by"] = round(target_price - current, 2)

    return response

# ── 11. Nearby Fresh Mandi ────────────────────────────────────────────────────
@app.get("/mandi/nearby-fresh")
def nearby_fresh(state: str, market: str, commodity: str):
    s, m, c = resolve_params(state, market, commodity)
    df = get_nearest_reporting_mandi(s, c, exclude_market=m)
    if df is None or df.empty:
        return {"state": s, "commodity": c, "alternatives": []}
    df["arrival_date"] = df["arrival_date"].astype(str)
    return {"state": s, "commodity": c, "excluded_market": m, "alternatives": df.to_dict("records")}
