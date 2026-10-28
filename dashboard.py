import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib

st.set_page_config(
    page_title="Energy Mismatch Dashboard",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main { background-color: #0b1220; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }
    h1, h2, h3 { color: #e6edf7; font-family: 'Segoe UI', sans-serif; }
    .stMarkdown p, .stMarkdown li { color: #c8d2e0; }
    .section-card {
        background: #111a2e;
        border: 1px solid #1f2a44;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-blue   { background: #1e3a8a; color: #dbeafe; }
    .pill-green  { background: #064e3b; color: #d1fae5; }
    .pill-amber  { background: #78350f; color: #fde68a; }
    .pill-red    { background: #7f1d1d; color: #fecaca; }
    .metric-big {
        font-size: 2.4rem; font-weight: 700; color: #e6edf7;
    }
    .metric-label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    hr { border-color: #1f2a44; }
    </style>
    """,
    unsafe_allow_html=True,
)


ROOT = Path(__file__).parent
OUT_MODEL = ROOT / "output" 
OUT_COMP = ROOT / "output"/ "ComparisonResults"
OUT_EVAL = ROOT / "output"/ "model_output"         
OUT_SHAP  = ROOT / "SHAP_Analysis"   
MODEL_PATH = ROOT / "xgb_model_fixed.json"
THR_PATH   = ROOT / "output"/ "model_output"/ "dynamic_thresholds_fixed.csv"


def show_image(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: `{path.relative_to(ROOT)}`")



@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_resource
def load_optimal_threshold():
    thr_path = ROOT / "output/model_output/threshold.pkl"
    if thr_path.exists():
        data = joblib.load(thr_path)
        return data.get('optimal_threshold', 0.5)
    return 0.5

@st.cache_data
def load_threshold_table():
    if not THR_PATH.exists():
        return None
    return pd.read_csv(THR_PATH)


booster = load_model()
thr_df = load_threshold_table()
opt_thr = load_optimal_threshold()


def get_hour_block(hour: int) -> int:
    return hour // 4

def get_regime(clouds_all: float) -> str:
    if clouds_all <= 25: return "clear"
    if clouds_all <= 65: return "partial"
    return "overcast"

def lookup_threshold_row(season: int, hour: int, clouds_all: float):
    
    if thr_df is None:
        return None
    hb  = get_hour_block(hour)
    reg = get_regime(clouds_all)
    mask = (
        (thr_df["season"] == season) &
        (thr_df["hour_block"] == hb) &
        (thr_df["weather_regime"] == reg)
    )
    sub = thr_df[mask]
    if len(sub) == 0:
        
        sub = thr_df[thr_df["season"] == season]
    return sub.iloc[0] if len(sub) else None


def derive_features(wind_onshore: float, row) -> dict:
    
    mean = float(row["proxy_mean"])
    std  = float(row["proxy_std"])
    thr_high = mean + 1.5 * std
    thr_low  = max(mean - 1.5 * std, 0.0)
    return {
        "proxy_mean": mean,
        "proxy_std":  std,
        "thr_high":   thr_high,
        "thr_low":    thr_low,
        "thr_range":  thr_high - thr_low,
        "ineff_rate": float(row.get("ineff_rate", 0.25)),
        "wind_above_high": max(wind_onshore - thr_high, 0.0),
        "wind_below_low":  max(thr_low - wind_onshore, 0.0),
        "wind_deviation":  wind_onshore - mean,
        "wind_zscore":     (wind_onshore - mean) / (std + 1e-6),
        "exceeds_threshold": int(wind_onshore > thr_high),
        "below_threshold":   int(wind_onshore < thr_low),
    }



st.markdown("#  Energy Mismatch Detection Dashboard")

st.markdown("## 1. Preprocessing")
st.markdown(
    """
    <div class="section-card">
    Hourly Spain ENTSO-E (2015–2018) energy + weather data is cleaned, merged on
    timestamp, missing data imputed, and included with calendar features
    hour, season and weather.
    Output of this stage feeds the comparison and modelling steps below.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("## 2 Comparison with other models (With vs Without dynamic thresholds)")
st.markdown(
    "Difference between **dynamic thresholds** (adaptive to season, hour-block, cloud regime) "
    "and a single **fixed global threshold**. Larger deltas = where dynamic thresholding pays off."
)
show_image(OUT_COMP /"threshold_delta.png", "Threshold delta: dynamic vs fixed")


st.markdown("## 3. XGBoost model performance")
c1, c2 = st.columns(2)
with c1:
    show_image(OUT_EVAL /"evaluation_dashboard_fixed.png", "Evaluation dashboard")
with c2:
    show_image(OUT_EVAL /"xgb_training_curves_fixed.png", "Training curves (logloss / AUC)")


st.markdown("## 4. Explainability using SHAP")

st.markdown("### Global importance")
c1, c2 = st.columns(2)
with c1: show_image(OUT_SHAP / "ShapSummary.png", "SHAP summary (Beeswarm Plot)")
with c2: show_image(OUT_SHAP / "TopFeatures.png", "Top 15 features (mean |SHAP|)")

st.markdown("### Dependence plots")
c1, c2, c3 = st.columns(3)
with c1: show_image(OUT_SHAP / "SHAP_generation hydro water reservoir.png", "Generation Hydro Water Reservoir")
with c2: show_image(OUT_SHAP / "SHAP_generation solar.png", "Generation Solar")
with c3: show_image(OUT_SHAP / "SHAP_generation wind onshore.png", "Generation Wind Onshore")

st.markdown("### Temporal analysis plots")
c1, c2 = st.columns(2)
with c1: show_image(OUT_SHAP / "HourlyAnalysis.png",   "Hourly Analysis Plot")
with c2: show_image(OUT_SHAP / "SeasonalAnalysis.png", "Seasonal Analysis Plot")


st.markdown("## 5. Live Mismatch Predictor")
st.markdown(
    "Thresholds **adapt dynamically** to the season, hour-block "
    "and cloud conditions you choose and based on the `compute_dynamic_thresholds()` function in `model.py` the probability of mismatch is computed"
)

left, right = st.columns([1, 1])

with left:
    st.markdown("### Inputs")
    season = st.selectbox("Season", [0, 1, 2, 3],
                          format_func=lambda x: ["Winter", "Spring", "Summer", "Autumn"][x])
    hour = st.slider("Hour of day",  0, 23, 14)
    clouds_all = st.slider("Cloud cover (%)",  0, 100, 40)
    wind_on  = st.slider("Wind onshore (MW)",  0, 18000, 6000, step=100)
    solar = st.slider("Solar generation (MW)", 0, 8000, 2500, step=100)
    gas  = st.slider("Fossil gas (MW)",  0, 20000, 6000, step=100)
    load  = st.slider("Total load (MW)", 15000, 45000, 28000, step=500)
    wind_speed = st.slider("Wind speed (m/s)",  0.0, 25.0, 7.0, step=0.5)
    temp_c  = st.slider("Temperature (°C)", -5.0, 40.0, 18.0, step=0.5)

with right:
    st.markdown("### Dynamic threshold (recomputed live)")

    row = lookup_threshold_row(season, hour, clouds_all)
    if row is None:
        st.error("Threshold table not loaded. Place `dynamic_thresholds_fixed.csv`")
        st.stop()

    feats = derive_features(wind_on, row)

    a, b, c = st.columns(3)
    a.metric("Dynamic threshold_low",  f"{feats['thr_low']:.0f} MW")
    b.metric("Dynamic threshold_high", f"{feats['thr_high']:.0f} MW")
    c.metric("Z-score",  f"{feats['wind_zscore']:+.2f}")

    a, b = st.columns(2)
    a.metric("Cloud cover",        get_regime(clouds_all))
    b.metric("Hour block",    get_hour_block(hour))
    
    st.markdown("### Mismatch probability")

    if booster is not None:
        feat_names = booster.get_booster().feature_names
        renewable = max(wind_on + solar, 0)
        load_ratio = renewable / max(load, 1)
        
        full = {
            "hour": hour, "season": season,
            "hour_block": get_hour_block(hour),
            "clouds_all": clouds_all, "wind_speed": wind_speed, "temp": temp_c + 273.15,
            "generation_wind_onshore": wind_on,
            "generation_solar": solar,
            "generation_fossil_gas": gas,
            "total_load_actual": load,
            "load_ratio": load_ratio,
            **feats,
        }
        x = np.array([[full.get(n, 0.0) for n in feat_names]], dtype=float)
        prob = float(booster.predict(xgb.DMatrix(x, feature_names=feat_names))[0])
    else:
        renewable = max(wind_on + solar, 0)
        load_ratio = renewable / max(load, 1)
        z = (-1.6
            + 0.85 * abs(feats["wind_zscore"])
            + 1.9 * feats["exceeds_threshold"]
            + 1.4 * feats["below_threshold"]
            + 1.8 * max(load_ratio - 0.55, 0)
            + 2.2 * feats["ineff_rate"])
        prob = 1 / (1 + np.exp(-z))

    is_high = prob >= opt_thr
    color   = "pill-red" if is_high else "pill-green"
    label   = "MISMATCH LIKELY" if is_high else "STABLE"

    st.markdown(
        f"<div class='metric-big'>{prob*100:.1f}%</div>"
        f"<div class='metric-label'>optimal threshold = {opt_thr:.2f}</div>"
        f"<div style='margin-top:8px;'><span class='pill {color}'>{label}</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(prob, 1.0))


st.markdown("## 6. Feature ranges that drive mismatch most")
st.markdown(
    "Derived from SHAP values in `explain.py` (top features) and the dynamic-threshold "
    "logic in `model.py`."
)

guide = pd.DataFrame([
    ["wind_zscore",  "|z| > 1.5",  "Wind far from seasonal/hourly mean - strong mismatch signal"],
    ["wind_above_high",  "> 0 MW (any overshoot)", "Wind exceeds adaptive thr_high"],
    ["wind_below_low", "> 0 MW (any shortfall)", "Wind below adaptive thr_low"],
    ["exceeds_threshold",  "= 1", "Hard breach above thr_high"],
    ["below_threshold", "= 1",  "Hard breach below thr_low"],
    ["load_ratio (ren/load)", "> 0.55",  "Renewables dominate - balancing harder"],
    ["ineff_rate (regime)",  "> 0.30", "Overcast / winter regimes carry higher base risk"],
    ["clouds_all",  "> 65 %",  "Switches regime to 'overcast' - wider thresholds"],
    ["hour_block", "4–5 (16:00–24:00)", "Evening ramp, demand peak"],
    ["season",  "0 = Winter", "Highest variance and ineff_rate"],
], columns=["Feature", "Critical range", "Why it matters"])
st.dataframe(guide, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Streamlit    Model: XGBoost      Explainability: SHAP")
