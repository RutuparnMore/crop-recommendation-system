import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List

from utils.soil_service import SoilService
from utils.weather_service import WeatherService
from utils.ml_engine import CropRecommendationEngine, AGRONOMIC_CROP_PROFILES
from utils.feasibility_engine import FeasibilityEngine

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="KrishiMitra AI - Climate & Soil Crop Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Agritech Theme) ---
st.markdown("""
<style>
    /* Main container enhancements */
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        color: #e8f5e9;
        font-weight: 400;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #dcedc8;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #558b2f;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 4px 0;
    }
    .crop-card {
        background: #ffffff;
        border: 2px solid #c8e6c9;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 3px 10px rgba(46, 125, 50, 0.08);
        transition: transform 0.2s ease;
    }
    .crop-card:hover {
        transform: translateY(-2px);
        border-color: #4caf50;
    }
    .rank-badge {
        background: #2e7d32;
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    .tag-badge {
        background: #e8f5e9;
        color: #2e7d32;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-right: 6px;
    }
    .pro-box {
        background-color: #f1f8e9;
        border-left: 5px solid #558b2f;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .con-box {
        background-color: #ffebee;
        border-left: 5px solid #d32f2f;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .infra-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .tip-box {
        background-color: #fffde7;
        border-left: 5px solid #fbc02d;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Singleton Services Initialization ---
@st.cache_resource
def load_services():
    soil_svc = SoilService(csv_path="data/soil_database.csv")
    ml_svc = CropRecommendationEngine()
    feas_svc = FeasibilityEngine(ml_engine=ml_svc)
    return soil_svc, ml_svc, feas_svc

soil_service, ml_engine, feasibility_engine = load_services()

# --- Sidebar: Configuration & Data Source Controls ---
st.sidebar.image("https://img.icons8.com/color/96/sprout.png", width=64)
st.sidebar.title("KrishiMitra Controls")
st.sidebar.markdown("**Intelligent Agronomic Advisory**")

st.sidebar.subheader("📡 Weather API Settings")
openweather_api_key = st.sidebar.text_input(
    "OpenWeatherMap API Key (Optional)",
    type="password",
    help="Enter your OpenWeatherMap API key. If left blank, the app automatically uses the live Open-Meteo meteorological service with zero setup."
)

weather_service = WeatherService(openweather_api_key=openweather_api_key)

st.sidebar.divider()
st.sidebar.subheader("🗂️ Soil Database Status")
if soil_service.df is not None and not soil_service.df.empty:
    st.sidebar.success(f"✓ Loaded {len(soil_service.df)} Talukas across Maharashtra")
    with st.sidebar.expander("🔍 View Soil Dataset Samples"):
        st.dataframe(soil_service.df[["Taluka", "District", "Soil Type", "pH"]].head(10), use_container_width=True)
else:
    st.sidebar.warning("Soil CSV not found, operating in heuristic regional mode.")

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Agronomic Engine Features:**
    - 🌦️ Real-time Geocoded Meteorology
    - 🧪 Local Soil Profile Auto-mapping
    - 🤖 Scikit-Learn Random Forest Classifier
    - 📊 Multi-Factor Agronomic Compatibility
    - 🛠️ Crop Stress Test & Feasibility Radar
    """
)

# --- Hero Header ---
st.markdown("""
<div class="main-header">
    <h1>🌾 KrishiMitra AI : Climate & Soil Crop Advisory</h1>
    <p>Empowering farmers with precision agricultural science: Real-time weather telemetry, localized soil data, and machine learning crop recommendations.</p>
</div>
""", unsafe_allow_html=True)

# --- SECTION 1: Location & Environmental Telemetry ---
st.subheader("📍 Step 1: Farm Location & Environmental Telemetry")

loc_col1, loc_col2 = st.columns([2, 1])

with loc_col1:
    location_mode = st.radio(
        "Choose Location Input Mode:",
        ["Select from Maharashtra Database (100+ Talukas)", "Enter Any Custom Location (All India / Global)"],
        horizontal=True
    )

    all_db_locations = soil_service.get_all_locations()
    location_displays = [item["display"] for item in all_db_locations]

    if location_mode == "Select from Maharashtra Database (100+ Talukas)" and location_displays:
        # Default to Baramati (Pune) or first item
        default_idx = 43 if len(location_displays) > 43 else 0
        selected_display = st.selectbox(
            "Select Taluka / District:",
            options=location_displays,
            index=default_idx,
            help="Select your taluka. Soil parameters will be automatically extracted from the local database."
        )
        selected_location_query = selected_display.split("(")[0].strip()
    else:
        selected_location_query = st.text_input(
            "Enter Location Name (Village, Taluka, District, or City):",
            value="Nashik",
            placeholder="e.g. Pune, Jalgaon, Ratnagiri, Satara, Solapur, Varanasi..."
        )

with loc_col2:
    st.write("")
    st.write("")
    fetch_btn = st.button("🚀 Fetch Telemetry & Analyze", type="primary", use_container_width=True)

# Session state caching for current telemetry
if "telemetry_data" not in st.session_state or fetch_btn:
    with st.spinner("Fetching meteorological telemetry and localized soil profile..."):
        weather_info = weather_service.fetch_weather(selected_location_query)
        soil_info = soil_service.fetch_soil_profile(selected_location_query)
        st.session_state["telemetry_data"] = {
            "weather": weather_info,
            "soil": soil_info,
            "query": selected_location_query
        }

current_data = st.session_state["telemetry_data"]
weather = current_data["weather"]
soil = current_data["soil"]

# Telemetry Status Pills
status_c1, status_c2 = st.columns(2)
with status_c1:
    st.info(f"🌤️ **Weather Source:** {weather['source']} | **Station:** {weather['location_name']}")
with status_c2:
    if soil.get("matched_in_database"):
        st.success(f"🌱 **Soil Source:** {soil['source']} | **Zone:** {soil['location_name']}")
    else:
        st.warning(f"🌱 **Soil Source:** {soil['source']}")

# Telemetry Metrics Grid
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Temperature</div>
        <div class="metric-value">{weather['temperature']}°C</div>
        <span style="font-size: 0.8rem; color: #666;">{weather['weather_description']}</span>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rel. Humidity</div>
        <div class="metric-value">{weather['humidity']}%</div>
        <span style="font-size: 0.8rem; color: #666;">Atmospheric</span>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rainfall</div>
        <div class="metric-value">{weather['rainfall']} mm</div>
        <span style="font-size: 0.8rem; color: #666;">Agro-Index</span>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Soil pH</div>
        <div class="metric-value">{soil['ph']}</div>
        <span style="font-size: 0.8rem; color: #666;">Range: {soil.get('ph_range', '6.5-7.5')}</span>
    </div>
    """, unsafe_allow_html=True)
with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Nitrogen (N)</div>
        <div class="metric-value">{soil['N']:.0f}</div>
        <span style="font-size: 0.8rem; color: #666;">kg/ha</span>
    </div>
    """, unsafe_allow_html=True)
with m6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Phosphorus (P)</div>
        <div class="metric-value">{soil['P']:.0f}</div>
        <span style="font-size: 0.8rem; color: #666;">kg/ha</span>
    </div>
    """, unsafe_allow_html=True)
with m7:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Potassium (K)</div>
        <div class="metric-value">{soil['K']:.0f}</div>
        <span style="font-size: 0.8rem; color: #666;">kg/ha</span>
    </div>
    """, unsafe_allow_html=True)

# Soil Details Summary Bar
st.markdown(f"""
<div style="background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 8px; padding: 10px 16px; margin: 16px 0;">
    <b>🌍 Local Soil Type:</b> {soil['soil_type']} &nbsp;|&nbsp; 
    <b>🧪 Health Assessment:</b> {soil['overall_quality']} &nbsp;|&nbsp; 
    <b>⚠️ Nutrients / Contaminants:</b> {soil.get('nutrient_summary', 'Normal')}
</div>
""", unsafe_allow_html=True)

# Option to fine-tune with farmer's actual Soil Health Card (SHC) report
with st.expander("⚙️ Optional: Fine-Tune Parameters (If you have a Soil Health Card report)"):
    st.caption("Adjust sliders below if your laboratory soil testing report shows specific values:")
    adj_col1, adj_col2, adj_col3, adj_col4 = st.columns(4)
    with adj_col1:
        custom_n = st.slider("Nitrogen (N kg/ha)", 10.0, 200.0, float(soil['N']), 5.0)
    with adj_col2:
        custom_p = st.slider("Phosphorus (P kg/ha)", 5.0, 150.0, float(soil['P']), 5.0)
    with adj_col3:
        custom_k = st.slider("Potassium (K kg/ha)", 10.0, 250.0, float(soil['K']), 5.0)
    with adj_col4:
        custom_ph = st.slider("Soil pH Level", 4.5, 9.5, float(soil['ph']), 0.1)

    # Allow overriding active parameters
    active_n = custom_n
    active_p = custom_p
    active_k = custom_k
    active_ph = custom_ph
    active_temp = weather['temperature']
    active_hum = weather['humidity']
    active_rain = weather['rainfall']
    active_soil_type = soil['soil_type']

st.divider()

# --- SECTION 2: AI Crop Recommendation Engine ---
st.subheader("🤖 Step 2: AI Top 3 Recommended Crops")
st.markdown("Ranked using a **Scikit-Learn Random Forest Classifier** combined with **Agronomic Multi-Factor Biological Matching**.")

# Run recommendation engine
top_recommendations = ml_engine.recommend_top_crops(
    n=active_n,
    p=active_p,
    k=active_k,
    temp=active_temp,
    hum=active_hum,
    ph=active_ph,
    rain=active_rain,
    soil_type=active_soil_type,
    top_k=3
)

# Display Top 3 Cards
rec_cols = st.columns(3)

for idx, rec in enumerate(top_recommendations):
    with rec_cols[idx]:
        st.markdown(f"""
        <div class="crop-card">
            <span class="rank-badge">RANK #{idx + 1}</span>
            <span class="tag-badge">{rec['category']}</span>
            <h2 style="margin: 10px 0 2px 0; color: #1b5e20;">{rec['crop']}</h2>
            <div style="font-size: 1.05rem; font-weight: 600; color: #43a047; margin-bottom: 10px;">{rec['marathi_name']}</div>
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.9rem;">
                    <span>Overall Suitability</span>
                    <span style="color: #2e7d32;">{rec['score']}%</span>
                </div>
            </div>
            <p style="font-size: 0.85rem; color: #555; margin-bottom: 6px;">
                🗓️ <b>Season:</b> {rec['season']}<br>
                💧 <b>Water Requirement:</b> {rec['water_need']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(1.0, rec['score'] / 100.0))

        with st.expander(f"📖 Scientific Rationale for {rec['crop']}", expanded=True):
            st.markdown(f"> {rec['reasoning']}")

# Multi-Factor Agronomic Radar Comparison
st.markdown("#### 📊 Environmental Suitability Comparison")
radar_col1, radar_col2 = st.columns([3, 2])

with radar_col1:
    categories = ['Temperature', 'Rainfall', 'Soil pH', 'Humidity', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    fig = go.Figure()

    colors = ['#2e7d32', '#1976d2', '#f57c00']
    for idx, rec in enumerate(top_recommendations):
        f_scores = [rec['feature_scores'].get(c, 70.0) for c in categories]
        f_scores.append(f_scores[0])  # Close radar loop
        fig.add_trace(go.Scatterpolar(
            r=f_scores,
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{rec['crop']} ({rec['score']}%)",
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=360
    )
    st.plotly_chart(fig, use_container_width=True)

with radar_col2:
    st.markdown("##### 🔍 Why This Hybrid Model Excels")
    st.markdown("""
    - **Dual Validation:** Rather than acting as a black box, our engine pairs Scikit-Learn probabilistic weights with ICAR biological tolerance limits.
    - **Soil-Climate Coupling:** Considers both rhizosphere chemistry (pH, N, P, K) and ambient meteorological stress (evapotranspiration, temperature extremes).
    - **Actionable Alignment:** Crops scoring >75% can be sown with standard farm practices, while crops between 50-75% require targeted irrigation or fertilizer intervention.
    """)

st.divider()

# --- SECTION 3: Custom Crop Feasibility Analyzer ---
st.subheader("🎯 Step 3: Custom Crop Feasibility Analyzer (Pros & Cons)")
st.markdown("Have a specific crop in mind? Test whether your climate and soil can support it before investing capital.")

custom_col1, custom_col2 = st.columns([2, 1])

with custom_col1:
    # Farmer choice chips
    preset_crops = ["Sugarcane", "Cotton", "Soybean", "Onion", "Grapes", "Pomegranate", "Banana", "Rice (Paddy)", "Wheat", "Tomato"]
    user_crop_input = st.selectbox(
        "Do you have a specific crop in mind?",
        options=preset_crops + ["Other (Type custom crop name)"],
        index=0
    )

    if user_crop_input == "Other (Type custom crop name)":
        user_crop_input = st.text_input("Enter the crop name you wish to evaluate:", placeholder="e.g. Turmeric, Ginger, Papaya, Coffee, Watermelon...")

with custom_col2:
    st.write("")
    st.write("")
    analyze_crop_btn = st.button("🔬 Stress-Test This Crop", type="primary", use_container_width=True)

# Perform Feasibility Analysis
feasibility_result = feasibility_engine.analyze_feasibility(
    crop_input=user_crop_input,
    weather={
        "temperature": active_temp,
        "humidity": active_hum,
        "rainfall": active_rain
    },
    soil={
        "soil_type": active_soil_type,
        "ph": active_ph,
        "N": active_n,
        "P": active_p,
        "K": active_k,
        "deficiencies": soil.get("deficiencies", []),
        "risks": soil.get("risks", []),
        "overall_quality": soil.get("overall_quality", "")
    }
)

# Display Feasibility Verdict Banner
v_score = feasibility_result["feasibility_score"]
v_badge = feasibility_result["verdict_badge"]

if v_badge == "success":
    st.success(f"### ✅ Viability Verdict: {feasibility_result['verdict']} (Score: {v_score}%)")
elif v_badge == "warning":
    st.warning(f"### ⚠️ Viability Verdict: {feasibility_result['verdict']} (Score: {v_score}%)")
else:
    st.error(f"### ❌ Viability Verdict: {feasibility_result['verdict']} (Score: {v_score}%)")

st.markdown(f"**Crop Selected:** `{feasibility_result['crop_name']}` ({feasibility_result['marathi_name']}) &nbsp;|&nbsp; **Category:** `{feasibility_result['category']}` &nbsp;|&nbsp; **Water Need:** `{feasibility_result['water_need']}`")

# Pros and Cons Two-Column Display
pro_col, con_col = st.columns(2)

with pro_col:
    st.markdown("#### 🟢 Positives (Pros)")
    if feasibility_result["positives"]:
        for pro in feasibility_result["positives"]:
            st.markdown(f"""<div class="pro-box">✅ {pro}</div>""", unsafe_allow_html=True)
    else:
        st.info("No significant natural advantages found for this crop in current conditions.")

with con_col:
    st.markdown("#### 🔴 Negatives (Cons & Risks)")
    if feasibility_result["negatives"]:
        for con in feasibility_result["negatives"]:
            st.markdown(f"""<div class="con-box">⚠️ {con}</div>""", unsafe_allow_html=True)
    else:
        st.success("No critical environmental bottlenecks detected!")

# Necessary Infrastructure & Mitigation
if feasibility_result["infrastructure"]:
    st.markdown("#### 🏗️ Recommended Irrigation & Protective Infrastructure")
    for infra in feasibility_result["infrastructure"]:
        st.markdown(f"""<div class="infra-box">🛠️ {infra}</div>""", unsafe_allow_html=True)

st.divider()

# --- SECTION 4: Yield Optimization & Actionable Tips ---
st.subheader("💡 Step 4: Yield Optimization & Actionable Agronomic Tips")
st.markdown("Targeted, science-backed field recommendations tailored to your soil chemistry and selected crop.")

tip_col1, tip_col2 = st.columns([1, 1])

with tip_col1:
    st.markdown(f"#### 🌾 Tailored Advice for `{feasibility_result['crop_name']}`")
    if feasibility_result["yield_tips"]:
        for tip in feasibility_result["yield_tips"]:
            st.markdown(f"""<div class="tip-box">🌱 {tip}</div>""", unsafe_allow_html=True)
    else:
        st.info("Standard package of practices applies.")

with tip_col2:
    st.markdown("#### 🧪 Soil Health & Correction Protocols")
    # Soil specific recommendations
    st_lower = active_soil_type.lower()
    st_def = [d.lower() for d in soil.get("deficiencies", [])]
    st_risk = [r.lower() for r in soil.get("risks", [])]

    soil_protocols = []

    # Zinc
    if any("zinc" in d for d in st_def) or "calcareous" in st_lower or "black" in st_lower:
        soil_protocols.append(
            "**Zinc Deficiency Correction:** Apply 25 kg/ha of Zinc Sulfate ($ZnSO_4$, 21%) into the soil during seedbed preparation. Avoid mixing directly with Phosphatic fertilizers (DAP/SSP) to prevent chemical precipitation."
        )

    # Salinity
    if any("salin" in r for r in st_risk) or "saline" in st_lower:
        soil_protocols.append(
            "**Salinity & Electrical Conductivity (EC) Management:** Apply 500–800 kg/acre Agricultural Gypsum ($CaSO_4$) to displace exchangeable sodium. Follow with furrow leaching and incorporate green manure (Dhaincha or Sunnhemp)."
        )

    # Lime reserve
    if "lime" in st_lower or active_ph > 8.0:
        soil_protocols.append(
            "**Lime Chlorosis & Iron Deficiency:** High calcium carbonate fixes iron and phosphorus. Apply elemental sulfur (25 kg/acre) or spray Chelated Iron (Fe-EDTA 12%) @ 1.5 g/liter water to prevent leaf yellowing."
        )

    # Laterite / Acidic
    if "laterite" in st_lower or active_ph < 6.2:
        soil_protocols.append(
            "**Soil Acidity Amelioration:** Apply Agricultural Lime or Dolomite (300 kg/acre) 3 weeks prior to sowing. Apply Single Super Phosphate (SSP) instead of DAP to supply both phosphorus and calcium."
        )

    # N-P-K balanced split
    soil_protocols.append(
        f"**Precision NPK Scheduling:** Current soil has N:{active_n:.0f}, P:{active_p:.0f}, K:{active_k:.0f} kg/ha. Apply 50% Nitrogen as basal and remaining 50% split at 30 & 60 days after sowing. Apply full Phosphorus and Potash at sowing."
    )

    for proto in soil_protocols:
        st.markdown(f"""<div class="tip-box">📋 {proto}</div>""", unsafe_allow_html=True)

st.divider()

# --- SECTION 5: Export / Download Farm Advisory Summary ---
st.subheader("📑 Farm Advisory Summary Report")
st.markdown("Download a formatted agricultural advisory report for documentation or sharing with field agronomists.")

report_md = f"""# KRISHIMITRA AI - CROP & SOIL ADVISORY REPORT
Generated for Farm Location: {current_data['query'].title()}
Soil Source: {soil['source']}
Weather Telemetry Source: {weather['source']}

==================================================
1. ENVIRONMENTAL & SOIL TELEMETRY
- Ambient Temperature: {weather['temperature']} °C ({weather['weather_description']})
- Atmospheric Relative Humidity: {weather['humidity']} %
- Agro-Rainfall Index: {weather['rainfall']} mm
- Soil Classification: {active_soil_type}
- Soil pH Reaction: {active_ph:.1f}
- Soil Nitrogen (N): {active_n:.1f} kg/ha
- Soil Phosphorus (P): {active_p:.1f} kg/ha
- Soil Potassium (K): {active_k:.1f} kg/ha
- Soil Quality Assessment: {soil.get('overall_quality', 'Normal')}

==================================================
2. TOP 3 SCIENTIFIC CROP RECOMMENDATIONS
Rank 1: {top_recommendations[0]['crop']} ({top_recommendations[0]['marathi_name']}) - Suitability: {top_recommendations[0]['score']}%
Reasoning: {top_recommendations[0]['reasoning']}

Rank 2: {top_recommendations[1]['crop']} ({top_recommendations[1]['marathi_name']}) - Suitability: {top_recommendations[1]['score']}%
Reasoning: {top_recommendations[1]['reasoning']}

Rank 3: {top_recommendations[2]['crop']} ({top_recommendations[2]['marathi_name']}) - Suitability: {top_recommendations[2]['score']}%
Reasoning: {top_recommendations[2]['reasoning']}

==================================================
3. CUSTOM CROP FEASIBILITY EVALUATION
Target Crop: {feasibility_result['crop_name']} ({feasibility_result['marathi_name']})
Viability Score: {feasibility_result['feasibility_score']}%
Verdict: {feasibility_result['verdict']}

Positives (Pros):
{chr(10).join(['- ' + p for p in feasibility_result['positives']])}

Negatives (Cons / Risks):
{chr(10).join(['- ' + c for c in feasibility_result['negatives']])}

Recommended Infrastructure:
{chr(10).join(['- ' + i for i in feasibility_result['infrastructure']])}

==================================================
4. ACTIONABLE YIELD OPTIMIZATION TIPS
{chr(10).join(['- ' + t for t in feasibility_result['yield_tips']])}

Soil Health Protocols:
{chr(10).join(['- ' + p for p in soil_protocols])}

Report generated by KrishiMitra AI - Precision Agritech System.
"""

st.download_button(
    label="📥 Download Printable Farm Advisory Report (.txt)",
    data=report_md,
    file_name=f"krishimitra_advisory_{current_data['query'].lower().replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True
)

st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; margin-top: 30px;">
    KrishiMitra AI Precision Agriculture Platform • Built with Streamlit, Scikit-Learn, and Meteorological APIs • Designed for Indian Farming Communities
</div>
""", unsafe_allow_html=True)
