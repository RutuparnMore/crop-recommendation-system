# 🌾 KrishiMitra AI: Climate & Soil-Driven Crop Advisory System

**KrishiMitra AI** is a state-of-the-art Agritech decision support platform built using **Python, Streamlit, Pandas, and Scikit-learn**. It empowers farmers, agronomists, and agricultural extension workers to make scientifically backed crop selection choices by bridging live meteorological telemetry, localized soil health mapping, and explainable Machine Learning.

---

## 🏗️ Architecture & Component Overview

```
d:/crop recommandation/
├── app.py                     # Streamlit frontend & interactive dashboard
├── requirements.txt           # Python dependency specifications
├── test_suite.py              # Automated validation test suite
├── README.md                  # Comprehensive technical documentation
├── data/
│   ├── soil_database.csv      # Localized soil database (100+ Maharashtra talukas)
│   └── crop_rf_model.joblib   # Trained Scikit-Learn Random Forest model
├── utils/
│   ├── __init__.py
│   ├── soil_service.py        # Soil profile parser, nutrient derivation & heuristics
│   ├── weather_service.py     # Multi-engine meteorological service (OpenWeatherMap + Open-Meteo)
│   ├── ml_engine.py           # ML classifier, agronomic compatibility & explainability logic
│   └── feasibility_engine.py  # Custom crop stress test (Pros, Cons, Infrastructure, Yield Tips)
└── .streamlit/
    └── config.toml            # Agritech green UI theme configuration
```

---

## 🌟 Core Features

### 1. Location-Based Telemetry & Soil Mapping
- **Dual Location Input Modes:**
  - **Database Mode:** Instant autocomplete selector across **100+ talukas and districts in Maharashtra** (Thane, Palghar, Raigad, Ratnagiri, Sindhudurg, Pune, Satara, Sangli, Solapur, Kolhapur, Ahmednagar, Dhule, Jalgaon, Nandurbar, Sambhajinagar, Jalna, Beed, Dharashiv, Nanded, Latur, Parbhani, Hingoli, Amravati, Akola, Buldhana, Washim, Yavatmal, Nagpur, Wardha, Bhandara, Gondia, Chandrapur, Gadchiroli).
  - **Free-Text Search:** Enter any village, taluka, district, or city across India or globally.
- **Meteorological Data:**
  - Real-time temperature (°C), relative humidity (%), and agro-rainfall index (mm).
- **Localized Soil Parameters:**
  - Extracts Soil Type, pH range, median pH, and derives quantitative **Nitrogen (N)**, **Phosphorus (P)**, and **Potassium (K)** values in kg/ha.
  - Detects micronutrient deficiencies (e.g. Zinc deficiency in calcareous black soils) and environmental hazards (salinity in coastal alluvial soils).
  - **Soil Health Card Override:** Interactive sliders allowing farmers to fine-tune N, P, K, and pH if they have exact laboratory soil test results.
  - **Zero-Dependency Resilience:** If a queried location is not present in the CSV database, the app automatically deploys an agro-climatic heuristic model with zero crashes.

### 2. AI Crop Recommendation Engine (Scikit-learn + Agronomic Model)
- Hybrid recommendation system:
  1. **Scikit-learn Random Forest Classifier:** Evaluates 7-dimensional non-linear interactions across `[N, P, K, Temperature, Humidity, pH, Rainfall]`.
  2. **Biological Envelope Compatibility:** Evaluates crop survival and flowering tolerances based on ICAR (Indian Council of Agricultural Research) & FAO agronomic standards.
- Outputs the **Top 3 Recommended Crops** with:
  - Ranked confidence score (%)
  - Regional/Marathi vernacular names
  - Growing season (Kharif, Rabi, Summer, Perennial)
  - Water requirement category
  - **Deep Scientific Reasoning:** Dynamically explains *why* the crop is recommended for the exact local soil and weather conditions.
  - Interactive multi-dimensional radar comparison chart.

### 3. Custom Crop Feasibility Analyzer (Pros & Cons)
- Secondary input field: *"Do you have a specific crop in mind?"* (e.g. Sugarcane, Cotton, Soybean, Onion, Grapes, Pomegranate, Banana, Rice, Wheat, Tomato, or any custom crop).
- Conducts an agronomic stress test against the farmer's local soil and weather:
  - **Viability Score (0–100%) & Verdict Banner** (Highly Feasible / Moderately Feasible / High-Risk).
  - 🟢 **Positives (Pros):** Natural environmental advantages (e.g., thermal window alignment, ideal soil pH, potassium sufficiency).
  - 🔴 **Negatives (Cons / Risks):** Limiting factors (e.g., rainfall deficit, alkaline chlorosis risk, soil salinity, moisture stress).
  - 🏗️ **Required Infrastructure & Mitigations:** Actionable infrastructure suggestions (e.g. drip irrigation with fertigation units, broad bed furrows, sub-surface salt leaching lines, shade nets).

### 4. Yield Optimization & Actionable Tips
- Tailored agricultural guidance for the selected crops and local soil quality:
  - **Zinc Deficiency Correction:** \(ZnSO_4\) application dosage and timing.
  - **Salinity Management:** Agricultural Gypsum (\(CaSO_4\)) application and green manuring.
  - **Soil Acidity / Alkalinity Amelioration:** Agricultural lime for acidic laterites; sulfur/FYM for calcareous black soils.
  - **Balanced NPK Scheduling:** Customized basal and split-dose fertilizer programs.
  - **Integrated Pest Management (IPM):** Pheromone traps, sticky sheets, and mulch specifications.
- **Exportable Advisory Report:** One-click download of the complete advisory as a formatted `.txt` report.

---

## 🔌 Connecting the Weather API & Soil Database

### A. Weather API Configuration
The application supports **two complementary weather services**:

1. **Free / Keyless Default (Open-Meteo):**
   - Works immediately out-of-the-box with **no registration, no keys, and no cost**.
   - Automatically geocodes location names, retrieves current temperature and humidity, and computes cumulative precipitation.

2. **OpenWeatherMap Integration:**
   - To use OpenWeatherMap, get a free API key at [openweathermap.org](https://openweathermap.org/api).
   - In the app sidebar, paste your API key into the **"OpenWeatherMap API Key (Optional)"** field.
   - The app will prioritize OpenWeatherMap live endpoints and fallback seamlessly to Open-Meteo if needed.
   - Alternatively, you can set an environment variable:
     ```bash
     export OPENWEATHER_API_KEY="your_api_key_here"
     ```

### B. Soil Dataset Configuration & Expansion
- The database is stored at `data/soil_database.csv`.
- Format:
  ```csv
  Village/Taluka,Soil Type,pH,Nutrient/Contamination Value
  Bhiwandi (Thane),Coastal Alluvial & Lateritic,6.0-7.5,"High Salinity risk, Moderate Phosphorus & Potash"
  Baramati (Pune),Medium Black (Vertisol) to Red Soil in Ghats,6.5-8.5,"Generally fertile, low N & P"
  Atpadi (Sangli),Medium to Deep Black,7.5-8.5,"High Calcium Carbonates, Rich in Potassium, Deficient in Zinc & N"
  ```
- **How it works:**
  - `utils/soil_service.py` parses `Village/Taluka` into Taluka and District components.
  - Parses pH ranges into min, max, and median values.
  - Converts qualitative descriptions (`"Deficient in Zinc & N"`, `"Rich in Potassium"`, etc.) into quantitative N, P, and K estimates (kg/ha) using agronomic rules.
  - To add new villages or districts, simply append new rows to `data/soil_database.csv`.
  - If a user enters a location not in the CSV, the system falls back to an agro-climatic heuristic model, ensuring the platform never breaks.

---

## 🚀 Installation & Running the Application

### 1. Clone / Navigate to Project Directory
```bash
cd "d:/crop recommandation"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_suite.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.
