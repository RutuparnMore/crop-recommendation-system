import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Attempt to import scikit-learn, with graceful fallback if Windows Application Control blocks its DLLs
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except (ImportError, Exception) as err:
    SKLEARN_AVAILABLE = False
    # Will use pure-NumPy statistical classification engine

# Agronomic Database of Crops: scientifically curated from ICAR & FAO parameters
AGRONOMIC_CROP_PROFILES = {
    "Cotton": {
        "marathi_name": "कापूस",
        "category": "Cash / Fibre Crop",
        "ideal_soil_types": ["Deep Black Cotton Soil", "Vertisol", "Medium Black", "Clay Loam"],
        "temp_range": (18.0, 22.0, 32.0, 38.0),      # min, opt_min, opt_max, max
        "humidity_range": (50.0, 60.0, 80.0, 90.0),
        "rainfall_range": (50.0, 65.0, 110.0, 180.0),
        "ph_range": (6.0, 6.5, 8.0, 8.5),
        "n_range": (60.0, 90.0, 130.0, 160.0),
        "p_range": (30.0, 40.0, 65.0, 90.0),
        "k_range": (20.0, 35.0, 60.0, 90.0),
        "water_need": "Moderate (600-800 mm total crop cycle)",
        "season": "Kharif",
        "rationale_template": "Cotton thrives in deep black soils that retain moisture through hot dry spells. Your soil's pH and potassium reserve support vigorous vegetative growth and boll formation."
    },
    "Soybean": {
        "marathi_name": "सोयाबीन",
        "category": "Oilseed / Legume",
        "ideal_soil_types": ["Medium Black", "Deep Black Cotton Soil", "Loam", "Clay Loam"],
        "temp_range": (18.0, 22.0, 30.0, 36.0),
        "humidity_range": (55.0, 65.0, 85.0, 92.0),
        "rainfall_range": (60.0, 75.0, 125.0, 175.0),
        "ph_range": (6.0, 6.5, 7.5, 8.2),
        "n_range": (20.0, 30.0, 50.0, 70.0),
        "p_range": (40.0, 55.0, 80.0, 100.0),
        "k_range": (30.0, 45.0, 65.0, 85.0),
        "water_need": "Moderate (450-650 mm)",
        "season": "Kharif",
        "rationale_template": "Soybean fixes atmospheric nitrogen via root nodules, making it ideal for soils with low baseline nitrogen. It thrives in well-aerated medium black soils with adequate phosphorus."
    },
    "Sugarcane": {
        "marathi_name": "ऊस",
        "category": "Cash / Sugar Crop",
        "ideal_soil_types": ["Deep Black Cotton Soil", "Alluvial", "Clay Loam", "Medium Black"],
        "temp_range": (20.0, 24.0, 35.0, 40.0),
        "humidity_range": (65.0, 75.0, 88.0, 95.0),
        "rainfall_range": (120.0, 150.0, 260.0, 350.0),
        "ph_range": (6.2, 6.8, 8.0, 8.5),
        "n_range": (110.0, 140.0, 190.0, 240.0),
        "p_range": (50.0, 65.0, 90.0, 120.0),
        "k_range": (70.0, 90.0, 140.0, 180.0),
        "water_need": "Very High (1500-2500 mm total crop cycle - Requires Assured Irrigation)",
        "season": "Annual / Adsali",
        "rationale_template": "Sugarcane produces tremendous biomass and demands rich potassium and nitrogen levels alongside heavy irrigation or deep water-retaining soils."
    },
    "Rice (Paddy)": {
        "marathi_name": "भात / तांदूळ",
        "category": "Cereal Food Grain",
        "ideal_soil_types": ["Coastal Alluvial & Lateritic", "Coastal Clay", "Laterite", "Clayey"],
        "temp_range": (20.0, 24.0, 32.0, 38.0),
        "humidity_range": (75.0, 80.0, 95.0, 99.0),
        "rainfall_range": (150.0, 180.0, 280.0, 400.0),
        "ph_range": (5.0, 5.5, 6.8, 7.5),
        "n_range": (65.0, 80.0, 110.0, 140.0),
        "p_range": (35.0, 45.0, 65.0, 85.0),
        "k_range": (30.0, 40.0, 55.0, 75.0),
        "water_need": "Very High (Standing water / 1200-1800 mm)",
        "season": "Kharif",
        "rationale_template": "Rice excels in heavy soils and coastal lateritic clay that retain moisture or experience abundant rainfall. Slightly acidic to neutral pH provides superior nutrient uptake."
    },
    "Wheat": {
        "marathi_name": "गहू",
        "category": "Cereal Food Grain",
        "ideal_soil_types": ["Medium Black", "Alluvial", "Clay Loam"],
        "temp_range": (10.0, 14.0, 24.0, 28.0),
        "humidity_range": (35.0, 45.0, 65.0, 80.0),
        "rainfall_range": (30.0, 45.0, 85.0, 130.0),
        "ph_range": (6.0, 6.5, 7.5, 8.2),
        "n_range": (80.0, 100.0, 130.0, 160.0),
        "p_range": (40.0, 50.0, 70.0, 90.0),
        "k_range": (30.0, 40.0, 60.0, 80.0),
        "water_need": "Moderate (450-650 mm across 5-6 critical irrigations)",
        "season": "Rabi (Winter)",
        "rationale_template": "Wheat thrives under cool winter night temperatures and moderate humidity. Loam to medium black soils with steady nitrogen support high tillering and grain weight."
    },
    "Pigeonpeas (Tur/Arhar)": {
        "marathi_name": "तूर",
        "category": "Pulse / Legume",
        "ideal_soil_types": ["Medium Black", "Deep Black Cotton Soil", "Red Sandy / Hill Soil"],
        "temp_range": (20.0, 26.0, 35.0, 40.0),
        "humidity_range": (40.0, 50.0, 70.0, 85.0),
        "rainfall_range": (50.0, 65.0, 110.0, 160.0),
        "ph_range": (6.0, 6.5, 7.8, 8.4),
        "n_range": (15.0, 25.0, 45.0, 60.0),
        "p_range": (45.0, 60.0, 80.0, 100.0),
        "k_range": (20.0, 30.0, 45.0, 65.0),
        "water_need": "Low to Moderate (Deep taproot reaches subsoil moisture)",
        "season": "Kharif / Semi-Arid",
        "rationale_template": "Pigeonpea has deep taproots that extract moisture and minerals from subsoil depths, making it exceptionally resilient to erratic rain in black and loamy soils."
    },
    "Chickpea (Chana/Harbara)": {
        "marathi_name": "हरभरा",
        "category": "Pulse / Legume",
        "ideal_soil_types": ["Medium Black", "Deep Black Cotton Soil", "Clay Loam"],
        "temp_range": (12.0, 16.0, 25.0, 30.0),
        "humidity_range": (15.0, 25.0, 50.0, 70.0),
        "rainfall_range": (30.0, 45.0, 80.0, 110.0),
        "ph_range": (6.2, 6.8, 8.2, 8.7),
        "n_range": (20.0, 35.0, 50.0, 65.0),
        "p_range": (50.0, 65.0, 85.0, 105.0),
        "k_range": (60.0, 75.0, 95.0, 115.0),
        "water_need": "Low (250-400 mm / Conserved residual moisture)",
        "season": "Rabi",
        "rationale_template": "Chickpea requires dry, cool climates and excels in moisture-retentive black soils using residual monsoon moisture. High potassium promotes sturdy pods."
    },
    "Grapes": {
        "marathi_name": "द्राक्षे",
        "category": "Horticulture / Fruit",
        "ideal_soil_types": ["Medium Black", "Sandy Loam", "Gravelly Black"],
        "temp_range": (15.0, 22.0, 35.0, 42.0),
        "humidity_range": (40.0, 55.0, 75.0, 85.0),
        "rainfall_range": (40.0, 55.0, 80.0, 120.0),
        "ph_range": (6.5, 7.0, 8.2, 8.5),
        "n_range": (20.0, 35.0, 60.0, 85.0),
        "p_range": (90.0, 120.0, 150.0, 180.0),
        "k_range": (140.0, 180.0, 230.0, 280.0),
        "water_need": "Moderate (Precision Drip Irrigation Mandatory)",
        "season": "Perennial",
        "rationale_template": "Grapes require high potassium availability and well-drained soils with neutral to mildly calcareous pH. High rainfall at maturity is harmful, making dry spells ideal."
    },
    "Pomegranate": {
        "marathi_name": "डाळिंब",
        "category": "Horticulture / Fruit",
        "ideal_soil_types": ["Medium to Shallow Black", "Loam", "Well-drained Gravelly"],
        "temp_range": (18.0, 24.0, 36.0, 42.0),
        "humidity_range": (30.0, 45.0, 65.0, 80.0),
        "rainfall_range": (35.0, 50.0, 85.0, 130.0),
        "ph_range": (6.5, 7.0, 8.2, 8.5),
        "n_range": (20.0, 35.0, 55.0, 75.0),
        "p_range": (25.0, 40.0, 60.0, 80.0),
        "k_range": (40.0, 55.0, 80.0, 105.0),
        "water_need": "Low to Moderate (Extremely Drought Tolerant)",
        "season": "Perennial (Bahar treatment)",
        "rationale_template": "Pomegranate thrives in semi-arid zones with hot summers and low humidity. It performs best in medium, well-drained calcareous soils free of water stagnation."
    },
    "Banana": {
        "marathi_name": "केळी",
        "category": "Horticulture / Fruit",
        "ideal_soil_types": ["Deep Black Cotton Soil", "Alluvial Loam", "Clay Loam"],
        "temp_range": (20.0, 25.0, 33.0, 38.0),
        "humidity_range": (65.0, 75.0, 90.0, 98.0),
        "rainfall_range": (80.0, 100.0, 160.0, 220.0),
        "ph_range": (6.0, 6.5, 7.5, 8.2),
        "n_range": (90.0, 110.0, 140.0, 180.0),
        "p_range": (65.0, 80.0, 100.0, 125.0),
        "k_range": (50.0, 65.0, 90.0, 120.0),
        "water_need": "High (1800-2200 mm - Heavy feeder requiring regular drip cycles)",
        "season": "Annual (11-12 months)",
        "rationale_template": "Banana is a ravenous nutrient consumer requiring fertile, organic-rich soil, strong moisture availability, and warm tropical temperatures."
    },
    "Mango": {
        "marathi_name": "आंबा (Alphonso / Kesar)",
        "category": "Horticulture / Fruit",
        "ideal_soil_types": ["Laterite", "Laterite & Coastal Clay", "Red Sandy", "Deep Alluvial"],
        "temp_range": (22.0, 26.0, 36.0, 42.0),
        "humidity_range": (45.0, 55.0, 80.0, 90.0),
        "rainfall_range": (70.0, 90.0, 150.0, 250.0),
        "ph_range": (5.5, 6.0, 7.2, 7.8),
        "n_range": (20.0, 30.0, 50.0, 70.0),
        "p_range": (20.0, 30.0, 50.0, 70.0),
        "k_range": (25.0, 35.0, 55.0, 75.0),
        "water_need": "Moderate (Requires dry dry-spell for flowering flush)",
        "season": "Perennial",
        "rationale_template": "Renowned Konkan laterite soils and coastal climates offer perfect drainage and micro-climate for king of fruits (like Alphonso). Dry flowering season secures high fruit set."
    },
    "Coconut": {
        "marathi_name": "नारळ",
        "category": "Plantation / Cash Crop",
        "ideal_soil_types": ["Coastal Alluvial & Saline", "Laterite & Coastal Clay", "Sandy Loam"],
        "temp_range": (22.0, 26.0, 34.0, 38.0),
        "humidity_range": (70.0, 80.0, 95.0, 99.0),
        "rainfall_range": (120.0, 150.0, 260.0, 380.0),
        "ph_range": (5.5, 6.2, 7.8, 8.5),
        "n_range": (20.0, 35.0, 60.0, 80.0),
        "p_range": (15.0, 25.0, 45.0, 65.0),
        "k_range": (35.0, 50.0, 80.0, 110.0),
        "water_need": "High (Coastal high humidity & constant subsoil moisture)",
        "season": "Perennial",
        "rationale_template": "Coconut flourishes in coastal belts with saline/alluvial or lateritic soils, high atmospheric humidity, and steady subsoil water tables."
    },
    "Onion": {
        "marathi_name": "कांदा",
        "category": "Horticultural Vegetable",
        "ideal_soil_types": ["Medium Black", "Sandy Loam", "Silty Loam"],
        "temp_range": (14.0, 18.0, 30.0, 35.0),
        "humidity_range": (45.0, 55.0, 75.0, 85.0),
        "rainfall_range": (40.0, 55.0, 90.0, 130.0),
        "ph_range": (6.0, 6.5, 7.5, 8.2),
        "n_range": (75.0, 95.0, 125.0, 150.0),
        "p_range": (35.0, 50.0, 75.0, 95.0),
        "k_range": (55.0, 75.0, 105.0, 135.0),
        "water_need": "Moderate (Sensitive to waterlogging, requires friable soil)",
        "season": "Kharif / Late Kharif / Rabi",
        "rationale_template": "Onions demand friable, well-aerated medium black soils rich in potassium for solid bulb formation and skin color without bulb rotting."
    },
    "Maize (Corn)": {
        "marathi_name": "मका",
        "category": "Cereal / Fodder Grain",
        "ideal_soil_types": ["Medium Black", "Alluvial", "Clay Loam"],
        "temp_range": (18.0, 22.0, 30.0, 35.0),
        "humidity_range": (50.0, 60.0, 75.0, 85.0),
        "rainfall_range": (50.0, 70.0, 115.0, 160.0),
        "ph_range": (5.8, 6.5, 7.5, 8.0),
        "n_range": (70.0, 90.0, 120.0, 150.0),
        "p_range": (40.0, 55.0, 75.0, 95.0),
        "k_range": (20.0, 30.0, 45.0, 65.0),
        "water_need": "Moderate (500-700 mm)",
        "season": "Kharif / Rabi",
        "rationale_template": "Maize is a C4 high-efficiency photosynthetic crop that responds aggressively to nitrogen in deep, well-drained loams and black soils."
    },
    "Sorghum (Jowar)": {
        "marathi_name": "ज्वारी",
        "category": "Cereal / Dryland Millet",
        "ideal_soil_types": ["Medium to Shallow Black", "Deep Black Cotton Soil", "Vertisol"],
        "temp_range": (20.0, 26.0, 35.0, 42.0),
        "humidity_range": (30.0, 45.0, 65.0, 80.0),
        "rainfall_range": (35.0, 50.0, 85.0, 130.0),
        "ph_range": (6.0, 6.5, 8.2, 8.7),
        "n_range": (50.0, 70.0, 95.0, 120.0),
        "p_range": (25.0, 35.0, 55.0, 75.0),
        "k_range": (30.0, 45.0, 65.0, 85.0),
        "water_need": "Low (Extremely drought resistant / Ideal for dryland farming)",
        "season": "Kharif / Rabi",
        "rationale_template": "Jowar is the lifeline of dryland farmers. Its xerophytic nature withstands moisture stress and produces hardy grain even on shallow black soils with high calcium carbonate."
    },
    "Pearl Millet (Bajra)": {
        "marathi_name": "बाजरी",
        "category": "Millet / Coarse Grain",
        "ideal_soil_types": ["Light to Medium Black", "Red Sandy / Hill Soil", "Sandy Loam"],
        "temp_range": (24.0, 28.0, 38.0, 44.0),
        "humidity_range": (25.0, 40.0, 60.0, 75.0),
        "rainfall_range": (25.0, 40.0, 70.0, 110.0),
        "ph_range": (6.5, 7.0, 8.2, 8.8),
        "n_range": (40.0, 60.0, 85.0, 110.0),
        "p_range": (20.0, 30.0, 50.0, 70.0),
        "k_range": (25.0, 35.0, 55.0, 75.0),
        "water_need": "Very Low (250-350 mm)",
        "season": "Kharif",
        "rationale_template": "Bajra thrives in high temperatures and low rainfall on lighter or sandy soils where other commercial crops fail. High nutrient use efficiency."
    },
    "Groundnut": {
        "marathi_name": "भुईमूग",
        "category": "Oilseed / Legume",
        "ideal_soil_types": ["Red Sandy / Hill Soil", "Sandy Loam", "Medium Black (Light)"],
        "temp_range": (22.0, 26.0, 34.0, 38.0),
        "humidity_range": (45.0, 55.0, 75.0, 85.0),
        "rainfall_range": (50.0, 65.0, 105.0, 150.0),
        "ph_range": (6.0, 6.5, 7.5, 8.0),
        "n_range": (20.0, 30.0, 45.0, 60.0),
        "p_range": (35.0, 50.0, 75.0, 95.0),
        "k_range": (30.0, 45.0, 65.0, 85.0),
        "water_need": "Moderate (400-600 mm - Friable soil needed for peg penetration)",
        "season": "Kharif / Summer",
        "rationale_template": "Groundnut requires loose, friable sandy loam or light black soil so underground pegs can easily penetrate and expand into healthy pods."
    },
    "Watermelon": {
        "marathi_name": "कलिंगड",
        "category": "Horticultural Fruit / Cucurbit",
        "ideal_soil_types": ["Sandy Loam", "Alluvial", "River Basin Loam"],
        "temp_range": (22.0, 26.0, 34.0, 40.0),
        "humidity_range": (50.0, 65.0, 85.0, 92.0),
        "rainfall_range": (35.0, 50.0, 80.0, 120.0),
        "ph_range": (6.0, 6.5, 7.2, 7.8),
        "n_range": (70.0, 90.0, 120.0, 150.0),
        "p_range": (20.0, 30.0, 50.0, 70.0),
        "k_range": (40.0, 55.0, 75.0, 95.0),
        "water_need": "Moderate (Drip irrigation with dry sunny atmosphere)",
        "season": "Zaid / Summer",
        "rationale_template": "Warm sunny days and warm nights maximize sugar accumulation (Brix level). Requires light, porous soil without water saturation."
    },
    "Papaya": {
        "marathi_name": "पपई",
        "category": "Horticulture / Fruit",
        "ideal_soil_types": ["Medium Black", "Alluvial Loam", "Sandy Loam"],
        "temp_range": (22.0, 26.0, 35.0, 40.0),
        "humidity_range": (60.0, 70.0, 88.0, 95.0),
        "rainfall_range": (80.0, 110.0, 180.0, 250.0),
        "ph_range": (6.2, 6.8, 7.5, 8.0),
        "n_range": (45.0, 60.0, 90.0, 120.0),
        "p_range": (40.0, 60.0, 85.0, 110.0),
        "k_range": (45.0, 60.0, 85.0, 110.0),
        "water_need": "Moderate to High (Extremely vulnerable to root rot/waterlogging)",
        "season": "Semi-Perennial",
        "rationale_template": "Papaya grows rapidly and delivers heavy fruit load in tropical climates, but strictly demands raised beds or quick-draining soil to prevent collar rot."
    },
    "Tomato": {
        "marathi_name": "टोमॅटो",
        "category": "Vegetable / Cash Horticulture",
        "ideal_soil_types": ["Medium Black", "Loam", "Red Loam"],
        "temp_range": (16.0, 20.0, 30.0, 35.0),
        "humidity_range": (50.0, 60.0, 75.0, 85.0),
        "rainfall_range": (40.0, 60.0, 100.0, 140.0),
        "ph_range": (6.0, 6.5, 7.5, 8.2),
        "n_range": (80.0, 100.0, 130.0, 160.0),
        "p_range": (50.0, 70.0, 95.0, 120.0),
        "k_range": (60.0, 80.0, 110.0, 140.0),
        "water_need": "Moderate (Regular drip schedule / Calcium for blossom-end rot)",
        "season": "All-season (Kharif/Rabi/Summer)",
        "rationale_template": "High-value horticulture crop that flourishes in fertile, well-drained loams and black soils. High potash ensures firm skin and extended transport shelf-life."
    }
}


class PureNumpyCropClassifier:
    """
    High-performance pure-Python/NumPy statistical classifier (Gaussian Kernel & Normalized Distance).
    Ensures 100% operational resilience when Windows Application Control blocks third-party C-extension DLLs.
    """

    def __init__(self, crops_profiles: Dict[str, Any]):
        self.classes_ = list(crops_profiles.keys())
        self.profiles = crops_profiles
        self._build_centroids()

    def _build_centroids(self):
        """Builds optimal parameter centroids and tolerance bandwidths for each crop."""
        self.feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.centroids = {}
        self.bandwidths = {}

        for crop, prof in self.profiles.items():
            # Midpoints of optimal ranges
            t_mid = (prof["temp_range"][1] + prof["temp_range"][2]) / 2.0
            h_mid = (prof["humidity_range"][1] + prof["humidity_range"][2]) / 2.0
            r_mid = (prof["rainfall_range"][1] + prof["rainfall_range"][2]) / 2.0
            p_mid = (prof["ph_range"][1] + prof["ph_range"][2]) / 2.0
            n_mid = (prof["n_range"][1] + prof["n_range"][2]) / 2.0
            phos_mid = (prof["p_range"][1] + prof["p_range"][2]) / 2.0
            k_mid = (prof["k_range"][1] + prof["k_range"][2]) / 2.0

            # Tolerance spread
            t_span = max(5.0, (prof["temp_range"][3] - prof["temp_range"][0]) / 2.0)
            h_span = max(10.0, (prof["humidity_range"][3] - prof["humidity_range"][0]) / 2.0)
            r_span = max(20.0, (prof["rainfall_range"][3] - prof["rainfall_range"][0]) / 2.0)
            p_span = max(0.5, (prof["ph_range"][3] - prof["ph_range"][0]) / 2.0)
            n_span = max(15.0, (prof["n_range"][3] - prof["n_range"][0]) / 2.0)
            phos_span = max(10.0, (prof["p_range"][3] - prof["p_range"][0]) / 2.0)
            k_span = max(15.0, (prof["k_range"][3] - prof["k_range"][0]) / 2.0)

            self.centroids[crop] = np.array([n_mid, phos_mid, k_mid, t_mid, h_mid, p_mid, r_mid], dtype=float)
            self.bandwidths[crop] = np.array([n_span, phos_span, k_span, t_span, h_span, p_span, r_span], dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Computes softmax probability distributions across all candidate crops."""
        x_vec = np.array([
            X["N"].iloc[0], X["P"].iloc[0], X["K"].iloc[0],
            X["temperature"].iloc[0], X["humidity"].iloc[0],
            X["ph"].iloc[0], X["rainfall"].iloc[0]
        ], dtype=float)

        log_likelihoods = []
        for crop in self.classes_:
            mu = self.centroids[crop]
            sigma = self.bandwidths[crop]
            # Normalized Mahalanobis-like squared distance
            normalized_diff = (x_vec - mu) / sigma
            dist_sq = np.sum(normalized_diff ** 2)
            log_likelihoods.append(-0.5 * dist_sq)

        log_arr = np.array(log_likelihoods)
        # Softmax with numerical stability
        max_log = np.max(log_arr)
        exp_vals = np.exp((log_arr - max_log) / 1.5)
        probs = exp_vals / np.sum(exp_vals)
        return np.array([probs])


class CropRecommendationEngine:
    """Hybrid Machine Learning and Agronomic Multi-Dimensional Biological Engine."""

    MODEL_PATH = "data/crop_rf_model.joblib"

    def __init__(self):
        self.crops = list(AGRONOMIC_CROP_PROFILES.keys())
        self.model = None
        self.engine_type = "Pure-NumPy Agronomic ML"
        self._initialize_model()

    def _generate_synthetic_training_dataset(self, samples_per_crop: int = 120) -> pd.DataFrame:
        """
        Generates a scientifically grounded agronomic dataset based on the exact
        ICAR/FAO biological envelopes for each crop, including natural environmental noise.
        """
        np.random.seed(42)
        records = []

        for crop_name, profile in AGRONOMIC_CROP_PROFILES.items():
            t_min, t_opt_min, t_opt_max, t_max = profile["temp_range"]
            h_min, h_opt_min, h_opt_max, h_max = profile["humidity_range"]
            r_min, r_opt_min, r_opt_max, r_max = profile["rainfall_range"]
            p_min, p_opt_min, p_opt_max, p_max = profile["ph_range"]
            n_min, n_opt_min, n_opt_max, n_max = profile["n_range"]
            phos_min, phos_opt_min, phos_opt_max, phos_max = profile["p_range"]
            k_min, k_opt_min, k_opt_max, k_max = profile["k_range"]

            for _ in range(samples_per_crop):
                is_optimal = np.random.rand() < 0.75

                if is_optimal:
                    temp = np.random.uniform(t_opt_min, t_opt_max)
                    hum = np.random.uniform(h_opt_min, h_opt_max)
                    rain = np.random.uniform(r_opt_min, r_opt_max)
                    ph = np.random.uniform(p_opt_min, p_opt_max)
                    n = np.random.uniform(n_opt_min, n_opt_max)
                    p_val = np.random.uniform(phos_opt_min, phos_opt_max)
                    k_val = np.random.uniform(k_opt_min, k_opt_max)
                else:
                    temp = np.random.uniform(t_min, t_max)
                    hum = np.random.uniform(h_min, h_max)
                    rain = np.random.uniform(r_min, r_max)
                    ph = np.random.uniform(p_min, p_max)
                    n = np.random.uniform(n_min, n_max)
                    p_val = np.random.uniform(phos_min, phos_max)
                    k_val = np.random.uniform(k_min, k_max)

                records.append({
                    "N": round(max(5.0, n), 1),
                    "P": round(max(5.0, p_val), 1),
                    "K": round(max(5.0, k_val), 1),
                    "temperature": round(temp, 1),
                    "humidity": round(hum, 1),
                    "ph": round(ph, 2),
                    "rainfall": round(max(10.0, rain), 1),
                    "label": crop_name
                })

        return pd.DataFrame(records)

    def _initialize_model(self) -> None:
        """Loads or trains ML model. Uses Scikit-learn if permitted by OS, otherwise pure-NumPy."""
        os.makedirs(os.path.dirname(self.MODEL_PATH) or "data", exist_ok=True)

        if SKLEARN_AVAILABLE:
            try:
                if os.path.exists(self.MODEL_PATH):
                    self.model = joblib.load(self.MODEL_PATH)
                    self.engine_type = "Scikit-Learn Random Forest"
                    return
                
                df = self._generate_synthetic_training_dataset(samples_per_crop=140)
                X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
                y = df["label"]

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1))
                ])
                pipeline.fit(X, y)
                self.model = pipeline
                self.engine_type = "Scikit-Learn Random Forest"
                try:
                    joblib.dump(self.model, self.MODEL_PATH)
                except Exception:
                    pass
                return
            except Exception as e:
                print(f"Scikit-learn initialization warning: {e}. Falling back to NumPy Classifier.")

        # Robust Fallback to Pure-NumPy Engine
        self.model = PureNumpyCropClassifier(AGRONOMIC_CROP_PROFILES)
        self.engine_type = "Pure-NumPy Agronomic ML (Policy-Safe)"

    def _calculate_feature_compatibility(self, val: float, val_range: Tuple[float, float, float, float]) -> float:
        """
        Calculates a continuous suitability score (0.0 to 1.0) for an agronomic feature.
        val_range = (min_tolerable, opt_min, opt_max, max_tolerable)
        """
        v_min, v_opt_min, v_opt_max, v_max = val_range

        if v_opt_min <= val <= v_opt_max:
            return 1.0
        elif v_min <= val < v_opt_min:
            return 0.5 + 0.5 * ((val - v_min) / (v_opt_min - v_min + 1e-5))
        elif v_opt_max < val <= v_max:
            return 0.5 + 0.5 * ((v_max - val) / (v_max - v_opt_max + 1e-5))
        else:
            dist = min(abs(val - v_min), abs(val - v_max))
            decay = max(0.05, 0.45 - (dist / (v_max - v_min + 1e-5)) * 0.4)
            return decay

    def evaluate_crop_suitability(
        self,
        crop_name: str,
        n: float,
        p: float,
        k: float,
        temp: float,
        hum: float,
        ph: float,
        rain: float,
        soil_type: str = ""
    ) -> Dict[str, Any]:
        """Calculates precise multi-dimensional scientific compatibility for a specific crop."""
        if crop_name not in AGRONOMIC_CROP_PROFILES:
            return {"score": 50.0, "details": {}}

        prof = AGRONOMIC_CROP_PROFILES[crop_name]

        n_score = self._calculate_feature_compatibility(n, prof["n_range"])
        p_score = self._calculate_feature_compatibility(p, prof["p_range"])
        k_score = self._calculate_feature_compatibility(k, prof["k_range"])
        t_score = self._calculate_feature_compatibility(temp, prof["temp_range"])
        h_score = self._calculate_feature_compatibility(hum, prof["humidity_range"])
        ph_score = self._calculate_feature_compatibility(ph, prof["ph_range"])
        r_score = self._calculate_feature_compatibility(rain, prof["rainfall_range"])

        soil_bonus = 0.0
        soil_lower = soil_type.lower()
        for ideal_st in prof["ideal_soil_types"]:
            if any(term in soil_lower for term in ideal_st.lower().split()):
                soil_bonus = 0.08
                break

        weights = {
            "temperature": 0.20,
            "rainfall": 0.22,
            "ph": 0.16,
            "humidity": 0.14,
            "nitrogen": 0.10,
            "phosphorus": 0.09,
            "potassium": 0.09
        }

        raw_score = (
            t_score * weights["temperature"] +
            r_score * weights["rainfall"] +
            ph_score * weights["ph"] +
            h_score * weights["humidity"] +
            n_score * weights["nitrogen"] +
            p_score * weights["phosphorus"] +
            k_score * weights["potassium"] +
            soil_bonus
        )

        final_score = min(99.0, max(15.0, raw_score * 100.0))

        return {
            "score": round(final_score, 1),
            "feature_scores": {
                "Temperature": round(t_score * 100, 1),
                "Rainfall": round(r_score * 100, 1),
                "Soil pH": round(ph_score * 100, 1),
                "Humidity": round(h_score * 100, 1),
                "Nitrogen (N)": round(n_score * 100, 1),
                "Phosphorus (P)": round(p_score * 100, 1),
                "Potassium (K)": round(k_score * 100, 1),
            },
            "profile": prof
        }

    def generate_recommendation_reasoning(
        self,
        crop_name: str,
        n: float,
        p: float,
        k: float,
        temp: float,
        hum: float,
        ph: float,
        rain: float,
        soil_type: str,
        suitability: Dict[str, Any]
    ) -> str:
        """Synthesizes dynamic, explainable agricultural reasoning for the recommendation."""
        prof = AGRONOMIC_CROP_PROFILES.get(crop_name, {})
        st_name = soil_type if soil_type else "your local soil"

        reasons = []

        # Soil factor
        if "black" in soil_type.lower() and crop_name in ["Cotton", "Soybean", "Sorghum (Jowar)", "Chickpea (Chana/Harbara)"]:
            reasons.append(f"your **{st_name}** provides deep root anchorage and high moisture retention")
        elif "laterite" in soil_type.lower() and crop_name in ["Mango", "Coconut", "Rice (Paddy)"]:
            reasons.append(f"your **{st_name}** provides the superior drainage and organic matter that {crop_name} thrives in")
        elif ph >= 7.0 and ph <= 8.2:
            reasons.append(f"the soil pH of **{ph:.1f}** is well-buffered for optimal micronutrient uptake")
        else:
            reasons.append(f"the current soil conditions match the root growth envelope for {crop_name}")

        # Weather factor
        t_min, t_opt_min, t_opt_max, t_max = prof.get("temp_range", (15, 20, 30, 35))
        if t_opt_min <= temp <= t_opt_max:
            reasons.append(f"current temperature (**{temp:.1f}°C**) is in the optimal vegetative zone ({t_opt_min:.0f}–{t_opt_max:.0f}°C)")
        else:
            reasons.append(f"temperature (**{temp:.1f}°C**) remains within acceptable seasonal limits")

        # Water/Rainfall factor
        r_min, r_opt_min, r_opt_max, r_max = prof.get("rainfall_range", (40, 60, 120, 180))
        if rain < 70 and crop_name in ["Pigeonpeas (Tur/Arhar)", "Sorghum (Jowar)", "Pearl Millet (Bajra)", "Pomegranate", "Chickpea (Chana/Harbara)"]:
            reasons.append(f"this crop has exceptional drought tolerance suited for your expected rainfall of **{rain:.0f} mm**")
        elif rain >= 120 and crop_name in ["Rice (Paddy)", "Sugarcane", "Banana", "Coconut"]:
            reasons.append(f"the high moisture level and expected rainfall (**{rain:.0f} mm**) satisfy its substantial water appetite")
        else:
            reasons.append(f"regional rainfall conditions (**{rain:.0f} mm**) support steady growth with standard irrigation")

        # Potassium factor
        if k >= 65 and crop_name in ["Grapes", "Onion", "Cotton", "Sugarcane"]:
            reasons.append(f"the high potassium reserve (**{k:.0f} kg/ha**) enhances fruit quality and boll strength")

        reason_str = "; furthermore, ".join(reasons)
        return f"{crop_name} is recommended because {reason_str}. {prof.get('rationale_template', '')}"

    def recommend_top_crops(
        self,
        n: float,
        p: float,
        k: float,
        temp: float,
        hum: float,
        ph: float,
        rain: float,
        soil_type: str = "",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Runs the Machine Learning classifier and agronomic evaluator
        to return the top K scientifically ranked crops with reasoning.
        """
        features_df = pd.DataFrame([{
            "N": n, "P": p, "K": k,
            "temperature": temp, "humidity": hum,
            "ph": ph, "rainfall": rain
        }])

        ml_probs = {}
        if self.model is not None:
            classes = self.model.classes_
            probs = self.model.predict_proba(features_df)[0]
            for c, pr in zip(classes, probs):
                ml_probs[c] = float(pr)

        candidates = []
        for crop_name in self.crops:
            eval_res = self.evaluate_crop_suitability(
                crop_name=crop_name,
                n=n, p=p, k=k,
                temp=temp, hum=hum,
                ph=ph, rain=rain,
                soil_type=soil_type
            )

            ml_p = ml_probs.get(crop_name, 0.05)
            ml_score = ml_p * 100.0
            agri_score = eval_res["score"]

            blended_score = round(0.40 * ml_score + 0.60 * agri_score, 1)

            reasoning = self.generate_recommendation_reasoning(
                crop_name=crop_name,
                n=n, p=p, k=k,
                temp=temp, hum=hum,
                ph=ph, rain=rain,
                soil_type=soil_type,
                suitability=eval_res
            )

            candidates.append({
                "crop": crop_name,
                "marathi_name": eval_res["profile"].get("marathi_name", ""),
                "category": eval_res["profile"].get("category", ""),
                "score": blended_score,
                "ml_probability": round(ml_p * 100, 1),
                "agronomic_compatibility": agri_score,
                "feature_scores": eval_res["feature_scores"],
                "reasoning": reasoning,
                "season": eval_res["profile"].get("season", ""),
                "water_need": eval_res["profile"].get("water_need", "")
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
