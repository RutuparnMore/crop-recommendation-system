import re
import os
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

DEFAULT_SOIL_PROFILES = {
    "Deep Black Cotton Soil (Vertisol)": {
        "ph": 7.8, "N": 40.0, "P": 45.0, "K": 85.0,
        "texture": "Clayey", "organic_matter": "Moderate",
        "deficiencies": ["Zinc", "Nitrogen"],
        "risks": ["Waterlogging under excessive rain", "Lime-induced chlorosis"],
        "quality": "Highly Fertile, High Moisture Retention, Calcareous"
    },
    "Medium Black Soil": {
        "ph": 7.4, "N": 50.0, "P": 40.0, "K": 65.0,
        "texture": "Clay Loam", "organic_matter": "Moderate",
        "deficiencies": ["Nitrogen", "Phosphorus"],
        "risks": ["Moderate drought sensitivity in shallow layers"],
        "quality": "Fertile, Good Cation Exchange Capacity"
    },
    "Laterite & Acidic Soil": {
        "ph": 6.0, "N": 45.0, "P": 25.0, "K": 32.0,
        "texture": "Porous Sandy Clay Loam", "organic_matter": "High",
        "deficiencies": ["Phosphorus (fixed)", "Potash", "Calcium"],
        "risks": ["Soil acidity", "High phosphorus fixation", "Nutrient leaching"],
        "quality": "Well-drained, Rich in Organic Matter, Acidic"
    },
    "Coastal Alluvial & Saline": {
        "ph": 7.1, "N": 58.0, "P": 48.0, "K": 52.0,
        "texture": "Alluvial Loam / Silty Clay", "organic_matter": "Moderate",
        "deficiencies": ["Micronutrients (Zinc, Boron)"],
        "risks": ["High salinity risk", "Poor aeration if waterlogged"],
        "quality": "Moderate Fertility, Salinity Prone, High Water Table"
    },
    "Red Sandy / Hill Soil": {
        "ph": 6.8, "N": 42.0, "P": 30.0, "K": 45.0,
        "texture": "Sandy Loam to Gravelly", "organic_matter": "Low to Moderate",
        "deficiencies": ["Nitrogen", "Phosphorus", "Humus"],
        "risks": ["Low moisture holding capacity", "High runoff erosion"],
        "quality": "Porous, Highly Permeable, Low Fertility"
    },
    "Standard Balanced Agricultural Loam": {
        "ph": 6.8, "N": 60.0, "P": 50.0, "K": 50.0,
        "texture": "Loam", "organic_matter": "Moderate",
        "deficiencies": [],
        "risks": [],
        "quality": "Balanced Agronomic Baseline"
    }
}


class SoilService:
    """Service to load, query, and derive localized soil properties."""

    def __init__(self, csv_path: str = "data/soil_database.csv"):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Loads and normalizes the soil database CSV."""
        if not os.path.exists(self.csv_path):
            self.df = pd.DataFrame()
            return

        try:
            df = pd.read_csv(self.csv_path)
            # Standardize columns
            df.columns = [c.strip() for c in df.columns]

            # Parse Taluka and District from "Village/Taluka" e.g., "Bhiwandi (Thane)"
            taluka_list = []
            district_list = []
            for item in df.get("Village/Taluka", []):
                item_str = str(item).strip()
                match = re.search(r"^(.*?)\s*\((.*?)\)$", item_str)
                if match:
                    taluka_list.append(match.group(1).strip())
                    district_list.append(match.group(2).strip())
                else:
                    taluka_list.append(item_str)
                    district_list.append("Unknown")

            df["Taluka"] = taluka_list
            df["District"] = district_list

            # Parse pH range into numeric min, max, average
            ph_min_list = []
            ph_max_list = []
            ph_avg_list = []
            for ph_val in df.get("pH", []):
                ph_str = str(ph_val).strip()
                ph_match = re.search(r"([\d\.]+)\s*-\s*([\d\.]+)", ph_str)
                if ph_match:
                    p_min = float(ph_match.group(1))
                    p_max = float(ph_match.group(2))
                    ph_min_list.append(p_min)
                    ph_max_list.append(p_max)
                    ph_avg_list.append(round((p_min + p_max) / 2.0, 2))
                else:
                    try:
                        p_val = float(ph_str)
                        ph_min_list.append(p_val)
                        ph_max_list.append(p_val)
                        ph_avg_list.append(p_val)
                    except ValueError:
                        ph_min_list.append(6.8)
                        ph_max_list.append(7.2)
                        ph_avg_list.append(7.0)

            df["pH_min"] = ph_min_list
            df["pH_max"] = ph_max_list
            df["pH_avg"] = ph_avg_list

            self.df = df
        except Exception as e:
            print(f"Error loading soil database: {e}")
            self.df = pd.DataFrame()

    def get_all_locations(self) -> List[Dict[str, str]]:
        """Returns a list of all available locations for search and autocomplete."""
        if self.df is None or self.df.empty:
            return []

        locations = []
        for _, row in self.df.iterrows():
            taluka = row["Taluka"]
            district = row["District"]
            display = f"{taluka} ({district})" if district != "Unknown" else taluka
            locations.append({
                "taluka": taluka,
                "district": district,
                "display": display,
                "raw": row["Village/Taluka"]
            })
        return locations

    def _estimate_nutrients_from_text(self, nutrient_text: str, soil_type: str, ph_avg: float) -> Tuple[float, float, float, List[str], List[str], str]:
        """Derives estimated N, P, K (kg/ha), deficiencies, and risks based on qualitative soil records."""
        text = (nutrient_text or "").lower()
        stype = (soil_type or "").lower()

        # Defaults
        N = 50.0
        P = 40.0
        K = 55.0
        deficiencies: List[str] = []
        risks: List[str] = []
        overall = "Moderately Fertile Agricultural Soil"

        # Salinity & Coastal
        if "salinity" in text or "saline" in stype:
            risks.append("High Salinity Risk / Electrical Conductivity (EC)")
            risks.append("Root zone sodium toxicity")
            N = 55.0
            P = 48.0
            K = 50.0
            overall = "Coastal Alluvial / Saline Soil (Requires Gypsum & Flushing)"

        # Calcareous / Deep Black Cotton Soil
        if "calcium carbonate" in text or "deep black" in stype or "lime" in text:
            if "deficient in zinc" in text:
                deficiencies.append("Zinc (Zn)")
            if "deficient in" in text and "n" in text:
                deficiencies.append("Nitrogen (N)")
            if "rich in potassium" in text:
                K = 85.0
            elif "moderate potassium" in text:
                K = 55.0
            
            if "low nitrogen" in text:
                N = 40.0
            else:
                N = 45.0

            if "moderate phosphorus" in text:
                P = 48.0
            elif "low" in text and "p" in text:
                P = 32.0
            else:
                P = 45.0

            risks.append("High lime reserve (Lime-induced iron/zinc chlorosis)")
            overall = "Deep Black Vertisol (High Moisture Retention & Lime Reserve)"

        # Lateritic & Acidic
        elif "laterite" in stype or "lateritic" in stype or "coastal clay" in stype:
            if "low nitrogen" in text:
                deficiencies.append("Nitrogen (N)")
                N = 42.0
            if "low phosphorus" in text or "phosphorus" in text:
                deficiencies.append("Available Phosphorus (P)")
                P = 24.0
            if "low potash" in text:
                deficiencies.append("Potassium (K)")
                K = 32.0
            risks.append("Acidic pH (< 6.5) causing Phosphorus fixation")
            risks.append("Nutrient leaching during heavy monsoon rainfall")
            overall = "Acidic Laterite / Coastal Clay (High Organic Matter, High Leaching)"

        # General fertile low N & P
        elif "generally fertile" in text:
            N = 48.0
            P = 34.0
            K = 65.0
            deficiencies.append("Nitrogen (N)")
            deficiencies.append("Phosphorus (P)")
            overall = "Fertile Medium Black Soil (Responsive to Balanced NPK)"

        # Fallback based on soil type
        elif "black" in stype:
            N, P, K = 45.0, 42.0, 75.0
            overall = "Medium Black Vertisol"
        elif "red" in stype or "sandy" in stype:
            N, P, K = 40.0, 28.0, 40.0
            overall = "Red Sandy Soil / Hill Loam (Fast Drainage)"
            deficiencies.append("Nitrogen (N)")
        elif "alluvial" in stype or "loam" in stype:
            N, P, K = 58.0, 45.0, 52.0
            overall = "Alluvial Loam Soil"

        return N, P, K, deficiencies, risks, overall

    def fetch_soil_profile(self, location_query: str) -> Dict[str, Any]:
        """
        Fetches soil profile for a location query.
        If found in CSV, returns extracted & derived values.
        If not found, synthesizes a scientifically sound profile based on agro-heuristics.
        """
        clean_query = location_query.strip().lower()

        # Check in DataFrame
        matched_row = None
        if self.df is not None and not self.df.empty:
            # 1. Exact taluka match
            m1 = self.df[self.df["Taluka"].str.lower() == clean_query]
            if not m1.empty:
                matched_row = m1.iloc[0]
            else:
                # 2. Taluka substring match
                m2 = self.df[self.df["Taluka"].str.lower().str.contains(clean_query, regex=False)]
                if not m2.empty:
                    matched_row = m2.iloc[0]
                else:
                    # 3. District match
                    m3 = self.df[self.df["District"].str.lower() == clean_query]
                    if not m3.empty:
                        matched_row = m3.iloc[0]
                    else:
                        m4 = self.df[self.df["District"].str.lower().str.contains(clean_query, regex=False)]
                        if not m4.empty:
                            matched_row = m4.iloc[0]

        if matched_row is not None:
            taluka = matched_row["Taluka"]
            district = matched_row["District"]
            soil_type = matched_row["Soil Type"]
            ph_range = matched_row["pH"]
            ph_avg = float(matched_row["pH_avg"])
            raw_nutrients = matched_row["Nutrient/Contamination Value"]

            N, P, K, deficiencies, risks, quality = self._estimate_nutrients_from_text(
                raw_nutrients, soil_type, ph_avg
            )

            return {
                "location_name": f"{taluka}, {district}" if district != "Unknown" else taluka,
                "taluka": taluka,
                "district": district,
                "matched_in_database": True,
                "soil_type": soil_type,
                "ph": ph_avg,
                "ph_range": ph_range,
                "N": N,
                "P": P,
                "K": K,
                "nutrient_summary": raw_nutrients,
                "deficiencies": deficiencies,
                "risks": risks,
                "overall_quality": quality,
                "source": "Local Agricultural Soil Database (Soil Health Map)"
            }

        # Fallback heuristic: not in CSV database
        return self._fallback_profile(location_query)

    def _fallback_profile(self, location_query: str) -> Dict[str, Any]:
        """Provides an intelligent heuristic soil profile when the location isn't in the CSV."""
        query = location_query.lower()
        
        # Heuristics based on common agro-regions
        if any(term in query for term in ["konkan", "goa", "kerala", "coastal", "alibaug", "ratnagiri"]):
            profile = DEFAULT_SOIL_PROFILES["Laterite & Acidic Soil"]
            soil_type = "Laterite & Coastal Acidic Soil"
        elif any(term in query for term in ["vidarbha", "marathwada", "deccan", "black", "cotton", "telangana", "malwa"]):
            profile = DEFAULT_SOIL_PROFILES["Deep Black Cotton Soil (Vertisol)"]
            soil_type = "Deep Black Cotton Soil (Vertisol)"
        elif any(term in query for term in ["punjab", "haryana", "ganga", "up", "bihar", "alluvial"]):
            profile = DEFAULT_SOIL_PROFILES["Standard Balanced Agricultural Loam"]
            soil_type = "Fertile Alluvial Loam Soil"
        else:
            profile = DEFAULT_SOIL_PROFILES["Medium Black Soil"]
            soil_type = "Medium Black Clay Loam"

        return {
            "location_name": location_query.title() if location_query else "Custom Location",
            "taluka": location_query.title() if location_query else "Custom",
            "district": "Identified Agro-Zone",
            "matched_in_database": False,
            "soil_type": soil_type,
            "ph": profile["ph"],
            "ph_range": f"{profile['ph'] - 0.5:.1f} - {profile['ph'] + 0.5:.1f}",
            "N": profile["N"],
            "P": profile["P"],
            "K": profile["K"],
            "nutrient_summary": f"Estimated baseline for {soil_type} in {location_query.title()}",
            "deficiencies": profile["deficiencies"],
            "risks": profile["risks"],
            "overall_quality": profile["quality"],
            "source": "Agro-Climatic Regional Heuristic (Customizable via Soil Sliders)"
        }
