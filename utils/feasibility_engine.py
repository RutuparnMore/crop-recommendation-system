import re
from typing import Dict, Any, List
from utils.ml_engine import AGRONOMIC_CROP_PROFILES, CropRecommendationEngine

class FeasibilityEngine:
    """Analyzes custom farmer crop choice viability and generates actionable agronomic guidance."""

    def __init__(self, ml_engine: CropRecommendationEngine):
        self.ml_engine = ml_engine

    def match_crop_name(self, query: str) -> str:
        """Finds closest matching known crop name or returns clean query."""
        clean = query.strip().lower()
        for k in AGRONOMIC_CROP_PROFILES.keys():
            if clean in k.lower() or k.lower() in clean:
                return k
        
        # Check aliases
        aliases = {
            "chana": "Chickpea (Chana/Harbara)",
            "harbara": "Chickpea (Chana/Harbara)",
            "tur": "Pigeonpeas (Tur/Arhar)",
            "arhar": "Pigeonpeas (Tur/Arhar)",
            "jowar": "Sorghum (Jowar)",
            "bajra": "Pearl Millet (Bajra)",
            "paddy": "Rice (Paddy)",
            "rice": "Rice (Paddy)",
            "corn": "Maize (Corn)",
            "makka": "Maize (Corn)",
            "kanda": "Onion",
            "tamatar": "Tomato",
            "draksha": "Grapes",
            "dalimb": "Pomegranate",
            "keli": "Banana",
            "amba": "Mango",
            "naral": "Coconut",
            "us": "Sugarcane",
            "kapas": "Cotton",
            "soyabean": "Soybean"
        }
        for alias, target in aliases.items():
            if alias in clean:
                return target

        return query.title()

    def analyze_feasibility(
        self,
        crop_input: str,
        weather: Dict[str, Any],
        soil: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Conducts deep agronomic stress test on user-selected crop against weather and soil.
        Returns pros, cons, required infrastructure, and yield tips.
        """
        crop_name = self.match_crop_name(crop_input)
        temp = float(weather.get("temperature", 26.0))
        hum = float(weather.get("humidity", 60.0))
        rain = float(weather.get("rainfall", 90.0))

        ph = float(soil.get("ph", 7.0))
        n = float(soil.get("N", 50.0))
        p = float(soil.get("P", 40.0))
        k = float(soil.get("K", 50.0))
        stype = str(soil.get("soil_type", ""))
        deficiencies = soil.get("deficiencies", [])
        risks = soil.get("risks", [])

        # Check if crop exists in profile database
        is_known = crop_name in AGRONOMIC_CROP_PROFILES
        if not is_known:
            # Fallback estimation for unknown crop
            return self._analyze_unknown_crop(crop_name, weather, soil)

        profile = AGRONOMIC_CROP_PROFILES[crop_name]
        suit_res = self.ml_engine.evaluate_crop_suitability(
            crop_name=crop_name,
            n=n, p=p, k=k,
            temp=temp, hum=hum,
            ph=ph, rain=rain,
            soil_type=stype
        )
        score = suit_res["score"]

        positives: List[str] = []
        negatives: List[str] = []
        infrastructure: List[str] = []
        yield_tips: List[str] = []

        # 1. Temperature Analysis
        t_min, t_opt_min, t_opt_max, t_max = profile["temp_range"]
        if t_opt_min <= temp <= t_opt_max:
            positives.append(
                f"**Thermal Window Perfect:** Current temperature ({temp:.1f}°C) is right inside the prime vegetative and flowering range ({t_opt_min:.0f}–{t_opt_max:.0f}°C)."
            )
        elif t_min <= temp < t_opt_min:
            negatives.append(
                f"**Sub-optimal Cool Temperature:** Current temperature ({temp:.1f}°C) is below ideal ({t_opt_min:.0f}°C), potentially slowing germination and vegetative growth."
            )
        elif t_opt_max < temp <= t_max:
            negatives.append(
                f"**Heat Stress Alert:** Temperature ({temp:.1f}°C) approaches the upper tolerance limit ({t_max:.0f}°C), which may induce flower drop or pollen desiccation."
            )
        else:
            negatives.append(
                f"**Severe Temperature Mismatch:** {temp:.1f}°C is outside biological tolerance bounds ({t_min:.0f}–{t_max:.0f}°C)."
            )

        # 2. Rainfall & Water Requirement Analysis
        r_min, r_opt_min, r_opt_max, r_max = profile["rainfall_range"]
        if r_opt_min <= rain <= r_opt_max:
            positives.append(
                f"**Rainfall Balance:** Expected rainfall of {rain:.0f} mm satisfies natural crop moisture demands without waterlogging."
            )
        elif rain < r_opt_min:
            deficit = r_opt_min - rain
            negatives.append(
                f"**Moisture Deficit ({deficit:.0f} mm below optimal):** Crop requires {r_opt_min:.0f}–{r_opt_max:.0f} mm, while your region expects ~{rain:.0f} mm."
            )
            infrastructure.append(
                "**Drip Irrigation System:** Install an inline drip irrigation network with fertigation Venturi to supply regular water without waste."
            )
            infrastructure.append(
                "**Farm Pond / Water Harvesting:** Construct a polythene-lined farm pond (Shet Tale) to store supplementary irrigation for critical dry spells."
            )
        else:
            negatives.append(
                f"**Excess Moisture / Waterlogging Hazard:** Region receives {rain:.0f} mm (crop threshold is {r_opt_max:.0f} mm). Heavy rain risks fungal root rots and collar rot."
            )
            infrastructure.append(
                "**Broad Bed Furrow (BBF) / Raised Beds:** Plant on raised ridges (BBF method) and dig perimeter drainage channels to drain surplus runoff immediately."
            )

        # 3. Soil Type & Texture Analysis
        soil_matches = any(term in stype.lower() for term in [s.lower() for s in profile["ideal_soil_types"]])
        if soil_matches:
            positives.append(
                f"**Soil Texture Synergy:** Your local {stype} matches the preferred soil typology ({', '.join(profile['ideal_soil_types'])})."
            )
        else:
            negatives.append(
                f"**Soil Physical Constraints:** {crop_name} prefers {', '.join(profile['ideal_soil_types'])}, whereas your soil is {stype}."
            )

        # 4. Soil pH Analysis
        p_min, p_opt_min, p_opt_max, p_max = profile["ph_range"]
        if p_opt_min <= ph <= p_opt_max:
            positives.append(
                f"**Optimal Soil Reaction:** pH of {ph:.1f} is well within the sweet spot ({p_opt_min:.1f}–{p_opt_max:.1f}), preventing nutrient lockup."
            )
        elif ph < p_opt_min:
            negatives.append(
                f"**Soil Acidity Barrier:** pH ({ph:.1f}) is acidic for {crop_name} (tolerates {p_opt_min:.1f}+), which locks up Phosphorus and Molybdenum."
            )
            yield_tips.append(
                f"**Soil Neutralization (Liming):** Apply 250–400 kg/acre Agricultural Lime ($CaCO_3$) or Dolomite before sowing to raise soil pH toward 6.5."
            )
        else:
            negatives.append(
                f"**Alkaline / Calcareous Soil Challenge:** pH ({ph:.1f}) is alkaline (above {p_opt_max:.1f}), which triggers lime-induced iron and zinc fixation."
            )
            yield_tips.append(
                "**Alkalinity Mitigation:** Apply elemental sulfur (20-30 kg/acre) or decomposed Farmyard Manure (FYM) mixed with green manure (Dhaincha) to lower rhizosphere pH."
            )

        # 5. Specific Nutrient Analysis (N, P, K)
        n_min, n_opt_min, n_opt_max, n_max = profile["n_range"]
        if n >= n_opt_min:
            positives.append(f"**Nitrogen Reserve:** Available nitrogen ({n:.0f} kg/ha) supports vigorous vegetative canopy formation.")
        else:
            negatives.append(f"**Low Nitrogen Availability:** Soil has only {n:.0f} kg/ha (ideal is {n_opt_min:.0f}–{n_opt_max:.0f} kg/ha).")
            yield_tips.append(
                f"**Nitrogen Management:** Split Urea application into 3 stages (Basal, Vegetative, Flowering). Treat seeds with *Azotobacter* or *Rhizobium* culture (250g/10kg seeds)."
            )

        p_min_t, p_opt_min_t, p_opt_max_t, p_max_t = profile["p_range"]
        if p >= p_opt_min_t:
            positives.append(f"**Phosphorus Sufficiency:** Phosphorus level ({p:.0f} kg/ha) encourages root proliferation and early flowering.")
        else:
            yield_tips.append(
                f"**Phosphorus Supplementation:** Apply Single Super Phosphate (SSP) or DAP as a basal dose along with Phosphorus Solubilizing Bacteria (PSB) to dissolve bound phosphates."
            )

        k_min_t, k_opt_min_t, k_opt_max_t, k_max_t = profile["k_range"]
        if k >= k_opt_min_t:
            positives.append(f"**Rich Potassium Buffer:** High potassium ({k:.0f} kg/ha) boosts disease resistance, fruit size, and drought endurance.")
        else:
            yield_tips.append(
                f"**Potash Boost:** Apply Muriate of Potash (MOP / 0:0:60) or foliar spray of 0:52:34 / 0:0:50 during fruit enlargement."
            )

        # 6. Location-Specific Soil Risk Recommendations (Salinity, Zinc, etc.)
        for d in deficiencies:
            if "zinc" in d.lower():
                yield_tips.append(
                    "**Zinc Deficiency Correction:** Apply Zinc Sulfate ($ZnSO_4$, 21%) @ 10 kg/acre in basal soil application, or spray Chelated Zinc (Zn-EDTA 12%) @ 1g/liter water during active growth."
                )
            if "nitrogen" in d.lower() and not any("Nitrogen Management" in t for t in yield_tips):
                yield_tips.append(
                    "**Organic Nitrogen Boost:** Incorporate 5 tons/acre well-decomposed cow dung compost or vermicompost before land preparation."
                )

        for r in risks:
            if "salinity" in r.lower():
                negatives.append(
                    "**Salinity / High EC Hazard:** Elevated soil salinity causes osmotic stress, stunting root moisture intake."
                )
                infrastructure.append(
                    "**Subsurface Drainage Lines:** Install perforated PVC drainage pipes or deep trenches to flush out toxic sodium salts."
                )
                yield_tips.append(
                    "**Salinity Amelioration:** Apply Agricultural Gypsum ($CaSO_4$) @ 500 kg/acre followed by heavy freshwater leaching before sowing."
                )
            if "waterlogging" in r.lower() and not any("Raised Beds" in i for i in infrastructure):
                infrastructure.append(
                    "**Drainage Channels:** Establish deep perimeter drainage furrows to prevent water stagnation around root zones."
                )

        # General IPM / Crop Specific Tip
        yield_tips.append(
            f"**Integrated Pest Management (IPM):** Install yellow/blue sticky traps (10/acre) and pheromone traps to monitor pests early; practice mulching (organic straw or 25-micron silver-black plastic mulch) to conserve 40% soil moisture."
        )

        # Verdict assignment
        if score >= 75.0:
            verdict = "Highly Feasible (Scientifically Recommended)"
            verdict_badge = "success"
        elif score >= 55.0:
            verdict = "Moderately Feasible (Requires Irrigation & Nutrient Management)"
            verdict_badge = "warning"
        else:
            verdict = "Challenging / High-Risk (Requires Major Capital & Infrastructure)"
            verdict_badge = "danger"

        return {
            "crop_name": crop_name,
            "marathi_name": profile.get("marathi_name", ""),
            "category": profile.get("category", ""),
            "feasibility_score": score,
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "positives": positives,
            "negatives": negatives,
            "infrastructure": infrastructure,
            "yield_tips": yield_tips,
            "water_need": profile.get("water_need", "Standard Agricultural Water Need")
        }

    def _analyze_unknown_crop(
        self,
        crop_name: str,
        weather: Dict[str, Any],
        soil: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provides generalized agronomic feasibility assessment for uncatalogued custom crops."""
        temp = float(weather.get("temperature", 26.0))
        hum = float(weather.get("humidity", 60.0))
        rain = float(weather.get("rainfall", 90.0))
        ph = float(soil.get("ph", 7.0))
        stype = str(soil.get("soil_type", "Standard Loam"))

        positives = [
            f"**Climate Baseline:** Regional ambient temperature ({temp:.1f}°C) and humidity ({hum:.1f}%) support diverse sub-tropical crops.",
            f"**Soil Bed:** Current soil texture ({stype}) provides natural physical stability."
        ]
        negatives = [
            f"**Unverified Crop Requirements:** Exact nutritional envelope for '{crop_name}' should be cross-verified with your local Krishi Vigyan Kendra (KVK).",
            f"**Water Supply Risk:** Ensure total crop water requirement is budgeted against expected rainfall of {rain:.0f} mm."
        ]
        infrastructure = [
            "**Micro-Irrigation (Drip/Sprinkler):** Crucial to maintain controlled moisture levels.",
            "**Soil Moisture Sensors / Tensiometers:** To avoid overwatering in clay or underwatering in sandy soils."
        ]
        yield_tips = [
            "**Soil Testing:** Conduct a certified laboratory Soil Health Card test before major capital outlay.",
            "**Organic Enrichment:** Add 4–5 tonnes of Farmyard Manure (FYM) per acre to balance soil microbial life.",
            "**Preventive Plant Protection:** Use Neem cake (100 kg/acre) during basal preparation to deter soil-borne nematodes."
        ]

        return {
            "crop_name": crop_name,
            "marathi_name": "प्रायोगिक पीक (Experimental Crop)",
            "category": "Custom Crop Entry",
            "feasibility_score": 60.0,
            "verdict": "Moderately Feasible (Verify with Local Krishi Vigyan Kendra)",
            "verdict_badge": "warning",
            "positives": positives,
            "negatives": negatives,
            "infrastructure": infrastructure,
            "yield_tips": yield_tips,
            "water_need": "Dependent on specific crop physiology"
        }
