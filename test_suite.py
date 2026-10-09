"""Test suite for KrishiMitra Agritech System."""
import sys
from utils.soil_service import SoilService
from utils.weather_service import WeatherService
from utils.ml_engine import CropRecommendationEngine
from utils.feasibility_engine import FeasibilityEngine

def run_tests():
    print("Testing SoilService...")
    soil_svc = SoilService("data/soil_database.csv")
    print(f"Total locations loaded: {len(soil_svc.get_all_locations())}")
    
    # Test query inside database
    soil_pune = soil_svc.fetch_soil_profile("Baramati")
    print(f"Baramati Soil: {soil_pune['soil_type']}, pH={soil_pune['ph']}, N={soil_pune['N']}, P={soil_pune['P']}, K={soil_pune['K']}")
    assert soil_pune["matched_in_database"] is True, "Failed to match Baramati in DB"

    # Test query outside database
    soil_unknown = soil_svc.fetch_soil_profile("Patna")
    print(f"Patna Soil: {soil_unknown['soil_type']}, pH={soil_unknown['ph']}")
    assert soil_unknown["matched_in_database"] is False, "Patna should not match DB"

    print("\nTesting WeatherService...")
    weather_svc = WeatherService()
    # Test open-meteo query for Pune
    w_pune = weather_svc.fetch_weather("Pune")
    print(f"Pune Weather: Temp={w_pune['temperature']}°C, Humidity={w_pune['humidity']}%, Rain={w_pune['rainfall']}mm, Source={w_pune['source']}")
    assert "temperature" in w_pune and w_pune["temperature"] > -50, "Invalid weather"

    print("\nTesting CropRecommendationEngine & ML model...")
    ml_engine = CropRecommendationEngine()
    recs = ml_engine.recommend_top_crops(
        n=soil_pune["N"],
        p=soil_pune["P"],
        k=soil_pune["K"],
        temp=w_pune["temperature"],
        hum=w_pune["humidity"],
        ph=soil_pune["ph"],
        rain=w_pune["rainfall"],
        soil_type=soil_pune["soil_type"],
        top_k=3
    )
    print(f"Top 3 recommendations for Baramati:")
    for i, r in enumerate(recs):
        print(f"  #{i+1}: {r['crop']} ({r['score']}%) - ML Prob: {r['ml_probability']}%")
        print(f"       Reasoning: {r['reasoning'][:120]}...")
    assert len(recs) == 3, "Expected 3 crop recommendations"

    print("\nTesting FeasibilityEngine for custom crop (Sugarcane in low rainfall area)...")
    feas_engine = FeasibilityEngine(ml_engine)
    feas_result = feas_engine.analyze_feasibility(
        crop_input="Sugarcane",
        weather={"temperature": 32.0, "humidity": 45.0, "rainfall": 50.0},
        soil=soil_pune
    )
    print(f"Sugarcane Viability Score: {feas_result['feasibility_score']}%, Verdict: {feas_result['verdict']}")
    print(f"Positives count: {len(feas_result['positives'])}")
    print(f"Negatives count: {len(feas_result['negatives'])}")
    print(f"Infrastructure count: {len(feas_result['infrastructure'])}")
    print(f"Yield tips count: {len(feas_result['yield_tips'])}")
    assert len(feas_result["negatives"]) > 0, "Expected negatives for sugarcane in low rain"
    assert len(feas_result["infrastructure"]) > 0, "Expected irrigation infrastructure for sugarcane in dry condition"

    print("\nALL AGRONOMIC TEST SUITES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
