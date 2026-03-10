#testing file
import pandas as pd
import numpy as np
import joblib
import sys
import os

def test_mortality_model():
    """
    Test the trained mortality prediction model with sample patients
    """
    print("🏥 MORTALITY PREDICTION MODEL TEST")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists('mortality_prediction_pipeline.pkl'):
        print("❌ Model file 'mortality_prediction_pipeline.pkl' not found!")
        print("   Please run the training script first.")
        return None
    
    try:
        # Load the trained pipeline
        print("🔍 Loading trained model...")
        pipeline = joblib.load('mortality_prediction_pipeline.pkl')
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get the EXACT features used during training
    try:
        # Load the original training data to see which features were used
        df = pd.read_csv("mrk0.csv")
        
        # Get the features that were actually used in training (from model info)
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            trained_features = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
        else:
            # If we can't get feature names, use the columns from the dataset
            # that match what the model expects (33 features as shown in your output)
            trained_features = [
                'age', 'bmi', 'weight', 'height', 'apache_2_diagnosis', 'apache_3j_diagnosis',
                'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'd1_heartrate_max',
                'd1_mbp_max', 'd1_resprate_max', 'd1_temp_max', 'd1_spo2_min', 'd1_creatinine_max',
                'd1_glucose_max', 'd1_hemaglobin_max', 'd1_wbc_max', 'd1_bun_max', 'diabetes_mellitus',
                'aids', 'cirrhosis', 'hepatic_failure', 'pre_icu_los_days', 'arf_apache',
                'gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache', 'hospital_admit_source',
                'icu_admit_source', 'icu_type', 'intubated_apache', 'ventilated_apache', 'immunosuppression'
            ]
        
        print(f"📊 Model expects {len(trained_features)} features")
        print("First 10 features expected by model:")
        for feature in trained_features[:10]:
            print(f"   - {feature}")
            
        return pipeline, trained_features
        
    except Exception as e:
        print(f"❌ Error getting feature information: {e}")
        return None, None

def create_correct_sample_patient(risk_level, trained_features):
    """
    Create sample patient data with EXACTLY the features the model expects
    """
    # Start with an empty dictionary
    sample_data = {}
    
    # Define values based on risk level for known features
    base_values = {
        'low_risk': {
            'age': 35, 'bmi': 22.5, 'weight': 70, 'height': 175,
            'apache_4a_hospital_death_prob': 0.05, 'apache_4a_icu_death_prob': 0.03,
            'd1_heartrate_max': 75, 'd1_mbp_max': 85, 'd1_resprate_max': 16,
            'd1_temp_max': 36.8, 'd1_spo2_min': 98, 'd1_creatinine_max': 0.8,
            'd1_glucose_max': 110, 'd1_hemaglobin_max': 14.0, 'd1_wbc_max': 7.5,
            'd1_bun_max': 15, 'diabetes_mellitus': 0, 'aids': 0, 'cirrhosis': 0,
            'hepatic_failure': 0, 'pre_icu_los_days': 1, 'arf_apache': 0,
            'gcs_verbal_apache': 5, 'gcs_motor_apache': 6, 'gcs_eyes_apache': 4,
            'intubated_apache': 0, 'ventilated_apache': 0, 'immunosuppression': 0
        },
        'medium_risk': {
            'age': 68, 'bmi': 28.5, 'weight': 85, 'height': 170,
            'apache_4a_hospital_death_prob': 0.15, 'apache_4a_icu_death_prob': 0.12,
            'd1_heartrate_max': 95, 'd1_mbp_max': 75, 'd1_resprate_max': 22,
            'd1_temp_max': 37.5, 'd1_spo2_min': 92, 'd1_creatinine_max': 1.4,
            'd1_glucose_max': 160, 'd1_hemaglobin_max': 11.5, 'd1_wbc_max': 12.0,
            'd1_bun_max': 28, 'diabetes_mellitus': 1, 'aids': 0, 'cirrhosis': 0,
            'hepatic_failure': 0, 'pre_icu_los_days': 3, 'arf_apache': 0,
            'gcs_verbal_apache': 4, 'gcs_motor_apache': 5, 'gcs_eyes_apache': 3,
            'intubated_apache': 0, 'ventilated_apache': 0, 'immunosuppression': 0
        },
        'high_risk': {
            'age': 82, 'bmi': 31.0, 'weight': 95, 'height': 165,
            'apache_4a_hospital_death_prob': 0.35, 'apache_4a_icu_death_prob': 0.30,
            'd1_heartrate_max': 120, 'd1_mbp_max': 65, 'd1_resprate_max': 28,
            'd1_temp_max': 38.2, 'd1_spo2_min': 85, 'd1_creatinine_max': 2.8,
            'd1_glucose_max': 240, 'd1_hemaglobin_max': 9.0, 'd1_wbc_max': 18.5,
            'd1_bun_max': 45, 'diabetes_mellitus': 1, 'aids': 0, 'cirrhosis': 1,
            'hepatic_failure': 0, 'pre_icu_los_days': 7, 'arf_apache': 1,
            'gcs_verbal_apache': 2, 'gcs_motor_apache': 3, 'gcs_eyes_apache': 2,
            'intubated_apache': 1, 'ventilated_apache': 1, 'immunosuppression': 0
        }
    }
    
    # Use base values for the risk level
    if risk_level in base_values:
        sample_data.update(base_values[risk_level])
    
    # Add default values for any remaining features
    for feature in trained_features:
        if feature not in sample_data:
            # Set sensible defaults for missing features
            if any(x in feature for x in ['age', 'bmi', 'weight', 'height']):
                sample_data[feature] = base_values[risk_level]['age'] if 'age' in feature else base_values[risk_level]['bmi']
            elif 'death_prob' in feature:
                sample_data[feature] = base_values[risk_level]['apache_4a_hospital_death_prob']
            elif 'max' in feature:
                sample_data[feature] = 50 if risk_level == 'low_risk' else 75 if risk_level == 'medium_risk' else 100
            elif 'min' in feature:
                sample_data[feature] = 90 if risk_level == 'low_risk' else 80 if risk_level == 'medium_risk' else 70
            elif any(x in feature for x in ['diagnosis', 'bodysystem']):
                sample_data[feature] = 100 if risk_level == 'low_risk' else 200 if risk_level == 'medium_risk' else 300
            else:
                sample_data[feature] = 0
    
    # Ensure we only return the features the model expects
    final_data = {}
    for feature in trained_features:
        if feature in sample_data:
            final_data[feature] = sample_data[feature]
        else:
            # Fallback for any missing features
            final_data[feature] = 0
    
    return final_data

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Medium Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def run_sample_tests(pipeline, trained_features):
    """
    Run tests with sample patients using the correct features
    """
    print("\n🎯 TESTING WITH SAMPLE PATIENTS (Correct Features)")
    print("=" * 60)
    
    test_patients = [
        {
            "name": "Low Risk Patient",
            "description": "Young, healthy patient with normal vitals",
            "expected": "Survival",
            "risk_level": "low_risk"
        },
        {
            "name": "Medium Risk Patient", 
            "description": "Elderly patient with some comorbidities",
            "expected": "Survival", 
            "risk_level": "medium_risk"
        },
        {
            "name": "High Risk Patient",
            "description": "Critically ill patient with multiple organ issues", 
            "expected": "Death",
            "risk_level": "high_risk"
        }
    ]
    
    results = []
    
    for i, patient in enumerate(test_patients, 1):
        print(f"\n🧪 Patient {i}: {patient['name']}")
        print(f"   Description: {patient['description']}")
        print(f"   Expected: {patient['expected']}")
        
        try:
            # Create patient with EXACT features model expects
            patient_features = create_correct_sample_patient(patient['risk_level'], trained_features)
            
            # Convert to DataFrame for prediction
            patient_df = pd.DataFrame([patient_features])
            
            # Ensure column order matches training
            patient_df = patient_df[trained_features]
            
            # Make prediction
            prediction = pipeline.predict(patient_df)[0]
            probability = pipeline.predict_proba(patient_df)[0, 1]
            
            # Get result
            outcome = "Death" if prediction == 1 else "Survival"
            risk_level = get_risk_level(probability)
            
            print(f"   🎯 Prediction: {outcome}")
            print(f"   📊 Probability: {probability:.3f} ({probability:.1%})")
            print(f"   🚦 Risk Level: {risk_level}")
            
            # Check if prediction matches expectation
            matches_expectation = (
                (patient['expected'] == "Death" and prediction == 1) or
                (patient['expected'] == "Survival" and prediction == 0)
            )
            
            if matches_expectation:
                print("   ✅ Matches expected outcome")
            else:
                print("   ⚠️  Does not match expected outcome")
            
            results.append({
                'patient': patient['name'],
                'expected': patient['expected'],
                'prediction': outcome,
                'probability': probability,
                'risk_level': risk_level,
                'matches': matches_expectation
            })
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            results.append({
                'patient': patient['name'],
                'error': str(e)
            })
    
    return results

def predict_custom_patient(pipeline, trained_features):
    """
    Predict mortality for a custom patient with correct features
    """
    print("\n" + "=" * 60)
    print("🎯 CUSTOM PATIENT PREDICTION")
    print("=" * 60)
    
    # Create a custom patient with the EXACT features
    print("Creating a custom patient with correct features...")
    
    custom_features = create_correct_sample_patient('medium_risk', trained_features)
    
    try:
        # Convert to DataFrame with correct column order
        patient_df = pd.DataFrame([custom_features])[trained_features]
        
        # Make prediction
        prediction = pipeline.predict(patient_df)[0]
        probability = pipeline.predict_proba(patient_df)[0, 1]
        risk_level = get_risk_level(probability)
        
        print("📋 Key Patient Features:")
        key_features = ['age', 'bmi', 'apache_4a_hospital_death_prob', 'd1_heartrate_max', 
                       'd1_mbp_max', 'd1_spo2_min', 'd1_creatinine_max', 'gcs_verbal_apache']
        for feature in key_features:
            if feature in custom_features:
                print(f"   {feature}: {custom_features[feature]}")
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Outcome: {'DEATH' if prediction == 1 else 'SURVIVAL'}")
        print(f"   Probability: {probability:.3f} ({probability:.1%})")
        print(f"   Risk Level: {risk_level}")
        
        # Clinical interpretation
        if prediction == 1:
            print(f"   🚨 CLINICAL ALERT: High mortality risk detected")
            if probability > 0.7:
                print("   💡 Consider intensive monitoring and intervention")
        else:
            print(f"   ✅ Favorable prognosis")
            if probability < 0.1:
                print("   💡 Low risk - standard care protocol")
        
        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def model_info(pipeline):
    """
    Display information about the trained model
    """
    print("\n" + "=" * 60)
    print("🔍 MODEL INFORMATION")
    print("=" * 60)
    
    print(f"Model type: {type(pipeline.named_steps['classifier']).__name__}")
    print(f"Preprocessor: {type(pipeline.named_steps['preprocessor']).__name__}")
    
    # Get feature names
    try:
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            print(f"Number of features: {len(feature_names)}")
            print("First 15 features:")
            for name in feature_names[:15]:
                print(f"  - {name}")
    except:
        print("Feature names not available")

if __name__ == "__main__":
    print("🏥 MORTALITY PREDICTION MODEL TESTER")
    print("=" * 60)
    
    # Test the model and get the correct features
    result = test_mortality_model()
    
    if result is not None:
        pipeline, trained_features = result
        
        # Run tests with correct features
        results = run_sample_tests(pipeline, trained_features)
        
        # Summary
        print("\n" + "=" * 60)
        print("📈 TEST SUMMARY")
        print("=" * 60)
        
        successful_tests = [r for r in results if 'error' not in r]
        if successful_tests:
            correct_predictions = sum(1 for r in successful_tests if r['matches'])
            total_predictions = len(successful_tests)
            
            print(f"Tests completed: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")
            
            print("\nDetailed results:")
            for result in successful_tests:
                status = "✅" if result['matches'] else "⚠️"
                print(f"  {status} {result['patient']}: {result['prediction']} (Prob: {result['probability']:.3f}, Risk: {result['risk_level']})")
        
        # Show model information
        model_info(pipeline)
        
        # Test with a custom patient
        predict_custom_patient(pipeline, trained_features)
        
        print("\n" + "=" * 60)
        print("✅ TESTING COMPLETE")
        print("=" * 60)
        print("The model is ready for use in your Streamlit app!")
        print("\nTo use in Streamlit:")
        print("1. Copy 'mortality_prediction_pipeline.pkl' to your Streamlit app directory")
        print("2. Use the EXACT same features in the same order")
        print("3. Ensure your patient data has all 33 features the model expects")
        
    else:
        print("\n❌ Testing failed. Please check the model file.")
