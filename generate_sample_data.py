import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_surgery_mortality_data(n_samples=100, alive_ratio=0.6):
    """
    Generate realistic surgery mortality dataset with 60% alive, 40% dead
    """
    np.random.seed(42)
    
    # Feature names
    feature_names = [
        'Patient_Age',
        'BMI',
        'ASA_Physical_Status',
        'Emergency_Surgery',
        'Preop_Hemoglobin',
        'Preop_Albumin',
        'Preop_Creatinine',
        'Comorbidity_Index',
        'Surgical_Duration_Min',
        'Estimated_Blood_Loss_ml',
        'Diabetes_Mellitus',
        'Hypertension',
        'Cardiac_Disease',
        'Respiratory_Disease',
        'Renal_Disease',
        'Liver_Disease',
        'Cancer_Presence',
        'Metastatic_Disease',
        'Previous_Abdominal_Surgery',
        'Functional_Status'
    ]
    
    n_alive = int(n_samples * alive_ratio)
    n_dead = n_samples - n_alive
    
    data = []
    
    # Generate ALIVE patients (60%)
    for i in range(n_alive):
        patient = {
            'Patient_ID': f"ALIVE_{i+1:03d}",
            'Patient_Age': np.random.randint(25, 70),  # Younger for alive
            'BMI': np.random.normal(25, 4),  # Normal BMI
            'ASA_Physical_Status': np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]),  # Lower ASA
            'Emergency_Surgery': 0,  # Mostly elective
            'Preop_Hemoglobin': np.random.normal(13.5, 1.5),  # Normal Hb
            'Preop_Albumin': np.random.normal(3.8, 0.5),  # Normal albumin
            'Preop_Creatinine': np.random.normal(0.9, 0.3),  # Normal creatinine
            'Comorbidity_Index': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),  # Low comorbidity
            'Surgical_Duration_Min': np.random.randint(45, 180),  # Shorter surgeries
            'Estimated_Blood_Loss_ml': np.random.randint(50, 400),  # Less blood loss
            'Diabetes_Mellitus': np.random.choice([0, 1], p=[0.8, 0.2]),
            'Hypertension': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Cardiac_Disease': np.random.choice([0, 1], p=[0.9, 0.1]),
            'Respiratory_Disease': np.random.choice([0, 1], p=[0.85, 0.15]),
            'Renal_Disease': np.random.choice([0, 1], p=[0.9, 0.1]),
            'Liver_Disease': np.random.choice([0, 1], p=[0.95, 0.05]),
            'Cancer_Presence': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Metastatic_Disease': 0,  # No metastasis for alive
            'Previous_Abdominal_Surgery': np.random.choice([0, 1], p=[0.6, 0.4]),
            'Functional_Status': np.random.choice([1, 2], p=[0.8, 0.2]),  # Better functional status
            'Mortality_Outcome': 0  # ALIVE
        }
        data.append(patient)
    
    # Generate DEAD patients (40%)
    for i in range(n_dead):
        patient = {
            'Patient_ID': f"DEAD_{i+1:03d}",
            'Patient_Age': np.random.randint(65, 90),  # Older for dead
            'BMI': np.random.normal(28, 5),  # Higher BMI
            'ASA_Physical_Status': np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2]),  # Higher ASA
            'Emergency_Surgery': np.random.choice([0, 1], p=[0.3, 0.7]),  # Mostly emergency
            'Preop_Hemoglobin': np.random.normal(10.5, 2.0),  # Lower Hb
            'Preop_Albumin': np.random.normal(2.8, 0.7),  # Lower albumin
            'Preop_Creatinine': np.random.normal(1.8, 0.8),  # Higher creatinine
            'Comorbidity_Index': np.random.choice([3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.2]),  # High comorbidity
            'Surgical_Duration_Min': np.random.randint(120, 360),  # Longer surgeries
            'Estimated_Blood_Loss_ml': np.random.randint(300, 1200),  # More blood loss
            'Diabetes_Mellitus': np.random.choice([0, 1], p=[0.3, 0.7]),
            'Hypertension': np.random.choice([0, 1], p=[0.2, 0.8]),
            'Cardiac_Disease': np.random.choice([0, 1], p=[0.4, 0.6]),
            'Respiratory_Disease': np.random.choice([0, 1], p=[0.5, 0.5]),
            'Renal_Disease': np.random.choice([0, 1], p=[0.6, 0.4]),
            'Liver_Disease': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Cancer_Presence': np.random.choice([0, 1], p=[0.4, 0.6]),
            'Metastatic_Disease': np.random.choice([0, 1], p=[0.6, 0.4]),
            'Previous_Abdominal_Surgery': np.random.choice([0, 1], p=[0.4, 0.6]),
            'Functional_Status': np.random.choice([3, 4], p=[0.6, 0.4]),  # Poor functional status
            'Mortality_Outcome': 1  # DEAD
        }
        data.append(patient)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure values are within realistic ranges
    df['BMI'] = df['BMI'].clip(15, 45)
    df['Preop_Hemoglobin'] = df['Preop_Hemoglobin'].clip(6, 18)
    df['Preop_Albumin'] = df['Preop_Albumin'].clip(1.5, 5.0)
    df['Preop_Creatinine'] = df['Preop_Creatinine'].clip(0.5, 5.0)
    df['ASA_Physical_Status'] = df['ASA_Physical_Status'].astype(int)
    df['Emergency_Surgery'] = df['Emergency_Surgery'].astype(int)
    
    # Round numerical values
    float_columns = ['BMI', 'Preop_Hemoglobin', 'Preop_Albumin', 'Preop_Creatinine']
    for col in float_columns:
        df[col] = df[col].round(2)
    
    return df

def create_sample_patients_csv():
    """Generate and save the sample dataset"""
    print("Generating surgery mortality dataset...")
    
    # Generate data
    df = generate_surgery_mortality_data(n_samples=100, alive_ratio=0.6)
    
    # Display summary
    print(f"Dataset shape: {df.shape}")
    print(f"Alive patients (0): {(df['Mortality_Outcome'] == 0).sum()}")
    print(f"Dead patients (1): {(df['Mortality_Outcome'] == 1).sum()}")
    print(f"Alive/Dead ratio: {(df['Mortality_Outcome'] == 0).sum()}/{(df['Mortality_Outcome'] == 1).sum()}")
    
    # Save to CSV
    csv_filename = "surgery_mortality_dataset.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ Dataset saved as: {csv_filename}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head().to_string())
    
    # Display basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total patients: {len(df)}")
    print(f"Alive (0): {(df['Mortality_Outcome'] == 0).sum()} ({df['Mortality_Outcome'].value_counts(normalize=True)[0]*100:.1f}%)")
    print(f"Dead (1): {(df['Mortality_Outcome'] == 1).sum()} ({df['Mortality_Outcome'].value_counts(normalize=True)[1]*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    df = create_sample_patients_csv()
    
    # Also create a version without Patient_ID and Outcome for prediction testing
    features_df = df.drop(['Patient_ID', 'Mortality_Outcome'], axis=1)
    features_df.to_csv("surgery_mortality_features_only.csv", index=False)
    print("✅ Features-only dataset saved as: surgery_mortality_features_only.csv")
    
    print("\n🎯 Sample patients from the dataset:")
    sample_alive = df[df['Mortality_Outcome'] == 0].head(2)
    sample_dead = df[df['Mortality_Outcome'] == 1].head(2)
    
    print("\nSample ALIVE patients:")
    print(sample_alive[['Patient_ID', 'Patient_Age', 'ASA_Physical_Status', 'Comorbidity_Index']].to_string(index=False))
    
    print("\nSample DEAD patients:")
    print(sample_dead[['Patient_ID', 'Patient_Age', 'ASA_Physical_Status', 'Comorbidity_Index']].to_string(index=False))