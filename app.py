#sample testing
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ===============================================================
# 🏥 MORTALITY PREDICTION APP
# ===============================================================

# Page configuration
st.set_page_config(
    page_title="Hospital Mortality Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .medium-risk {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .low-risk {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .outcome-alive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .outcome-dead {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .sample-patient {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #1f77b4;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================
# 📊 LOAD MODEL & DATA
# ===============================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('mortality_prediction_pipeline.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'mortality_prediction_pipeline.pkl' exists in the same directory")
        return None

def get_feature_info():
    """Define the clinical parameters with proper display names"""
    features_info = {
        # Demographics
        'age': {'display_name': 'Patient Age', 'type': 'number', 'min': 18, 'max': 120, 'default': 55, 'step': 1, 'help': 'Patient age in years'},
        'bmi': {'display_name': 'Body Mass Index', 'type': 'number', 'min': 10.0, 'max': 60.0, 'default': 26.0, 'step': 0.1, 'help': 'Body Mass Index'},
        'weight': {'display_name': 'Weight (kg)', 'type': 'number', 'min': 30, 'max': 200, 'default': 70, 'step': 1, 'help': 'Patient weight in kilograms'},
        'height': {'display_name': 'Height (cm)', 'type': 'number', 'min': 100, 'max': 220, 'default': 170, 'step': 1, 'help': 'Patient height in centimeters'},
        
        # Clinical Scores
        'apache_4a_hospital_death_prob': {'display_name': 'APACHE IVa Death Probability', 'type': 'number', 'min': 0.0, 'max': 1.0, 'default': 0.15, 'step': 0.01, 'help': 'APACHE IVa predicted hospital mortality probability'},
        'apache_4a_icu_death_prob': {'display_name': 'APACHE IVa ICU Death Probability', 'type': 'number', 'min': 0.0, 'max': 1.0, 'default': 0.12, 'step': 0.01, 'help': 'APACHE IVa predicted ICU mortality probability'},
        
        # Vital Signs (First 24 hours)
        'd1_heartrate_max': {'display_name': 'Max Heart Rate (bpm)', 'type': 'number', 'min': 40, 'max': 200, 'default': 85, 'step': 1, 'help': 'Maximum heart rate in first 24 hours'},
        'd1_mbp_max': {'display_name': 'Max Mean Blood Pressure (mmHg)', 'type': 'number', 'min': 40, 'max': 180, 'default': 80, 'step': 1, 'help': 'Maximum mean arterial pressure'},
        'd1_resprate_max': {'display_name': 'Max Respiratory Rate (breaths/min)', 'type': 'number', 'min': 8, 'max': 50, 'default': 18, 'step': 1, 'help': 'Maximum respiratory rate'},
        'd1_temp_max': {'display_name': 'Max Temperature (°C)', 'type': 'number', 'min': 32.0, 'max': 42.0, 'default': 37.0, 'step': 0.1, 'help': 'Maximum body temperature'},
        'd1_spo2_min': {'display_name': 'Min Oxygen Saturation (%)', 'type': 'number', 'min': 60, 'max': 100, 'default': 95, 'step': 1, 'help': 'Minimum oxygen saturation level'},
        
        # Laboratory Values
        'd1_creatinine_max': {'display_name': 'Max Creatinine (mg/dL)', 'type': 'number', 'min': 0.3, 'max': 10.0, 'default': 1.2, 'step': 0.1, 'help': 'Maximum creatinine level'},
        'd1_glucose_max': {'display_name': 'Max Glucose (mg/dL)', 'type': 'number', 'min': 50, 'max': 500, 'default': 140, 'step': 1, 'help': 'Maximum blood glucose level'},
        'd1_hemaglobin_max': {'display_name': 'Max Hemoglobin (g/dL)', 'type': 'number', 'min': 5.0, 'max': 20.0, 'default': 12.5, 'step': 0.1, 'help': 'Maximum hemoglobin level'},
        'd1_wbc_max': {'display_name': 'Max White Blood Count (×10³/μL)', 'type': 'number', 'min': 1.0, 'max': 50.0, 'default': 8.5, 'step': 0.1, 'help': 'Maximum white blood cell count'},
        'd1_bun_max': {'display_name': 'Max Blood Urea Nitrogen (mg/dL)', 'type': 'number', 'min': 5, 'max': 150, 'default': 20, 'step': 1, 'help': 'Maximum blood urea nitrogen level'},
        
        # Comorbidities
        'diabetes_mellitus': {'display_name': 'Diabetes Mellitus', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'History of diabetes mellitus'},
        'aids': {'display_name': 'AIDS/HIV', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'History of AIDS or HIV'},
        'cirrhosis': {'display_name': 'Liver Cirrhosis', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'History of liver cirrhosis'},
        'hepatic_failure': {'display_name': 'Hepatic Failure', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'History of hepatic failure'},
        
        # ICU Status
        'pre_icu_los_days': {'display_name': 'Pre-ICU Length of Stay (days)', 'type': 'number', 'min': 0, 'max': 30, 'default': 2, 'step': 1, 'help': 'Days in hospital before ICU admission'},
        'arf_apache': {'display_name': 'Acute Renal Failure', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'Acute renal failure at admission'},
        'gcs_verbal_apache': {'display_name': 'GCS Verbal Score', 'type': 'number', 'min': 1, 'max': 5, 'default': 4, 'step': 1, 'help': 'Glasgow Coma Scale - Verbal response (1-5)'},
        'gcs_motor_apache': {'display_name': 'GCS Motor Score', 'type': 'number', 'min': 1, 'max': 6, 'default': 5, 'step': 1, 'help': 'Glasgow Coma Scale - Motor response (1-6)'},
        'gcs_eyes_apache': {'display_name': 'GCS Eyes Score', 'type': 'number', 'min': 1, 'max': 4, 'default': 3, 'step': 1, 'help': 'Glasgow Coma Scale - Eye opening (1-4)'},
        'intubated_apache': {'display_name': 'Mechanically Ventilated', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'Mechanical ventilation at admission'},
        'ventilated_apache': {'display_name': 'Ventilator Support', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'Ventilator support required'},
        'immunosuppression': {'display_name': 'Immunosuppression', 'type': 'binary', 'options': ['No', 'Yes'], 'default': 0, 'help': 'Immunosuppressed state'},
        
        # Additional features
        'apache_2_diagnosis': {'display_name': 'APACHE II Diagnosis Code', 'type': 'number', 'min': 1, 'max': 1000, 'default': 100, 'step': 1, 'help': 'APACHE II diagnosis code'},
        'apache_3j_diagnosis': {'display_name': 'APACHE IIIj Diagnosis Code', 'type': 'number', 'min': 1, 'max': 1000, 'default': 100, 'step': 1, 'help': 'APACHE IIIj diagnosis code'},
        'hospital_admit_source': {'display_name': 'Hospital Admission Source', 'type': 'number', 'min': 1, 'max': 10, 'default': 3, 'step': 1, 'help': 'Source of hospital admission'},
        'icu_admit_source': {'display_name': 'ICU Admission Source', 'type': 'number', 'min': 1, 'max': 10, 'default': 3, 'step': 1, 'help': 'Source of ICU admission'},
        'icu_type': {'display_name': 'ICU Type', 'type': 'number', 'min': 1, 'max': 10, 'default': 3, 'step': 1, 'help': 'Type of ICU unit'}
    }
    return features_info

def get_sample_patients():
    """Define sample patients with realistic clinical values"""
    sample_patients = {
        'Low Risk Patient': {
            'description': 'Young patient with normal vitals and minimal comorbidities',
            'expected_outcome': 'Survival',
            'features': {
                'age': 35, 'bmi': 22.5, 'weight': 70, 'height': 175,
                'apache_4a_hospital_death_prob': 0.05, 'apache_4a_icu_death_prob': 0.03,
                'd1_heartrate_max': 75, 'd1_mbp_max': 85, 'd1_resprate_max': 16,
                'd1_temp_max': 36.8, 'd1_spo2_min': 98, 'd1_creatinine_max': 0.8,
                'd1_glucose_max': 110, 'd1_hemaglobin_max': 14.0, 'd1_wbc_max': 7.5,
                'd1_bun_max': 15, 'diabetes_mellitus': 0, 'aids': 0, 'cirrhosis': 0,
                'hepatic_failure': 0, 'pre_icu_los_days': 1, 'arf_apache': 0,
                'gcs_verbal_apache': 5, 'gcs_motor_apache': 6, 'gcs_eyes_apache': 4,
                'intubated_apache': 0, 'ventilated_apache': 0, 'immunosuppression': 0,
                'apache_2_diagnosis': 100, 'apache_3j_diagnosis': 100,
                'hospital_admit_source': 3, 'icu_admit_source': 3, 'icu_type': 3
            }
        },
        'Medium Risk Patient': {
            'description': 'Elderly patient with controlled comorbidities',
            'expected_outcome': 'Survival',
            'features': {
                'age': 68, 'bmi': 28.5, 'weight': 85, 'height': 170,
                'apache_4a_hospital_death_prob': 0.15, 'apache_4a_icu_death_prob': 0.12,
                'd1_heartrate_max': 95, 'd1_mbp_max': 75, 'd1_resprate_max': 22,
                'd1_temp_max': 37.5, 'd1_spo2_min': 92, 'd1_creatinine_max': 1.4,
                'd1_glucose_max': 160, 'd1_hemaglobin_max': 11.5, 'd1_wbc_max': 12.0,
                'd1_bun_max': 28, 'diabetes_mellitus': 1, 'aids': 0, 'cirrhosis': 0,
                'hepatic_failure': 0, 'pre_icu_los_days': 3, 'arf_apache': 0,
                'gcs_verbal_apache': 4, 'gcs_motor_apache': 5, 'gcs_eyes_apache': 3,
                'intubated_apache': 0, 'ventilated_apache': 0, 'immunosuppression': 0,
                'apache_2_diagnosis': 200, 'apache_3j_diagnosis': 200,
                'hospital_admit_source': 4, 'icu_admit_source': 4, 'icu_type': 3
            }
        },
        'High Risk Patient': {
            'description': 'Critically ill patient with multiple organ dysfunction',
            'expected_outcome': 'Mortality',
            'features': {
                'age': 82, 'bmi': 31.0, 'weight': 95, 'height': 165,
                'apache_4a_hospital_death_prob': 0.35, 'apache_4a_icu_death_prob': 0.30,
                'd1_heartrate_max': 120, 'd1_mbp_max': 65, 'd1_resprate_max': 28,
                'd1_temp_max': 38.2, 'd1_spo2_min': 85, 'd1_creatinine_max': 2.8,
                'd1_glucose_max': 240, 'd1_hemaglobin_max': 9.0, 'd1_wbc_max': 18.5,
                'd1_bun_max': 45, 'diabetes_mellitus': 1, 'aids': 0, 'cirrhosis': 1,
                'hepatic_failure': 0, 'pre_icu_los_days': 7, 'arf_apache': 1,
                'gcs_verbal_apache': 2, 'gcs_motor_apache': 3, 'gcs_eyes_apache': 2,
                'intubated_apache': 1, 'ventilated_apache': 1, 'immunosuppression': 0,
                'apache_2_diagnosis': 300, 'apache_3j_diagnosis': 300,
                'hospital_admit_source': 1, 'icu_admit_source': 1, 'icu_type': 1
            }
        }
    }
    return sample_patients

# ===============================================================
# 🎯 PREDICTION FUNCTIONS
# ===============================================================

def predict_mortality(features_dict, model, feature_names):
    """Make mortality prediction and return binary outcome"""
    try:
        # Convert dictionary to array in correct order
        features_array = np.array([features_dict[feature] for feature in feature_names]).reshape(1, -1)
        
        # Get prediction (0 or 1) and probability
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0, 1]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "High Risk", "high-risk"
    elif probability >= 0.3:
        return "Medium Risk", "medium-risk"
    else:
        return "Low Risk", "low-risk"

def get_outcome_display(prediction, probability):
    """Get display information for the outcome"""
    if prediction == 1:  # Dead
        return {
            'text': '❌ HIGH MORTALITY RISK',
            'description': 'Elevated probability of hospital mortality',
            'class': 'outcome-dead',
            'icon': '❌',
            'color': 'red'
        }
    else:  # Alive (prediction == 0)
        return {
            'text': '✅ FAVORABLE OUTCOME',
            'description': 'Low probability of hospital mortality',
            'class': 'outcome-alive',
            'icon': '✅',
            'color': 'green'
        }

# ===============================================================
# 📈 VISUALIZATION FUNCTIONS
# ===============================================================

def create_probability_gauge(probability, prediction):
    """Create a gauge chart for mortality probability"""
    if prediction == 1:  # High risk
        gauge_color = "red"
    else:  # Low risk
        gauge_color = "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Mortality Probability (%)"},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    return fig

# ===============================================================
# 🏠 MAIN APP
# ===============================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Hospital Mortality Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Load model and feature information
    model = load_model()
    features_info = get_feature_info()
    sample_patients = get_sample_patients()
    
    if model is None:
        return
    
    # Get the actual feature names from the model
    try:
        if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = list(model.named_steps['preprocessor'].get_feature_names_out())
        else:
            # Fallback to our defined features
            feature_names = list(features_info.keys())
    except:
        feature_names = list(features_info.keys())
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Single Patient Assessment", "Batch Prediction", "Model Information"]
    )
    
    # ===============================================================
    # 👤 SINGLE PATIENT ASSESSMENT
    # ===============================================================
    
    if app_mode == "Single Patient Assessment":
        st.header("👤 Patient Clinical Assessment")
        
        # Sample Patients Section
        st.subheader("🎯 Quick Patient Examples")
        st.info("Select a sample patient to auto-fill the form:")
        
        # Create columns for sample patients
        cols = st.columns(3)
        selected_sample = None
        
        for i, (patient_name, patient_info) in enumerate(sample_patients.items()):
            with cols[i]:
                if st.button(
                    f"**{patient_name}**\n\n{patient_info['description']}",
                    key=f"sample_{i}",
                    use_container_width=True,
                    help=f"Expected: {patient_info['expected_outcome']}"
                ):
                    selected_sample = patient_name
        
        # Display which sample is selected
        if selected_sample:
            st.success(f"✅ **{selected_sample}** loaded! Scroll down to see the filled form.")
        
        # Initialize patient features
        patient_features = {}
        
        # Create input form with organized sections
        with st.form("patient_form"):
            st.subheader("Patient Clinical Parameters")
            
            # Demographics Section
            with st.expander("📊 Demographics & Basic Info", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                demo_features = ['age', 'bmi', 'weight', 'height']
                
                for i, feature in enumerate(demo_features):
                    with [col1, col2, col3, col4][i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        if info['type'] == 'number':
                            # Ensure step is same type as min/max
                            step = info.get('step', 1)
                            if isinstance(info['min'], float) or isinstance(info['max'], float):
                                step = float(step)
                            else:
                                step = int(step)
                                
                            patient_features[feature] = st.number_input(
                                info['display_name'],
                                min_value=info['min'],
                                max_value=info['max'],
                                value=default_value,
                                step=step,
                                help=info['help']
                            )
            
            # Clinical Scores Section
            with st.expander("📈 Clinical Severity Scores", expanded=True):
                col1, col2 = st.columns(2)
                score_features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
                
                for i, feature in enumerate(score_features):
                    with [col1, col2][i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        patient_features[feature] = st.number_input(
                            info['display_name'],
                            min_value=info['min'],
                            max_value=info['max'],
                            value=default_value,
                            step=info.get('step', 0.01),
                            help=info['help']
                        )
            
            # Vital Signs Section
            with st.expander("💓 Vital Signs (First 24 Hours)", expanded=True):
                cols = st.columns(5)
                vital_features = ['d1_heartrate_max', 'd1_mbp_max', 'd1_resprate_max', 'd1_temp_max', 'd1_spo2_min']
                
                for i, feature in enumerate(vital_features):
                    with cols[i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        # Ensure step is same type as min/max
                        step = info.get('step', 1)
                        if isinstance(info['min'], float) or isinstance(info['max'], float):
                            step = float(step)
                        else:
                            step = int(step)
                            
                        patient_features[feature] = st.number_input(
                            info['display_name'],
                            min_value=info['min'],
                            max_value=info['max'],
                            value=default_value,
                            step=step,
                            help=info['help']
                        )
            
            # Laboratory Values Section
            with st.expander("🔬 Laboratory Values", expanded=False):
                cols = st.columns(5)
                lab_features = ['d1_creatinine_max', 'd1_glucose_max', 'd1_hemaglobin_max', 'd1_wbc_max', 'd1_bun_max']
                
                for i, feature in enumerate(lab_features):
                    with cols[i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        # Ensure step is same type as min/max
                        step = info.get('step', 0.1)
                        if isinstance(info['min'], float) or isinstance(info['max'], float):
                            step = float(step)
                        else:
                            step = int(step)
                            
                        patient_features[feature] = st.number_input(
                            info['display_name'],
                            min_value=info['min'],
                            max_value=info['max'],
                            value=default_value,
                            step=step,
                            help=info['help']
                        )
            
            # Comorbidities Section
            with st.expander("🩺 Comorbidities", expanded=False):
                cols = st.columns(4)
                comorbidity_features = ['diabetes_mellitus', 'aids', 'cirrhosis', 'hepatic_failure']
                
                for i, feature in enumerate(comorbidity_features):
                    with cols[i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        patient_features[feature] = st.selectbox(
                            info['display_name'],
                            options=[0, 1],
                            index=default_value,
                            format_func=lambda x: info['options'][x],
                            help=info['help']
                        )
            
            # ICU Status Section
            with st.expander("🏥 ICU Status & Scores", expanded=False):
                cols = st.columns(4)
                icu_features = ['pre_icu_los_days', 'arf_apache', 'intubated_apache', 'ventilated_apache']
                
                for i, feature in enumerate(icu_features):
                    with cols[i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        if info['type'] == 'binary':
                            patient_features[feature] = st.selectbox(
                                info['display_name'],
                                options=[0, 1],
                                index=default_value,
                                format_func=lambda x: info['options'][x],
                                help=info['help']
                            )
                        else:
                            # Ensure step is same type as min/max
                            step = info.get('step', 1)
                            if isinstance(info['min'], float) or isinstance(info['max'], float):
                                step = float(step)
                            else:
                                step = int(step)
                                
                            patient_features[feature] = st.number_input(
                                info['display_name'],
                                min_value=info['min'],
                                max_value=info['max'],
                                value=default_value,
                                step=step,
                                help=info['help']
                            )
                
                # GCS Scores
                st.markdown("**Glasgow Coma Scale:**")
                gcs_cols = st.columns(3)
                gcs_features = ['gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache']
                
                for i, feature in enumerate(gcs_features):
                    with gcs_cols[i]:
                        info = features_info[feature]
                        if selected_sample:
                            default_value = sample_patients[selected_sample]['features'][feature]
                        else:
                            default_value = info['default']
                        
                        patient_features[feature] = st.number_input(
                            info['display_name'],
                            min_value=info['min'],
                            max_value=info['max'],
                            value=default_value,
                            step=1,  # GCS scores are integers
                            help=info['help']
                        )
            
            # Additional features section for remaining features
            remaining_features = set(feature_names) - set(patient_features.keys())
            if remaining_features:
                with st.expander("⚙️ Additional Parameters", expanded=False):
                    cols = st.columns(3)
                    remaining_list = list(remaining_features)
                    
                    for i, feature in enumerate(remaining_list):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if feature in features_info:
                                info = features_info[feature]
                                if selected_sample and feature in sample_patients[selected_sample]['features']:
                                    default_value = sample_patients[selected_sample]['features'][feature]
                                else:
                                    default_value = info['default']
                                
                                if info['type'] == 'binary':
                                    patient_features[feature] = st.selectbox(
                                        info['display_name'],
                                        options=[0, 1],
                                        index=default_value,
                                        format_func=lambda x: info['options'][x],
                                        help=info['help']
                                    )
                                else:
                                    # Ensure step is same type as min/max
                                    step = info.get('step', 1)
                                    if isinstance(info['min'], float) or isinstance(info['max'], float):
                                        step = float(step)
                                    else:
                                        step = int(step)
                                        
                                    patient_features[feature] = st.number_input(
                                        info['display_name'],
                                        min_value=info['min'],
                                        max_value=info['max'],
                                        value=default_value,
                                        step=step,
                                        help=info['help']
                                    )
            
            # Submit button - FIXED: Use st.form_submit_button()
            submitted = st.form_submit_button("🔍 Predict Mortality Risk")
        
        # Make prediction when form is submitted or sample is selected
        if submitted or selected_sample:
            with st.spinner('Analyzing clinical parameters...'):
                # Ensure we have all required features
                for feature in feature_names:
                    if feature not in patient_features:
                        # Set default value for any missing features
                        if feature in features_info:
                            patient_features[feature] = features_info[feature]['default']
                        else:
                            patient_features[feature] = 0
                
                prediction, probability = predict_mortality(patient_features, model, feature_names)
                
                if prediction is not None:
                    risk_level, risk_class = get_risk_level(probability)
                    outcome_info = get_outcome_display(prediction, probability)
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Display results in two columns
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # Outcome Display
                        st.markdown(f"""
                        <div class="{outcome_info['class']}">
                            <h2>{outcome_info['icon']} {outcome_info['text']}</h2>
                            <h3>Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}</h3>
                            <p><strong>Mortality Probability:</strong> {probability:.1%}</p>
                            <p><strong>Risk Level:</strong> {risk_level}</p>
                            <p>{outcome_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Clinical recommendations
                        st.subheader("📋 Clinical Implications")
                        if prediction == 1:  # High risk
                            st.error("""
                            **🚨 High Alert - Intensive Monitoring Recommended**
                            
                            **Consider:**
                            - Enhanced hemodynamic monitoring
                            - Frequent laboratory assessment
                            - Early specialist consultation
                            - ICU-level care if not already admitted
                            - Family discussion about goals of care
                            """)
                        else:
                            if risk_level == "Medium Risk":
                                st.warning("""
                                **⚠️ Moderate Risk - Close Observation**
                                
                                **Recommended:**
                                - Standard monitoring protocols
                                - Regular vital sign checks
                                - Watch for clinical deterioration
                                - Consider step-down unit placement
                                """)
                            else:
                                st.success("""
                                **✅ Low Risk - Standard Care**
                                
                                **Standard Protocol:**
                                - Routine clinical monitoring
                                - Standard nursing care
                                - Regular physician assessment
                                - Progressive mobility as tolerated
                                """)
                    
                    with res_col2:
                        # Probability gauge
                        st.plotly_chart(create_probability_gauge(probability, prediction), use_container_width=True)
                        
                        # Key parameters summary
                        st.subheader("📊 Key Clinical Parameters")
                        key_params = ['age', 'apache_4a_hospital_death_prob', 'd1_heartrate_max', 
                                    'd1_mbp_max', 'd1_spo2_min', 'd1_creatinine_max', 'gcs_verbal_apache']
                        
                        summary_data = []
                        for param in key_params:
                            if param in patient_features:
                                display_name = features_info[param]['display_name']
                                value = patient_features[param]
                                summary_data.append({'Parameter': display_name, 'Value': value})
                        
                        if summary_data:
                            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # ===============================================================
    # 📊 BATCH PREDICTION
    # ===============================================================
    
    elif app_mode == "Batch Prediction":
        st.header("📊 Batch Patient Assessment")
        
        st.info("""
        **Upload a CSV file with clinical parameters.** 
        The file should include the features used during model training.
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                batch_data = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(batch_data.head())
                
                # Check if required features are present
                missing_features = set(feature_names) - set(batch_data.columns)
                extra_features = set(batch_data.columns) - set(feature_names)
                
                if missing_features:
                    st.warning(f"⚠️ Missing features: {list(missing_features)[:5]}...")
                if extra_features:
                    st.info(f"ℹ️ Extra features will be ignored: {list(extra_features)[:5]}...")
                
                if st.button("🔍 Process Batch Prediction"):
                    with st.spinner('Processing patient data...'):
                        # Use only the features the model expects
                        available_features = [f for f in feature_names if f in batch_data.columns]
                        
                        if len(available_features) < len(feature_names) * 0.8:
                            st.error("❌ Too many missing features for reliable predictions")
                        else:
                            # Make predictions
                            predictions = model.predict(batch_data[available_features])
                            probabilities = model.predict_proba(batch_data[available_features])[:, 1]
                            
                            # Add predictions to dataframe
                            results_df = batch_data.copy()
                            results_df['Mortality_Prediction'] = predictions
                            results_df['Mortality_Probability'] = probabilities
                            results_df['Risk_Level'] = results_df['Mortality_Probability'].apply(
                                lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.3 else 'Low'
                            )
                            
                            # Display results
                            st.success(f"✅ Processed {len(results_df)} patients")
                            
                            # Summary statistics
                            st.subheader("📈 Prediction Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            high_risk_count = (results_df['Risk_Level'] == 'High').sum()
                            medium_risk_count = (results_df['Risk_Level'] == 'Medium').sum()
                            low_risk_count = (results_df['Risk_Level'] == 'Low').sum()
                            
                            col1.metric("High Risk Patients", high_risk_count)
                            col2.metric("Medium Risk Patients", medium_risk_count)
                            col3.metric("Low Risk Patients", low_risk_count)
                            
                            # Display results table
                            st.subheader("🔍 Detailed Predictions")
                            display_cols = ['Mortality_Prediction', 'Mortality_Probability', 'Risk_Level'] + available_features[:10]
                            st.dataframe(results_df[display_cols].head(20))
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="mortality_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # ===============================================================
    # ℹ️ MODEL INFORMATION
    # ===============================================================
    
    else:
        st.header("ℹ️ Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Overview")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Purpose:** Hospital mortality risk prediction")
            st.write("**Features:** Clinical parameters from ICU patients")
            st.write("**Training:** Historical ICU patient data")
            
            st.subheader("Clinical Parameters")
            features_info = get_feature_info()
            for category in ['Demographics', 'Clinical Scores', 'Vital Signs', 'Laboratory Values']:
                with st.expander(f"📁 {category}"):
                    # Filter features by category
                    if category == 'Demographics':
                        features = ['age', 'bmi', 'weight', 'height']
                    elif category == 'Clinical Scores':
                        features = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
                    elif category == 'Vital Signs':
                        features = [f for f in features_info.keys() if f.startswith('d1_') and ('max' in f or 'min' in f) and f not in ['d1_creatinine_max', 'd1_glucose_max', 'd1_hemaglobin_max', 'd1_wbc_max', 'd1_bun_max']]
                    else:
                        features = [f for f in features_info.keys() if f.startswith('d1_') and f in ['d1_creatinine_max', 'd1_glucose_max', 'd1_hemaglobin_max', 'd1_wbc_max', 'd1_bun_max']]
                    
                    for feature in features:
                        if feature in features_info:
                            st.write(f"• {features_info[feature]['display_name']}")
        
        with col2:
            st.subheader("Sample Patient Profiles")
            for patient_name, patient_info in sample_patients.items():
                with st.expander(f"👤 {patient_name}"):
                    st.write(f"**Description:** {patient_info['description']}")
                    st.write(f"**Expected Outcome:** {patient_info['expected_outcome']}")
                    
                    # Show key parameters
                    key_params = ['age', 'apache_4a_hospital_death_prob', 'd1_heartrate_max', 'd1_spo2_min']
                    params_text = []
                    for param in key_params:
                        if param in patient_info['features']:
                            display_name = features_info[param]['display_name']
                            value = patient_info['features'][param]
                            params_text.append(f"{display_name}: {value}")
                    
                    st.write("**Key Parameters:** " + " | ".join(params_text))
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ Clinical Decision Support Tool**
        
        This prediction model is intended for clinical decision support only. 
        Healthcare professionals should use clinical judgment in conjunction with 
        this tool when making patient care decisions. Predictions are probabilistic 
        estimates based on historical data and should not be used as sole criteria 
        for clinical decisions.
        """)

# ===============================================================
# 🚀 RUN THE APP
# ===============================================================

if __name__ == "__main__":
    main()
