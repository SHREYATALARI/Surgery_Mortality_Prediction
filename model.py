import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Load dataset

# %%
df = pd.read_csv("mrk0.csv")
print("Dataset shape:", df.shape)
df.head()

# %% [markdown]
# # MAP TO SURGERY PARAMETERS

# %%
# Map your existing columns to surgery-like parameters
# We'll select the most relevant clinical features for mortality prediction

# Clinical parameters that are similar to our surgery parameters
CLINICAL_PARAMETERS_MAPPING = {
    # Demographics and basic info
    'age': 'Patient_Age',
    'bmi': 'BMI',
    'weight': 'Weight',
    'height': 'Height',
    
    # Clinical scores and diagnoses
    'apache_2_diagnosis': 'APACHE_2_Diagnosis',
    'apache_3j_diagnosis': 'APACHE_3J_Diagnosis', 
    'apache_4a_hospital_death_prob': 'APACHE_4a_Death_Probability',
    'apache_4a_icu_death_prob': 'APACHE_4a_ICU_Death_Probability',
    
    # Vital signs - using D1 (first day) measurements as baseline
    'd1_heartrate_max': 'Max_Heart_Rate',
    'd1_mbp_max': 'Max_Mean_BP',
    'd1_resprate_max': 'Max_Resp_Rate', 
    'd1_temp_max': 'Max_Temperature',
    'd1_spo2_min': 'Min_Oxygen_Saturation',
    
    # Lab values - using D1 measurements
    'd1_creatinine_max': 'Max_Creatinine',
    'd1_glucose_max': 'Max_Glucose',
    'd1_hemaglobin_max': 'Max_Hemoglobin',
    'd1_wbc_max': 'Max_WBC',
    'd1_bun_max': 'Max_BUN',
    
    # Comorbidities
    'diabetes_mellitus': 'Diabetes_Mellitus',
    'aids': 'AIDS',
    'cirrhosis': 'Cirrhosis',
    'hepatic_failure': 'Hepatic_Failure',
    
    # ICU status
    'pre_icu_los_days': 'Pre_ICU_LOS_Days',
    'arf_apache': 'Acute_Renal_Failure',
    'gcs_verbal_apache': 'GCS_Verbal',
    'gcs_motor_apache': 'GCS_Motor',
    'gcs_eyes_apache': 'GCS_Eyes'
}

print("🔍 MAPPING CLINICAL PARAMETERS")
print("="*60)

# Find which mapped parameters exist in your dataset
available_mapped_params = []
for original_col, mapped_name in CLINICAL_PARAMETERS_MAPPING.items():
    if original_col in df.columns:
        available_mapped_params.append(original_col)
        print(f"✅ {original_col} -> {mapped_name}")

print(f"\n📊 Total mapped parameters: {len(available_mapped_params)}")

# Also include some additional important parameters that might be available
additional_params = [
    'hospital_admit_source', 'icu_admit_source', 'icu_type', 
    'intubated_apache', 'ventilated_apache', 'immunosuppression'
]

available_additional = [p for p in additional_params if p in df.columns]
print(f"📋 Additional available parameters: {len(available_additional)}")

# Combine all available parameters
all_available_params = available_mapped_params + available_additional

print(f"\n🎯 TOTAL FEATURES AVAILABLE: {len(all_available_params)}")

# %% [markdown]
# # EDA

# %%
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes.value_counts())

# %% [markdown]
# # Target Variable Distribution

# %%
if 'hospital_death' in df.columns:
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='hospital_death', data=df)
    plt.title('Hospital Death Distribution')
    plt.xlabel('Death (0=No, 1=Yes)')
    
    plt.subplot(1, 2, 2)
    df['hospital_death'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    plt.title('Death Percentage')
    
    plt.tight_layout()
    plt.show()

    target_stats = df['hospital_death'].value_counts()
    print("Target Distribution:")
    print(f"Survived (0): {target_stats[0]:,} patients ({target_stats[0]/len(df)*100:.1f}%)")
    print(f"Died (1): {target_stats[1]:,} patients ({target_stats[1]/len(df)*100:.1f}%)")
    
    # Check class balance
    if target_stats[1] / len(df) < 0.1:
        print("⚠️  Highly imbalanced dataset - will use class weights in models")
else:
    print("❌ 'hospital_death' column not found!")

# %% [markdown]
# # Missing Values Analysis

# %%
print("🔍 MISSING VALUES ANALYSIS")
print("="*50)

missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100

# Show features with missing values
missing_data = pd.DataFrame({
    'Missing_Count': missing_summary,
    'Missing_Percentage': missing_percentage
}).sort_values('Missing_Percentage', ascending=False)

missing_data = missing_data[missing_data['Missing_Count'] > 0]

if len(missing_data) > 0:
    print("Features with missing values:")
    print(missing_data.head(20))  # Show top 20
    
    # Check if our selected features have high missingness
    selected_missing = missing_data[missing_data.index.isin(all_available_params)]
    if len(selected_missing) > 0:
        print(f"\n⚠️  Selected features with missing values:")
        print(selected_missing)
else:
    print("✅ No missing values found!")

# %% [markdown]
# # Feature Distributions

# %%
print("📊 FEATURE DISTRIBUTIONS")
print("="*50)

# Plot distributions for key numeric features
numeric_features = df[all_available_params].select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_features) > 0:
    # Plot first 12 numeric features
    plot_features = numeric_features[:12]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(plot_features):
        if i < len(axes):
            df[feature].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue')
            axes[i].set_title(f'{feature}\n(n={df[feature].notna().sum()})')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
    
    # Remove empty subplots
    for i in range(len(plot_features), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()
    
    print(f"📈 Displayed distributions for {len(plot_features)} numeric features")
else:
    print("No numeric features available for visualization")

# %% [markdown]
# # Correlation Analysis

# %%
print("🔗 CORRELATION ANALYSIS")
print("="*50)

if 'hospital_death' in df.columns and len(numeric_features) > 0:
    # Calculate correlations with target
    correlation_with_target = df[numeric_features + ['hospital_death']].corr()['hospital_death'].abs().sort_values(ascending=False)
    
    # Remove target itself
    correlation_with_target = correlation_with_target[1:]
    
    print("Top 15 features correlated with mortality:")
    print(correlation_with_target.head(15))
    
    # Plot top correlated features
    top_correlated = correlation_with_target.head(10).index.tolist()
    
    if len(top_correlated) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[top_correlated + ['hospital_death']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Top Features Correlation with Mortality')
        plt.tight_layout()
        plt.show()
    
    # Identify strongly correlated features (> 0.1 correlation)
    strong_correlations = correlation_with_target[correlation_with_target > 0.1]
    print(f"\n✅ Features with strong correlation (>0.1): {len(strong_correlations)}")
    if len(strong_correlations) > 0:
        print(strong_correlations)
    
else:
    print("Cannot compute correlations - missing target or numeric features")

# %% [markdown]
# # Data Preprocessing

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("🔧 DATA PREPROCESSING")
print("="*50)

if 'hospital_death' in df.columns and len(all_available_params) > 0:
    
    # Use available parameters as features
    X = df[all_available_params]
    y = df['hospital_death']
    
    print(f"✅ Using {len(all_available_params)} features for modeling")
    print("Selected features:")
    for i, feature in enumerate(all_available_params, 1):
        mapped_name = CLINICAL_PARAMETERS_MAPPING.get(feature, feature)
        print(f"  {i:2d}. {feature} -> {mapped_name}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set:  {X_test.shape[0]} samples")
    print(f"   Features:     {X_train.shape[1]}")
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📈 Data types:")
    print(f"   Numeric features:    {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    # Create preprocessing pipeline
    if len(categorical_cols) > 0:
        # If we have categorical features, we need more complex preprocessing
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
    else:
        # Only numeric features - simpler pipeline
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # For simple pipeline without ColumnTransformer
        feature_names = numeric_cols
    
    print(f"✅ Preprocessing complete!")
    print(f"   Processed training shape: {X_train_processed.shape}")
    print(f"   Processed testing shape:  {X_test_processed.shape}")
    
else:
    print("❌ Cannot proceed - missing target variable or features")
    raise ValueError("Required columns not found")

# %% [markdown]
# # Model Training - Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

print("🧪 LOGISTIC REGRESSION")
print("="*50)

model_lr = LogisticRegression(
    class_weight='balanced',
    solver='liblinear',
    max_iter=2000,
    random_state=42
)

model_lr.fit(X_train_processed, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_train_processed)
y_prob_lr = model_lr.predict_proba(X_train_processed)[:, 1]

# Evaluation
print("TRAINING RESULTS:")
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_train, y_pred_lr))
train_auc = roc_auc_score(y_train, y_prob_lr)
print(f"Training ROC-AUC: {train_auc:.4f}")

# Test set evaluation
y_pred_lr_test = model_lr.predict(X_test_processed)
y_prob_lr_test = model_lr.predict_proba(X_test_processed)[:, 1]
test_auc = roc_auc_score(y_test, y_prob_lr_test)
test_accuracy = accuracy_score(y_test, y_pred_lr_test)

print(f"\nTEST RESULTS:")
print(f"Test ROC-AUC:     {test_auc:.4f}")
print(f"Test Accuracy:    {test_accuracy:.4f}")
print(f"Model converged in {model_lr.n_iter_[0]} iterations")

# %% [markdown]
# # Model Training - Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

print("🌲 RANDOM FOREST")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,  # Reduced for faster training
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_processed)
y_prob_rf = rf_model.predict_proba(X_test_processed)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
rf_auc = roc_auc_score(y_test, y_prob_rf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"ROC-AUC:  {rf_auc:.4f}")
print(f"Accuracy: {rf_accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names[:len(rf_model.feature_importances_)],  # Ensure same length
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10).to_string(index=False))

# %% [markdown]
# # Model Comparison

# %%
print("📊 MODEL COMPARISON")
print("="*50)

models_summary = {
    'Logistic Regression': {
        'auc': test_auc,
        'accuracy': test_accuracy
    },
    'Random Forest': {
        'auc': rf_auc, 
        'accuracy': rf_accuracy
    }
}

comparison_df = pd.DataFrame(models_summary).T
print(comparison_df.round(4))

# Find best model
best_model_name = max(models_summary.keys(), key=lambda x: models_summary[x]['auc'])
best_auc = models_summary[best_model_name]['auc']

print(f"\n🎯 BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})")

# %% [markdown]
# # Final Model Saving

# %%
import joblib

print("💾 SAVING MODEL")
print("="*50)

# Select best model
if best_model_name == 'Logistic Regression':
    best_model = model_lr
else:
    best_model = rf_model

# Create production pipeline
production_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Retrain on full training data (X_train, y_train already defined)
production_pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(production_pipeline, 'mortality_prediction_pipeline1.pkl')
print("✅ Production pipeline saved: 'mortality_prediction_pipeline1.pkl'")

# Test the saved pipeline
loaded_pipeline = joblib.load('mortality_prediction_pipeline1.pkl')

# Create a sample patient for testing (using mean values)
sample_patient = X_train.median().to_frame().T
sample_pred = loaded_pipeline.predict(sample_patient)
sample_prob = loaded_pipeline.predict_proba(sample_patient)[0, 1]

print(f"🧪 Sample test prediction:")
print(f"   Prediction: {'Death' if sample_pred[0] == 1 else 'Survival'}")
print(f"   Probability: {sample_prob:.3f}")

print(f"\n🎉 SUCCESS: Model trained on {len(all_available_params)} clinical parameters!")
print(f"   Best model: {best_model_name}")
print(f"   Test AUC: {best_auc:.4f}")