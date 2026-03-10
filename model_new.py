#testing for pr changes
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
# # Model Training - Enhanced with XGBoost and Neural Network

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import time

# Import TensorFlow/Keras for Neural Network
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Neural Network will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Dictionary to store all models and their results
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle class imbalance
    )
}

# Store results
results = {}

print("🧪 ENHANCED MODEL TRAINING WITH XGBOOST AND NEURAL NETWORK")
print("="*70)

# Train and evaluate traditional models
for model_name, model in models.items():
    print(f"\n🎯 Training {model_name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train_processed, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    train_time = time.time() - start_time
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'train_time': train_time,
        'model': model
    }
    
    print(f"✅ {model_name} completed in {train_time:.2f} seconds")
    print(f"   ROC-AUC: {results[model_name]['roc_auc']:.4f}")
    print(f"   Accuracy: {results[model_name]['accuracy']:.4f}")
    print(f"   F1-Score: {results[model_name]['f1_score']:.4f}")

# %% [markdown]
# # Neural Network Implementation

# %%
if TENSORFLOW_AVAILABLE:
    print("\n🧠 TRAINING FULLY CONNECTED NEURAL NETWORK")
    print("="*50)
    
    # Define neural network architecture
    def create_neural_network(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    # Create and train neural network
    nn_model = create_neural_network(X_train_processed.shape[1])
    
    print("Neural Network Architecture:")
    nn_model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
    ]
    
    # Train neural network
    start_time = time.time()
    
    history = nn_model.fit(
        X_train_processed, y_train,
        validation_data=(X_test_processed, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}  # Handle class imbalance
    )
    
    train_time = time.time() - start_time
    
    # Neural network predictions
    y_prob_nn = nn_model.predict(X_test_processed).flatten()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)
    
    # Calculate metrics for neural network
    results['Neural Network'] = {
        'accuracy': accuracy_score(y_test, y_pred_nn),
        'precision': precision_score(y_test, y_pred_nn, zero_division=0),
        'recall': recall_score(y_test, y_pred_nn, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_nn, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_nn),
        'train_time': train_time,
        'model': nn_model,
        'history': history
    }
    
    print(f"✅ Neural Network completed in {train_time:.2f} seconds")
    print(f"   ROC-AUC: {results['Neural Network']['roc_auc']:.4f}")
    print(f"   Accuracy: {results['Neural Network']['accuracy']:.4f}")
    print(f"   F1-Score: {results['Neural Network']['f1_score']:.4f}")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0,1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # AUC
    axes[1,0].plot(history.history['auc'], label='Training AUC')
    axes[1,0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1,0].set_title('Model AUC')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('AUC')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1,1].plot(history.history['lr'], label='Learning Rate')
        axes[1,1].set_title('Learning Rate')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

else:
    print("⚠️  Neural Network skipped - TensorFlow not available")

# %% [markdown]
# # Enhanced Model Comparison

# %%
print("📊 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
print("="*70)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('roc_auc', ascending=False)

print("\nOverall Performance Metrics (Sorted by ROC-AUC):")
print(results_df[['roc_auc', 'accuracy', 'f1_score', 'precision', 'recall', 'train_time']].round(4))

# Plot performance comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ROC-AUC Comparison
axes[0,0].barh(results_df.index, results_df['roc_auc'], color='skyblue')
axes[0,0].set_xlabel('ROC-AUC Score')
axes[0,0].set_title('Model Performance (ROC-AUC)')
axes[0,0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random Guess')
for i, v in enumerate(results_df['roc_auc']):
    axes[0,0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Accuracy Comparison
axes[0,1].barh(results_df.index, results_df['accuracy'], color='lightgreen')
axes[0,1].set_xlabel('Accuracy Score')
axes[0,1].set_title('Model Performance (Accuracy)')
for i, v in enumerate(results_df['accuracy']):
    axes[0,1].text(v + 0.01, i, f'{v:.3f}', va='center')

# F1-Score Comparison
axes[0,2].barh(results_df.index, results_df['f1_score'], color='salmon')
axes[0,2].set_xlabel('F1-Score')
axes[0,2].set_title('Model Performance (F1-Score)')
for i, v in enumerate(results_df['f1_score']):
    axes[0,2].text(v + 0.01, i, f'{v:.3f}', va='center')

# Precision Comparison
axes[1,0].barh(results_df.index, results_df['precision'], color='gold')
axes[1,0].set_xlabel('Precision Score')
axes[1,0].set_title('Model Performance (Precision)')
for i, v in enumerate(results_df['precision']):
    axes[1,0].text(v + 0.01, i, f'{v:.3f}', va='center')

# Recall Comparison
axes[1,1].barh(results_df.index, results_df['recall'], color='lightcoral')
axes[1,1].set_xlabel('Recall Score')
axes[1,1].set_title('Model Performance (Recall)')
for i, v in enumerate(results_df['recall']):
    axes[1,1].text(v + 0.01, i, f'{v:.3f}', va='center')

# Training Time Comparison
axes[1,2].barh(results_df.index, results_df['train_time'], color='violet')
axes[1,2].set_xlabel('Training Time (seconds)')
axes[1,2].set_title('Training Time Comparison')
for i, v in enumerate(results_df['train_time']):
    axes[1,2].text(v + 0.01, i, f'{v:.2f}s', va='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Feature Importance Analysis

# %%
print("🔝 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from tree-based models
tree_based_models = ['Random Forest', 'XGBoost']

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for idx, model_name in enumerate(tree_based_models):
    if model_name in models:
        model = models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance DataFrame
            feat_imp_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            # Plot
            axes[idx].barh(feat_imp_df['feature'], feat_imp_df['importance'], color='lightseagreen')
            axes[idx].set_title(f'Top 10 Features - {model_name}')
            axes[idx].set_xlabel('Importance Score')
            
            # Add value labels
            for i, v in enumerate(feat_imp_df['importance']):
                axes[idx].text(v + 0.001, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.show()

# Display top features for each model
print("\n📋 TOP 5 FEATURES FOR TREE-BASED MODELS:")
for model_name in tree_based_models:
    if model_name in models:
        model = models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False).head(5)
            
            print(f"\n{model_name}:")
            for _, row in feat_imp_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

# %% [markdown]
# # Confusion Matrix for All Models

# %%
print("📈 CONFUSION MATRICES FOR ALL MODELS")
print("="*50)

# Create subplots for confusion matrices
n_models = len(results)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (model_name, result) in enumerate(results.items()):
    if idx < 4:  # Limit to 4 subplots
        if model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
            y_pred = (result['model'].predict(X_test_processed).flatten() > 0.5).astype(int)
        else:
            y_pred = result['model'].predict(X_test_processed)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f'{model_name}\nAUC: {result["roc_auc"]:.3f}')
        
        # Add labels
        axes[idx].set_xticklabels(['Survived', 'Died'])
        axes[idx].set_yticklabels(['Survived', 'Died'])

# Remove empty subplots if any
for idx in range(n_models, 4):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Print classification reports for all models
print("\n📋 DETAILED CLASSIFICATION REPORTS:")
for model_name, result in results.items():
    if model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
        y_pred = (result['model'].predict(X_test_processed).flatten() > 0.5).astype(int)
    else:
        y_pred = result['model'].predict(X_test_processed)
    
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

# %% [markdown]
# # Final Model Selection and Deployment

# %%
print("🏆 SELECTING BEST MODEL FOR DEPLOYMENT")
print("="*50)

# Find best model based on ROC-AUC
best_model_name = results_df.index[0]
best_model = results[best_model_name]['model']
best_auc = results_df.iloc[0]['roc_auc']

print(f"🎯 BEST PERFORMING MODEL: {best_model_name}")
print(f"📊 Best ROC-AUC Score: {best_auc:.4f}")
print(f"⚡ Training Time: {results_df.iloc[0]['train_time']:.2f} seconds")
print(f"🎯 Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
print(f"📈 F1-Score: {results_df.iloc[0]['f1_score']:.4f}")

# Compare performance improvement
print(f"\n🔍 PERFORMANCE COMPARISON:")
baseline_auc = results['Logistic Regression']['roc_auc']
improvement = best_auc - baseline_auc
print(f"   Logistic Regression (Baseline) AUC: {baseline_auc:.4f}")
print(f"   {best_model_name} AUC: {best_auc:.4f}")
print(f"   Improvement: +{improvement:.4f} ({improvement/baseline_auc*100:.1f}%)")

# %% [markdown]
# # Final Model Saving with Enhanced Pipeline

# %%
import joblib
from datetime import datetime

print("💾 SAVING ENHANCED PRODUCTION PIPELINE")
print("="*50)

# Create comprehensive production pipeline
production_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Retrain on full training data (only for non-neural network models)
if best_model_name != 'Neural Network':
    production_pipeline.fit(X_train, y_train)

# Save the pipeline with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'mortality_prediction_pipeline_enhanced_{timestamp}.pkl'

if best_model_name != 'Neural Network':
    joblib.dump(production_pipeline, model_filename)
    print(f"✅ Production pipeline saved: '{model_filename}'")
else:
    # For neural network, save separately
    nn_filename = f'neural_network_model_{timestamp}.h5'
    best_model.save(nn_filename)
    # Save preprocessor separately
    joblib.dump(preprocessor, f'preprocessor_{timestamp}.pkl')
    print(f"✅ Neural Network model saved: '{nn_filename}'")
    print(f"✅ Preprocessor saved: 'preprocessor_{timestamp}.pkl'")

# Also save the results and metadata
metadata = {
    'model_name': best_model_name,
    'performance_metrics': results[best_model_name],
    'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names,
    'training_date': timestamp,
    'dataset_shape': X_train.shape,
    'best_auc': best_auc,
    'all_results': results_df.to_dict()
}

joblib.dump(metadata, f'model_metadata_{timestamp}.pkl')
print(f"✅ Model metadata saved: 'model_metadata_{timestamp}.pkl'")

# Test the saved pipeline
print(f"\n🧪 SAMPLE PREDICTIONS TEST:")
sample_patients = [
    X_train.median().to_frame().T,  # Average patient
    X_train.quantile(0.25).to_frame().T,  # Lower risk patient
    X_train.quantile(0.75).to_frame().T   # Higher risk patient
]

patient_types = ['Average', 'Lower Risk', 'Higher Risk']

for i, (patient, p_type) in enumerate(zip(sample_patients, patient_types)):
    if best_model_name == 'Neural Network' and TENSORFLOW_AVAILABLE:
        # Preprocess for neural network
        patient_processed = preprocessor.transform(patient)
        sample_prob = best_model.predict(patient_processed).flatten()[0]
    else:
        sample_prob = production_pipeline.predict_proba(patient)[0, 1]
    
    sample_pred = 1 if sample_prob > 0.5 else 0
    
    print(f"   {p_type} Patient:")
    print(f"      Prediction: {'🚨 HIGH RISK' if sample_pred == 1 else '✅ LOW RISK'}")
    print(f"      Mortality Probability: {sample_prob:.3f}")
    print(f"      Risk Level: {'High' if sample_prob > 0.5 else 'Low'}")

print(f"\n🎉 SUCCESS: Enhanced model training completed!")
print(f"   Total models trained: {len(models) + (1 if TENSORFLOW_AVAILABLE else 0)}")
print(f"   Best model: {best_model_name}")
print(f"   Best Test AUC: {best_auc:.4f}")
print(f"   Features used: {len(all_available_params)}")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")
