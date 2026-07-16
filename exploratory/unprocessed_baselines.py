import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost

# Load the data
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of fraudulent transactions: {df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean()*100:.3f}%")

# ============================================
# 1. DEFINE TRAIN/TEST SPLIT (TIME-BASED)
# ============================================
# Since Time is in seconds from first transaction, we'll use time-based split
# Training: first 70% of time period
# Test: last 30% of time period (with a delay gap)

def get_train_test_split(df, train_ratio=0.7, delay_ratio=0.1):
    """
    Split data based on time with a delay period
    
    Parameters:
    - train_ratio: proportion of time for training
    - delay_ratio: proportion of time for delay period
    """
    max_time = df['Time'].max()
    
    # Define time boundaries
    train_end_time = max_time * train_ratio
    delay_end_time = max_time * (train_ratio + delay_ratio)
    
    # Split data
    train_df = df[df['Time'] <= train_end_time].copy()
    test_df = df[df['Time'] > delay_end_time].copy()
    
    print(f"\nData split:")
    print(f"Training set: {len(train_df)} transactions, {train_df['Class'].sum()} frauds ({train_df['Class'].mean()*100:.3f}%)")
    print(f"Test set: {len(test_df)} transactions, {test_df['Class'].sum()} frauds ({test_df['Class'].mean()*100:.3f}%)")
    
    return train_df, test_df

train_df, test_df = get_train_test_split(df)

# ============================================
# 2. PREPARE FEATURES
# ============================================
# Use all V1-V28 features plus Amount (excluding Time and Class)
input_features = [col for col in df.columns if col not in ['Time', 'Class']]
output_feature = 'Class'

print(f"\nInput features: {len(input_features)} features")
print(f"Features: {input_features}")

# ============================================
# 3. SCALING FUNCTION
# ============================================
def scale_data(train_df, test_df, input_features):
    """Scale features using StandardScaler fitted on training data"""
    scaler = StandardScaler()
    
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    # Fit on training data and transform both sets
    train_df_scaled[input_features] = scaler.fit_transform(train_df[input_features])
    test_df_scaled[input_features] = scaler.transform(test_df[input_features])
    
    return train_df_scaled, test_df_scaled

# ============================================
# 4. MODEL TRAINING FUNCTION
# ============================================
def fit_model_and_get_predictions(classifier, train_df, test_df,
                                input_features, output_feature="Class", scale=True):
    """
    Train a classifier and get predictions
    
    Returns dictionary with:
    - classifier: trained model
    - predictions_train: training predictions
    - predictions_test: test predictions
    """
    # Scale data if requested
    if scale:
        train_df, test_df = scale_data(train_df, test_df, input_features)
    
    # Train the classifier
    classifier.fit(train_df[input_features], train_df[output_feature])
    
    # Get predictions (probabilities of fraud)
    predictions_test = classifier.predict_proba(test_df[input_features])[:, 1]
    
    predictions_train = classifier.predict_proba(train_df[input_features])[:, 1]
    
    model_and_predictions_dictionary = {
        'classifier': classifier,
        'predictions_test': predictions_test,
        'predictions_train': predictions_train,
    }
    
    return model_and_predictions_dictionary
# ============================================
# 5. PERFORMANCE METRICS
# ============================================
def performance_assessment(predictions_df, output_feature='Class',
                        prediction_feature='predictions', rounded=True):
    """
    Calculate AUC ROC and Average Precision
    
    Note: Card Precision@k is not applicable here as we don't have CUSTOMER_ID
    """
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], 
                                    predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], 
                                        predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]],
                            columns=['AUC ROC', 'Average Precision'])
    
    if rounded:
        performances = performances.round(3)
    
    return performances

# ============================================
# 6. TRAIN MULTIPLE MODELS
# ============================================
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

classifiers_dictionary = {
    'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
    'Decision Tree (depth=2)': DecisionTreeClassifier(max_depth=2, random_state=0),
    'Decision Tree (unlimited)': DecisionTreeClassifier(random_state=0),
    'Random Forest': RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1),
    'XGBoost': xgboost.XGBClassifier(random_state=0, n_jobs=-1, eval_metric='logloss')
}

fitted_models_and_predictions = {}

for classifier_name in classifiers_dictionary:
    print(f"\nTraining {classifier_name}...")
    model_and_predictions = fit_model_and_get_predictions(
        classifiers_dictionary[classifier_name], 
        train_df, 
        test_df,
        input_features=input_features,
        output_feature=output_feature,
        scale=True
    )
    fitted_models_and_predictions[classifier_name] = model_and_predictions

# ============================================
# 7. EVALUATE PERFORMANCE
# ============================================
def performance_assessment_model_collection(fitted_models_dict, 
                                        transactions_df,
                                        type_set='test'):
    """Evaluate all models and return performance DataFrame"""
    performances = pd.DataFrame()
    
    for classifier_name, model_and_predictions in fitted_models_dict.items():
        predictions_df = transactions_df.copy()
        predictions_df['predictions'] = model_and_predictions['predictions_' + type_set]
        
        performances_model = performance_assessment(
            predictions_df, 
            output_feature='Class',
            prediction_feature='predictions'
        )
        performances_model.index = [classifier_name]
        performances = pd.concat([performances, performances_model])
    
    return performances

print("\n" + "="*50)
print("TEST SET PERFORMANCE")
print("="*50)
test_performances = performance_assessment_model_collection(
    fitted_models_and_predictions, 
    test_df, 
    type_set='test'
)
print(test_performances)

print("\n" + "="*50)
print("TRAINING SET PERFORMANCE")
print("="*50)
train_performances = performance_assessment_model_collection(
    fitted_models_and_predictions, 
    train_df, 
    type_set='train'
)
print(train_performances)

# ============================================
# 9. VISUALIZE RESULTS
# ============================================
# Plot performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC ROC comparison
test_performances['AUC ROC'].plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_xlabel('AUC ROC Score')
axes[0].set_title('Test Set - AUC ROC Comparison')
axes[0].set_xlim([0, 1])
axes[0].grid(axis='x', alpha=0.3)
# Average Precision comparison
test_performances['Average Precision'].plot(kind='barh', ax=axes[1], color='lightcoral')
axes[1].set_xlabel('Average Precision Score')
axes[1].set_title('Test Set - Average Precision Comparison')
axes[1].set_xlim([0, 1])
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"✓ Best AUC ROC: {test_performances['AUC ROC'].max():.3f}")
print(f"✓ Best Average Precision: {test_performances['Average Precision'].max():.3f}")
best_model = test_performances['Average Precision'].idxmax()
print(f"✓ Best overall model: {best_model}")