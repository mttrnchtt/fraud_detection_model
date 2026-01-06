import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of fraudulent transactions: {df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean()*100:.3f}%")

# ============================================
# 2. PREQUENTIAL SPLIT FUNCTION
# ============================================
def prequential_split_indices(df, n_folds=4, delta_train_pct=0.10, 
                            delta_delay_pct=0.05, delta_test_pct=0.05):
    """
    Create prequential (rolling window) train-test splits based on time
    
    Parameters:
    - n_folds: number of validation folds
    - delta_train_pct: percentage of data for training in each fold
    - delta_delay_pct: percentage of data for delay period
    - delta_test_pct: percentage of data for testing in each fold
    """
    # First, reset index to ensure we have a proper integer index
    df_indexed = df.reset_index(drop=True)
    max_time = df_indexed['Time'].max()
    min_time = df_indexed['Time'].min()
    
    split_indices = []
    
    for fold in range(n_folds):
        # Calculate time boundaries for this fold
        fold_start = min_time + fold * delta_test_pct * (max_time - min_time)
        train_start = fold_start
        train_end = train_start + (delta_train_pct * (max_time - min_time))
        delay_end = train_end + (delta_delay_pct * (max_time - min_time))
        test_end = delay_end + (delta_test_pct * (max_time - min_time))
        
        # Get indices as numpy arrays of integers
        train_idx = np.where((df_indexed['Time'] >= train_start) & (df_indexed['Time'] < train_end))[0]
        test_idx = np.where((df_indexed['Time'] >= delay_end) & (df_indexed['Time'] < test_end))[0]
        
        # Only add if both train and test have data
        if len(train_idx) > 0 and len(test_idx) > 0:
            split_indices.append((train_idx, test_idx))
            print(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    
    return split_indices

# ============================================
# 3. PREQUENTIAL GRID SEARCH FUNCTION
# ============================================
def prequential_grid_search(df, classifier, input_features, output_feature,
                            parameters, scoring, n_folds=4,
                            delta_train_pct=0.10, delta_delay_pct=0.05, 
                            delta_test_pct=0.05,
                            performance_metrics=['roc_auc', 'average_precision'],
                            search_type='grid', n_iter=10, random_state=0,
                            n_jobs=-1):
    """
    Perform prequential validation with grid or random search
    """
    # Create pipeline with scaler
    estimators = [
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ]
    pipe = Pipeline(estimators)
    
    # Get prequential split indices
    split_indices = prequential_split_indices(df, n_folds, delta_train_pct,
                                             delta_delay_pct, delta_test_pct)
    
    # Perform grid or random search
    if search_type == 'grid':
        search = GridSearchCV(pipe, parameters, scoring=scoring, 
                             cv=split_indices, refit=False, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(pipe, parameters, scoring=scoring,
                                   cv=split_indices, refit=False, 
                                   n_iter=n_iter, random_state=random_state,
                                   n_jobs=n_jobs)
    
    # Fit
    X = df[input_features]
    y = df[output_feature]
    
    print(f"  Running {search_type} search with {len(parameters[list(parameters.keys())[0]])} parameter combinations...")
    search.fit(X, y)
    
    # Extract results
    results_df = pd.DataFrame()
    for metric in performance_metrics:
        metric_name = metric.replace('_', ' ').title()
        results_df[f'{metric_name} Mean'] = search.cv_results_[f'mean_test_{metric}']
        results_df[f'{metric_name} Std'] = search.cv_results_[f'std_test_{metric}']
    
    results_df['Parameters'] = search.cv_results_['params']
    results_df['Fit Time'] = search.cv_results_['mean_fit_time']
    
    return results_df

# ============================================
# 4. MODEL SELECTION WRAPPER
# ============================================
def model_selection_wrapper(df, classifier, input_features, output_feature,
                           parameters, scoring,
                           n_folds=4, delta_train_pct=0.10,
                           delta_delay_pct=0.05, delta_test_pct=0.05,
                           performance_metrics=['roc_auc', 'average_precision'],
                           search_type='grid', n_iter=10, n_jobs=-1):
    """
    Wrapper that performs both validation and test set evaluation
    """
    # Reset index to ensure proper integer indexing
    df = df.reset_index(drop=True)
    
    # Split data into validation period and test period
    max_time = df['Time'].max()
    min_time = df['Time'].min()
    
    # Use first 60% for validation, last 20% for test
    validation_end = min_time + (max_time - min_time) * 0.60
    test_start = min_time + (max_time - min_time) * 0.70
    
    df_validation = df[df['Time'] <= validation_end].copy().reset_index(drop=True)
    df_test = df[df['Time'] >= test_start].copy().reset_index(drop=True)
    
    print(f"\nValidation set: {len(df_validation)} transactions")
    print(f"Test set: {len(df_test)} transactions")
    
    # Validation performance
    print("\n[VALIDATION SET]")
    results_validation = prequential_grid_search(
        df_validation, classifier, input_features, output_feature,
        parameters, scoring, n_folds, delta_train_pct, delta_delay_pct,
        delta_test_pct, performance_metrics, search_type, n_iter, n_jobs=n_jobs
    )
    results_validation = results_validation.add_suffix(' Validation')
    
    # Test performance
    print("\n[TEST SET]")
    results_test = prequential_grid_search(
        df_test, classifier, input_features, output_feature,
        parameters, scoring, n_folds, delta_train_pct, delta_delay_pct,
        delta_test_pct, performance_metrics, search_type, n_iter, n_jobs=n_jobs
    )
    results_test = results_test.add_suffix(' Test')
    
    # Combine results
    results_validation = results_validation.drop(
        columns=[c for c in results_validation.columns if 'Parameters' in c or 'Fit Time' in c]
    )
    combined_results = pd.concat([results_test, results_validation], axis=1)
    
    return combined_results

# ============================================
# 5. SUMMARY PERFORMANCE FUNCTION
# ============================================
def get_summary_performances(results_df, param_col='Parameters Summary'):
    """Extract best parameters and performances"""
    metrics = ['Roc Auc', 'Average Precision']
    summary = pd.DataFrame(columns=metrics)
    
    results_df = results_df.reset_index(drop=True)
    
    best_params = []
    validation_perf = []
    test_perf = []
    optimal_params = []
    optimal_test_perf = []
    
    for metric in metrics:
        # Best validation parameters
        idx_best_val = results_df[f'{metric} Mean Validation'].idxmax()
        best_params.append(results_df[param_col].iloc[idx_best_val])
        
        val_mean = results_df[f'{metric} Mean Validation'].iloc[idx_best_val]
        val_std = results_df[f'{metric} Std Validation'].iloc[idx_best_val]
        validation_perf.append(f"{val_mean:.3f}+/-{val_std:.3f}")
        
        test_mean = results_df[f'{metric} Mean Test'].iloc[idx_best_val]
        test_std = results_df[f'{metric} Std Test'].iloc[idx_best_val]
        test_perf.append(f"{test_mean:.3f}+/-{test_std:.3f}")
        
        # Optimal test parameters
        idx_opt_test = results_df[f'{metric} Mean Test'].idxmax()
        optimal_params.append(results_df[param_col].iloc[idx_opt_test])
        
        opt_mean = results_df[f'{metric} Mean Test'].iloc[idx_opt_test]
        opt_std = results_df[f'{metric} Std Test'].iloc[idx_opt_test]
        optimal_test_perf.append(f"{opt_mean:.3f}+/-{opt_std:.3f}")
    
    summary.loc['Best Validation Parameters'] = best_params
    summary.loc['Validation Performance'] = validation_perf
    summary.loc['Test Performance'] = test_perf
    summary.loc['Optimal Parameters'] = optimal_params
    summary.loc['Optimal Test Performance'] = optimal_test_perf
    
    return summary

# ============================================
# 6. DEFINE FEATURES AND METRICS
# ============================================
input_features = [col for col in df.columns if col not in ['Time', 'Class']]
output_feature = 'Class'

print(f"\nNumber of input features: {len(input_features)}")

# Scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}

performance_metrics = ['roc_auc', 'average_precision']

# ============================================
# 7. MODEL SELECTION - DECISION TREE
# ============================================
print("\n" + "="*80)
print("MODEL SELECTION: DECISION TREE")
print("="*80)

classifier_dt = DecisionTreeClassifier()
parameters_dt = {
    'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'clf__random_state': [0]
}

start_time = time.time()
results_dt = model_selection_wrapper(
    df, classifier_dt, input_features, output_feature,
    parameters_dt, scoring, n_folds=3, n_jobs=-1
)
execution_time_dt = time.time() - start_time

# Add parameter summary
params_dict = dict(results_dt['Parameters Test'])
results_dt['Parameters Summary'] = [params_dict[i]['clf__max_depth'] 
                                    for i in range(len(params_dict))]

print(f"\n✓ Decision Tree completed in {execution_time_dt:.2f}s")
print("\nTop 5 configurations by validation AUC ROC:")
print(results_dt.nsmallest(5, 'Roc Auc Mean Validation')[
    ['Parameters Summary', 'Roc Auc Mean Validation', 'Roc Auc Mean Test']
].to_string())

summary_dt = get_summary_performances(results_dt)
print("\nSummary:")
print(summary_dt)

# ============================================
# 8. MODEL SELECTION - LOGISTIC REGRESSION
# ============================================
print("\n" + "="*80)
print("MODEL SELECTION: LOGISTIC REGRESSION")
print("="*80)

classifier_lr = LogisticRegression(max_iter=1000)
parameters_lr = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__random_state': [0]
}

start_time = time.time()
results_lr = model_selection_wrapper(
    df, classifier_lr, input_features, output_feature,
    parameters_lr, scoring, n_folds=3, n_jobs=-1
)
execution_time_lr = time.time() - start_time

params_dict = dict(results_lr['Parameters Test'])
results_lr['Parameters Summary'] = [params_dict[i]['clf__C'] 
                                    for i in range(len(params_dict))]

print(f"\n✓ Logistic Regression completed in {execution_time_lr:.2f}s")
summary_lr = get_summary_performances(results_lr)
print("\nSummary:")
print(summary_lr)

# ============================================
# 9. MODEL SELECTION - RANDOM FOREST
# ============================================
print("\n" + "="*80)
print("MODEL SELECTION: RANDOM FOREST")
print("="*80)

classifier_rf = RandomForestClassifier()
parameters_rf = {
    'clf__max_depth': [5, 10, 20],
    'clf__n_estimators': [50, 100],
    'clf__random_state': [0],
    'clf__n_jobs': [-1]
}

start_time = time.time()
results_rf = model_selection_wrapper(
    df, classifier_rf, input_features, output_feature,
    parameters_rf, scoring, n_folds=3, n_jobs=1  # n_jobs=1 for outer loop
)
execution_time_rf = time.time() - start_time

params_dict = dict(results_rf['Parameters Test'])
results_rf['Parameters Summary'] = [
    f"{params_dict[i]['clf__n_estimators']}/{params_dict[i]['clf__max_depth']}"
    for i in range(len(params_dict))
]

print(f"\n✓ Random Forest completed in {execution_time_rf:.2f}s")
summary_rf = get_summary_performances(results_rf)
print("\nSummary:")
print(summary_rf)

# ============================================
# 10. MODEL SELECTION - XGBOOST
# ============================================
print("\n" + "="*80)
print("MODEL SELECTION: XGBOOST")
print("="*80)

classifier_xgb = xgboost.XGBClassifier(eval_metric='logloss')
parameters_xgb = {
    'clf__max_depth': [3, 6, 9],
    'clf__n_estimators': [50, 100],
    'clf__learning_rate': [0.1, 0.3],
    'clf__random_state': [0],
    'clf__n_jobs': [-1]
}

start_time = time.time()
results_xgb = model_selection_wrapper(
    df, classifier_xgb, input_features, output_feature,
    parameters_xgb, scoring, n_folds=3, n_jobs=1
)
execution_time_xgb = time.time() - start_time

params_dict = dict(results_xgb['Parameters Test'])
results_xgb['Parameters Summary'] = [
    f"{params_dict[i]['clf__n_estimators']}/{params_dict[i]['clf__learning_rate']}/{params_dict[i]['clf__max_depth']}"
    for i in range(len(params_dict))
]

print(f"\n✓ XGBoost completed in {execution_time_xgb:.2f}s")
summary_xgb = get_summary_performances(results_xgb)
print("\nSummary:")
print(summary_xgb)

# ============================================
# 11. COMPARISON OF ALL MODELS
# ============================================
print("\n" + "="*80)
print("COMPARISON OF ALL MODELS")
print("="*80)

results_dict = {
    'Decision Tree': results_dt,
    'Logistic Regression': results_lr,
    'Random Forest': results_rf,
    'XGBoost': results_xgb
}

execution_times = {
    'Decision Tree': execution_time_dt,
    'Logistic Regression': execution_time_lr,
    'Random Forest': execution_time_rf,
    'XGBoost': execution_time_xgb
}

# Extract best performances for each model
comparison_df = pd.DataFrame()
for model_name, results in results_dict.items():
    best_idx = results['Roc Auc Mean Validation'].idxmax()
    
    comparison_df.loc[model_name, 'AUC ROC (Val)'] = results['Roc Auc Mean Validation'].iloc[best_idx]
    comparison_df.loc[model_name, 'AUC ROC (Test)'] = results['Roc Auc Mean Test'].iloc[best_idx]
    comparison_df.loc[model_name, 'AP (Val)'] = results['Average Precision Mean Validation'].iloc[best_idx]
    comparison_df.loc[model_name, 'AP (Test)'] = results['Average Precision Mean Test'].iloc[best_idx]
    comparison_df.loc[model_name, 'Execution Time (s)'] = execution_times[model_name]

print(comparison_df.round(3))

# ============================================
# 12. VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC ROC comparison
ax = axes[0, 0]
x = np.arange(len(results_dict))
width = 0.35
ax.bar(x - width/2, comparison_df['AUC ROC (Val)'], width, label='Validation', alpha=0.8)
ax.bar(x + width/2, comparison_df['AUC ROC (Test)'], width, label='Test', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('AUC ROC')
ax.set_title('AUC ROC Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Average Precision comparison
ax = axes[0, 1]
ax.bar(x - width/2, comparison_df['AP (Val)'], width, label='Validation', alpha=0.8)
ax.bar(x + width/2, comparison_df['AP (Test)'], width, label='Test', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Average Precision')
ax.set_title('Average Precision Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Execution times
ax = axes[1, 0]
ax.bar(comparison_df.index, comparison_df['Execution Time (s)'], color='steelblue', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Time (seconds)')
ax.set_title('Execution Time Comparison')
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Decision tree depth analysis
ax = axes[1, 1]
ax.plot(results_dt['Parameters Summary'], results_dt['Roc Auc Mean Validation'], 
        'o-', label='Validation', linewidth=2)
ax.plot(results_dt['Parameters Summary'], results_dt['Roc Auc Mean Test'], 
        's-', label='Test', linewidth=2)
ax.set_xlabel('Tree Depth')
ax.set_ylabel('AUC ROC')
ax.set_title('Decision Tree: AUC ROC vs Depth')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 13. FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

best_model = comparison_df['AUC ROC (Test)'].idxmax()
print(f"\n✓ Best model (by AUC ROC on test): {best_model}")
print(f"  - AUC ROC: {comparison_df.loc[best_model, 'AUC ROC (Test)']:.3f}")
print(f"  - Average Precision: {comparison_df.loc[best_model, 'AP (Test)']:.3f}")
print(f"  - Execution time: {comparison_df.loc[best_model, 'Execution Time (s)']:.2f}s")