import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
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
# 1. LOAD PREPROCESSED DATA
# ============================================
def load_processed_data(processed_dir: str):
    """
    Load X and y arrays from preprocessed directory
    """
    data_dir = Path(processed_dir)
    
    # Load features
    X_train = np.load(data_dir / "X_train.npz")['data']
    X_val = np.load(data_dir / "X_val.npz")['data']
    X_test = np.load(data_dir / "X_test.npz")['data']
    
    # Load targets
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    # Load manifest for info
    with open(data_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    
    print("="*80)
    print("LOADED PREPROCESSED DATA")
    print("="*80)
    print(f"Train set: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"Val set:   X_val shape={X_val.shape}, y_val shape={y_val.shape}")
    print(f"Test set:  X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    print(f"\nClass distribution (train): {manifest['class_balance']['train_pos']} positive, {manifest['class_balance']['train_neg']} negative")
    print(f"Class distribution (val):   {manifest['class_balance']['val_pos']} positive, {manifest['class_balance']['val_neg']} negative")
    print(f"Class distribution (test):  {manifest['class_balance']['test_pos']} positive, {manifest['class_balance']['test_neg']} negative")
    print(f"\nFeatures: {manifest['features']}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, manifest

# ============================================
# 2. CUSTOM CV SPLITTER FOR VALIDATION/TEST
# ============================================
class FixedSplit:
    """Custom CV splitter that uses fixed validation and test splits"""
    def __init__(self, n_splits=1):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
    def split(self, X, y, groups=None):
        """Yield single train/val split (test set is used separately)"""
        n_samples = X.shape[0]
        # Simple 80/20 split within the validation set
        split_point = int(0.8 * n_samples)
        train_idx = np.arange(split_point)
        val_idx = np.arange(split_point, n_samples)
        yield train_idx, val_idx

# ============================================
# 3. SCORING METRICS
# ============================================
scoring = {
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}

# ============================================
# 4. MODEL SELECTION FUNCTION
# ============================================
def perform_model_selection(X_train, y_train, X_val, y_val, X_test, y_test,
                        classifier, parameters, model_name):
    """
    Perform model selection using validation set and evaluate on test set
    """
    print(f"\n{'='*80}")
    print(f"MODEL SELECTION: {model_name}")
    print(f"{'='*80}")
    
    # Create pipeline with scaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])
    
    # Create custom CV splitter
    cv_splitter = FixedSplit(n_splits=1)
    
    # Perform grid or random search on validation set
    print(f"\n[1/3] Fitting grid search on validation set...")
    
    search = GridSearchCV(pipe, parameters, scoring=scoring, 
                            cv=cv_splitter, refit=False, verbose=1)    
    start_time = time.time()
    search.fit(X_val, y_val)
    val_time = time.time() - start_time
    
    # Extract validation results
    results_df = pd.DataFrame()
    for metric in ['roc_auc', 'average_precision']:
        metric_name = metric.replace('_', ' ').title()
        results_df[f'{metric_name} Val'] = search.cv_results_['mean_test_' + metric]
        results_df[f'{metric_name} Val Std'] = search.cv_results_['std_test_' + metric]
    results_df['Parameters'] = search.cv_results_['params']
    results_df['Val Fit Time'] = search.cv_results_['mean_fit_time']
    
    print(f"  ✓ Completed in {val_time:.2f}s")
    print(f"  ✓ Best validation AUC ROC: {results_df['Roc Auc Val'].max():.4f}")
    
    # Evaluate best parameters on test set
    print(f"\n[2/3] Evaluating best parameters on test set...")
    best_idx = results_df['Roc Auc Val'].idxmax()
    best_params = results_df['Parameters'].iloc[best_idx]
    
    # Train best model on full validation set
    pipe_best = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])
    # Set best parameters
    pipe_best.set_params(**best_params)
    
    pipe_best.fit(X_val, y_val)
    
    # Predict on test set
    y_pred_proba_test = pipe_best.predict_proba(X_test)[:, 1]
    
    # Calculate test metrics
    test_auc = metrics.roc_auc_score(y_test, y_pred_proba_test)
    test_ap = metrics.average_precision_score(y_test, y_pred_proba_test)
    
    results_df['Roc Auc Test'] = np.nan
    results_df.loc[best_idx, 'Roc Auc Test'] = test_auc
    results_df['Average Precision Test'] = np.nan
    results_df.loc[best_idx, 'Average Precision Test'] = test_ap
    
    print(f"  ✓ Best model test AUC ROC: {test_auc:.4f}")
    print(f"  ✓ Best model test AUPRC: {test_ap:.4f}")
    
    print(f"\n[3/3] Top 5 configurations by validation AUC ROC:")
    print(results_df.nsmallest(5, 'Roc Auc Val')[
        ['Parameters', 'Roc Auc Val', 'Roc Auc Test']
    ].to_string(index=False))
    
    # Summary for best model
    print(f"\n{'─'*80}")
    print(f"BEST MODEL FOR {model_name}")
    print(f"{'─'*80}")
    print(f"Parameters: {best_params}")
    print(f"Validation AUC ROC:     {results_df['Roc Auc Val'].iloc[best_idx]:.4f}")
    print(f"Test AUC ROC:           {test_auc:.4f}")
    print(f"Test Average Precision: {test_ap:.4f}")
    
    return results_df, best_params, test_auc, test_ap, pipe_best

# ============================================
# 5. LOAD DATA
# ============================================
# Update this path to your processed data directory
processed_data_dir = "data_processed"  # Change this path
X_train, X_val, X_test, y_train, y_val, y_test, manifest = load_processed_data(processed_data_dir)

# ============================================
# 6. MODEL SELECTION - DECISION TREE
# ============================================
classifier_dt = DecisionTreeClassifier(random_state=0)
parameters_dt = {
    'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'clf__random_state': [0]
}

results_dt, best_params_dt, test_auc_dt, test_ap_dt, model_dt = perform_model_selection(
    X_train, y_train, X_val, y_val, X_test, y_test,
    classifier_dt, parameters_dt, "DECISION TREE")

# ============================================
# 7. MODEL SELECTION - LOGISTIC REGRESSION
# ============================================
classifier_lr = LogisticRegression(max_iter=1000, random_state=0)
parameters_lr = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__random_state': [0]
}

results_lr, best_params_lr, test_auc_lr, test_ap_lr, model_lr = perform_model_selection(
    X_train, y_train, X_val, y_val, X_test, y_test,
    classifier_lr, parameters_lr, "LOGISTIC REGRESSION")

# ============================================
# 8. MODEL SELECTION - RANDOM FOREST
# ============================================
classifier_rf = RandomForestClassifier(random_state=0)
parameters_rf = {
    'clf__max_depth': [5, 10, 15, 20],
    'clf__n_estimators': [50, 100],
    'clf__random_state': [0],
    'clf__n_jobs': [-1]
}

results_rf, best_params_rf, test_auc_rf, test_ap_rf, model_rf = perform_model_selection(
    X_train, y_train, X_val, y_val, X_test, y_test,
    classifier_rf, parameters_rf, "RANDOM FOREST")

# ============================================
# 9. MODEL SELECTION - XGBOOST
# ============================================
classifier_xgb = xgboost.XGBClassifier(eval_metric='logloss', random_state=0)
parameters_xgb = {
    'clf__max_depth': [3, 6, 9],
    'clf__n_estimators': [50, 100],
    'clf__learning_rate': [0.1, 0.3],
    'clf__random_state': [0],
    'clf__n_jobs': [-1]
}

results_xgb, best_params_xgb, test_auc_xgb, test_ap_xgb, model_xgb = perform_model_selection(
    X_train, y_train, X_val, y_val, X_test, y_test,
    classifier_xgb, parameters_xgb, "XGBOOST")

# ============================================
# 10. COMPARISON OF ALL MODELS
# ============================================
print(f"\n{'='*80}")
print("FINAL COMPARISON - ALL MODELS")
print(f"{'='*80}\n")

comparison_data = {
    'Model': ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost'],
    'Test AUC ROC': [test_auc_dt, test_auc_lr, test_auc_rf, test_auc_xgb],
    'Test AUPRC': [test_ap_dt, test_ap_lr, test_ap_rf, test_ap_xgb]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test AUC ROC', ascending=False)

print(comparison_df.to_string(index=False))

# ============================================
# 11. DIAGNOSTICS
# ============================================
print(f"\n{'='*80}")
print("DATA DIAGNOSTICS")
print(f"{'='*80}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"\nTrain class balance: {y_train.sum()} fraud / {len(y_train)} total ({100*y_train.sum()/len(y_train):.2f}%)")
print(f"Val class balance: {y_val.sum()} fraud / {len(y_val)} total ({100*y_val.sum()/len(y_val):.2f}%)")
print(f"Test class balance: {y_test.sum()} fraud / {len(y_test)} total ({100*y_test.sum()/len(y_test):.2f}%)")

# ============================================
# 12. VISUALIZATIONS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC ROC comparison
ax = axes[0, 0]
ax.bar(comparison_df['Model'], comparison_df['Test AUC ROC'], 
    color=['#2ecc71'],
    alpha=0.8, edgecolor='black')
ax.set_ylabel('Test AUC ROC', fontsize=12)
ax.set_title('Test AUC ROC Comparison', fontsize=14, fontweight='bold')
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random baseline')
ax.legend()

# Average Precision comparison
ax = axes[0, 1]
ax.bar(comparison_df['Model'], comparison_df['Test AUPRC'], 
    color=['#2ecc71'],
    alpha=0.8, edgecolor='black')
ax.set_ylabel('Test AUPRC', fontsize=12)
ax.set_title('Test AUPRC Comparison', fontsize=14, fontweight='bold')
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Decision tree depth analysis - simplified
ax = axes[1, 0]
try:
    if len(results_dt) > 0:
        dt_depths = []
        dt_val_aucs = []
        for i, row in results_dt.iterrows():
            params = row['Parameters']
            if 'clf__max_depth' in params:
                dt_depths.append(params['clf__max_depth'])
                dt_val_aucs.append(row['Roc Auc Val'])
        
        if dt_depths:
            sorted_pairs = sorted(zip(dt_depths, dt_val_aucs))
            dt_depths, dt_val_aucs = zip(*sorted_pairs)
            ax.plot(dt_depths, dt_val_aucs, 'o-', 
                    label='Validation', linewidth=2, markersize=8, color='#3498db')
            ax.scatter([dt_depths[np.argmax(dt_val_aucs)]], [max(dt_val_aucs)], 
                    color='#2ecc71', s=200, marker='*', label='Best', zorder=5)
            ax.set_xlabel('Tree Depth', fontsize=12)
            ax.set_ylabel('AUC ROC', fontsize=12)
            ax.set_title('Decision Tree: AUC ROC vs Depth', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
except Exception as e:
    print(f"Warning: Could not plot DT analysis: {e}")
    ax.text(0.5, 0.5, 'Could not plot DT analysis', ha='center', va='center')

# Class distribution
ax = axes[1, 1]
sets = ['Train', 'Val', 'Test']
fraud_counts = [y_train.sum(), y_val.sum(), y_test.sum()]
legit_counts = [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()]

x_pos = np.arange(len(sets))
width = 0.35

ax.bar(x_pos - width/2, fraud_counts, width, label='Fraud', alpha=0.8, color='#e74c3c')
ax.bar(x_pos + width/2, legit_counts, width, label='Legitimate', alpha=0.8, color='#2ecc71')
ax.set_xlabel('Dataset Split', fontsize=12)
ax.set_ylabel('Number of Transactions', fontsize=12)
ax.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(sets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Model selection with preprocessed data completed!")