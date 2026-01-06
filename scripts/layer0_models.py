import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================
# BEST MODEL DEFINITIONS
# ============================================

class BestEnsembleModels:
    @staticmethod
    def get_balanced_random_forest():
        """
        Best Balanced Random Forest
        - Sampling strategy: 0.5
        - Performance: AUC ROC=0.983, AP=0.753, BA=0.893
        """
        model = BalancedRandomForestClassifier(
            max_depth=5,
            n_estimators=50,
            sampling_strategy=0.5,
            random_state=0,
            n_jobs=-1
        )
        return model
    
    @staticmethod
    def get_weighted_xgboost():
        """
        Best Weighted XGBoost
        - Scale pos weight: 50
        - Performance: AUC ROC=0.981, AP=0.757, BA=0.865
        """
        model = xgboost.XGBClassifier(
            learning_rate=0.3,
            max_depth=6,
            n_estimators=50,
            scale_pos_weight=50,
            random_state=0,
            n_jobs=-1,
            eval_metric='logloss'
        )
        return model
    
    @staticmethod
    def get_weighted_logistic_regression():
        """
        Weighted Logistic Regression
        - Class weight: balanced
        - Performance: AUC ROC=0.977, AP=0.690, BA=0.909
        """
        model = LogisticRegression(
            C=0.1,
            random_state=0,
            max_iter=1000,
            class_weight='balanced'
        )
        return model
    
    @staticmethod
    def create_pipeline(model, scale=True):
        """Create sklearn pipeline with optional scaling"""
        if scale:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
        else:
            return Pipeline([('clf', model)])
    
    @staticmethod
    def get_all_models():
        """Get all three best models as a dictionary"""
        return {
            'Balanced Random Forest': BestEnsembleModels.get_balanced_random_forest(),
            'Weighted XGBoost': BestEnsembleModels.get_weighted_xgboost(),
            'Weighted Logistic Regression': BestEnsembleModels.get_weighted_logistic_regression()
        }

# ============================================
# MODEL PERSISTENCE
# ============================================

class ModelPersistence:
    """Save and load trained models"""
    
    @staticmethod
    def save_model(model, filepath):
        """Save trained model to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_all_models(models_dict, output_dir='saved_models'):
        """Save all models to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        for name, model in models_dict.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = output_dir / filename
            ModelPersistence.save_model(model, filepath)
            saved_paths[name] = str(filepath)
        
        return saved_paths

# ============================================
# EVALUATION UTILITIES
# ============================================

def load_data(processed_dir):
    """Load preprocessed data"""
    data_dir = Path(processed_dir)
    X_train = np.load(data_dir / "X_train.npz")['data']
    X_val = np.load(data_dir / "X_val.npz")['data']
    X_test = np.load(data_dir / "X_test.npz")['data']
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return {
        'AUC ROC': metrics.roc_auc_score(y_test, y_pred_proba),
        'Average Precision': metrics.average_precision_score(y_test, y_pred_proba),
    }

def print_metrics(model_name, metrics_dict):
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")
    for metric_name, value in metrics_dict.items():
        print(f"{metric_name:.<25} {value:.4f}")
    print(f"{'='*80}")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("="*80)
    print("BEST ENSEMBLE MODELS - TRAINING AND EVALUATION")
    print("="*80)
    
    # Step 1: Load data
    print("\n[1/5] Loading preprocessed data...")
    processed_dir = "data_processed"
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(processed_dir)
    
    # Step 2: Get model definitions
    print("\n[2/5] Initializing best ensemble models...")
    models = BestEnsembleModels.get_all_models()
    
    # Step 3: Train and evaluate each model
    print("\n[3/5] Training and evaluating models...")
    trained_models = {}
    results = []
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Create pipeline with scaling
        pipeline = BestEnsembleModels.create_pipeline(model, scale=True)
        
        # Train
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        
        # Evaluate on test set
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        results.append({
            'Model': model_name,
            **{k: f"{v:.4f}" for k, v in test_metrics.items()}
        })
    
    # Step 4: Save trained models
    print("\n[4/5] Saving trained models to disk...")
    saved_paths = ModelPersistence.save_all_models(trained_models, output_dir='saved_models')
    
    # HOW TO Load model
    # loaded_model = ModelPersistence.load_model(saved_paths['Weighted XGBoost'])
    
    
    # Display results table
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL MODELS")
    print("="*80 + "\n")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ Models saved to: saved_models/")
    print("="*80)
    
    return trained_models, results_df


# ============================================
# MODEL INFORMATION
# ============================================

def print_model_info():
    """Print detailed information about each best model"""
    
    print("\n" + "="*80)
    print("BEST ENSEMBLE MODELS - CONFIGURATION DETAILS")
    print("="*80 + "\n")
    
    # Balanced Random Forest
    print("1. BALANCED RANDOM FOREST")
    print("-" * 80)
    print("   Parameters:")
    print("     - max_depth: 5")
    print("     - n_estimators: 50")
    print("     - sampling_strategy: 0.5")
    print("     - random_state: 0")
    print()
    
    # Weighted XGBoost
    print("2. WEIGHTED XGBOOST")
    print("-" * 80)
    print("   Parameters:")
    print("     - learning_rate: 0.3")
    print("     - max_depth: 6")
    print("     - n_estimators: 50")
    print("     - scale_pos_weight: 50")
    print("     - random_state: 0")
    print()
    
    # Weighted Logistic Regression
    print("3. WEIGHTED LOGISTIC REGRESSION")
    print("-" * 80)
    print("   Parameters:")
    print("     - C: 0.1")
    print("     - class_weight: 'balanced'")
    print("     - max_iter: 1000")
    print("     - random_state: 0")
    print()
    print("="*80 + "\n")

# ============================================
# RUN MAIN
# ============================================

if __name__ == "__main__":
    # Run main training pipeline
    trained_models, results = main()
    
    # Print model information
    #print_model_info()