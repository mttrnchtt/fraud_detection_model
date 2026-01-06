from typing import Any
from sklearn.neural_network import MLPClassifier


def create_mlp_model(config: dict) -> MLPClassifier:
    """
    Create and configure MLPClassifier from config.
    
    Args:
        config: Dictionary containing model, optimizer, and training parameters
        
    Returns:
        Configured MLPClassifier ready for training
    """
    # Extract config sections
    model_config = config['model']
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    # Handle early stopping configuration
    early_stopping_enabled = training_config['early_stopping']['monitor'] is not None
    early_stopping_patience = training_config['early_stopping']['patience'] if early_stopping_enabled else 10
    
    # Map config to MLPClassifier parameters
    model = MLPClassifier(
        # Model architecture
        hidden_layer_sizes=tuple(model_config['hidden_dims']),
        activation=model_config['activation'],  # 'relu' for hidden layers
        # Note: output_activation is always sigmoid for binary classification in MLPClassifier
        
        # Optimizer settings
        solver=optimizer_config['name'],  # 'adam'
        learning_rate_init=optimizer_config['lr'],  # 0.001
        alpha=optimizer_config['weight_decay'],  # L2 regularization instead of dropout
        # Note: betas [0.9, 0.999] are handled internally by Adam solver
        
        # Training settings
        max_iter=training_config['epochs'],  # 50
        batch_size=training_config['batch_size'],  # 1024
        shuffle=False,  # Disable shuffling to preserve chronological order
        early_stopping=False,  # We'll handle early stopping manually with our val set
        validation_fraction=0,  # Don't use built-in validation - we have separate val set
        n_iter_no_change=early_stopping_patience,
        
        # Learning rate scheduling (built into MLPClassifier)
        learning_rate='adaptive',  # Reduces learning rate when loss plateaus
        
        # Class balancing - MLPClassifier doesn't have class_weight, so we'll handle this in training
        # Note: pos_weight=518.7 will be handled via sample weights during fit()
        
        # Other settings
        random_state=config['experiment']['seed'],
        verbose=False,
        warm_start=False,
        
        # Note: use_batchnorm and dropout are not supported in MLPClassifier
        # They use L2 regularization (alpha) instead
    )
    
    return model


def get_model_info(model: MLPClassifier) -> dict:
    """
    Extract model architecture information for logging.
    
    Args:
        model: Trained MLPClassifier
        
    Returns:
        Dictionary with model information
    """
    return {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'solver': model.solver,
        'learning_rate_init': model.learning_rate_init,
        'alpha': model.alpha,
        'max_iter': model.max_iter,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'classes_': model.classes_.tolist() if hasattr(model, 'classes_') else None,
    }