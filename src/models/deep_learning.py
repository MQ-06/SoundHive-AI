"""Deep learning models for beehive health monitoring"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def build_dnn(
    input_dim: int,
    num_classes: int,
    hidden_layers: list = [256, 128, 64],
    dropout_rate: float = 0.3,
    activation: str = 'relu',
    optimizer: str = 'adam'
) -> Sequential:
    """
    Build a Fully Connected Deep Neural Network.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_layers: List of neurons per hidden layer
        dropout_rate: Dropout rate for regularization
        activation: Activation function for hidden layers
        optimizer: Optimizer for training
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Input and first hidden layer
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss_function = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss_function = 'categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    
    return model


def train_dnn(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: int = 1
) -> Tuple[Sequential, dict]:
    """
    Train a Deep Neural Network.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels (can be integer or one-hot encoded)
        X_test: Test features
        y_test: Test labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        verbose: Verbosity level
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Ensure labels are properly formatted
    if len(y_train.shape) == 1:
        # Integer labels - convert to one-hot if multi-class
        num_classes = len(np.unique(y_train))
        if num_classes > 2:
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test_cat = to_categorical(y_test, num_classes=num_classes)
        else:
            y_test_cat = y_test
    else:
        y_test_cat = y_test
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    return model, history.history

