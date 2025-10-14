"""
Model training pipeline for air quality forecasting.
Implements multiple ML models including Random Forest, XGBoost, LSTM, and CNN-LSTM.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

# Import deep learning libraries with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    print("TensorFlow not available. LSTM and CNN-LSTM models will be disabled.")

# Define a placeholder class for type hints when TensorFlow is not available
class _DummyKerasModel:
    pass

if TF_AVAILABLE:
    KerasModel = keras.Model
else:
    KerasModel = _DummyKerasModel

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Model training class for air quality forecasting.
    Supports multiple model types including traditional ML and deep learning models.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize ModelTrainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize model storage
        self.models = {}
        self.model_performance = {}
        self.scalers = {}
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.get('training', {}).get('random_state', 42))
        if TF_AVAILABLE:
            tf.random.set_seed(self.config.get('training', {}).get('random_state', 42))
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.get('logging', {}).get('level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                self.config.get('logging', {}).get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_training_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                            target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with target values
            target_column: Name of the target column to predict
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Remove non-feature columns
        exclude_cols = ['datetime', 'site_id', 'year', 'month', 'day', 'hour', 'date']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Prepare features
        X = features_df[feature_cols].copy()
        
        # Filter out object/string columns and keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if object_cols:
            self.logger.warning(f"Excluding {len(object_cols)} non-numeric columns: {object_cols[:5]}...")
            X = X[numeric_cols]
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Ensure all data is numeric
        X = X.astype(float)
        
        # Prepare targets
        if target_column in targets_df.columns:
            y = targets_df[target_column].values
        else:
            raise ValueError(f"Target column '{target_column}' not found in targets DataFrame")
        
        # Remove rows with missing targets
        mask = ~np.isnan(y)
        X = X.loc[mask]
        y = y[mask]
        
        # Final check - ensure X is all numeric
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            self.logger.error(f"Non-numeric columns found in final features: {X.dtypes[~X.dtypes.apply(lambda x: np.issubdtype(x, np.number))]}")
            # Convert any remaining non-numeric columns
            for col in X.columns:
                if not np.issubdtype(X[col].dtype, np.number):
                    X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)  # Fill any NaNs introduced by conversion
        
        final_feature_cols = X.columns.tolist()
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X.values, y, final_feature_cols
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = None, test_split: float = None) -> Dict[str, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Features array
            y: Target array
            validation_split: Proportion for validation set
            test_split: Proportion for test set
            
        Returns:
            Dictionary containing train/val/test splits
        """
        if validation_split is None:
            validation_split = self.config.get('training', {}).get('validation_split', 0.2)
        if test_split is None:
            test_split = self.config.get('training', {}).get('test_split', 0.2)
        
        random_state = self.config.get('training', {}).get('random_state', 42)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
        
        # Second split: separate train and validation from temp
        val_size = validation_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
        
        self.logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None,
                           model_name: str = "random_forest") -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name to store the model
            
        Returns:
            Trained Random Forest model
        """
        rf_config = self.config.get('models', {}).get('random_forest', {})
        
        model = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Train the model
        self.logger.info("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        # Store model and evaluate
        self.models[model_name] = model
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.model_performance[model_name] = {'val_r2': val_score}
            self.logger.info(f"Random Forest validation R²: {val_score:.4f}")
        
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None,
                     model_name: str = "xgboost") -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name to store the model
            
        Returns:
            Trained XGBoost model
        """
        xgb_config = self.config.get('models', {}).get('xgboost', {})
        
        model = xgb.XGBRegressor(
            n_estimators=xgb_config.get('n_estimators', 200),
            max_depth=xgb_config.get('max_depth', 8),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=xgb_config.get('random_state', 42)
        )
        
        # Prepare evaluation set for early stopping
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train the model
        self.logger.info("Training XGBoost model...")
        # Note: early_stopping_rounds parameter name changed in newer XGBoost versions
        try:
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        except TypeError:
            # Try with newer parameter name
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        # Store model and evaluate
        self.models[model_name] = model
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            self.model_performance[model_name] = {'val_r2': val_score}
            self.logger.info(f"XGBoost validation R²: {val_score:.4f}")
        
        return model
    
    def create_lstm_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> KerasModel:
        """
        Create LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_dim: Number of output dimensions
            
        Returns:
            Compiled LSTM model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        lstm_config = self.config.get('models', {}).get('lstm', {})
        
        model = keras.Sequential([
            layers.LSTM(
                lstm_config.get('hidden_size', 64),
                return_sequences=True,
                input_shape=input_shape
            ),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            
            layers.LSTM(
                lstm_config.get('hidden_size', 64),
                return_sequences=False
            ),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lstm_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequence_data(self, X: np.ndarray, y: np.ndarray, 
                            sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for sequence models (LSTM, CNN-LSTM).
        
        Args:
            X: Features array
            y: Targets array
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   model_name: str = "lstm") -> Optional[KerasModel]:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name to store the model
            
        Returns:
            Trained LSTM model or None if TensorFlow not available
        """
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available. Skipping LSTM training.")
            return None
        
        lstm_config = self.config.get('models', {}).get('lstm', {})
        sequence_length = lstm_config.get('sequence_length', 24)
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self.prepare_sequence_data(X_train, y_train, sequence_length)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequence_data(X_val, y_val, sequence_length)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Create and train model
        input_shape = (sequence_length, X_train.shape[1])
        model = self.create_lstm_model(input_shape)
        
        self.logger.info("Training LSTM model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=lstm_config.get('early_stopping_patience', 10),
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=lstm_config.get('epochs', 100),
            batch_size=lstm_config.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model
        self.models[model_name] = model
        
        # Store training history
        if validation_data is not None:
            val_loss = min(history.history['val_loss'])
            self.model_performance[model_name] = {'val_loss': val_loss}
            self.logger.info(f"LSTM validation loss: {val_loss:.4f}")
        
        return model
    
    def create_cnn_lstm_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> KerasModel:
        """
        Create CNN-LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            output_dim: Number of output dimensions
            
        Returns:
            Compiled CNN-LSTM model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN-LSTM models")
        
        cnn_lstm_config = self.config.get('models', {}).get('cnn_lstm', {})
        
        model = keras.Sequential([
            # Reshape for CNN
            layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
            
            # CNN layers
            layers.Conv2D(
                filters=cnn_lstm_config.get('cnn_filters', [32, 64])[0],
                kernel_size=(cnn_lstm_config.get('cnn_kernel_size', 3), 1),
                activation='relu'
            ),
            layers.Conv2D(
                filters=cnn_lstm_config.get('cnn_filters', [32, 64])[1] if len(cnn_lstm_config.get('cnn_filters', [32, 64])) > 1 else 64,
                kernel_size=(cnn_lstm_config.get('cnn_kernel_size', 3), 1),
                activation='relu'
            ),
            layers.MaxPooling2D(pool_size=(2, 1)),
            layers.Dropout(cnn_lstm_config.get('dropout', 0.2)),
            
            # Reshape for LSTM
            layers.Reshape((-1, cnn_lstm_config.get('cnn_filters', [32, 64])[-1])),
            
            # LSTM layers
            layers.LSTM(
                cnn_lstm_config.get('lstm_hidden_size', 50),
                return_sequences=False
            ),
            layers.Dropout(cnn_lstm_config.get('dropout', 0.2)),
            
            # Dense layers
            layers.Dense(25, activation='relu'),
            layers.Dense(output_dim)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cnn_lstm_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_cnn_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       model_name: str = "cnn_lstm") -> Optional[KerasModel]:
        """
        Train CNN-LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name to store the model
            
        Returns:
            Trained CNN-LSTM model or None if TensorFlow not available
        """
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available. Skipping CNN-LSTM training.")
            return None
        
        cnn_lstm_config = self.config.get('models', {}).get('cnn_lstm', {})
        sequence_length = cnn_lstm_config.get('sequence_length', 24)
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self.prepare_sequence_data(X_train, y_train, sequence_length)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequence_data(X_val, y_val, sequence_length)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Create and train model
        input_shape = (sequence_length, X_train.shape[1])
        model = self.create_cnn_lstm_model(input_shape)
        
        self.logger.info("Training CNN-LSTM model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=cnn_lstm_config.get('early_stopping_patience', 10),
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=cnn_lstm_config.get('epochs', 100),
            batch_size=cnn_lstm_config.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model
        self.models[model_name] = model
        
        # Store training history
        if validation_data is not None:
            val_loss = min(history.history['val_loss'])
            self.model_performance[model_name] = {'val_loss': val_loss}
            self.logger.info(f"CNN-LSTM validation loss: {val_loss:.4f}")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Training all available models...")
        
        # Train traditional ML models
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train deep learning models if TensorFlow is available
        if TF_AVAILABLE:
            self.train_lstm(X_train, y_train, X_val, y_val)
            self.train_cnn_lstm(X_train, y_train, X_val, y_val)
        else:
            self.logger.warning("TensorFlow not available. Skipping deep learning models.")
        
        self.logger.info(f"Training completed for {len(self.models)} models")
        return self.models
    
    def save_model(self, model_name: str, save_path: str = None):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model to save
            save_path: Path to save the model (optional)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if save_path is None:
            models_dir = Path(self.config.get('paths', {}).get('models', 'models/'))
            models_dir.mkdir(exist_ok=True)
            save_path = models_dir / f"{model_name}_model"
        
        model = self.models[model_name]
        
        # Save based on model type
        if hasattr(model, 'save'):  # TensorFlow/Keras model
            model.save(f"{save_path}.h5")
        else:  # Scikit-learn or XGBoost model
            joblib.dump(model, f"{save_path}.pkl")
        
        self.logger.info(f"Model '{model_name}' saved to {save_path}")
    
    def load_model(self, model_name: str, load_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            load_path: Path to load the model from
        """
        if load_path.endswith('.h5') and TF_AVAILABLE:
            # Load TensorFlow/Keras model
            model = keras.models.load_model(load_path)
        elif load_path.endswith('.pkl'):
            # Load Scikit-learn or XGBoost model
            model = joblib.load(load_path)
        else:
            raise ValueError(f"Unsupported model file format: {load_path}")
        
        self.models[model_name] = model
        self.logger.info(f"Model '{model_name}' loaded from {load_path}")
    
    def get_model_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all trained models.
        
        Returns:
            Dictionary with model summaries
        """
        summary = {}
        
        for name, model in self.models.items():
            model_info = {
                'type': type(model).__name__,
                'parameters': getattr(model, 'get_params', lambda: {})(),
            }
            
            if name in self.model_performance:
                model_info['performance'] = self.model_performance[name]
            
            summary[name] = model_info
        
        return summary
