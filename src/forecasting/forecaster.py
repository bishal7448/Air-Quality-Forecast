"""
Forecasting pipeline for automated air quality predictions.
Provides 24-48 hour forecasts with hourly intervals and confidence intervals.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
from datetime import datetime, timedelta
import joblib
import warnings

# Import libraries with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')

class AirQualityForecaster:
    """
    Air quality forecaster class that provides automated short-term predictions.
    Supports ensemble forecasting and confidence interval estimation.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize AirQualityForecaster with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize storage for models and scalers
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
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
    
    def load_models(self, models_dict: Dict[str, Any], scalers_dict: Dict[str, Any] = None,
                   feature_names: List[str] = None):
        """
        Load trained models and scalers for forecasting.
        
        Args:
            models_dict: Dictionary of trained models
            scalers_dict: Dictionary of scalers (optional)
            feature_names: List of feature names
        """
        self.models = models_dict
        if scalers_dict:
            self.scalers = scalers_dict
        if feature_names:
            self.feature_names = feature_names
        
        self.is_fitted = True
        self.logger.info(f"Loaded {len(self.models)} models for forecasting")
    
    def load_models_from_disk(self, model_paths: Dict[str, str], 
                             scaler_path: str = None,
                             feature_names_path: str = None):
        """
        Load models from disk.
        
        Args:
            model_paths: Dictionary mapping model names to file paths
            scaler_path: Path to scaler file (optional)
            feature_names_path: Path to feature names file (optional)
        """
        self.models = {}
        
        for model_name, path in model_paths.items():
            try:
                if path.endswith('.h5') and TF_AVAILABLE:
                    # Load TensorFlow/Keras model
                    model = keras.models.load_model(path)
                elif path.endswith('.pkl'):
                    # Load Scikit-learn or XGBoost model
                    model = joblib.load(path)
                else:
                    self.logger.warning(f"Unsupported model format for {model_name}: {path}")
                    continue
                
                self.models[model_name] = model
                self.logger.info(f"Loaded model {model_name} from {path}")
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
        
        # Load scaler if provided
        if scaler_path and Path(scaler_path).exists():
            try:
                self.scalers = joblib.load(scaler_path)
                self.logger.info(f"Loaded scalers from {scaler_path}")
            except Exception as e:
                self.logger.error(f"Error loading scalers: {e}")
        
        # Load feature names if provided
        if feature_names_path and Path(feature_names_path).exists():
            try:
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                self.logger.info(f"Loaded {len(self.feature_names)} feature names")
            except Exception as e:
                self.logger.error(f"Error loading feature names: {e}")
        
        self.is_fitted = len(self.models) > 0
    
    def _prepare_sequence_data(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Prepare sequence data for LSTM models."""
        if len(X) < sequence_length:
            # Pad with the last available values if not enough history
            padding = np.repeat(X[-1:], sequence_length - len(X), axis=0)
            X_padded = np.vstack([padding, X])
        else:
            X_padded = X
        
        # Create sequences
        X_seq = []
        for i in range(sequence_length, len(X_padded) + 1):
            X_seq.append(X_padded[i-sequence_length:i])
        
        return np.array(X_seq)
    
    def predict_single_model(self, model, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Make prediction using a single model.
        
        Args:
            model: Trained model
            X: Input features
            model_name: Name of the model
            
        Returns:
            Predictions array
        """
        try:
            # Check if model requires sequence input
            is_sequence = 'lstm' in model_name.lower()
            
            if is_sequence:
                # Get sequence length from config
                if 'lstm' in model_name.lower():
                    config_key = 'lstm' if model_name == 'lstm' else 'cnn_lstm'
                    sequence_length = self.config.get('models', {}).get(config_key, {}).get('sequence_length', 24)
                else:
                    sequence_length = 24
                
                X_seq = self._prepare_sequence_data(X, sequence_length)
                predictions = model.predict(X_seq)
                
                # Handle different output shapes
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                
            else:
                predictions = model.predict(X)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making prediction with {model_name}: {e}")
            return np.array([])
    
    def ensemble_predict(self, X: np.ndarray, ensemble_method: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            X: Input features
            ensemble_method: Method for combining predictions ("mean", "median", "weighted_mean")
            
        Returns:
            Tuple of (predictions, std_deviations)
        """
        if not self.is_fitted:
            raise ValueError("No models loaded for forecasting")
        
        all_predictions = []
        
        for model_name, model in self.models.items():
            predictions = self.predict_single_model(model, X, model_name)
            
            if len(predictions) > 0:
                # Ensure predictions have the same length
                if len(all_predictions) > 0 and len(predictions) != len(all_predictions[0]):
                    min_len = min(len(predictions), len(all_predictions[0]))
                    predictions = predictions[:min_len]
                    all_predictions = [pred[:min_len] for pred in all_predictions]
                
                all_predictions.append(predictions)
        
        if not all_predictions:
            raise ValueError("No valid predictions from any model")
        
        all_predictions = np.array(all_predictions)
        
        # Combine predictions
        if ensemble_method == "mean":
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif ensemble_method == "median":
            ensemble_pred = np.median(all_predictions, axis=0)
        elif ensemble_method == "weighted_mean":
            # Simple equal weighting for now
            ensemble_pred = np.mean(all_predictions, axis=0)
        else:
            self.logger.warning(f"Unknown ensemble method: {ensemble_method}, using mean")
            ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Calculate standard deviation across models
        ensemble_std = np.std(all_predictions, axis=0)
        
        return ensemble_pred, ensemble_std
    
    def forecast_single_step(self, X: np.ndarray, 
                           ensemble_method: str = "mean") -> Dict[str, np.ndarray]:
        """
        Make single-step forecast.
        
        Args:
            X: Input features for prediction
            ensemble_method: Method for ensemble prediction
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        # Make ensemble prediction
        predictions, std_devs = self.ensemble_predict(X, ensemble_method)
        
        # Calculate confidence intervals
        confidence_intervals = self.config.get('forecasting', {}).get('confidence_intervals', [0.05, 0.95])
        lower_ci = np.percentile([predictions - 1.96 * std_devs, predictions], 
                                confidence_intervals[0] * 100, axis=0)
        upper_ci = np.percentile([predictions + 1.96 * std_devs, predictions], 
                                confidence_intervals[1] * 100, axis=0)
        
        return {
            'predictions': predictions,
            'std_deviations': std_devs,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        }
    
    def forecast_multi_step(self, X_init: np.ndarray, forecast_hours: int = 24,
                          ensemble_method: str = "mean", 
                          recursive: bool = True) -> Dict[str, np.ndarray]:
        """
        Make multi-step forecast.
        
        Args:
            X_init: Initial input features
            forecast_hours: Number of hours to forecast
            ensemble_method: Method for ensemble prediction
            recursive: Whether to use recursive forecasting
            
        Returns:
            Dictionary with forecast results
        """
        if not recursive:
            # Direct multi-step prediction (if supported by models)
            return self.forecast_single_step(X_init, ensemble_method)
        
        # Recursive forecasting
        all_predictions = []
        all_std_devs = []
        all_lower_ci = []
        all_upper_ci = []
        
        X_current = X_init.copy()
        
        for hour in range(forecast_hours):
            # Make prediction for current step
            forecast_result = self.forecast_single_step(X_current[-1:], ensemble_method)
            
            all_predictions.extend(forecast_result['predictions'])
            all_std_devs.extend(forecast_result['std_deviations'])
            all_lower_ci.extend(forecast_result['lower_ci'])
            all_upper_ci.extend(forecast_result['upper_ci'])
            
            # Update features for next step (simplified approach)
            # In practice, you'd need to properly update meteorological forecasts
            if len(X_current) >= 24:  # Keep sliding window
                X_current = X_current[1:]
            
            # Create next step features (this is a simplified approach)
            next_features = X_current[-1].copy()
            # Update with prediction (assuming O3 and NO2 forecasts are in the features)
            # This would need to be adapted based on actual feature structure
            
            X_current = np.vstack([X_current, next_features.reshape(1, -1)])
        
        return {
            'predictions': np.array(all_predictions),
            'std_deviations': np.array(all_std_devs),
            'lower_ci': np.array(all_lower_ci),
            'upper_ci': np.array(all_upper_ci)
        }
    
    def create_forecast_dataframe(self, forecast_result: Dict[str, np.ndarray],
                                start_datetime: datetime, target_name: str = "Target",
                                forecast_intervals: int = 1) -> pd.DataFrame:
        """
        Create a formatted DataFrame from forecast results.
        
        Args:
            forecast_result: Dictionary with forecast results
            start_datetime: Starting datetime for the forecast
            target_name: Name of the target variable
            forecast_intervals: Interval between forecasts in hours
            
        Returns:
            DataFrame with formatted forecast results
        """
        num_predictions = len(forecast_result['predictions'])
        
        # Create datetime index
        datetime_index = [start_datetime + timedelta(hours=int(i * forecast_intervals)) 
                         for i in range(num_predictions)]
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'datetime': datetime_index,
            f'{target_name}_prediction': forecast_result['predictions'],
            f'{target_name}_std': forecast_result['std_deviations'],
            f'{target_name}_lower_ci': forecast_result['lower_ci'],
            f'{target_name}_upper_ci': forecast_result['upper_ci']
        })
        
        return forecast_df
    
    def forecast_for_site(self, site_data: pd.DataFrame, site_id: int,
                         target_columns: List[str] = ["O3_target", "NO2_target"],
                         forecast_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for a specific site.
        
        Args:
            site_data: DataFrame with site data (features)
            site_id: Site identifier
            target_columns: List of target variables to forecast
            forecast_hours: Number of hours to forecast
            
        Returns:
            Dictionary mapping target names to forecast DataFrames
        """
        if not self.is_fitted:
            raise ValueError("No models loaded for forecasting")
        
        # Prepare features (remove non-feature columns)
        exclude_cols = ['datetime', 'site_id', 'year', 'month', 'day', 'hour'] + target_columns
        feature_cols = [col for col in site_data.columns if col not in exclude_cols]
        
        if self.feature_names:
            # Use stored feature names if available
            available_features = [col for col in self.feature_names if col in site_data.columns]
            X = site_data[available_features].values
        else:
            X = site_data[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features if scaler is available
        if 'features' in self.scalers:
            X = self.scalers['features'].transform(X)
        
        forecasts = {}
        ensemble_method = self.config.get('forecasting', {}).get('ensemble_methods', ["mean"])[0]
        
        for target_name in target_columns:
            self.logger.info(f"Generating forecast for {target_name} at site {site_id}")
            
            try:
                # Generate forecast
                forecast_result = self.forecast_multi_step(
                    X, forecast_hours=forecast_hours, 
                    ensemble_method=ensemble_method
                )
                
                # Create forecast DataFrame
                start_datetime = datetime.now()  # In practice, use actual forecast start time
                forecast_df = self.create_forecast_dataframe(
                    forecast_result, start_datetime, target_name
                )
                
                # Add site information
                forecast_df['site_id'] = site_id
                
                forecasts[target_name] = forecast_df
                
            except Exception as e:
                self.logger.error(f"Error forecasting {target_name} for site {site_id}: {e}")
        
        return forecasts
    
    def forecast_all_sites(self, data_dict: Dict[int, pd.DataFrame],
                          target_columns: List[str] = ["O3_target", "NO2_target"],
                          forecast_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all sites.
        
        Args:
            data_dict: Dictionary mapping site IDs to DataFrames
            target_columns: List of target variables to forecast
            forecast_hours: Number of hours to forecast
            
        Returns:
            Dictionary mapping target names to combined forecast DataFrames
        """
        all_forecasts = {target: [] for target in target_columns}
        
        for site_id, site_data in data_dict.items():
            self.logger.info(f"Processing site {site_id}")
            
            try:
                site_forecasts = self.forecast_for_site(
                    site_data, site_id, target_columns, forecast_hours
                )
                
                # Collect forecasts for each target
                for target_name, forecast_df in site_forecasts.items():
                    all_forecasts[target_name].append(forecast_df)
                    
            except Exception as e:
                self.logger.error(f"Error processing site {site_id}: {e}")
        
        # Combine forecasts for each target
        combined_forecasts = {}
        for target_name, forecast_list in all_forecasts.items():
            if forecast_list:
                combined_forecasts[target_name] = pd.concat(forecast_list, ignore_index=True)
                self.logger.info(f"Combined forecasts for {target_name}: {len(combined_forecasts[target_name])} records")
        
        return combined_forecasts
    
    def save_forecasts(self, forecasts: Dict[str, pd.DataFrame], 
                      output_dir: str = "results/forecasts/",
                      timestamp: str = None):
        """
        Save forecast results to files.
        
        Args:
            forecasts: Dictionary of forecast DataFrames
            output_dir: Directory to save forecasts
            timestamp: Timestamp string for filename (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target_name, forecast_df in forecasts.items():
            filename = f"{target_name}_forecast_{timestamp}.csv"
            filepath = output_path / filename
            
            try:
                forecast_df.to_csv(filepath, index=False)
                self.logger.info(f"Saved forecast for {target_name} to {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving forecast for {target_name}: {e}")
    
    def create_forecast_summary(self, forecasts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a summary of forecast results.
        
        Args:
            forecasts: Dictionary of forecast DataFrames
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for target_name, forecast_df in forecasts.items():
            if not forecast_df.empty:
                pred_col = f"{target_name}_prediction"
                std_col = f"{target_name}_std"
                
                summary = {
                    'target': target_name,
                    'num_sites': forecast_df['site_id'].nunique() if 'site_id' in forecast_df.columns else 1,
                    'num_forecasts': len(forecast_df),
                    'mean_prediction': forecast_df[pred_col].mean(),
                    'std_prediction': forecast_df[pred_col].std(),
                    'mean_uncertainty': forecast_df[std_col].mean() if std_col in forecast_df.columns else 0,
                    'min_prediction': forecast_df[pred_col].min(),
                    'max_prediction': forecast_df[pred_col].max()
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def get_forecast_status(self) -> Dict[str, Any]:
        """
        Get current status of the forecaster.
        
        Returns:
            Dictionary with forecaster status
        """
        status = {
            'is_fitted': self.is_fitted,
            'num_models': len(self.models),
            'available_models': list(self.models.keys()),
            'has_scalers': len(self.scalers) > 0,
            'num_features': len(self.feature_names),
            'config_loaded': bool(self.config)
        }
        
        return status
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for forecasting.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['datetime']
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for minimum data requirements
        if len(data) < 24:
            self.logger.warning("Input data has less than 24 hours of data")
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            self.logger.warning(f"Columns with >50% missing data: {high_missing.index.tolist()}")
        
        return True
