"""
Utility functions and helpers for the air quality forecasting project.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
from datetime import datetime, timedelta
import joblib
import warnings

warnings.filterwarnings('ignore')

class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration to YAML file."""
        try:
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    @staticmethod
    def update_config(config_path: str, updates: Dict):
        """Update configuration with new values."""
        config = ConfigManager.load_config(config_path)
        config.update(updates)
        ConfigManager.save_config(config, config_path)

class LoggingUtils:
    """Logging utilities."""
    
    @staticmethod
    def setup_logger(name: str, level: str = "INFO", 
                    log_file: str = None, format_str: str = None) -> logging.Logger:
        """Setup a logger with specified configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

class DataUtils:
    """Data handling utilities."""
    
    @staticmethod
    def create_datetime_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
        """Create basic datetime features from datetime column."""
        df_temp = df.copy()
        df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
        
        df_temp['year'] = df_temp[datetime_col].dt.year
        df_temp['month'] = df_temp[datetime_col].dt.month
        df_temp['day'] = df_temp[datetime_col].dt.day
        df_temp['hour'] = df_temp[datetime_col].dt.hour
        df_temp['day_of_week'] = df_temp[datetime_col].dt.dayofweek
        df_temp['day_of_year'] = df_temp[datetime_col].dt.dayofyear
        
        return df_temp
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate',
                             numeric_only: bool = True) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        df_clean = df.copy()
        
        if numeric_only:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            target_cols = numeric_cols
        else:
            target_cols = df_clean.columns
        
        if strategy == 'interpolate':
            df_clean[target_cols] = df_clean[target_cols].interpolate(method='time')
        elif strategy == 'forward_fill':
            df_clean[target_cols] = df_clean[target_cols].fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_clean[target_cols] = df_clean[target_cols].fillna(method='bfill')
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=target_cols)
        elif strategy == 'mean':
            df_clean[target_cols] = df_clean[target_cols].fillna(df_clean[target_cols].mean())
        elif strategy == 'median':
            df_clean[target_cols] = df_clean[target_cols].fillna(df_clean[target_cols].median())
        
        return df_clean
    
    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.Series:
        """Detect outliers in data series."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from specified columns."""
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                outliers = DataUtils.detect_outliers(df_clean[col], method, threshold)
                df_clean = df_clean[~outliers]
        
        return df_clean
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], 
                           lags: List[int], group_col: str = None) -> pd.DataFrame:
        """Create lag features for specified columns."""
        df_lagged = df.copy()
        
        for col in columns:
            if col in df_lagged.columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    if group_col:
                        df_lagged[lag_col_name] = df_lagged.groupby(group_col)[col].shift(lag)
                    else:
                        df_lagged[lag_col_name] = df_lagged[col].shift(lag)
        
        return df_lagged
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str], 
                               windows: List[int], statistics: List[str] = ['mean'],
                               group_col: str = None) -> pd.DataFrame:
        """Create rolling window features."""
        df_rolling = df.copy()
        
        for col in columns:
            if col in df_rolling.columns:
                for window in windows:
                    for stat in statistics:
                        feature_name = f"{col}_rolling_{window}_{stat}"
                        
                        if group_col:
                            grouped = df_rolling.groupby(group_col)[col]
                        else:
                            grouped = df_rolling[col]
                        
                        if stat == 'mean':
                            if group_col:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).mean()
                            else:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).mean()
                        elif stat == 'std':
                            if group_col:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).std()
                            else:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).std()
                        elif stat == 'min':
                            if group_col:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).min()
                            else:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).min()
                        elif stat == 'max':
                            if group_col:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).max()
                            else:
                                df_rolling[feature_name] = grouped.rolling(window=window, min_periods=1).max()
        
        return df_rolling

class ModelUtils:
    """Model management utilities."""
    
    @staticmethod
    def save_model(model, filepath: str, model_type: str = 'auto'):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if model_type == 'auto':
            # Auto-detect model type
            if hasattr(model, 'save'):  # TensorFlow/Keras
                model.save(filepath)
            else:  # Scikit-learn/XGBoost
                joblib.dump(model, filepath)
        elif model_type == 'keras':
            model.save(filepath)
        elif model_type == 'sklearn':
            joblib.dump(model, filepath)
        elif model_type == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def load_model(filepath: str, model_type: str = 'auto'):
        """Load model from disk."""
        if model_type == 'auto':
            # Auto-detect based on file extension
            if filepath.endswith('.h5'):
                try:
                    import tensorflow as tf
                    return tf.keras.models.load_model(filepath)
                except ImportError:
                    raise ImportError("TensorFlow required for loading .h5 models")
            elif filepath.endswith('.pkl'):
                return joblib.load(filepath)
            else:
                # Try joblib first, then pickle
                try:
                    return joblib.load(filepath)
                except:
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
        elif model_type == 'keras':
            import tensorflow as tf
            return tf.keras.models.load_model(filepath)
        elif model_type == 'sklearn':
            return joblib.load(filepath)
        elif model_type == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model) -> Dict[str, Any]:
        """Get information about a model."""
        info = {
            'type': type(model).__name__,
            'module': type(model).__module__
        }
        
        # Add model-specific information
        if hasattr(model, 'get_params'):
            info['parameters'] = model.get_params()
        
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
        
        if hasattr(model, 'summary'):
            info['is_keras_model'] = True
        
        return info

class FileUtils:
    """File handling utilities."""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]):
        """Ensure directory exists."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """Save data to JSON file."""
        FileUtils.ensure_dir(Path(filepath).parent)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: str, **kwargs):
        """Save DataFrame to CSV file."""
        FileUtils.ensure_dir(Path(filepath).parent)
        df.to_csv(filepath, index=False, **kwargs)
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from CSV file."""
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(filepath)
    
    @staticmethod
    def get_file_modified_time(filepath: str) -> datetime:
        """Get file modification time."""
        return datetime.fromtimestamp(os.path.getmtime(filepath))
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern."""
        path = Path(directory)
        return [str(f) for f in path.glob(pattern) if f.is_file()]

class ValidationUtils:
    """Data validation utilities."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None,
                          min_rows: int = None, max_missing_pct: float = None) -> Dict[str, Any]:
        """Validate DataFrame and return validation results."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic info
        results['info']['shape'] = df.shape
        results['info']['columns'] = list(df.columns)
        results['info']['dtypes'] = df.dtypes.to_dict()
        
        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                results['valid'] = False
                results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check minimum rows
        if min_rows and len(df) < min_rows:
            results['valid'] = False
            results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # Check missing values
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        if max_missing_pct:
            high_missing = missing_counts[missing_counts / total_rows * 100 > max_missing_pct]
            if not high_missing.empty:
                results['warnings'].append(
                    f"Columns with >{max_missing_pct}% missing data: {high_missing.index.tolist()}"
                )
        
        results['info']['missing_counts'] = missing_counts.to_dict()
        results['info']['missing_percentages'] = (missing_counts / total_rows * 100).to_dict()
        
        return results
    
    @staticmethod
    def validate_model_input(X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Validate model input data."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Validate features
        results['info']['X_shape'] = X.shape
        results['info']['X_dtype'] = str(X.dtype)
        
        if np.isnan(X).any():
            results['warnings'].append("Features contain NaN values")
        
        if np.isinf(X).any():
            results['warnings'].append("Features contain infinite values")
        
        # Validate targets if provided
        if y is not None:
            results['info']['y_shape'] = y.shape
            results['info']['y_dtype'] = str(y.dtype)
            
            if len(X) != len(y):
                results['valid'] = False
                results['errors'].append(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
            
            if np.isnan(y).any():
                results['warnings'].append("Targets contain NaN values")
        
        return results

class PerformanceUtils:
    """Performance monitoring utilities."""
    
    @staticmethod
    def time_function(func):
        """Decorator to time function execution."""
        import time
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def monitor_memory_usage():
        """Monitor current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total'] = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        except ImportError:
            pass
        
        return info

class PlottingUtils:
    """Plotting utilities."""
    
    @staticmethod
    def setup_plot_style(style: str = 'seaborn-v0_8'):
        """Setup matplotlib plotting style."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use(style)
        sns.set_palette("husl")
    
    @staticmethod
    def save_plot(fig, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """Save matplotlib figure."""
        FileUtils.ensure_dir(Path(filepath).parent)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    
    @staticmethod
    def create_comparison_table(data: Dict[str, Any]) -> str:
        """Create a formatted comparison table."""
        if not data:
            return "No data to display"
        
        # Find the maximum width for each column
        keys = list(data.keys())
        if not keys:
            return "No data to display"
        
        # Assume all values are dictionaries with the same keys
        first_item = list(data.values())[0]
        if not isinstance(first_item, dict):
            return str(data)
        
        columns = list(first_item.keys())
        
        # Create header
        header = "| " + " | ".join([f"{col:>12}" for col in ["Item"] + columns]) + " |"
        separator = "|" + "|".join(["-" * 14 for _ in range(len(columns) + 1)]) + "|"
        
        # Create rows
        rows = []
        for key, values in data.items():
            row_data = [f"{key:>12}"]
            for col in columns:
                value = values.get(col, "N/A")
                if isinstance(value, float):
                    row_data.append(f"{value:>12.4f}")
                else:
                    row_data.append(f"{str(value):>12}")
            rows.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join([header, separator] + rows)
