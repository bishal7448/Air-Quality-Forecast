# Air Quality Forecasting System - API & Interface Documentation

## Overview

This document provides comprehensive documentation for all the APIs and interfaces in the Air Quality Forecasting System. It covers internal component interfaces, external APIs, data formats, and integration points.

## Table of Contents

1. [Component Interfaces](#component-interfaces)
2. [Data Formats](#data-formats)
3. [Configuration Interface](#configuration-interface)
4. [Web Dashboard API](#web-dashboard-api)
5. [External Integration APIs](#external-integration-apis)
6. [Error Handling](#error-handling)
7. [Usage Examples](#usage-examples)

## Component Interfaces

### 1. DataLoader Interface

**Module**: `src/data_preprocessing/data_loader.py`

#### Core Methods

```python
class DataLoader:
    def __init__(self, config_path: str = "configs/config.yaml")
    
    def load_site_coordinates() -> pd.DataFrame:
        """
        Load monitoring site coordinates.
        
        Returns:
            DataFrame with columns: ['Site', 'Latitude_N', 'Longitude_E']
        """
    
    def load_training_data(site_id: Optional[int] = None) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Load training data for specified site(s).
        
        Args:
            site_id: Specific site ID (1-7) or None for all sites
            
        Returns:
            Single DataFrame for specific site or Dict of DataFrames for all sites
        """
    
    def load_test_data(site_id: Optional[int] = None) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Load unseen input data for forecasting.
        
        Args:
            site_id: Specific site ID (1-7) or None for all sites
            
        Returns:
            Single DataFrame for specific site or Dict of DataFrames for all sites
        """
    
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
    
    def combine_all_sites(data_dict: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from multiple sites.
        
        Args:
            data_dict: Dictionary of site_id -> DataFrame
            
        Returns:
            Combined DataFrame with site_id column
        """
```

#### Data Schema

**Input CSV Format**:
```
datetime,year,month,day,hour,T_forecast,q_forecast,u_forecast,v_forecast,w_forecast,
NO2_satellite,HCHO_satellite,NO2_HCHO_ratio,O3_forecast,NO2_forecast,O3_target,NO2_target,site_id
```

**Output DataFrame Schema**:
```python
{
    'datetime': pd.Timestamp,
    'year': int,
    'month': int,
    'day': int,
    'hour': int,
    'T_forecast': float,      # Temperature forecast (°C)
    'q_forecast': float,      # Humidity forecast
    'u_forecast': float,      # U-wind component (m/s)
    'v_forecast': float,      # V-wind component (m/s)
    'w_forecast': float,      # W-wind component (m/s)
    'NO2_satellite': float,   # Satellite NO₂ concentration
    'HCHO_satellite': float,  # Satellite HCHO concentration
    'NO2_HCHO_ratio': float,  # Ratio of NO₂/HCHO
    'O3_forecast': float,     # O₃ forecast value
    'NO2_forecast': float,    # NO₂ forecast value
    'O3_target': float,       # Ground-truth O₃ (training only)
    'NO2_target': float,      # Ground-truth NO₂ (training only)
    'site_id': int           # Site identifier (1-7)
}
```

### 2. FeatureEngineer Interface

**Module**: `src/feature_engineering/feature_engineer.py`

#### Core Methods

```python
class FeatureEngineer:
    def __init__(self, config_path: str = "configs/config.yaml")
    
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features with cyclical encoding.
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with added temporal features:
            - hour_sin, hour_cos: Cyclical hour encoding
            - month_sin, month_cos: Cyclical month encoding
            - day_of_year_sin, day_of_year_cos: Cyclical day encoding
            - season: Season indicator (0-3)
            - is_weekend: Weekend flag
            - is_morning_rush, is_evening_rush: Rush hour flags
        """
    
    def create_lag_features(df: pd.DataFrame, group_col: str = 'site_id') -> pd.DataFrame:
        """
        Create lag features for historical dependencies.
        
        Args:
            df: DataFrame with time series data
            group_col: Column to group by for lag calculation
            
        Returns:
            DataFrame with lag features:
            - {variable}_lag_{hours}h: Historical values at specified lags
            
        Configuration:
            features.lag_features.lags: [1, 2, 3, 6, 12, 24]
            features.lag_features.variables: ["O3_target", "NO2_target", "T_forecast"]
        """
    
    def create_rolling_features(df: pd.DataFrame, group_col: str = 'site_id') -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with time series data
            group_col: Column to group by for rolling calculation
            
        Returns:
            DataFrame with rolling features:
            - {variable}_rolling_{window}h_{statistic}: Rolling statistics
            
        Configuration:
            features.rolling_features.windows: [3, 6, 12, 24]
            features.rolling_features.statistics: ["mean", "std", "min", "max"]
        """
    
    def create_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create meteorological derived features.
        
        Args:
            df: DataFrame with meteorological variables
            
        Returns:
            DataFrame with meteorological features:
            - wind_speed: Derived from u,v components
            - wind_direction: Wind direction in degrees
            - stability_indicator: Atmospheric stability measure
        """
    
    def create_satellite_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create satellite-derived features.
        
        Args:
            df: DataFrame with satellite data
            
        Returns:
            DataFrame with satellite features:
            - satellite_data_availability: Data availability score
            - satellite_quality_flag: Quality indicator
        """
    
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with interaction features
        """
```

### 3. ModelTrainer Interface

**Module**: `src/models/model_trainer.py`

#### Core Methods

```python
class ModelTrainer:
    def __init__(self, config_path: str = "configs/config.yaml")
    
    def prepare_training_data(features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                            target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with target values
            target_column: Name of target column ('O3_target' or 'NO2_target')
            
        Returns:
            Tuple of (X, y, feature_names)
        """
    
    def split_data(X: np.ndarray, y: np.ndarray, 
                   validation_split: float = None, 
                   test_split: float = None) -> Dict[str, np.ndarray]:
        """
        Split data chronologically for time series.
        
        Args:
            X: Feature array
            y: Target array
            validation_split: Validation proportion (default from config)
            test_split: Test proportion (default from config)
            
        Returns:
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
    
    def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of trained models: {model_name: model_object}
        """
    
    def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestRegressor:
    def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> xgb.XGBRegressor:
    def train_lstm(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> keras.Model:
    def train_cnn_lstm(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> keras.Model:
```

### 4. ModelEvaluator Interface

**Module**: `src/evaluation/model_evaluator.py`

#### Core Methods

```python
class ModelEvaluator:
    def __init__(self, config_path: str = "configs/config.yaml")
    
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name for logging
            
        Returns:
            Dictionary of metrics:
            {
                'rmse': float,
                'mae': float,
                'r2': float,
                'mape': float,
                'bias': float,
                'correlation': float,
                'accuracy_10percent': float,
                'accuracy_20percent': float
            }
        """
    
    def compare_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray,
                      target_name: str = "Target") -> pd.DataFrame:
        """
        Compare multiple models on same test set.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            target_name: Target variable name
            
        Returns:
            DataFrame with comparison results
        """
    
    def create_evaluation_report(results: Dict, target_name: str) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            target_name: Target variable name
            
        Returns:
            Markdown formatted report string
        """
```

### 5. AirQualityForecaster Interface

**Module**: `src/forecasting/forecaster.py`

#### Core Methods

```python
class AirQualityForecaster:
    def __init__(self, config_path: str = "configs/config.yaml")
    
    def load_models(models_dict: Dict[str, Any], scalers_dict: Dict[str, Any] = None,
                   feature_names: List[str] = None):
        """
        Load trained models and scalers.
        
        Args:
            models_dict: Dictionary of trained models
            scalers_dict: Dictionary of scalers (optional)
            feature_names: List of feature names
        """
    
    def forecast_for_site(site_data: pd.DataFrame, site_id: int,
                         target_columns: List[str], 
                         forecast_hours: int = 24) -> pd.DataFrame:
        """
        Generate forecast for specific site.
        
        Args:
            site_data: Input data for forecasting
            site_id: Site identifier
            target_columns: List of targets to forecast
            forecast_hours: Number of hours to forecast
            
        Returns:
            DataFrame with columns:
            - datetime: Forecast timestamps
            - forecast: Predicted values
            - lower_bound: Lower confidence bound
            - upper_bound: Upper confidence bound
            - pollutant: Pollutant name
            - site_id: Site identifier
        """
    
    def ensemble_predict(X: np.ndarray, ensemble_method: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            X: Input features
            ensemble_method: Combination method ("mean", "median", "weighted_mean")
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
```

## Data Formats

### 1. Training Data Format

**File**: `site_{1-7}_train_data.csv`

```csv
datetime,year,month,day,hour,T_forecast,q_forecast,u_forecast,v_forecast,w_forecast,NO2_satellite,HCHO_satellite,NO2_HCHO_ratio,O3_forecast,NO2_forecast,O3_target,NO2_target
2019-01-01 00:00:00,2019,1,1,0,15.2,0.65,-2.3,1.8,0.1,25.4,12.8,1.98,35.6,28.3,42.1,31.5
2019-01-01 01:00:00,2019,1,1,1,14.8,0.67,-2.1,1.9,0.1,24.9,12.6,1.97,34.2,27.8,40.8,30.2
```

### 2. Forecast Data Format

**File**: `site_{1-7}_unseen_input_data.csv`

```csv
datetime,year,month,day,hour,T_forecast,q_forecast,u_forecast,v_forecast,w_forecast,NO2_satellite,HCHO_satellite,NO2_HCHO_ratio,O3_forecast,NO2_forecast
2024-07-01 00:00:00,2024,7,1,0,28.5,0.72,1.2,-0.8,0.2,18.7,8.9,2.10,52.3,22.1
2024-07-01 01:00:00,2024,7,1,1,28.2,0.74,1.1,-0.9,0.2,18.2,8.7,2.09,51.8,21.8
```

### 3. Site Coordinates Format

**File**: `lat_lon_sites.txt`

```
Site	Latitude_N	Longitude_E
1	28.7041	77.1025
2	28.5355	77.3910
3	28.6139	77.2090
4	28.7041	77.1025
5	28.4595	77.0266
6	28.5821	77.0880
7	28.6877	77.2273
```

### 4. Forecast Output Format

**JSON Response**:
```json
{
    "site_id": 1,
    "pollutant": "O3",
    "forecast_generated": "2024-07-01T12:00:00Z",
    "forecast_horizon_hours": 24,
    "model_used": "xgboost",
    "confidence_level": 0.95,
    "forecasts": [
        {
            "datetime": "2024-07-01T13:00:00Z",
            "forecast": 45.7,
            "lower_bound": 38.2,
            "upper_bound": 53.1,
            "hour_ahead": 1
        },
        {
            "datetime": "2024-07-01T14:00:00Z",
            "forecast": 48.3,
            "lower_bound": 40.8,
            "upper_bound": 55.8,
            "hour_ahead": 2
        }
    ],
    "health_assessment": {
        "status": "GOOD",
        "recommendation": "Normal outdoor activities",
        "average_level": 45.7,
        "peak_level": 58.9
    },
    "metadata": {
        "features_used": 187,
        "training_data_period": "2019-2024",
        "model_performance": {
            "rmse": 12.3,
            "r2": 0.85
        }
    }
}
```

## Configuration Interface

### YAML Configuration Schema

**File**: `configs/config.yaml`

```yaml
# Project Information
project:
  name: "Air Quality Forecasting for Delhi"
  version: "1.0.0"
  description: "AI/ML-based short-term air quality forecasting system"

# Data Configuration
data:
  raw_data_path: "data/"
  site_coordinates: "lat_lon_sites.txt"
  train_files:
    - "site_1_train_data.csv"
    - "site_2_train_data.csv"
    # ... additional sites
  test_files:
    - "site_1_unseen_input_data.csv"
    - "site_2_unseen_input_data.csv"
    # ... additional sites

# Preprocessing Configuration
preprocessing:
  missing_value_strategy: "interpolate"  # "interpolate", "forward_fill", "mean"
  outlier_detection: "iqr"               # "iqr", "zscore", "isolation_forest"
  scaling_method: "standard"             # "standard", "minmax", "robust"

# Feature Engineering Configuration
features:
  lag_features:
    lags: [1, 2, 3, 6, 12, 24]
    variables: ["O3_target", "NO2_target", "T_forecast", "q_forecast"]
  
  rolling_features:
    windows: [3, 6, 12, 24]
    statistics: ["mean", "std", "min", "max"]
    variables: ["O3_target", "NO2_target", "T_forecast", "NO2_satellite"]
  
  temporal_features:
    cyclical_encoding: true
    rush_hour_indicators: true
    seasonal_indicators: true

# Model Configuration
models:
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42
  
  xgboost:
    n_estimators: 200
    max_depth: 8
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
  
  lstm:
    sequence_length: 24
    hidden_size: 64
    dropout: 0.2
    learning_rate: 0.001
    epochs: 100
    batch_size: 32
  
  cnn_lstm:
    sequence_length: 24
    conv_filters: [32, 64]
    conv_kernel_size: 3
    lstm_units: 50
    dropout: 0.2
    learning_rate: 0.001
    epochs: 100

# Training Configuration
training:
  validation_split: 0.2
  test_split: 0.2
  random_state: 42
  cross_validation:
    method: "time_series_split"
    n_splits: 5

# Forecasting Configuration
forecasting:
  forecast_horizons: [24, 48]
  confidence_intervals: [0.05, 0.95]
  ensemble_methods: ["mean", "median", "weighted_mean"]
  health_thresholds:
    O3:
      good: 55
      moderate: 100
    NO2:
      good: 53
      moderate: 100

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/air_quality_forecast.log"
  max_file_size: "10MB"
  backup_count: 5

# Output Configuration
output:
  results_dir: "results/"
  models_dir: "models/"
  plots_dir: "results/plots/"
  reports_dir: "results/reports/"
  forecasts_dir: "results/forecasts/"
```

## Web Dashboard API

### Streamlit Interface

**File**: `app.py`

#### Session State Variables

```python
st.session_state = {
    'data_loaded': bool,
    'combined_data': pd.DataFrame,
    'enhanced_data': pd.DataFrame,
    'models_trained': Dict[str, Any],
    'forecasts': Dict[str, pd.DataFrame],
    'site_coords': pd.DataFrame
}
```

#### Dashboard Methods

```python
class AirQualityDashboard:
    def load_data() -> pd.DataFrame:
        """Load and cache air quality data."""
    
    def create_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for modeling."""
    
    def train_models_for_target(target_name: str, enhanced_data: pd.DataFrame) -> Dict:
        """Train models for specific target variable."""
    
    def generate_forecast(target_name: str, model_info: Dict, hours: int = 24) -> pd.DataFrame:
        """Generate forecast for target variable."""
    
    def get_health_status(pollutant: str, avg_level: float) -> Tuple[str, str, str]:
        """Get health status and advice based on pollutant levels."""
    
    def create_air_quality_heatmap(data: pd.DataFrame, coords_df: pd.DataFrame,
                                  pollutant: str = 'O3_target') -> pd.DataFrame:
        """Create heatmap data for visualization."""
```

### FastAPI Production Interface

**Proposed API Endpoints**:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Air Quality Forecasting API", version="1.0.0")

class ForecastRequest(BaseModel):
    site_id: int
    forecast_hours: int = 24
    pollutants: List[str] = ["O3", "NO2"]
    confidence_level: float = 0.95

class ForecastResponse(BaseModel):
    site_id: int
    forecasts: List[Dict]
    health_assessment: Dict
    metadata: Dict

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate air quality forecast for specified site."""

@app.get("/sites")
async def get_available_sites():
    """Get list of available monitoring sites."""

@app.get("/health/{site_id}")
async def get_current_health_status(site_id: int):
    """Get current health status for specific site."""

@app.get("/models")
async def get_model_info():
    """Get information about available models and their performance."""
```

## External Integration APIs

### 1. Meteorological Data APIs

```python
class MeteorologicalAPI:
    def get_weather_forecast(lat: float, lon: float, hours: int = 48) -> Dict:
        """
        Get weather forecast from external API.
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Forecast horizon
            
        Returns:
            Dictionary with meteorological variables
        """
```

### 2. Satellite Data APIs

```python
class SatelliteDataAPI:
    def get_satellite_observations(lat: float, lon: float, 
                                 start_date: str, end_date: str) -> Dict:
        """
        Get satellite observations from external API.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with satellite measurements
        """
```

### 3. Health Advisory APIs

```python
class HealthAdvisoryAPI:
    def get_health_recommendations(o3_level: float, no2_level: float) -> Dict:
        """
        Get health recommendations based on pollutant levels.
        
        Args:
            o3_level: O₃ concentration (μg/m³)
            no2_level: NO₂ concentration (μg/m³)
            
        Returns:
            Dictionary with health status and recommendations
        """
```

## Error Handling

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict] = None
    timestamp: str
    request_id: str

# Common Error Types
class ValidationError(Exception):
    """Data validation errors"""
    
class ModelError(Exception):
    """Model training/prediction errors"""
    
class DataError(Exception):
    """Data loading/processing errors"""
    
class ConfigurationError(Exception):
    """Configuration-related errors"""
```

### Error Handling Examples

```python
try:
    data = data_loader.load_training_data(site_id=1)
except FileNotFoundError as e:
    logger.error(f"Training data file not found: {e}")
    raise DataError(f"Cannot load data for site {site_id}")

try:
    model = trainer.train_xgboost(X_train, y_train)
except Exception as e:
    logger.error(f"XGBoost training failed: {e}")
    # Fallback to simpler model
    model = trainer.train_random_forest(X_train, y_train)
```

## Usage Examples

### 1. Complete Pipeline Example

```python
from src.data_preprocessing.data_loader import DataLoader
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.evaluation.model_evaluator import ModelEvaluator
from src.forecasting.forecaster import AirQualityForecaster

# Initialize components
data_loader = DataLoader("configs/config.yaml")
feature_engineer = FeatureEngineer("configs/config.yaml")
model_trainer = ModelTrainer("configs/config.yaml")
model_evaluator = ModelEvaluator("configs/config.yaml")
forecaster = AirQualityForecaster("configs/config.yaml")

# Load and process data
coords_df = data_loader.load_site_coordinates()
train_data = data_loader.load_training_data()
combined_data = data_loader.combine_all_sites(train_data)

# Feature engineering
enhanced_data = feature_engineer.create_temporal_features(combined_data)
enhanced_data = feature_engineer.create_lag_features(enhanced_data)
enhanced_data = feature_engineer.create_rolling_features(enhanced_data)

# Model training for O₃
features_df = enhanced_data.drop(['O3_target', 'NO2_target'], axis=1, errors='ignore')
targets_df = enhanced_data[['O3_target', 'NO2_target']]

X, y, feature_names = model_trainer.prepare_training_data(features_df, targets_df, 'O3_target')
data_splits = model_trainer.split_data(X, y)

models = model_trainer.train_all_models(
    data_splits['X_train'], data_splits['y_train'],
    data_splits['X_val'], data_splits['y_val']
)

# Evaluation
comparison_results = model_evaluator.compare_models(
    models, data_splits['X_test'], data_splits['y_test'], "O3"
)

# Forecasting
forecaster.load_models(models, None, feature_names)
test_data = data_loader.load_test_data(site_id=1)
forecast_df = forecaster.forecast_for_site(
    test_data, site_id=1, target_columns=['O3_target'], forecast_hours=24
)

print("Forecast generated:", forecast_df.head())
```

### 2. Custom Model Integration

```python
# Add custom model to ModelTrainer
class CustomModelTrainer(ModelTrainer):
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb
        
        # Get configuration
        params = self.config.get('models', {}).get('lightgbm', {})
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)] if X_val is not None else None,
                 early_stopping_rounds=10,
                 verbose=False)
        
        # Store model
        self.models['lightgbm'] = model
        return model

# Use custom trainer
trainer = CustomModelTrainer("configs/config.yaml")
models = trainer.train_all_models(X_train, y_train, X_val, y_val)
```

### 3. Real-time Forecasting API

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load models once at startup
models = {}
scalers = {}
feature_names = []

@app.on_event("startup")
async def load_models():
    global models, scalers, feature_names
    
    models['O3'] = joblib.load('models/xgboost_O3.pkl')
    models['NO2'] = joblib.load('models/random_forest_NO2.pkl')
    scalers = joblib.load('models/scalers.pkl')
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

@app.post("/forecast/{site_id}")
async def generate_forecast(site_id: int, data: dict):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Feature engineering (simplified)
        input_df['hour_sin'] = np.sin(2 * np.pi * input_df['hour'] / 24)
        input_df['hour_cos'] = np.cos(2 * np.pi * input_df['hour'] / 24)
        # ... additional feature engineering
        
        # Scale features
        features = input_df[feature_names]
        scaled_features = scalers.transform(features)
        
        # Generate predictions
        predictions = {}
        for pollutant, model in models.items():
            pred = model.predict(scaled_features)[0]
            predictions[pollutant] = {
                'forecast': float(pred),
                'confidence_interval': [float(pred * 0.9), float(pred * 1.1)]
            }
        
        return {
            'site_id': site_id,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

This comprehensive API documentation provides all the interfaces and integration points needed to work with the Air Quality Forecasting System, enabling both internal development and external integration.
