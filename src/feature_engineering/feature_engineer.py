"""
Feature engineering module for air quality forecasting.
Creates temporal, lag, rolling, and meteorological features.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for air quality data.
    Creates temporal, lag, rolling, and domain-specific features.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize FeatureEngineer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
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
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime column.
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        df_features = df.copy()
        
        if 'datetime' not in df_features.columns:
            self.logger.error("DataFrame must contain 'datetime' column")
            raise ValueError("DataFrame must contain 'datetime' column")
        
        # Ensure datetime is properly formatted
        df_features['datetime'] = pd.to_datetime(df_features['datetime'])
        
        # Extract basic temporal features
        df_features['hour'] = df_features['datetime'].dt.hour
        df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
        df_features['month'] = df_features['datetime'].dt.month
        df_features['day_of_year'] = df_features['datetime'].dt.dayofyear
        df_features['week_of_year'] = df_features['datetime'].dt.isocalendar().week
        
        # Create season feature
        df_features['season'] = df_features['month'].apply(self._get_season)
        
        # Weekend indicator
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        # Rush hour indicators
        df_features['is_morning_rush'] = ((df_features['hour'] >= 7) & (df_features['hour'] <= 9)).astype(int)
        df_features['is_evening_rush'] = ((df_features['hour'] >= 17) & (df_features['hour'] <= 19)).astype(int)
        
        # Cyclical encoding for temporal features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        self.logger.info("Created temporal features")
        return df_features
    
    def _get_season(self, month: int) -> int:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def create_lag_features(self, df: pd.DataFrame, group_col: str = 'site_id') -> pd.DataFrame:
        """
        Create lag features for specified variables.
        
        Args:
            df: DataFrame with time series data
            group_col: Column to group by (e.g., site_id)
            
        Returns:
            DataFrame with lag features added
        """
        df_lagged = df.copy()
        
        # Get lag configuration
        lag_config = self.config.get('features', {}).get('lag_features', {})
        lags = lag_config.get('lags', [1, 2, 3, 6, 12, 24])
        variables = lag_config.get('variables', [])
        
        # Filter variables that exist in the DataFrame
        available_vars = [var for var in variables if var in df_lagged.columns]
        
        # Sort by datetime within each group for proper lag calculation
        df_lagged = df_lagged.sort_values(['site_id', 'datetime']).reset_index(drop=True)
        
        for var in available_vars:
            for lag in lags:
                lag_col_name = f"{var}_lag_{lag}h"
                df_lagged[lag_col_name] = df_lagged.groupby(group_col)[var].shift(lag)
        
        self.logger.info(f"Created lag features for {len(available_vars)} variables with {len(lags)} lags")
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame, group_col: str = 'site_id') -> pd.DataFrame:
        """
        Create rolling window statistics features.
        
        Args:
            df: DataFrame with time series data
            group_col: Column to group by (e.g., site_id)
            
        Returns:
            DataFrame with rolling features added
        """
        df_rolling = df.copy()
        
        # Get rolling configuration
        rolling_config = self.config.get('features', {}).get('rolling_features', {})
        windows = rolling_config.get('windows', [3, 6, 12, 24])
        statistics = rolling_config.get('statistics', ['mean', 'std', 'min', 'max'])
        variables = rolling_config.get('variables', [])
        
        # Filter variables that exist in the DataFrame
        available_vars = [var for var in variables if var in df_rolling.columns]
        
        # Sort by datetime within each group for proper rolling calculation
        df_rolling = df_rolling.sort_values(['site_id', 'datetime']).reset_index(drop=True)
        
        for var in available_vars:
            for window in windows:
                grouped = df_rolling.groupby(group_col)[var]
                
                for stat in statistics:
                    col_name = f"{var}_rolling_{window}h_{stat}"
                    
                    if stat == 'mean':
                        df_rolling[col_name] = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                    elif stat == 'std':
                        df_rolling[col_name] = grouped.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
                    elif stat == 'min':
                        df_rolling[col_name] = grouped.rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
                    elif stat == 'max':
                        df_rolling[col_name] = grouped.rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
        
        self.logger.info(f"Created rolling features for {len(available_vars)} variables with {len(windows)} windows")
        return df_rolling
    
    def create_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific meteorological features.
        
        Args:
            df: DataFrame with meteorological data
            
        Returns:
            DataFrame with meteorological features added
        """
        df_met = df.copy()
        
        # Wind speed and direction
        if 'u_forecast' in df_met.columns and 'v_forecast' in df_met.columns:
            df_met['wind_speed'] = np.sqrt(df_met['u_forecast']**2 + df_met['v_forecast']**2)
            df_met['wind_direction'] = np.arctan2(df_met['v_forecast'], df_met['u_forecast'])
            df_met['wind_direction_deg'] = np.degrees(df_met['wind_direction'])
            df_met['wind_direction_deg'] = (df_met['wind_direction_deg'] + 360) % 360
        
        # Temperature features
        if 'T_forecast' in df_met.columns:
            # Temperature category
            df_met['temp_category'] = pd.cut(df_met['T_forecast'], 
                                           bins=[-np.inf, 10, 20, 30, np.inf],
                                           labels=[0, 1, 2, 3]).astype('float')  # Cold, Cool, Warm, Hot
            
            # Diurnal temperature variation (requires grouping by date)
            if 'datetime' in df_met.columns:
                df_met['date'] = df_met['datetime'].dt.strftime('%Y-%m-%d')
                df_met['temp_daily_mean'] = df_met.groupby(['site_id', 'date'])['T_forecast'].transform('mean')
                df_met['temp_deviation_from_daily_mean'] = df_met['T_forecast'] - df_met['temp_daily_mean']
        
        # Humidity features
        if 'q_forecast' in df_met.columns:
            # Humidity category
            df_met['humidity_category'] = pd.cut(df_met['q_forecast'], 
                                               bins=[-np.inf, 5, 10, 15, np.inf],
                                               labels=[0, 1, 2, 3]).astype('float')  # Dry, Moderate, Humid, Very Humid
        
        # Atmospheric stability indicators
        if 'w_forecast' in df_met.columns:
            # Vertical wind component categories
            df_met['vertical_stability'] = pd.cut(df_met['w_forecast'], 
                                                bins=[-np.inf, -1, 1, np.inf],
                                                labels=[0, 1, 2]).astype('float')  # Sinking, Neutral, Rising
        
        # Combined meteorological indices
        if all(col in df_met.columns for col in ['T_forecast', 'q_forecast', 'wind_speed']):
            # Simple air quality index based on meteorological conditions
            # Higher wind speeds and lower temperatures typically favor dispersion
            df_met['met_dispersion_index'] = (df_met['wind_speed'] * 10) / (df_met['T_forecast'] + 1)
        
        # Pollution accumulation potential
        if 'wind_speed' in df_met.columns:
            df_met['stagnation_potential'] = (df_met['wind_speed'] < 2).astype(int)
        
        self.logger.info("Created meteorological features")
        return df_met
    
    def create_satellite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from satellite data.
        
        Args:
            df: DataFrame with satellite data
            
        Returns:
            DataFrame with satellite features added
        """
        df_sat = df.copy()
        
        # Handle missing satellite data
        satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
        available_sat_cols = [col for col in satellite_cols if col in df_sat.columns]
        
        for col in available_sat_cols:
            # Create missingness indicator
            df_sat[f"{col}_missing"] = df_sat[col].isnull().astype(int)
            
            # Forward fill for missing values (satellites don't measure continuously)
            df_sat[f"{col}_filled"] = df_sat.groupby('site_id')[col].ffill()
            
            # Backward fill remaining missing values
            df_sat[f"{col}_filled"] = df_sat.groupby('site_id')[f"{col}_filled"].bfill()
        
        # Create ratios and interactions
        if 'NO2_satellite' in df_sat.columns and 'HCHO_satellite' in df_sat.columns:
            # Avoid division by zero
            df_sat['NO2_HCHO_ratio'] = np.where(
                df_sat['HCHO_satellite_filled'] != 0,
                df_sat['NO2_satellite_filled'] / df_sat['HCHO_satellite_filled'],
                0
            )
        
        # Satellite data availability score
        satellite_data_cols = [col for col in available_sat_cols if not col.endswith('_filled')]
        if satellite_data_cols:
            df_sat['satellite_data_availability'] = df_sat[satellite_data_cols].count(axis=1) / len(satellite_data_cols)
        
        self.logger.info("Created satellite features")
        return df_sat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with interaction features added
        """
        df_interact = df.copy()
        
        # Temperature and humidity interactions
        if 'T_forecast' in df_interact.columns and 'q_forecast' in df_interact.columns:
            df_interact['temp_humidity_interaction'] = df_interact['T_forecast'] * df_interact['q_forecast']
        
        # Wind and temperature interactions
        if 'wind_speed' in df_interact.columns and 'T_forecast' in df_interact.columns:
            df_interact['wind_temp_interaction'] = df_interact['wind_speed'] * df_interact['T_forecast']
        
        # Forecast and satellite interactions
        if 'NO2_forecast' in df_interact.columns and 'NO2_satellite_filled' in df_interact.columns:
            df_interact['NO2_forecast_satellite_diff'] = df_interact['NO2_forecast'] - df_interact['NO2_satellite_filled']
            df_interact['NO2_forecast_satellite_ratio'] = np.where(
                df_interact['NO2_satellite_filled'] != 0,
                df_interact['NO2_forecast'] / df_interact['NO2_satellite_filled'],
                1
            )
        
        # Hour and meteorological interactions
        if 'hour' in df_interact.columns and 'T_forecast' in df_interact.columns:
            df_interact['hour_temp_interaction'] = df_interact['hour'] * df_interact['T_forecast']
        
        self.logger.info("Created interaction features")
        return df_interact
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all feature types in the correct order.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features created
        """
        self.logger.info("Starting comprehensive feature engineering...")
        
        # Start with original data
        df_all_features = df.copy()
        
        # Create temporal features first (needed for other features)
        df_all_features = self.create_temporal_features(df_all_features)
        
        # Create meteorological features
        df_all_features = self.create_meteorological_features(df_all_features)
        
        # Create satellite features
        df_all_features = self.create_satellite_features(df_all_features)
        
        # Create lag features (requires sorted data)
        df_all_features = self.create_lag_features(df_all_features)
        
        # Create rolling features (requires sorted data)
        df_all_features = self.create_rolling_features(df_all_features)
        
        # Create interaction features last
        df_all_features = self.create_interaction_features(df_all_features)
        
        # Remove intermediate columns if needed
        intermediate_cols = ['date', 'temp_daily_mean']
        cols_to_drop = [col for col in intermediate_cols if col in df_all_features.columns]
        if cols_to_drop:
            df_all_features = df_all_features.drop(columns=cols_to_drop)
        
        self.logger.info(f"Feature engineering completed. Final shape: {df_all_features.shape}")
        return df_all_features
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for analysis and interpretation.
        
        Returns:
            Dictionary of feature group names and their column patterns
        """
        feature_groups = {
            'temporal': ['hour', 'day_of_week', 'month', 'season', 'is_weekend', 'is_morning_rush', 'is_evening_rush',
                        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'],
            'meteorological': ['T_forecast', 'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast',
                             'wind_speed', 'wind_direction', 'temp_category', 'humidity_category', 'vertical_stability',
                             'met_dispersion_index', 'stagnation_potential'],
            'satellite': ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite', 'NO2_HCHO_ratio',
                         'satellite_data_availability'],
            'forecast_inputs': ['O3_forecast', 'NO2_forecast'],
            'lag_features': 'lag_',  # Pattern for lag features
            'rolling_features': 'rolling_',  # Pattern for rolling features
            'interaction_features': 'interaction',  # Pattern for interaction features
        }
        
        return feature_groups
    
    def select_features_by_importance(self, df: pd.DataFrame, feature_importance: Dict[str, float], 
                                    top_k: int = 50) -> List[str]:
        """
        Select top K most important features.
        
        Args:
            df: DataFrame with features
            feature_importance: Dictionary of feature names and importance scores
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        # Filter importance scores for features that exist in the DataFrame
        available_importance = {k: v for k, v in feature_importance.items() if k in df.columns}
        
        # Sort features by importance and select top K
        sorted_features = sorted(available_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:top_k]]
        
        self.logger.info(f"Selected top {len(selected_features)} features out of {len(available_importance)} available")
        
        return selected_features
