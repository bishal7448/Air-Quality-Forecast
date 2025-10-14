"""
Data loading and preprocessing utilities for air quality forecasting.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Data loader for air quality forecasting data.
    Handles loading, cleaning, and basic preprocessing of training and test data.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize DataLoader with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize scalers dictionary for different features
        self.scalers = {}
        
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
    
    def load_site_coordinates(self) -> pd.DataFrame:
        """Load site coordinates from text file."""
        try:
            coord_file = Path(self.config['data']['raw_data_path']) / self.config['data']['site_coordinates']
            
            # Read the coordinate file
            coords_df = pd.read_csv(coord_file, sep='\t', skiprows=1, 
                                  names=['Site', 'Latitude_N', 'Longitude_E'])
            
            # Clean the data - remove the row number prefix
            coords_df['Site'] = coords_df['Site'].astype(str).str.extract(r'(\d+)')[0].astype(int)
            
            self.logger.info(f"Loaded coordinates for {len(coords_df)} sites")
            return coords_df
            
        except Exception as e:
            self.logger.error(f"Error loading site coordinates: {e}")
            raise
    
    def load_training_data(self, site_id: Optional[int] = None) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Load training data for specified site(s).
        
        Args:
            site_id: Specific site ID to load. If None, loads all sites.
            
        Returns:
            DataFrame for single site or dictionary of DataFrames for all sites.
        """
        try:
            data_path = Path(self.config['data']['raw_data_path'])
            
            if site_id is not None:
                # Load specific site
                filename = f"site_{site_id}_train_data.csv"
                file_path = data_path / filename
                
                if not file_path.exists():
                    raise FileNotFoundError(f"Training data file not found: {file_path}")
                
                df = pd.read_csv(file_path)
                df['site_id'] = site_id
                
                # Create datetime column
                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                
                self.logger.info(f"Loaded training data for site {site_id}: {len(df)} records")
                return df
            
            else:
                # Load all sites
                all_data = {}
                for i in range(1, 8):  # Sites 1-7
                    try:
                        site_data = self.load_training_data(site_id=i)
                        all_data[i] = site_data
                    except FileNotFoundError:
                        self.logger.warning(f"Training data for site {i} not found, skipping...")
                
                self.logger.info(f"Loaded training data for {len(all_data)} sites")
                return all_data
                
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            raise
    
    def load_test_data(self, site_id: Optional[int] = None) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Load test (unseen input) data for specified site(s).
        
        Args:
            site_id: Specific site ID to load. If None, loads all sites.
            
        Returns:
            DataFrame for single site or dictionary of DataFrames for all sites.
        """
        try:
            data_path = Path(self.config['data']['raw_data_path'])
            
            if site_id is not None:
                # Load specific site
                filename = f"site_{site_id}_unseen_input_data.csv"
                file_path = data_path / filename
                
                if not file_path.exists():
                    raise FileNotFoundError(f"Test data file not found: {file_path}")
                
                df = pd.read_csv(file_path)
                df['site_id'] = site_id
                
                # Create datetime column
                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                
                self.logger.info(f"Loaded test data for site {site_id}: {len(df)} records")
                return df
            
            else:
                # Load all sites
                all_data = {}
                for i in range(1, 8):  # Sites 1-7
                    try:
                        site_data = self.load_test_data(site_id=i)
                        all_data[i] = site_data
                    except FileNotFoundError:
                        self.logger.warning(f"Test data for site {i} not found, skipping...")
                
                self.logger.info(f"Loaded test data for {len(all_data)} sites")
                return all_data
                
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise
    
    def combine_all_sites(self, data_dict: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from all sites into a single DataFrame."""
        try:
            combined_df = pd.concat(data_dict.values(), ignore_index=True)
            self.logger.info(f"Combined data from {len(data_dict)} sites: {len(combined_df)} total records")
            return combined_df
        except Exception as e:
            self.logger.error(f"Error combining site data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        missing_strategy = self.config.get('preprocessing', {}).get('missing_value_strategy', 'interpolate')
        
        if missing_strategy == 'interpolate':
            # Interpolate missing values for numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
            
        elif missing_strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
            
        elif missing_strategy == 'drop':
            df_clean = df_clean.dropna()
        
        # Handle outliers if enabled
        if self.config.get('preprocessing', {}).get('outlier_detection', False):
            df_clean = self._handle_outliers(df_clean)
        
        self.logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using specified method."""
        method = self.config.get('preprocessing', {}).get('outlier_method', 'iqr')
        df_clean = df.copy()
        
        # Get numeric columns excluding datetime and identifiers
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        exclude_cols = ['year', 'month', 'day', 'hour', 'site_id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if method == 'iqr':
            for col in numeric_cols:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        elif method == 'zscore':
            for col in numeric_cols:
                if col in df_clean.columns:
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < 3]
        
        return df_clean
    
    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Get feature columns
        feature_cols = (
            self.config['features']['meteorological'] +
            self.config['features']['satellite'] +
            self.config['features']['forecast_inputs']
        )
        
        # Get target columns
        target_cols = self.config['features']['targets']
        
        # Filter available columns
        available_feature_cols = [col for col in feature_cols if col in df.columns]
        available_target_cols = [col for col in target_cols if col in df.columns]
        
        # Extract features and targets
        features_df = df[available_feature_cols + ['datetime', 'site_id']].copy()
        
        if available_target_cols:
            targets_df = df[available_target_cols + ['datetime', 'site_id']].copy()
        else:
            # For test data without targets
            targets_df = pd.DataFrame()
        
        self.logger.info(f"Prepared features: {len(available_feature_cols)} columns, "
                        f"targets: {len(available_target_cols)} columns")
        
        return features_df, targets_df
    
    def scale_features(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None, 
                      fit_scaler: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scale features using specified scaling method.
        
        Args:
            train_df: Training data
            test_df: Test data (optional)
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled DataFrame(s)
        """
        scaling_method = self.config.get('preprocessing', {}).get('scaling_method', 'standard')
        
        # Get numeric columns to scale
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['site_id', 'year', 'month', 'day', 'hour']
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
        train_scaled = train_df.copy()
        
        if fit_scaler:
            # Fit scaler on training data
            train_scaled[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
            self.scalers['features'] = scaler
        else:
            # Use existing scaler
            if 'features' in self.scalers:
                train_scaled[cols_to_scale] = self.scalers['features'].transform(train_df[cols_to_scale])
        
        if test_df is not None:
            test_scaled = test_df.copy()
            if 'features' in self.scalers:
                test_cols_to_scale = [col for col in cols_to_scale if col in test_df.columns]
                test_scaled[test_cols_to_scale] = self.scalers['features'].transform(test_df[test_cols_to_scale])
            
            self.logger.info(f"Scaled features using {scaling_method} scaler")
            return train_scaled, test_scaled
        
        return train_scaled
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, subfolder: str = "processed"):
        """Save processed data to file."""
        try:
            save_path = Path(self.config['data']['raw_data_path']) / subfolder
            save_path.mkdir(exist_ok=True)
            
            full_path = save_path / filename
            df.to_csv(full_path, index=False)
            
            self.logger.info(f"Saved processed data to {full_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the dataset."""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'date_range': None
        }
        
        if 'datetime' in df.columns:
            summary['date_range'] = {
                'start': df['datetime'].min(),
                'end': df['datetime'].max(),
                'duration_days': (df['datetime'].max() - df['datetime'].min()).days
            }
        
        return summary
