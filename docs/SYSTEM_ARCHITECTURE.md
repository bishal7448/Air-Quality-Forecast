# Air Quality Forecasting System - Complete Architecture

## Overview

The Air Quality Forecasting System is a comprehensive AI/ML-based solution for predicting ground-level O₃ and NO₂ concentrations in Delhi. This document provides a complete architectural overview of how the system works, from data ingestion to real-time forecasting.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES LAYER                        │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Satellite Data │ Meteorological  │    Ground Monitoring        │
│  • NO₂, HCHO    │  • Temperature  │    • Historical O₃/NO₂      │
│  • Ratios       │  • Humidity     │    • Site coordinates       │
│                 │  • Wind (u,v,w) │    • Multi-year data        │
└─────────────────┴─────────────────┴─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  DataLoader (src/data_preprocessing/data_loader.py)            │
│  • Multi-site data loading                                     │
│  • Data validation & cleaning                                  │
│  • Missing value imputation                                    │
│  • Temporal alignment                                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  FeatureEngineer (src/feature_engineering/feature_engineer.py) │
│  • Temporal features (cyclical encoding)                       │
│  • Lag features (1,2,3,6,12,24 hours)                         │
│  • Rolling statistics (mean, std, min, max)                    │
│  • Meteorological indices                                      │
│  • Satellite-derived features                                  │
│  • Interaction terms                                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING LAYER                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Traditional ML │  Deep Learning  │    Model Management         │
│  • Random Forest│  • LSTM         │    • Cross-validation       │
│  • XGBoost      │  • CNN-LSTM     │    • Hyperparameter tuning  │
│                 │                 │    • Model persistence      │
└─────────────────┴─────────────────┴─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ModelEvaluator (src/evaluation/model_evaluator.py)           │
│  • Comprehensive metrics (RMSE, MAE, R², MAPE)                │
│  • Cross-validation & temporal splitting                       │
│  • Residual analysis                                          │
│  • Feature importance analysis                                │
│  • Visualization & reporting                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FORECASTING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  AirQualityForecaster (src/forecasting/forecaster.py)         │
│  • Real-time predictions                                       │
│  • Multi-horizon forecasts (24-48h)                           │
│  • Confidence intervals                                       │
│  • Ensemble predictions                                       │
│  • Health impact assessment                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               USER INTERFACE LAYER                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Web Dashboard  │  Jupyter        │    Command Line             │
│  • app.py       │  • Notebooks    │    • Demo scripts           │
│  • Streamlit    │  • Interactive  │    • Batch processing       │
│  • Interactive  │    analysis     │    • API endpoints          │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Core Components

### 1. Data Preprocessing Layer

**Module**: `src/data_preprocessing/data_loader.py`
**Class**: `DataLoader`

**Responsibilities**:
- Load training and test data from multiple monitoring sites
- Handle missing values using interpolation or forward-fill
- Validate data quality and detect outliers
- Create datetime columns and ensure temporal consistency
- Normalize and scale features

**Key Methods**:
- `load_site_coordinates()`: Load monitoring site coordinates
- `load_training_data(site_id)`: Load training data for specific site
- `load_test_data(site_id)`: Load unseen input data for forecasting
- `clean_data()`: Handle missing values and outliers
- `combine_all_sites()`: Merge data from multiple sites

**Data Flow**:
```
CSV Files → DataLoader → Cleaned DataFrame → Feature Engineering
```

### 2. Feature Engineering Layer

**Module**: `src/feature_engineering/feature_engineer.py`
**Class**: `FeatureEngineer`

**Responsibilities**:
- Create temporal features with cyclical encoding
- Generate lag features for historical dependencies
- Calculate rolling statistics over multiple windows
- Derive meteorological indices (wind speed, direction)
- Process satellite data and create derived features
- Generate interaction terms between variables

**Key Methods**:
- `create_temporal_features()`: Hour, day, season encoding
- `create_lag_features()`: Historical values at 1-24 hour lags
- `create_rolling_features()`: Moving averages and statistics
- `create_meteorological_features()`: Wind speed/direction, stability
- `create_satellite_features()`: Satellite data processing
- `create_interaction_features()`: Cross-feature interactions

**Feature Categories**:

1. **Temporal Features**:
   - Cyclical encoding: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
   - Calendar variables: `day_of_week`, `season`, `is_weekend`
   - Rush hour indicators: `is_morning_rush`, `is_evening_rush`

2. **Lag Features**:
   - Historical pollutant levels: `O3_target_lag_1h`, `NO2_target_lag_24h`
   - Meteorological lags: `T_forecast_lag_1h`, `q_forecast_lag_6h`

3. **Rolling Features**:
   - Moving averages: `T_forecast_rolling_12_mean`
   - Statistics: `NO2_satellite_rolling_6_std`, `wind_speed_rolling_24_max`

4. **Meteorological Features**:
   - Derived variables: `wind_speed`, `wind_direction`
   - Atmospheric stability indicators
   - Heat index, atmospheric pressure tendencies

5. **Satellite Features**:
   - Data availability scores
   - Missing data handling flags
   - Ratio calculations between different satellite measurements

### 3. Model Training Layer

**Module**: `src/models/model_trainer.py`
**Class**: `ModelTrainer`

**Responsibilities**:
- Train multiple model architectures
- Handle both traditional ML and deep learning models
- Perform hyperparameter optimization
- Manage model persistence and loading
- Prepare data for different model types

**Supported Models**:

1. **Random Forest**:
   - Ensemble method with feature importance
   - Robust to overfitting
   - Handles mixed data types well
   - Fast training and prediction

2. **XGBoost**:
   - Gradient boosting with regularization
   - Excellent predictive performance
   - Built-in feature selection
   - Early stopping support

3. **LSTM (Long Short-Term Memory)**:
   - Captures long-term temporal dependencies
   - Handles sequential patterns in time series
   - Memory cells for information retention
   - Suitable for complex temporal relationships

4. **CNN-LSTM Hybrid**:
   - Combines spatial and temporal pattern recognition
   - Convolutional layers extract local features
   - LSTM layers capture sequential dependencies
   - Advanced architecture for complex patterns

**Key Methods**:
- `prepare_training_data()`: Data preprocessing for model input
- `split_data()`: Temporal train/validation/test splits
- `train_all_models()`: Train multiple models simultaneously
- `train_random_forest()`, `train_xgboost()`: Specific model trainers
- `train_lstm()`, `train_cnn_lstm()`: Deep learning model trainers
- `prepare_sequence_data()`: Sequence preparation for LSTM models

### 4. Evaluation Layer

**Module**: `src/evaluation/model_evaluator.py`
**Class**: `ModelEvaluator`

**Responsibilities**:
- Comprehensive model evaluation with multiple metrics
- Cross-validation and temporal validation
- Residual analysis and diagnostic plots
- Feature importance analysis
- Model comparison and selection

**Evaluation Metrics**:

1. **Accuracy Metrics**:
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of Determination)
   - MAPE (Mean Absolute Percentage Error)

2. **Bias Analysis**:
   - Mean bias (systematic errors)
   - Normalized bias
   - Seasonal bias patterns

3. **Correlation Metrics**:
   - Pearson correlation
   - Spearman rank correlation
   - Statistical significance testing

4. **Practical Metrics**:
   - Accuracy within 10% and 20% bounds
   - Health-relevant threshold accuracy
   - Peak prediction accuracy

**Key Methods**:
- `calculate_metrics()`: Comprehensive metric calculation
- `evaluate_model()`: Single model evaluation
- `compare_models()`: Multi-model comparison
- `create_evaluation_report()`: Comprehensive reporting
- `plot_predictions_vs_actual()`: Visualization methods

### 5. Forecasting Layer

**Module**: `src/forecasting/forecaster.py`
**Class**: `AirQualityForecaster`

**Responsibilities**:
- Generate real-time air quality forecasts
- Provide 24-48 hour predictions with hourly resolution
- Calculate confidence intervals
- Ensemble multiple models for robust predictions
- Health impact assessment

**Forecasting Process**:

1. **Data Preparation**:
   - Latest meteorological forecasts
   - Recent ground measurements
   - Satellite observations

2. **Model Prediction**:
   - Individual model predictions
   - Ensemble combination
   - Uncertainty quantification

3. **Post-processing**:
   - Confidence interval calculation
   - Health status assessment
   - Forecast visualization

**Key Methods**:
- `load_models()`: Load trained models for forecasting
- `predict_single_model()`: Single model prediction
- `ensemble_predict()`: Combine multiple models
- `forecast_for_site()`: Site-specific forecasting
- `calculate_confidence_intervals()`: Uncertainty estimation

### 6. User Interface Layer

#### Web Dashboard (`app.py`)
- **Framework**: Streamlit
- **Features**:
  - Interactive data exploration
  - Real-time model training
  - Forecast visualization
  - Health impact display
  - Multi-site comparison

#### Jupyter Notebooks (`notebooks/`)
- **01_complete_workflow.ipynb**: Comprehensive analysis workflow
- Interactive exploration and analysis
- Step-by-step model development
- Detailed visualizations

#### Command Line Interface
- **run_simple_demo.py**: Quick demonstration
- **run_complete_demo.py**: Full pipeline execution
- **start_dashboard.py**: Launch web interface

## Data Flow Architecture

### Training Pipeline

```
Raw Data → Data Loading → Data Cleaning → Feature Engineering → Model Training → Model Evaluation → Model Selection
```

1. **Data Loading**: 
   - Multi-site CSV files loaded
   - Coordinates and metadata processed
   - Temporal alignment performed

2. **Data Cleaning**:
   - Missing value imputation
   - Outlier detection and handling
   - Data quality validation

3. **Feature Engineering**:
   - 50+ features created
   - Temporal, lag, rolling, meteorological features
   - Feature scaling and normalization

4. **Model Training**:
   - Multiple models trained simultaneously
   - Cross-validation performed
   - Hyperparameters optimized

5. **Model Evaluation**:
   - Comprehensive metrics calculated
   - Model comparison performed
   - Best model selected

### Forecasting Pipeline

```
Input Data → Feature Engineering → Model Prediction → Ensemble → Post-processing → Health Assessment → Output
```

1. **Input Data**:
   - Latest meteorological forecasts
   - Recent measurements
   - Satellite observations

2. **Feature Engineering**:
   - Same features as training
   - Real-time feature calculation
   - Missing data handling

3. **Model Prediction**:
   - Multiple model predictions
   - Uncertainty estimation
   - Quality checks

4. **Ensemble**:
   - Model combination
   - Weighted averaging
   - Confidence intervals

5. **Post-processing**:
   - Forecast formatting
   - Time series generation
   - Visualization preparation

6. **Health Assessment**:
   - WHO/EPA guideline comparison
   - Health status determination
   - Recommendation generation

## Configuration System

### Configuration File (`configs/config.yaml`)

The system uses a centralized YAML configuration file to manage all parameters:

```yaml
project:
  name: "Air Quality Forecasting for Delhi"
  version: "1.0.0"

data:
  raw_data_path: "data/"
  site_coordinates: "lat_lon_sites.txt"
  train_files: ["site_1_train_data.csv", ...]

models:
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5
  
  xgboost:
    n_estimators: 200
    max_depth: 8
    learning_rate: 0.1
  
  lstm:
    sequence_length: 24
    hidden_size: 64
    dropout: 0.2

features:
  lag_features:
    lags: [1, 2, 3, 6, 12, 24]
    variables: ["O3_target", "NO2_target", "T_forecast"]
  
  rolling_features:
    windows: [3, 6, 12, 24]
    statistics: ["mean", "std", "min", "max"]

forecasting:
  forecast_horizons: [24, 48]
  confidence_intervals: [0.05, 0.95]
  ensemble_methods: ["mean", "median"]
```

## System Design Patterns

### 1. Factory Pattern
- **ModelTrainer**: Creates different model types based on configuration
- **FeatureEngineer**: Creates different feature types dynamically

### 2. Strategy Pattern
- **Missing Value Handling**: Different strategies (interpolation, forward-fill, mean)
- **Ensemble Methods**: Different combination strategies (mean, median, weighted)

### 3. Observer Pattern
- **Logging System**: Centralized logging across all components
- **Progress Tracking**: Status updates during long operations

### 4. Configuration Pattern
- **Centralized Configuration**: YAML-based parameter management
- **Environment-specific Settings**: Development/production configurations

### 5. Pipeline Pattern
- **Data Processing Pipeline**: Sequential data transformations
- **Model Training Pipeline**: Standardized training workflow

## Error Handling and Resilience

### Data Quality Issues
- **Missing Data**: Multiple imputation strategies
- **Outliers**: IQR and Z-score detection
- **Data Validation**: Comprehensive quality checks

### Model Failures
- **Individual Model Failures**: Graceful degradation with ensemble
- **Training Failures**: Fallback to simpler models
- **Prediction Errors**: Confidence interval adjustment

### System Resilience
- **Logging**: Comprehensive error tracking
- **Recovery**: Automatic fallback mechanisms
- **Monitoring**: Performance degradation detection

## Performance Optimization

### Data Processing
- **Vectorization**: NumPy operations for speed
- **Chunking**: Process large datasets in chunks
- **Caching**: Cache expensive computations

### Model Training
- **Parallel Processing**: Multi-core utilization
- **Early Stopping**: Prevent overfitting and save time
- **Incremental Learning**: Update models with new data

### Memory Management
- **Data Types**: Optimal data types for memory efficiency
- **Garbage Collection**: Explicit cleanup of large objects
- **Streaming**: Process data in streams for large files

## Security Considerations

### Data Security
- **Input Validation**: Sanitize all inputs
- **File Access**: Restricted file system access
- **Data Anonymization**: No personal data collection

### Model Security
- **Model Validation**: Prevent model poisoning
- **Input Sanitization**: Validate forecast inputs
- **Access Control**: Secure model file access

## Scalability Architecture

### Horizontal Scaling
- **Multi-site Processing**: Independent site processing
- **Model Parallelization**: Train models in parallel
- **Distributed Computing**: Support for cluster deployment

### Vertical Scaling
- **Memory Optimization**: Efficient memory usage patterns
- **CPU Optimization**: Vectorized operations
- **GPU Support**: TensorFlow GPU acceleration

### Storage Scaling
- **Database Integration**: Support for external databases
- **Cloud Storage**: Integration with cloud storage services
- **Data Partitioning**: Time-based data partitioning

## Integration Points

### External APIs
- **Meteorological Services**: Weather forecast APIs
- **Satellite Data**: Earth observation APIs
- **Health Services**: Air quality health APIs

### Monitoring Systems
- **Performance Monitoring**: System performance tracking
- **Model Drift Detection**: Automated model performance monitoring
- **Alert Systems**: Automated alerting for system issues

### Export Formats
- **JSON**: API responses
- **CSV**: Data exports
- **PDF**: Report generation
- **HTML**: Interactive dashboards

## Deployment Architecture

### Development Environment
- **Local Development**: Complete local setup
- **Jupyter Integration**: Interactive development
- **Version Control**: Git integration

### Testing Environment
- **Unit Testing**: Component-level testing
- **Integration Testing**: End-to-end testing
- **Performance Testing**: Load and stress testing

### Production Environment
- **Containerization**: Docker deployment
- **API Services**: FastAPI production API
- **Monitoring**: Production monitoring setup
- **Backup**: Automated backup systems

## Future Architecture Considerations

### Machine Learning Operations (MLOps)
- **Model Versioning**: Track model versions
- **A/B Testing**: Compare model performance
- **Automated Retraining**: Scheduled model updates

### Real-time Processing
- **Streaming Data**: Real-time data ingestion
- **Event-driven Architecture**: Reactive system design
- **Edge Computing**: Local processing capabilities

### Advanced Analytics
- **Causal Inference**: Understanding pollution causes
- **Anomaly Detection**: Unusual pattern detection
- **Optimization**: Pollution control optimization

This architecture provides a robust, scalable, and maintainable foundation for air quality forecasting, with clear separation of concerns and well-defined interfaces between components.
