# 🌍 Air Quality Forecasting Dashboard

An interactive web application for visualizing and demonstrating the Delhi Air Quality Forecasting System. This dashboard provides real-time model training, forecasting, and health impact assessments for ground-level O₃ and NO₂ concentrations.

## 🚀 Quick Start

### Method 1: Using the Launcher Script (Recommended)
```bash
python start_dashboard.py
```

This will automatically:
- ✅ Check and install required packages
- 🔍 Verify data availability  
- 🚀 Launch the dashboard in your browser
- 📱 Open http://localhost:8501 automatically

### Method 2: Direct Streamlit Launch
```bash
# Install required packages first
pip install streamlit plotly pandas numpy scikit-learn xgboost

# Launch the dashboard
streamlit run app.py
```

## 📋 Prerequisites

### Required Python Packages
- `streamlit` - Web application framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting

### Optional Packages (for full functionality)
- `tensorflow` - Deep learning models (LSTM, CNN-LSTM)
- `lightgbm` - Additional ML models
- `catboost` - Additional ML models

### Data Requirements
- Air quality training data files (`data/site_*_train_data.csv`)
- Site coordinates file (`data/lat_lon_sites.txt`)
- Configuration file (`configs/config.yaml`)

## 🎯 Dashboard Features

### 🏠 Main Dashboard
- **📊 Data Overview**: Statistics and site information
- **🗺️ Site Locations**: Interactive map of monitoring sites
- **📈 Historical Trends**: Time series visualization of O₃ and NO₂
- **🤖 Model Training**: Real-time training of multiple ML models
- **🔮 Forecasting**: Generate 24-48 hour predictions
- **🏥 Health Assessment**: Air quality health recommendations

### 🔬 Model Comparison
- **📊 Performance Metrics**: RMSE, MAE, R² comparison
- **🎯 Prediction Analysis**: Scatter plots of predictions vs actual
- **🧠 Model Insights**: Feature importance and model characteristics
- **📈 Evaluation Charts**: Interactive performance visualizations

### ℹ️ System Information
- **🌍 Project Overview**: Comprehensive system documentation
- **🔬 Technical Architecture**: Model and data pipeline details
- **🏥 Health Guidelines**: Air quality standards and recommendations
- **🚀 Usage Instructions**: Step-by-step user guide

## 📱 Using the Dashboard

### Step 1: Load Data
1. Click **"📊 Load Data"** in the sidebar
2. Wait for data to load from all monitoring sites
3. Review data statistics and site information

### Step 2: Train Models
1. Select target pollutant (O₃ or NO₂) 
2. Click **"🚀 Train Models"**
3. Wait for feature engineering and model training
4. Review model performance metrics

### Step 3: Generate Forecasts
1. Adjust forecast horizon (6-48 hours)
2. Click **"Generate Forecasts"**
3. View interactive forecast charts
4. Check health impact assessments

### Step 4: Analyze Results
1. Navigate to **"🔬 Model Comparison"**
2. Compare model performance metrics
3. Examine prediction accuracy plots
4. Review feature importance (for tree-based models)

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Visualization**: Plotly interactive charts
- **Backend**: Python ML pipeline integration
- **Models**: Random Forest, XGBoost, LSTM, CNN-LSTM
- **Data Processing**: pandas, NumPy

### Performance Optimization
- **Caching**: Session state for data and models
- **Progressive Loading**: Step-by-step model training
- **Efficient Rendering**: Optimized chart generation
- **Memory Management**: Selective data loading

### Configuration
The dashboard uses the same configuration file as the main system:
- `configs/config.yaml` - Model parameters and settings
- Automatic fallback for missing configurations
- Environment-specific adaptations

## 🔧 Troubleshooting

### Common Issues

**"Modules not loaded" error:**
```bash
# Ensure you're in the correct directory
cd air_quality_forecast

# Install missing dependencies
pip install -r requirements.txt
```

**"No data could be loaded" warning:**
- Verify data files exist in `data/` directory
- Check file naming matches expected pattern
- Ensure `configs/config.yaml` exists

**Dashboard won't start:**
```bash
# Check if port 8501 is available
netstat -an | findstr 8501

# Try a different port
streamlit run app.py --server.port 8502
```

**Memory issues:**
- Close other applications
- Use fewer monitoring sites for training
- Reduce model complexity in configuration

### Browser Compatibility
- ✅ Chrome/Chromium (Recommended)
- ✅ Firefox
- ✅ Edge
- ⚠️ Safari (Limited testing)

### Port Configuration
Default: http://localhost:8501

To use a different port:
```bash
streamlit run app.py --server.port 8080
```

## 📊 Data Format Requirements

### Training Data Files
Expected format: `site_{ID}_train_data.csv`
```csv
datetime,O3_target,NO2_target,T_forecast,humidity_forecast,...
2019-01-01 00:00:00,45.2,38.7,15.3,65.8,...
```

### Site Coordinates
File: `data/lat_lon_sites.txt`
```
Site,Latitude,Longitude
1,28.6139,77.2090
2,28.5355,77.3910
```

## 🎨 Customization

### Styling
Modify the CSS in `app.py` to change colors, fonts, and layout:
```python
st.markdown("""
<style>
    .main-header {
        color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

### Adding New Features
1. Create new page functions in `AirQualityDashboard` class
2. Add navigation options in sidebar
3. Implement visualization components
4. Test with sample data

## 📈 Performance Monitoring

### System Metrics
- **Memory Usage**: Monitor RAM consumption
- **CPU Usage**: Track processing load
- **Network**: Check data loading speed
- **Browser**: Monitor rendering performance

### Dashboard Metrics
- **Load Time**: Initial data loading
- **Training Time**: Model training duration  
- **Forecast Time**: Prediction generation
- **Render Time**: Chart display speed

## 🤝 Contributing

### Adding New Visualizations
1. Create new Plotly chart functions
2. Add to appropriate dashboard page
3. Include interactive controls
4. Test with multiple data scenarios

### Improving Performance
1. Optimize data loading strategies
2. Implement better caching mechanisms
3. Reduce chart rendering complexity
4. Add progressive loading features

## 📞 Support

### Getting Help
1. **Check Documentation**: Review README and inline comments
2. **Configuration Issues**: Verify `configs/config.yaml` settings
3. **Data Problems**: Validate data file formats and locations
4. **Performance Issues**: Monitor system resources

### Error Reporting
When reporting issues, include:
- Error messages and stack traces
- System specifications (OS, Python version)
- Data size and characteristics
- Browser type and version

## 🔗 Related Files

- `app.py` - Main dashboard application
- `start_dashboard.py` - Launcher script
- `requirements.txt` - Package dependencies
- `configs/config.yaml` - System configuration
- `src/` - Core ML pipeline modules

---

**🌟 Built for better air quality monitoring and public health protection!**

For more information about the underlying ML system, see the main [README.md](README.md) file.
