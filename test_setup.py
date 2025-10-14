#!/usr/bin/env python3
"""
Test script to verify the air quality forecasting project setup.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_imports():
    """Test if all required packages and modules can be imported."""
    print("🧪 Testing package imports...")
    
    # Test basic packages
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yaml
        import xgboost as xgb
        import sklearn
        print("✅ Core data science packages imported successfully")
    except ImportError as e:
        print(f"❌ Error importing core packages: {e}")
        return False
    
    # Test our modules
    try:
        from data_preprocessing.data_loader import DataLoader
        from feature_engineering.feature_engineer import FeatureEngineer
        from models.model_trainer import ModelTrainer
        from evaluation.model_evaluator import ModelEvaluator
        from forecasting.forecaster import AirQualityForecaster
        from utils.helpers import ConfigManager, DataUtils
        print("✅ Custom modules imported successfully")
    except ImportError as e:
        print(f"❌ Error importing custom modules: {e}")
        return False
    
    return True

def test_data_availability():
    """Test if all required data files are available."""
    print("\n📊 Testing data availability...")
    
    data_dir = project_root / 'data'
    required_files = [
        'lat_lon_sites.txt',
        'site_1_train_data.csv',
        'site_1_unseen_input_data.csv'
    ]
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    # Count all site files
    train_files = list(data_dir.glob('site_*_train_data.csv'))
    test_files = list(data_dir.glob('site_*_unseen_input_data.csv'))
    
    print(f"📈 Found {len(train_files)} training files")
    print(f"🔮 Found {len(test_files)} test files")
    
    return len(train_files) >= 7 and len(test_files) >= 7

def test_config():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    config_path = project_root / 'configs' / 'config.yaml'
    if not config_path.exists():
        print("❌ Configuration file missing")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['project', 'data', 'models', 'features']
        for section in required_sections:
            if section in config:
                print(f"✅ Config section '{section}' found")
            else:
                print(f"❌ Config section '{section}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of our modules."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test data loader
        from data_preprocessing.data_loader import DataLoader
        data_loader = DataLoader(config_path="configs/config.yaml")
        print("✅ DataLoader initialized successfully")
        
        # Test feature engineer
        from feature_engineering.feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer(config_path="configs/config.yaml")
        print("✅ FeatureEngineer initialized successfully")
        
        # Test model trainer (without TensorFlow for now)
        from models.model_trainer import ModelTrainer
        trainer = ModelTrainer(config_path="configs/config.yaml")
        print("✅ ModelTrainer initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        return False

def test_data_loading():
    """Test actual data loading with a small sample."""
    print("\n📁 Testing data loading...")
    
    try:
        from data_preprocessing.data_loader import DataLoader
        
        # Initialize data loader
        data_loader = DataLoader(config_path="configs/config.yaml")
        
        # Load site coordinates
        coords_df = data_loader.load_site_coordinates()
        print(f"✅ Loaded coordinates for {len(coords_df)} sites")
        
        # Load training data for site 1
        site_1_data = data_loader.load_training_data(site_id=1)
        print(f"✅ Loaded Site 1 training data: {site_1_data.shape}")
        print(f"   Columns: {list(site_1_data.columns)}")
        print(f"   Date range: {site_1_data['datetime'].min()} to {site_1_data['datetime'].max()}")
        
        # Get data summary
        summary = data_loader.get_data_summary(site_1_data)
        print(f"✅ Generated data summary: {summary['shape'][0]:,} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        print(f"   Exception details: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Air Quality Forecasting System Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Availability", test_data_availability),
        ("Configuration", test_config),
        ("Basic Functionality", test_basic_functionality),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/01_complete_workflow.ipynb")
        print("2. Or execute the workflow programmatically")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
