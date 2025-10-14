#!/usr/bin/env python3
"""
Air Quality Forecasting System - Comprehensive Setup Script
===========================================================

This script will automatically prepare your system to run the Air Quality 
Forecasting project. It performs the following tasks:

1. ✅ System compatibility check
2. 📦 Install Python dependencies 
3. 🔧 Setup project directories
4. 📊 Validate data files
5. ⚙️ Test module imports
6. 🚀 Launch system tests
7. 🎯 Provide next steps

Usage:
    python setup.py

Author: Air Quality Forecasting Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import importlib.util
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AirQualitySetup:
    """Comprehensive setup manager for Air Quality Forecasting System."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_log = []
        self.errors = []
        self.warnings = []
        
        # Change to project directory
        os.chdir(self.project_root)
        
        print(self._color_text("🌍 AIR QUALITY FORECASTING SYSTEM SETUP", Colors.HEADER + Colors.BOLD))
        print("=" * 70)
        print("🎯 Automated Environment Setup and Validation")
        print(f"📁 Project Directory: {self.project_root}")
        print(f"🕒 Setup Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def _color_text(self, text, color):
        """Apply color to text if supported."""
        if platform.system() == "Windows":
            return text  # Windows might not support ANSI colors
        return f"{color}{text}{Colors.ENDC}"
    
    def _log(self, message, level="INFO"):
        """Log setup progress."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        
        if level == "ERROR":
            self.errors.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
    
    def _run_command(self, command, capture_output=True, check=True):
        """Run shell command safely."""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True, 
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            self._log(f"Command failed: {command} - {e}", "ERROR")
            return None
    
    def check_system_compatibility(self):
        """Check system requirements and compatibility."""
        print(f"\n{self._color_text('🔍 STEP 1: System Compatibility Check', Colors.OKBLUE)}")
        print("-" * 50)
        
        # Python version
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self._log("Python 3.8+ required", "ERROR")
            print(f"   {self._color_text('❌ Error: Python 3.8+ required', Colors.FAIL)}")
            return False
        else:
            print(f"   {self._color_text('✅ Python version compatible', Colors.OKGREEN)}")
        
        # Operating System
        system = platform.system()
        print(f"💻 Operating System: {system} {platform.release()}")
        print(f"   {self._color_text('✅ OS detected and supported', Colors.OKGREEN)}")
        
        # Available memory (rough estimate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"💾 Available RAM: {memory_gb:.1f} GB")
            if memory_gb < 4:
                self._log("Low memory detected - may affect performance", "WARNING")
                print(f"   {self._color_text('⚠️  Warning: 4GB+ RAM recommended', Colors.WARNING)}")
            else:
                print(f"   {self._color_text('✅ Sufficient memory available', Colors.OKGREEN)}")
        except ImportError:
            print("💾 Memory check: psutil not available (will be installed)")
        
        # Disk space
        try:
            disk_usage = os.statvfs(self.project_root) if hasattr(os, 'statvfs') else None
            if disk_usage:
                free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
                print(f"💿 Free Disk Space: {free_gb:.1f} GB")
                if free_gb < 2:
                    self._log("Low disk space - may cause issues", "WARNING")
        except:
            print("💿 Disk space: Unable to check")
        
        return True
    
    def setup_directories(self):
        """Create necessary project directories."""
        print(f"\n{self._color_text('📁 STEP 2: Directory Setup', Colors.OKBLUE)}")
        print("-" * 50)
        
        directories = [
            'logs',
            'models', 
            'results',
            'results/plots',
            'results/reports',
            'results/forecasts',
            'data/processed'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"📂 Created: {directory}")
                self._log(f"Created directory: {directory}")
            else:
                print(f"✅ Exists: {directory}")
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print(f"\n{self._color_text('📦 STEP 3: Installing Dependencies', Colors.OKBLUE)}")
        print("-" * 50)
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print(f"{self._color_text('❌ requirements.txt not found', Colors.FAIL)}")
            return False
        
        print("📋 Installing packages from requirements.txt...")
        print("   This may take several minutes depending on your internet connection.")
        
        # Upgrade pip first
        print("\n🔄 Upgrading pip...")
        pip_result = self._run_command(f'"{sys.executable}" -m pip install --upgrade pip')
        
        if pip_result and pip_result.returncode == 0:
            print(f"   {self._color_text('✅ Pip upgraded successfully', Colors.OKGREEN)}")
        else:
            print(f"   {self._color_text('⚠️  Pip upgrade warning (continuing)', Colors.WARNING)}")
        
        # Install requirements
        print("\n📦 Installing project dependencies...")
        install_result = self._run_command(
            f'"{sys.executable}" -m pip install -r "{requirements_file}"'
        )
        
        if install_result and install_result.returncode == 0:
            print(f"   {self._color_text('✅ Dependencies installed successfully', Colors.OKGREEN)}")
            self._log("All dependencies installed")
            return True
        else:
            print(f"   {self._color_text('❌ Some dependencies failed to install', Colors.FAIL)}")
            self._log("Dependency installation had issues", "WARNING")
            
            # Try installing critical packages individually
            critical_packages = [
                'pandas>=1.5.0',
                'numpy>=1.24.0', 
                'scikit-learn>=1.3.0',
                'xgboost>=1.7.0',
                'matplotlib>=3.7.0',
                'pyyaml>=6.0'
            ]
            
            print("\n🚨 Attempting to install critical packages individually...")
            for package in critical_packages:
                print(f"   Installing {package}...")
                result = self._run_command(
                    f'"{sys.executable}" -m pip install "{package}"',
                    capture_output=False
                )
                if result and result.returncode == 0:
                    print(f"     {self._color_text('✅ Success', Colors.OKGREEN)}")
                else:
                    print(f"     {self._color_text('❌ Failed', Colors.FAIL)}")
            
            return True  # Continue even if some packages failed
    
    def validate_data_files(self):
        """Check for required data files."""
        print(f"\n{self._color_text('📊 STEP 4: Data Files Validation', Colors.OKBLUE)}")
        print("-" * 50)
        
        data_dir = self.project_root / 'data'
        if not data_dir.exists():
            print(f"{self._color_text('❌ Data directory not found', Colors.FAIL)}")
            return False
        
        # Check for coordinate file
        coords_file = data_dir / 'lat_lon_sites.txt'
        if coords_file.exists():
            print(f"✅ Site coordinates: {coords_file.name}")
        else:
            print(f"❌ Missing: {coords_file.name}")
            self._log(f"Missing coordinates file: {coords_file}", "ERROR")
        
        # Check for training data files
        train_files = list(data_dir.glob('site_*_train_data.csv'))
        test_files = list(data_dir.glob('site_*_unseen_input_data.csv'))
        
        print(f"📈 Training files found: {len(train_files)}")
        print(f"🔮 Test files found: {len(test_files)}")
        
        if len(train_files) >= 1:
            print(f"   {self._color_text('✅ Training data available', Colors.OKGREEN)}")
            
            # Quick data validation on first file
            try:
                import pandas as pd
                sample_file = train_files[0]
                df = pd.read_csv(sample_file, nrows=5)
                print(f"   📝 Sample file shape: {df.shape}")
                print(f"   📋 Columns: {list(df.columns)}")
                
                # Check for key columns
                required_cols = ['datetime', 'O3_target', 'NO2_target']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"   ⚠️  Missing columns: {missing_cols}")
                    self._log(f"Missing required columns: {missing_cols}", "WARNING")
                else:
                    print(f"   {self._color_text('✅ Required columns present', Colors.OKGREEN)}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not validate data structure: {e}")
        else:
            print(f"   {self._color_text('⚠️  No training data found', Colors.WARNING)}")
            self._log("No training data files found", "WARNING")
        
        return True
    
    def test_imports(self):
        """Test critical package imports."""
        print(f"\n{self._color_text('🧪 STEP 5: Import Testing', Colors.OKBLUE)}")
        print("-" * 50)
        
        # Critical packages
        critical_imports = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('sklearn', None),
            ('matplotlib.pyplot', 'plt'),
            ('yaml', None),
            ('pathlib', None)
        ]
        
        optional_imports = [
            ('xgboost', 'xgb'),
            ('tensorflow', 'tf'),
            ('seaborn', 'sns'),
            ('plotly', None),
            ('streamlit', 'st')
        ]
        
        print("🔍 Testing critical imports...")
        critical_success = True
        for package, alias in critical_imports:
            try:
                if alias:
                    exec(f"import {package} as {alias}")
                else:
                    exec(f"import {package}")
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - FAILED")
                critical_success = False
                self._log(f"Critical import failed: {package}", "ERROR")
        
        print("\n🔍 Testing optional imports...")
        for package, alias in optional_imports:
            try:
                if alias:
                    exec(f"import {package} as {alias}")
                else:
                    exec(f"import {package}")
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ⚠️  {package} - Not available")
                self._log(f"Optional import not available: {package}", "WARNING")
        
        return critical_success
    
    def test_custom_modules(self):
        """Test project's custom modules."""
        print(f"\n{self._color_text('🔧 STEP 6: Custom Module Testing', Colors.OKBLUE)}")
        print("-" * 50)
        
        # Add src to path
        sys.path.insert(0, str(self.project_root / 'src'))
        
        modules_to_test = [
            ('data_preprocessing.data_loader', 'DataLoader'),
            ('feature_engineering.feature_engineer', 'FeatureEngineer'),
            ('models.model_trainer', 'ModelTrainer'),
            ('evaluation.model_evaluator', 'ModelEvaluator'),
            ('forecasting.forecaster', 'AirQualityForecaster'),
            ('utils.helpers', None)
        ]
        
        success_count = 0
        for module_name, class_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                if class_name:
                    getattr(module, class_name)
                print(f"   ✅ {module_name}")
                success_count += 1
            except (ImportError, AttributeError) as e:
                print(f"   ❌ {module_name} - {e}")
                self._log(f"Custom module failed: {module_name} - {e}", "ERROR")
        
        if success_count == len(modules_to_test):
            print(f"\n   {self._color_text('✅ All custom modules loaded successfully', Colors.OKGREEN)}")
            return True
        else:
            print(f"\n   {self._color_text('⚠️  Some modules had issues', Colors.WARNING)}")
            return False
    
    def run_basic_functionality_test(self):
        """Run a basic functionality test."""
        print(f"\n{self._color_text('⚡ STEP 7: Basic Functionality Test', Colors.OKBLUE)}")
        print("-" * 50)
        
        try:
            # Try to initialize core components
            sys.path.insert(0, str(self.project_root / 'src'))
            
            from data_preprocessing.data_loader import DataLoader
            from feature_engineering.feature_engineer import FeatureEngineer
            
            print("🔄 Initializing DataLoader...")
            data_loader = DataLoader(config_path="configs/config.yaml")
            print("   ✅ DataLoader initialized")
            
            print("🔄 Initializing FeatureEngineer...")
            feature_engineer = FeatureEngineer(config_path="configs/config.yaml")
            print("   ✅ FeatureEngineer initialized")
            
            # Try loading site coordinates if available
            try:
                coords_df = data_loader.load_site_coordinates()
                print(f"   ✅ Loaded {len(coords_df)} site coordinates")
            except Exception as e:
                print(f"   ⚠️  Could not load coordinates: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Functionality test failed: {e}")
            self._log(f"Basic functionality test failed: {e}", "ERROR")
            return False
    
    def create_setup_summary(self):
        """Create a setup summary report."""
        print(f"\n{self._color_text('📋 STEP 8: Setup Summary', Colors.OKBLUE)}")
        print("-" * 50)
        
        summary = {
            "setup_time": datetime.now().isoformat(),
            "project_path": str(self.project_root),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": f"{platform.system()} {platform.release()}",
            "setup_log": self.setup_log,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        # Save summary to file
        summary_file = self.project_root / 'setup_summary.json'
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"💾 Setup summary saved to: {summary_file}")
        except Exception as e:
            print(f"⚠️  Could not save setup summary: {e}")
        
        # Print summary
        print(f"\n📊 Setup Statistics:")
        print(f"   ✅ Steps completed successfully")
        print(f"   ⚠️  Warnings: {len(self.warnings)}")
        print(f"   ❌ Errors: {len(self.errors)}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings[:5]:  # Show first 5
                print(f"   - {warning}")
            if len(self.warnings) > 5:
                print(f"   ... and {len(self.warnings) - 5} more")
        
        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors[:5]:  # Show first 5
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more")
    
    def show_next_steps(self):
        """Show next steps to the user."""
        print(f"\n{self._color_text('🚀 NEXT STEPS', Colors.HEADER + Colors.BOLD)}")
        print("=" * 70)
        
        if len(self.errors) == 0:
            print(f"{self._color_text('🎉 Setup completed successfully!', Colors.OKGREEN)}")
            print("\nYou can now run the Air Quality Forecasting System:")
            print()
            print("1️⃣  Run the Interactive Dashboard:")
            print(f"   python start_dashboard.py")
            print(f"   {self._color_text('   (Recommended for beginners)', Colors.OKCYAN)}")
            print()
            print("2️⃣  Run Complete Demo:")
            print(f"   python run_complete_demo.py")
            print()
            print("3️⃣  Explore Jupyter Notebook:")
            print(f"   jupyter notebook notebooks/01_complete_workflow.ipynb")
            print()
            print("4️⃣  Run Quick Test:")
            print(f"   python test_setup.py")
            print()
            print("5️⃣  Use Project Launcher (Interactive Menu):")
            print(f"   python launcher.py")
            
        else:
            print(f"{self._color_text('⚠️  Setup completed with some issues', Colors.WARNING)}")
            print("\nBefore running the project, please:")
            print()
            print("1️⃣  Check the errors above and resolve them")
            print("2️⃣  Try running: python test_setup.py")
            print("3️⃣  Check setup_summary.json for detailed logs")
            print("4️⃣  Refer to SETUP_README.md for troubleshooting")
        
        print(f"\n{self._color_text('📚 Additional Resources:', Colors.OKBLUE)}")
        print("   📖 README.md - Project overview and usage")
        print("   📖 SETUP_README.md - Detailed setup instructions")
        print("   📖 WORKFLOW.md - Development workflow")
        print("   📖 DASHBOARD_README.md - Dashboard usage guide")
        
        print(f"\n{self._color_text('🆘 Need Help?', Colors.OKBLUE)}")
        print("   - Check the logs/ directory for detailed error logs")
        print("   - Run: python check_requirements.py for diagnostics")
        print("   - Review setup_summary.json for full details")
        
        print("\n" + "=" * 70)
        print(f"{self._color_text('Thank you for using the Air Quality Forecasting System!', Colors.OKGREEN)}")
        print("=" * 70)
    
    def run_setup(self):
        """Run the complete setup process."""
        start_time = time.time()
        
        steps = [
            ("System Compatibility Check", self.check_system_compatibility),
            ("Directory Setup", self.setup_directories),
            ("Install Dependencies", self.install_dependencies),
            ("Validate Data Files", self.validate_data_files),
            ("Test Imports", self.test_imports),
            ("Test Custom Modules", self.test_custom_modules),
            ("Basic Functionality Test", self.run_basic_functionality_test),
            ("Create Setup Summary", self.create_setup_summary)
        ]
        
        success = True
        for step_name, step_func in steps:
            try:
                self._log(f"Starting: {step_name}")
                result = step_func()
                if not result:
                    success = False
            except Exception as e:
                print(f"\n{self._color_text(f'❌ Error in {step_name}: {e}', Colors.FAIL)}")
                self._log(f"Error in {step_name}: {e}", "ERROR")
                success = False
        
        # Always show next steps
        self.show_next_steps()
        
        end_time = time.time()
        setup_duration = end_time - start_time
        print(f"\n⏱️  Total setup time: {setup_duration:.1f} seconds")
        
        return success

def main():
    """Main setup function."""
    try:
        setup = AirQualitySetup()
        success = setup.run_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠️  Setup interrupted by user{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"\n\n{Colors.FAIL}❌ Unexpected error during setup: {e}{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
