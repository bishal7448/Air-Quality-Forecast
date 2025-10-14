#!/usr/bin/env python3
"""
Air Quality Dashboard Launcher
==============================

Simple script to launch the Streamlit web dashboard for the 
Air Quality Forecasting System.

This script will:
1. Check for required dependencies
2. Install missing packages if needed
3. Launch the Streamlit dashboard
4. Open the browser automatically
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_and_install_requirements():
    """Check and install required packages."""
    print("🔍 Checking required packages...")
    
    # Essential packages for the dashboard
    required_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing {len(missing_packages)} missing packages...")
        installation_success = True
        for package in missing_packages:
            if not install_package(package):
                installation_success = False
        return installation_success
    else:
        print("✅ All required packages are already installed!")
        return True

def check_data_availability():
    """Check if data files are available."""
    data_dir = Path("data")
    config_file = Path("configs/config.yaml")
    
    if not data_dir.exists():
        print("⚠️ Data directory not found. Creating...")
        data_dir.mkdir(exist_ok=True)
        return False
    
    if not config_file.exists():
        print("⚠️ Configuration file not found.")
        return False
    
    # Check for at least one data file
    data_files = list(data_dir.glob("site_*.csv"))
    if not data_files:
        print("⚠️ No site data files found in data directory.")
        print("   Please ensure you have site training data files.")
        return False
    
    print(f"✅ Found {len(data_files)} data files")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n🚀 Launching Air Quality Forecasting Dashboard...")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    app_path = current_dir / "app.py"
    
    if not app_path.exists():
        print("❌ Dashboard app.py not found!")
        return False
    
    try:
        # Launch streamlit
        print("🌍 Starting Streamlit server...")
        print("📱 Dashboard will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("\n⚠️ To stop the server, press Ctrl+C in this terminal")
        print("=" * 60)
        
        # Small delay before opening browser
        time.sleep(2)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

def show_welcome_message():
    """Display welcome message and instructions."""
    print("""
🌍 AIR QUALITY FORECASTING DASHBOARD LAUNCHER
=============================================

This will launch the interactive web dashboard for the Delhi Air Quality 
Forecasting System. The dashboard includes:

📊 Interactive Data Exploration
🤖 Real-time Model Training & Evaluation  
🔮 24-48 Hour Pollution Forecasts
🏥 Health Impact Assessments
🗺️ Multi-site Monitoring Visualization

==============================================
""")

def main():
    """Main launcher function."""
    show_welcome_message()
    
    # Check and install requirements
    if not check_and_install_requirements():
        print("❌ Failed to install required packages. Please install manually.")
        return
    
    # Check data availability
    data_available = check_data_availability()
    if not data_available:
        print("\n⚠️ Some data files are missing, but you can still explore the dashboard.")
        print("   The system will show demo functionality where data is available.")
    
    # Launch dashboard
    success = launch_dashboard()
    
    if success:
        print("\n✅ Dashboard session completed successfully!")
    else:
        print("\n❌ Dashboard launch failed!")

if __name__ == "__main__":
    main()

# Note: If you encounter issues installing packages, try running the following commands from the main folder:
# python -c "import streamlit; print('streamlit: OK')"
# python -c "import plotly; print('plotly: OK')"
# python -c "import xgboost; print('xgboost: OK')"
