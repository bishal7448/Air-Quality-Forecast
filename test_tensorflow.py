#!/usr/bin/env python3
"""
Quick test to verify TensorFlow and deep learning models are available.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_tensorflow_models():
    """Test that TensorFlow models can be created."""
    print("🧪 Testing TensorFlow model availability...")
    
    try:
        from models.model_trainer import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer(config_path="configs/config.yaml")
        
        # Check if TensorFlow is available
        from models.model_trainer import TF_AVAILABLE
        print(f"✅ TensorFlow available: {TF_AVAILABLE}")
        
        if TF_AVAILABLE:
            # Test LSTM model creation
            try:
                lstm_model = trainer.create_lstm_model(input_shape=(24, 10))
                print("✅ LSTM model created successfully")
                print(f"   Model type: {type(lstm_model).__name__}")
                
                # Test CNN-LSTM model creation
                cnn_lstm_model = trainer.create_cnn_lstm_model(input_shape=(24, 10))
                print("✅ CNN-LSTM model created successfully")
                print(f"   Model type: {type(cnn_lstm_model).__name__}")
                
            except Exception as e:
                print(f"❌ Error creating deep learning models: {e}")
                return False
        else:
            print("❌ TensorFlow not available")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing TensorFlow models: {e}")
        return False

def main():
    """Run TensorFlow availability test."""
    print("🚀 Testing TensorFlow Model Availability")
    print("=" * 50)
    
    success = test_tensorflow_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TensorFlow models are fully available!")
        print("\nAll model types now supported:")
        print("✅ Random Forest")
        print("✅ XGBoost") 
        print("✅ LSTM")
        print("✅ CNN-LSTM")
    else:
        print("❌ TensorFlow models are not available")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
