import sys
import os
sys.path.insert(0, r'C:\Users\Admin\OneDrive\Documents\Major-Deepfake\backend')

def test_model_construction():
    """Test that SpectraShieldCNN can be instantiated."""
    from app.inference.model import SpectraShieldCNN
    model = SpectraShieldCNN()
    assert model is not None


def test_model_forward_pass():
    """Test that model produces output with correct shape."""
    import torch
    from app.inference.model import SpectraShieldCNN
    
    model = SpectraShieldCNN()
    model.eval()
    
    # Input: [batch, 1, 128, 128]
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model(x)
    
    # Output: single logit, shape [batch, 1]
    assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"


def test_model_missing_files():
    """Test that ModelLoader handles missing files gracefully."""
    from app.inference.model_loader import ModelLoader
    
    # Temporarily set model paths to non-existent
    import app.config as config_mod
    original_model_path = config_mod.Config.MODEL_PATH
    original_config_path = config_mod.Config.MODEL_CONFIG_PATH
    
    config_mod.Config.MODEL_PATH = "/nonexistent/best_model.pth"
    config_mod.Config.MODEL_CONFIG_PATH = "/nonexistent/model_config.json"
    
    try:
        # Model should not be loaded
        assert not ModelLoader.is_loaded(), "Model should not be loaded when files missing"
        
        # initialize should return None model and a device
        model, device = ModelLoader.initialize()
        assert model is None, "Model should be None when files missing"
        assert device is not None, "Device should not be None"
    finally:
        # Restore original paths
        config_mod.Config.MODEL_PATH = original_model_path
        config_mod.Config.MODEL_CONFIG_PATH = original_config_path