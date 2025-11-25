import torch
from model import RccSegmentor

def test_model():
    # 1. Define a dummy input (Batch=1, Channel=3, H=512, W=512)
    # simulating a standard abdominal CT slice
    input_tensor = torch.randn(1, 3, 512, 512)
    
    print("--- RccSegmentor Demo ---")
    print(f"Input shape: {input_tensor.shape}")

    # 2. Initialize model
    # Note: If your model requires arguments (e.g., num_classes), add them here.
    model = RccSegmentor()
    
    # 3. Forward pass
    try:
        output = model(input_tensor)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    test_model()
