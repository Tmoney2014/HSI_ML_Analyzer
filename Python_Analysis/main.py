import sys
import os
import argparse

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_hsi_data
from utils.band_selection import select_best_bands
from utils.model_trainer import train_model, export_model_for_csharp

def main():
    parser = argparse.ArgumentParser(description="HSI Analysis & Model Export Tool")
    parser.add_argument("--data_path", type=str, default="./data/sample_cube.hdr", help="Path to HSI header file")
    parser.add_argument("--output_path", type=str, default="./output/model_config.json", help="Path to save C# model config")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("🚀 [Step 1] Initializing HSI Analysis Pipeline...")
    
    # 1. Load Data
    cube, wavelengths = load_hsi_data(args.data_path)
    
    # 2. Band Selection
    # For extraction, we flatten the cube
    n_bands = 5
    selected_bands = select_best_bands(cube, n_bands=n_bands)
    
    # 3. Train Model
    print("🚀 [Step 3] Training SVM Model...")
    # Extract features based on selected bands
    # Shape: (H, W, B) -> (H*W, B_selected)
    h, w, b = cube.shape
    flat_cube = cube.reshape(-1, b)
    X_subset = flat_cube[:, selected_bands]
    
    # Generate synthetic labels for demonstration if no label file is provided
    # In a real scenario, you would load a label mask image here using args.label_path
    print("   [Info] Using synthetic labels for demonstration.")
    y_labels = np.random.randint(0, 2, size=h*w) 
    
    model = train_model(X_subset, y_labels)
    
    # 4. Export for C#
    print(f"🚀 [Step 4] Exporting model to {args.output_path}...")
    export_model_for_csharp(model, selected_bands, args.output_path)
    
    print("✅ Pipeline completed successfully.")
    print(f"✅ Configuration saved! You can now load '{args.output_path}' in C#.")

if __name__ == "__main__":
    main()
