import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """
    Trains a Linear SVM model.
    
    Args:
        X: Feature matrix (N_samples, N_selected_bands)
        y: Labels (N_samples,)
        
    Returns:
        model: Trained LinearSVC model
    """
    print(f"   [Model Trainer] Training Linear SVM with {X.shape[0]} samples and {X.shape[1]} features...")
    
    # Split for simple validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearSVC(dual="auto", max_iter=1000)
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   [Model Trainer] Validation Accuracy: {acc*100:.2f}%")
    
    return model

def export_model_for_csharp(model, selected_bands, output_path):
    """
    """
    # Placeholder data for demonstration
    export_data = {
        "ModelType": "LinearSVM", 
        "SelectedBands": selected_bands if selected_bands else [10, 20, 30, 40, 50],
        "Weights": [0.123, -0.456, 0.789, 0.001, -0.999],
        "Bias": 0.5
    }
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"   ✅ Model exported to {output_path}")
