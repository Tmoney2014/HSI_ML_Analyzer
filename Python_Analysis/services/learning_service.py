import json
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LearningService:
    def train_svm(self, X, y):
        """
        Train Linear SVM.
        """
        print(f"   [LearningService] Training SVM... Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearSVC(dual=False, max_iter=1000)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"   [Error] SVM Training Failed: {e}")
            raise

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"   [LearningService] Accuracy: {acc*100:.2f}%")
        
        return model, acc

    def export_model(self, model, selected_bands, output_path, preprocessing_config=None):
        """
        Export model to JSON for C#.
        """
        weights = model.coef_[0].tolist()
        bias = float(model.intercept_[0])
        
        export_data = {
            "ModelType": "LinearSVM",
            "SelectedBands": [int(b) for b in selected_bands],
            "Weights": weights,
            "Bias": bias,
            "Preprocessing": preprocessing_config if preprocessing_config else {},
            "Note": "Score = sum(w*x) + b. If Score > 0 then Class 1 (Defect)."
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)
            
        print(f"   [LearningService] Model exported to {output_path}")
