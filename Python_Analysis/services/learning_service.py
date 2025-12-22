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

    def export_model(self, model, selected_bands, output_path, preprocessing_config=None, use_ref=False, mask_rules=None):
        """
        Export model to JSON for C#.
        """
        weights = model.coef_[0].tolist()
        bias = float(model.intercept_[0])
        
        # Convert prep_chain to flat format for C# compatibility
        prep_flat = {
            "ApplySG": False,
            "SGWin": 5,
            "SGPoly": 2,
            "ApplyDeriv": False,
            "DerivOrder": 1,
            "ApplyL2": False,
            "ApplyMinMax": False,
            "ApplySNV": False,
            "ApplyCenter": False,
            "Mode": "Reflectance" if use_ref else "Raw",
            "MaskRules": mask_rules if mask_rules else "Mean"
        }
        
        if preprocessing_config:
            for step in preprocessing_config:
                name = step.get('name')
                p = step.get('params', {})
                if name == "SG":
                    prep_flat["ApplySG"] = True
                    prep_flat["SGWin"] = p.get('win', 5)
                    prep_flat["SGPoly"] = p.get('poly', 2)
                elif name == "SimpleDeriv":
                    prep_flat["ApplyDeriv"] = True
                    prep_flat["Gap"] = p.get('gap', 5) # Export Gap size
                elif name == "L2":
                    prep_flat["ApplyL2"] = True
                elif name == "MinMax":
                    prep_flat["ApplyMinMax"] = True
                elif name == "SNV":
                    prep_flat["ApplySNV"] = True
                elif name == "Center":
                    prep_flat["ApplyCenter"] = True
        
        export_data = {
            "ModelType": "LinearSVM",
            "SelectedBands": [int(b) for b in selected_bands],
            "Weights": weights,
            "Bias": bias,
            "Preprocessing": prep_flat,
            "Note": "판정 공식: Score = sum(w*x) + b. 만약 Score > 0 이면 '불량(Class 1)' 입니다."
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
            
        print(f"   [LearningService] Model exported to {output_path}")
