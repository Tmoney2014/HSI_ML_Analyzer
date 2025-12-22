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

    def export_model(self, model, selected_bands, output_path, preprocessing_config=None, use_ref=False, mask_rules=None, label_map=None, colors_map=None):
        """
        Export model to JSON for C#.
        """
        # Handle Binary vs Multi-Class Weights
        # Binary (2 classes): coef_ is (1, n_features), intercept_ is (1,)
        # Multi (3+ classes): coef_ is (n_classes, n_features), intercept_ is (n_classes,)
        
        if model.coef_.ndim > 1 and model.coef_.shape[0] > 1:
            # Multi-Class
            weights = model.coef_.tolist() # List of lists
            bias = model.intercept_.tolist() # List of floats
            is_multiclass = True
        else:
            # Binary
            weights = model.coef_[0].tolist()
            bias = float(model.intercept_[0])
            is_multiclass = False
        
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
                    prep_flat["DerivOrder"] = p.get('order', 1) # Export Order
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
            "IsMultiClass": is_multiclass,
            "Preprocessing": prep_flat,
            "Labels": label_map if label_map else {"0": "Normal", "1": "Defect"},
            "Colors": colors_map if colors_map else {"0": "#00FF00", "1": "#FF0000"},
            "Note": "Multi-Class: Weights is [Class][Feature]. Score[c] = sum(W[c]*x) + B[c]. Argmax(Score) -> Class ID." if is_multiclass else "Binary: Score > 0 -> Class 1"
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
            
        print(f"   [LearningService] Model exported to {output_path}")
