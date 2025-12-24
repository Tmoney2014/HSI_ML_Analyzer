import json
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LearningService:
    def train_model(self, X, y, model_type="Linear SVM", test_ratio=0.2):
        """
        Unified Factory Method for Model Training.
        
        Args:
            X: Data matrix
            y: Labels
            model_type: "Linear SVM", "PLS-DA", "LDA"
            test_ratio: Validation split ratio (0.1 ~ 0.5)
            
        Returns:
            model: Trained sklearn model object
            acc: Validation accuracy (0.0 ~ 1.0)
        """
        # Validate Ratio
        if test_ratio < 0.05 or test_ratio > 0.9: test_ratio = 0.2
        
        print(f"   [LearningService] Training {model_type}... Samples: {X.shape[0]}, Features: {X.shape[1]}, Split: {test_ratio}")
        
        if model_type == "Linear SVM":
            return self._train_svm(X, y, test_ratio)
        elif model_type == "PLS-DA":
            return self._train_pls(X, y, test_ratio)
        elif model_type == "LDA":
            return self._train_lda(X, y, test_ratio)
        else:
            print(f"   [Error] Unknown Model Type: {model_type}, falling back to SVM.")
            return self._train_svm(X, y, test_ratio)

    # --- Private Model Implementations ---

    def _train_svm(self, X, y, test_ratio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        model = LinearSVC(dual=False, max_iter=1000)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"   [LearningService] SVM Accuracy: {acc*100:.2f}%")
            return model, acc
        except Exception as e:
            print(f"   [Error] SVM Training Failed: {e}")
            raise

    def _train_lda(self, X, y, test_ratio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        model = LinearDiscriminantAnalysis() # Defaults are usually good
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"   [LearningService] LDA Accuracy: {acc*100:.2f}%")
            return model, acc
        except Exception as e:
            print(f"   [Error] LDA Training Failed: {e}")
            raise

    def _train_pls(self, X, y, test_ratio):
        """
        PLS-DA Implementation using PLSRegression.
        Note: PLS is a regression algo, so we need One-Hot Encoding for multi-class classification.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        
        # 1. Encode Y (Labels) -> One-Hot
        try:
            n_classes = len(np.unique(y))
            
            # Simple One-Hot Manual
            y_train_mat = np.zeros((len(y_train), n_classes))
            for i, val in enumerate(y_train):
                y_train_mat[i, int(val)] = 1
                
            y_test_mat = np.zeros((len(y_test), n_classes))
            for i, val in enumerate(y_test):
                y_test_mat[i, int(val)] = 1

            # 2. Train PLS
            n_components = min(X.shape[1], 10) 
            model = PLSRegression(n_components=X.shape[1], scale=False) 
            model.fit(X_train, y_train_mat)
            
            # 3. Predict & converting back to class
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            acc = accuracy_score(y_test, y_pred)
            print(f"   [LearningService] PLS-DA Accuracy: {acc*100:.2f}%")
            
            # Monkey Patching for Export Compatibility
            
            # SCENARIO 1: model.coef_ is (n_features, n_targets) -> Standard Sklearn
            # SCENARIO 2: model.coef_ is (n_targets, n_features) -> Some versions/configs?
            # We want export_coef_ to be (n_targets, n_features) for C# (Class, Band).
            
            coef = model.coef_
            n_features_in = X.shape[1]
            n_targets_in = n_classes
            
            if coef.shape[0] == n_features_in and coef.shape[1] == n_targets_in:
                # Shape (Feat, Targ) -> Transpose to (Targ, Feat)
                model.export_coef_ = coef.T
                # Bias = Y_mean - X_mean * Coef (Broadcasting: (T,) - (F,)@(F,T) -> (T,) - (T,) -> (T,))
                # np.dot(x_mean, coef) -> (Targ,)
                bias_correction = lambda xm, ym, c: ym - np.dot(xm, c)
            elif coef.shape[1] == n_features_in and coef.shape[0] == n_targets_in:
                # Shape (Targ, Feat) -> Use as is
                model.export_coef_ = coef
                # Bias = Y_mean - Coef * X_mean
                # np.dot(coef, x_mean) -> (Targ, Feat) @ (Feat,) -> (Targ,)
                bias_correction = lambda xm, ym, c: ym - np.dot(c, xm)
            else:
                 # Unexpected shape (maybe flattened?)
                 print(f"   [Warning] Unexpected PLS Coef Shape: {coef.shape}. features={n_features_in}, targets={n_targets_in}")
                 # Fallback: assume (Feat, Targ) if ambiguous or Try to reshape?
                 # Let's trust standard behavior if ambiguous, but the check above covers logical permutations.
                 model.export_coef_ = coef.T # Default behavior
                 bias_correction = lambda xm, ym, c: ym - np.dot(xm, c)

            # Scikit-Learn Version Compatibility for Means
            x_mean = getattr(model, 'x_mean_', getattr(model, '_x_mean', None))
            y_mean = getattr(model, 'y_mean_', getattr(model, '_y_mean', None))
            
            if x_mean is not None and y_mean is not None:
                try:
                    # Apply detected bias correction logic
                    model.export_intercept_ = bias_correction(x_mean, y_mean, coef)
                except Exception as ex:
                    print(f"   [Error] Bias calc failed: {ex}. Zero bias.")
                    model.export_intercept_ = np.zeros(n_targets_in)
            else:
                 print("   [Warning] PLS Means not found. Assuming Zero Bias.")
                 model.export_intercept_ = np.zeros(n_targets_in)
            
            return model, acc
            
        except Exception as e:
            print(f"   [Error] PLS-DA Training Failed: {e}")
            raise

    # ----------------------------------------

    def export_model(self, model, selected_bands, output_path, preprocessing_config=None, use_ref=False, mask_rules=None, label_map=None, colors_map=None, exclude_rules=None):
        """
        Export model to JSON for C#.
        Handles SVM, PLS-DA, and LDA (Linear Only).
        """
        model_type_name = type(model).__name__
        
        is_linear = True
        weights = []
        bias = []
        is_multiclass = False
        
        # 1. Extract Weights based on Model Type
        if isinstance(model, LinearSVC) or isinstance(model, LinearDiscriminantAnalysis):
            # Both LinearSVC and LDA share similar coef_ structure
            if model.coef_.ndim > 1 and model.coef_.shape[0] > 1:
                weights = model.coef_.tolist()
                bias = model.intercept_.tolist()
                is_multiclass = True
            else:
                # Binary or Single Class
                if model.coef_.ndim == 1:
                     weights = model.coef_.tolist()
                else:
                     weights = model.coef_[0].tolist()
                
                # Ensure bias is float
                if hasattr(model.intercept_, '__len__') and len(model.intercept_) > 0:
                   bias = float(model.intercept_[0])
                else: 
                   bias = float(model.intercept_) # Fallback
                is_multiclass = False

                
        elif isinstance(model, PLSRegression):
            # Use our monkey-patched attributes
            if hasattr(model, 'export_coef_') and hasattr(model, 'export_intercept_'):
                # export_coef_ is (n_classes, n_features)
                if model.export_coef_.shape[0] > 1:
                    weights = model.export_coef_.tolist()
                    bias = model.export_intercept_.tolist()
                    is_multiclass = True
                else: 
                    # Generally PLS-DA with >2 classes is multiclass
                    if model.export_coef_.shape[0] == 1:
                        weights = model.export_coef_[0].tolist()
                        bias = float(model.export_intercept_[0])
                        is_multiclass = False
                    else:
                        weights = model.export_coef_.tolist()
                        bias = model.export_intercept_.tolist()
                        is_multiclass = True
            else:
                 print("   [Warning] PLS Model missing export attributes.")
        
        else:
             print(f"   [Error] Unsupported model type for export: {model_type_name}")
             return
        
        # Convert prep_chain to flat format
        prep_flat = {
            "ApplySG": False, "SGWin": 5, "SGPoly": 2,
            "ApplyDeriv": False, "Gap": 5, "DerivOrder": 1,
            "ApplyL2": False, "ApplyMinMax": False, "ApplySNV": False, "ApplyCenter": False,
            "Mode": "Reflectance" if use_ref else "Raw",
            "MaskRules": mask_rules if mask_rules else "Mean"
        }
        
        if preprocessing_config:
            for step in preprocessing_config:
                name = step.get('name')
                p = step.get('params', {})
                if name == "SG":
                    prep_flat["ApplySG"] = True; prep_flat["SGWin"] = p.get('win', 5); prep_flat["SGPoly"] = p.get('poly', 2)
                elif name == "SimpleDeriv":
                    prep_flat["ApplyDeriv"] = True; prep_flat["Gap"] = p.get('gap', 5); prep_flat["DerivOrder"] = p.get('order', 1)
                elif name == "L2": prep_flat["ApplyL2"] = True
                elif name == "MinMax": prep_flat["ApplyMinMax"] = True
                elif name == "SNV": prep_flat["ApplySNV"] = True
                elif name == "Center": prep_flat["ApplyCenter"] = True
        
        export_data = {
            "ModelType": "LinearModel", # Unified name
            "OriginalType": model_type_name,
            "SelectedBands": [int(b) for b in selected_bands],
            "ExcludeBands": exclude_rules if exclude_rules else "",
            "Weights": weights,
            "Bias": bias,
            "IsMultiClass": is_multiclass,
            "Preprocessing": prep_flat,
            "Labels": label_map if label_map else {"0": "Normal", "1": "Defect"},
            "Colors": colors_map if colors_map else {"0": "#00FF00", "1": "#FF0000"}
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
            
        print(f"   [LearningService] Model exported to {output_path}")
