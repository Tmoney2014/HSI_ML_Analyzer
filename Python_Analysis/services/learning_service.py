import json
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifier, LogisticRegression  # AI가 수정함: Ridge, LogReg 추가

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
try:  # AI가 수정함: matplotlib 미설치 환경에서도 학습/내보내기 동작 보장
    import matplotlib.pyplot as plt  # AI가 수정함: 플롯 생성용 선택 import
except ImportError:  # AI가 수정함: matplotlib 없을 때 fallback 처리
    plt = None  # AI가 수정함: plot 비활성화 플래그
try:  # AI가 수정함: seaborn 선택 의존성도 안전하게 로드
    import seaborn as sns  # AI가 수정함: 더 나은 플롯 스타일 옵션
except ImportError:  # AI가 수정함: seaborn 없을 때 fallback 처리
    sns = None  # AI가 수정함: seaborn 비활성화 플래그
from config import get as cfg_get  # AI가 수정함: 설정 파일 사용

_HSI_CAMERA_BANDWIDTH_BPS = 940 * 1_000_000 / 8  # AI가 수정함: GigE Vision 940Mbps → bytes/sec
_HSI_PIXEL_BYTES = 2  # AI가 수정함: uint16 per pixel
_HSI_FRAME_WIDTH = 640  # AI가 수정함: pixels per spatial line
# TODO: move to config in v2  # AI가 수정함: 고정 FPS 추정값은 후속 버전에서 설정화

class LearningService:
    def train_model(self, X, y, model_type="Linear SVM", test_ratio=0.2, log_callback=None):
        """
        Unified Factory Method for Model Training.
        
        Args:
            X: Data matrix
            y: Labels
            model_type: "Linear SVM", "PLS-DA", "LDA"
            test_ratio: Validation split ratio (0.1 ~ 0.5)
            log_callback: Function(str) to print logs to UI
        """
        # Helper logging
        def log(msg):
            if log_callback: log_callback(msg)
            else: print(msg)

        # Validate Ratio
        if test_ratio < 0.05 or test_ratio > 0.9: test_ratio = 0.2
        
        log(f"   [LearningService] Training {model_type}... Samples: {X.shape[0]}, Features: {X.shape[1]}, Split: {test_ratio}")
        
        if model_type == "Linear SVM":
            return self._train_svm(X, y, test_ratio, log)
        elif model_type == "PLS-DA":
            return self._train_pls(X, y, test_ratio, log)
        elif model_type == "LDA":
            return self._train_lda(X, y, test_ratio, log)
        elif model_type == "Ridge Classifier":  # AI가 수정함: Ridge Classifier 추가
            return self._train_ridge(X, y, test_ratio, log)  # AI가 수정함: Ridge 학습 경로 연결
        elif model_type == "Logistic Regression":  # AI가 수정함: Logistic Regression 추가
            return self._train_logistic(X, y, test_ratio, log)  # AI가 수정함: LogReg 학습 경로 연결
        else:
            log(f"   [Error] Unknown Model Type: {model_type}, falling back to SVM.")
            return self._train_svm(X, y, test_ratio, log)

    # --- Private Model Implementations ---

    def _train_svm(self, X, y, test_ratio, log):
        # AI가 수정함: V2-STR — stratify=y 추가 (클래스 비율 보존), 샘플 부족 시 fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
        except ValueError:
            log("⚠️ [SVM] Stratified split 불가 (클래스 샘플 부족). 일반 split으로 진행합니다.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        model = LinearSVC(dual=False, max_iter=cfg_get('model', 'svm_max_iter', 1000), class_weight='balanced')  # AI가 수정함: class_weight='balanced' 추가 — 불균형 클래스 편향 방지
        try:
            model.fit(X_train, y_train)
            
            # AI가 수정함: 과적합 지표 추가
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            gap = (train_acc - test_acc) * 100
            f1 = f1_score(y_test, y_test_pred, average='macro')
            
            # 과적합 경고
            gap_warning = " ⚠️ 과적합 의심" if gap > 5 else ""
            log(f"   [SVM] Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap:.1f}%{gap_warning}")
            log(f"   [SVM] F1 (Macro): {f1:.3f}")
            
            metrics = {
                "TrainAccuracy": round(train_acc * 100, 2),
                "TestAccuracy": round(test_acc * 100, 2),
                "F1Score": round(f1, 4),
                "TotalSamples": int(len(y)),
                "TestSplit": test_ratio
            }
            return model, metrics
        except Exception as e:
            log(f"   [Error] SVM Training Failed: {e}")
            raise

    def _train_lda(self, X, y, test_ratio, log):
        # AI가 수정함: V2-STR — stratify=y 추가 (클래스 비율 보존), 샘플 부족 시 fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
        except ValueError:
            log("⚠️ [LDA] Stratified split 불가 (클래스 샘플 부족). 일반 split으로 진행합니다.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        model = LinearDiscriminantAnalysis() # Defaults are usually good
        try:
            model.fit(X_train, y_train)
            
            # AI가 수정함: 과적합 지표 추가
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            gap = (train_acc - test_acc) * 100
            f1 = f1_score(y_test, y_test_pred, average='macro')
            
            gap_warning = " ⚠️ 과적합 의심" if gap > 5 else ""
            log(f"   [LDA] Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap:.1f}%{gap_warning}")
            log(f"   [LDA] F1 (Macro): {f1:.3f}")
            
            metrics = {
                "TrainAccuracy": round(train_acc * 100, 2),
                "TestAccuracy": round(test_acc * 100, 2),
                "F1Score": round(f1, 4),
                "TotalSamples": int(len(y)),
                "TestSplit": test_ratio
            }
            return model, metrics
        except Exception as e:
            log(f"   [Error] LDA Training Failed: {e}")
            raise

    def _train_ridge(self, X, y, test_ratio, log):  # AI가 수정함: Ridge Classifier 학습 추가
        # AI가 수정함: _train_svm 패턴 동일 — stratify=y 추가, 샘플 부족 시 fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)  # AI가 수정함: stratified split 우선 적용
        except ValueError:
            log("⚠️ [Ridge] Stratified split 불가 (클래스 샘플 부족). 일반 split으로 진행합니다.")  # AI가 수정함: 경고 로그 추가
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)  # AI가 수정함: fallback split 적용
        model = RidgeClassifier(class_weight='balanced')  # AI가 수정함: class_weight='balanced' 불균형 클래스 대응
        try:
            model.fit(X_train, y_train)  # AI가 수정함: Ridge 학습 수행

            y_train_pred = model.predict(X_train)  # AI가 수정함: train 예측값 계산
            y_test_pred = model.predict(X_test)  # AI가 수정함: test 예측값 계산

            train_acc = accuracy_score(y_train, y_train_pred)  # AI가 수정함: train 정확도 계산
            test_acc = accuracy_score(y_test, y_test_pred)  # AI가 수정함: test 정확도 계산
            gap = (train_acc - test_acc) * 100  # AI가 수정함: 과적합 gap 계산
            f1 = f1_score(y_test, y_test_pred, average='macro')  # AI가 수정함: macro F1 계산

            gap_warning = " ⚠️ 과적합 의심" if gap > 5 else ""  # AI가 수정함: gap 경고 문자열 생성
            log(f"   [Ridge] Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap:.1f}%{gap_warning}")  # AI가 수정함: Ridge 성능 로그 출력
            log(f"   [Ridge] F1 (Macro): {f1:.3f}")  # AI가 수정함: Ridge F1 로그 출력

            metrics = {  # AI가 수정함: Ridge metrics 구성
                "TrainAccuracy": round(train_acc * 100, 2),  # AI가 수정함: train 정확도 저장
                "TestAccuracy": round(test_acc * 100, 2),  # AI가 수정함: test 정확도 저장
                "F1Score": round(f1, 4),  # AI가 수정함: F1 저장
                "TotalSamples": int(len(y)),  # AI가 수정함: 총 샘플 수 저장
                "TestSplit": test_ratio  # AI가 수정함: test 비율 저장
            }  # AI가 수정함: Ridge metrics 종료
            return model, metrics  # AI가 수정함: Ridge 모델과 메트릭 반환
        except Exception as e:
            log(f"   [Error] Ridge Training Failed: {e}")  # AI가 수정함: Ridge 학습 실패 로그
            raise  # AI가 수정함: 예외 재발생

    def _train_logistic(self, X, y, test_ratio, log):  # AI가 수정함: Logistic Regression 학습 추가
        # AI가 수정함: _train_svm 패턴 동일 — stratify=y 추가, 샘플 부족 시 fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)  # AI가 수정함: stratified split 우선 적용
        except ValueError:
            log("⚠️ [LogReg] Stratified split 불가 (클래스 샘플 부족). 일반 split으로 진행합니다.")  # AI가 수정함: 경고 로그 추가
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)  # AI가 수정함: fallback split 적용
        import sklearn  # AI가 수정함: 버전 분기용
        from packaging.version import Version as _V  # AI가 수정함: 버전 비교용
        _kwargs = {  # AI가 수정함: LogisticRegression 생성 kwargs 구성
            'max_iter': cfg_get('model', 'logistic_max_iter', 1000),  # AI가 수정함: 반복 횟수 설정값 사용
            'class_weight': 'balanced',  # AI가 수정함: 불균형 클래스 대응
            'solver': 'lbfgs',  # AI가 수정함: multinomial 호환 solver 지정
        }  # AI가 수정함: kwargs 초기화 종료
        if _V(sklearn.__version__) < _V('1.6.0'):  # AI가 수정함: sklearn < 1.6 에서만 multi_class 명시
            _kwargs['multi_class'] = 'multinomial'  # AI가 수정함: multinomial 고정
        model = LogisticRegression(**_kwargs)  # AI가 수정함: 버전 분기 적용
        try:
            model.fit(X_train, y_train)  # AI가 수정함: Logistic Regression 학습 수행
            assert (  # AI가 수정함: multinomial weight shape 검증
                model.coef_.shape[0] == len(np.unique(y))  # AI가 수정함: multinomial weight shape 검증
            ), f"LogReg coef shape mismatch: {model.coef_.shape[0]} != {len(np.unique(y))}"  # AI가 수정함: multinomial weight shape 검증

            y_train_pred = model.predict(X_train)  # AI가 수정함: train 예측값 계산
            y_test_pred = model.predict(X_test)  # AI가 수정함: test 예측값 계산

            train_acc = accuracy_score(y_train, y_train_pred)  # AI가 수정함: train 정확도 계산
            test_acc = accuracy_score(y_test, y_test_pred)  # AI가 수정함: test 정확도 계산
            gap = (train_acc - test_acc) * 100  # AI가 수정함: 과적합 gap 계산
            f1 = f1_score(y_test, y_test_pred, average='macro')  # AI가 수정함: macro F1 계산

            gap_warning = " ⚠️ 과적합 의심" if gap > 5 else ""  # AI가 수정함: gap 경고 문자열 생성
            log(f"   [LogReg] Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap:.1f}%{gap_warning}")  # AI가 수정함: LogReg 성능 로그 출력
            log(f"   [LogReg] F1 (Macro): {f1:.3f}")  # AI가 수정함: LogReg F1 로그 출력

            metrics = {  # AI가 수정함: LogReg metrics 구성
                "TrainAccuracy": round(train_acc * 100, 2),  # AI가 수정함: train 정확도 저장
                "TestAccuracy": round(test_acc * 100, 2),  # AI가 수정함: test 정확도 저장
                "F1Score": round(f1, 4),  # AI가 수정함: F1 저장
                "TotalSamples": int(len(y)),  # AI가 수정함: 총 샘플 수 저장
                "TestSplit": test_ratio  # AI가 수정함: test 비율 저장
            }  # AI가 수정함: LogReg metrics 종료
            return model, metrics  # AI가 수정함: LogReg 모델과 메트릭 반환
        except Exception as e:
            log(f"   [Error] Logistic Training Failed: {e}")  # AI가 수정함: Logistic 학습 실패 로그
            raise  # AI가 수정함: 예외 재발생

    def _train_pls(self, X, y, test_ratio, log):
        """
        PLS-DA Implementation using PLSRegression.
        Note: PLS is a regression algo, so we need One-Hot Encoding for multi-class classification.
        """
        # AI가 수정함: V2-STR — stratify=y 추가 (클래스 비율 보존), 샘플 부족 시 fallback
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
        except ValueError:
            log("⚠️ [PLS-DA] Stratified split 불가 (클래스 샘플 부족). 일반 split으로 진행합니다.")
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
            # AI가 수정함: n_components = min(X.shape[1], 10) 로 과도한 모델 방지
            n_components = min(X.shape[1], 10)
            model = PLSRegression(n_components=n_components, scale=False) 
            model.fit(X_train, y_train_mat)
            
            # 3. Predict & converting back to class
            y_train_probs = model.predict(X_train)
            y_train_pred = np.argmax(y_train_probs, axis=1)
            y_test_probs = model.predict(X_test)
            y_test_pred = np.argmax(y_test_probs, axis=1)
            
            # AI가 수정함: 과적합 지표 추가
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            gap = (train_acc - test_acc) * 100
            f1 = f1_score(y_test, y_test_pred, average='macro')
            
            gap_warning = " ⚠️ 과적합 의심" if gap > 5 else ""
            log(f"   [PLS-DA] Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}% | Gap: {gap:.1f}%{gap_warning}")
            log(f"   [PLS-DA] F1 (Macro): {f1:.3f}")
            
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

            # AI가 수정함: sklearn 버전 독립적 bias 계산 — L-1
            # sklearn 1.0+에서 x_mean_/_x_mean/x_offset_ 모두 제거됨.
            # X/Y 평균을 직접 계산하여 버전 종속성 완전 제거.
            x_mean = X_train.mean(axis=0)          # (n_features,)
            y_mean = y_train_mat.mean(axis=0)      # (n_targets,)
            try:
                model.export_intercept_ = bias_correction(x_mean, y_mean, coef)
            except Exception as ex:
                print(f"   [Error] Bias calc failed: {ex}. Zero bias.")
                model.export_intercept_ = np.zeros(n_targets_in)
            
            metrics = {
                "TrainAccuracy": round(train_acc * 100, 2),
                "TestAccuracy": round(test_acc * 100, 2),
                "F1Score": round(f1, 4),
                "TotalSamples": int(len(y)),
                "TestSplit": test_ratio
            }
            return model, metrics
            
        except Exception as e:
            print(f"   [Error] PLS-DA Training Failed: {e}")
            raise

    # ----------------------------------------

    def export_model(self, model, selected_bands, output_path, preprocessing_config=None, processing_mode="Raw", mask_rules=None, label_map=None, colors_map=None, exclude_rules=None, threshold=None, mean_spectrum=None, spa_scores=None, metrics=None, model_name="model", description="", total_bands=None):
        """
        Export model to JSON for C#.
        Handles SVM, PLS-DA, and LDA (Linear Only).
        Metrics: Optional performance dict from train_model
        """
        model_type_name = type(model).__name__
        
        # AI가 추가함: Timestamp for Tracking
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        is_linear = True
        weights = []
        bias = []
        is_multiclass = False
        
        # AI가 수정함: V2-SB — selected_bands 정규화 (sort + dedup)
        # SPA는 비정렬 순서로 반환 가능 → 정렬 후 Weights 열도 동일 순서로 재정렬 필요
        original_bands = list(selected_bands)
        selected_bands = sorted(set(int(b) for b in selected_bands))

        # AI가 수정함: V2-RRB — total_bands None일 때 mean_spectrum 길이로 자동 유추
        if total_bands is None and mean_spectrum is not None:
            total_bands = len(mean_spectrum)

        # 1. Extract Weights based on Model Type
        if isinstance(model, (LinearSVC, LinearDiscriminantAnalysis, RidgeClassifier, LogisticRegression)):  # AI가 수정함: Ridge, LogReg 추가 — isinstance 체인 확장
            # Both LinearSVC and LDA share similar coef_ structure
            if model.coef_.ndim > 1 and model.coef_.shape[0] > 1:
                raw_weights = np.array(model.coef_)  # shape: (n_classes, n_features_original)
                # AI가 수정함: V2-SB — SelectedBands 정렬 시 Weights 열 순서도 같이 재정렬
                # original_bands 순서 → selected_bands(정렬) 순서로 열 인덱스 매핑
                if original_bands != selected_bands:
                    band_to_col = {int(b): i for i, b in enumerate(original_bands)}
                    col_order = [band_to_col[b] for b in selected_bands if b in band_to_col]
                    raw_weights = raw_weights[:, col_order]
                weights = raw_weights.tolist()
                bias = model.intercept_.tolist()
                is_multiclass = True
                # AI가 수정함: assert → 조건부 경고로 교체 — n_components 커스텀 설정 시 크래시 방지
                if isinstance(model, LinearDiscriminantAnalysis):
                    n_classes_actual = model.coef_.shape[0]
                    n_classes_expected = len(np.unique(model.classes_))
                    if n_classes_actual != n_classes_expected:
                        import warnings
                        warnings.warn(
                            f"LDA coef_ shape mismatch: {n_classes_actual} rows != "
                            f"{n_classes_expected} classes. C# Weights 해석 불일치 가능.",
                            RuntimeWarning
                        )
            else:
                # Binary or Single Class
                if model.coef_.ndim == 1:
                     raw_coef = model.coef_
                else:
                     raw_coef = model.coef_[0]
                # AI가 수정함: V2-SB — binary 케이스도 Weights 열 재정렬
                if original_bands != selected_bands:
                    band_to_col = {int(b): i for i, b in enumerate(original_bands)}
                    col_order = [band_to_col[b] for b in selected_bands if b in band_to_col]
                    raw_coef = raw_coef[col_order]
                weights = raw_coef.tolist()
                
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
            "ApplySG": False, "SGWin": 5, "SGPoly": 2, "SGDeriv": 0,
            "ApplyDeriv": False, "Gap": 5, "DerivOrder": 1,
            "ApplyL2": False, "ApplyMinMax": False, "ApplySNV": False,
            "ApplyCenter": False, "ApplyMinSub": False,
            "ApplyAbsorbance": (processing_mode == "Absorbance"),
            "Mode": processing_mode,
            "MaskRules": mask_rules if mask_rules else "Mean",
            "Threshold": str(threshold) if threshold is not None else "0.0"
        }
        
        if preprocessing_config:
            for step in preprocessing_config:
                name = step.get('name')
                p = step.get('params', {})
                if name == "SG":
                    # AI가 수정함: SGDeriv 추가 — C# SavitzkyGolayProcessor 패리티
                    prep_flat["ApplySG"] = True; prep_flat["SGWin"] = p.get('win', 5); prep_flat["SGPoly"] = p.get('poly', 2); prep_flat["SGDeriv"] = p.get('deriv', 0)
                elif name == "SimpleDeriv":
                    prep_flat["ApplyDeriv"] = True; prep_flat["Gap"] = p.get('gap', 5); prep_flat["DerivOrder"] = p.get('order', 1)
                elif name == "L2": prep_flat["ApplyL2"] = True
                elif name == "MinMax": prep_flat["ApplyMinMax"] = True
                elif name == "SNV": prep_flat["ApplySNV"] = True
                elif name == "MinSub": prep_flat["ApplyMinSub"] = True  # AI가 추가함: MinSub export
                # Absorbance is handled by Mode

        # AI가 수정함: RequiredRawBands 계산 로직 개선 (Deriv Order & SG Window 고려)
        # Logic:
        # 1. Deriv(미분)가 있다면, 차수(Order)만큼 Base Band가 늘어남. (e.g. Band, Band+Gap, Band+2*Gap...)
        # 2. SG(골레이)가 있다면, 각 Base Band를 중심으로 Window Size만큼 주변 밴드가 필요함.
        # 3. 이 두 가지를 순차적으로 적용하여 최종 필요 밴드 집합을 구함.
        
        required_raw_bands = set()
        
        # Parameters (Strict Mode Validation)
        is_deriv = prep_flat.get("ApplyDeriv", False)
        
        # 1. Check Deriv Gap (SimpleDeriv 사용 시에만 gap 밴드 필요 — Absorbance 단독은 C#이 gap 접근 안 함)
        # AI가 수정함: is_absorbance 분기 제거 — SimpleDeriv 없이 Absorbance만 쓰면 gap 불필요
        deriv_gap = 5 # Default
        if is_deriv:
            if "Gap" not in prep_flat:
                 # 만약 UI가 Gap을 안 줬다면 에러
                 raise ValueError("Export Error: 'Gap' parameter is missing for Derivative!")
            deriv_gap = prep_flat["Gap"]
        
        deriv_order = prep_flat.get("DerivOrder", 1)
        
        is_sg = prep_flat.get("ApplySG", False)
        if is_sg and "SGWin" not in prep_flat:
             # SG 사용 시 Win 필수
             raise ValueError("Export Error: 'SGWin' parameter is missing for Savitzky-Golay!")
             
        sg_win = prep_flat.get("SGWin", 5)
        sg_radius = sg_win // 2 if is_sg else 0
        
        # Step 1: Core Base Bands (Deriv 또는 Absorbance gap 고려)
        # 만약 Deriv/Absorbance 미사용이면 base_bands = selected_bands
        base_bands = set()
        for b in selected_bands:
            base_idx = int(b)
            base_bands.add(base_idx)
            
            if is_deriv and deriv_gap > 0:
                # 1차 미분 -> b, b+Gap 필요
                # 2차 미분 -> b, b+Gap, b+2*Gap 필요 ...
                for k in range(1, deriv_order + 1):
                    base_bands.add(base_idx + (k * deriv_gap))
        
        # Step 2: Expand by SG Window (SG 고려)
        # 모든 Base Band에 대해 Radius 만큼 좌우로 확장
        for base in base_bands:
            required_raw_bands.add(base) # 자기 자신
            if sg_radius > 0:
                for offset in range(1, sg_radius + 1):
                    # AI가 수정함: 음수 인덱스 방지 (max(0, base - offset))
                    left_idx = max(0, base - offset)
                    required_raw_bands.add(left_idx)
                    required_raw_bands.add(base + offset)
        
        required_raw_bands_sorted = sorted(list(required_raw_bands))
        estimated_fps = _HSI_CAMERA_BANDWIDTH_BPS / (  # AI가 수정함: FPS 추정 계산
            max(len(required_raw_bands_sorted), 1) * _HSI_PIXEL_BYTES * _HSI_FRAME_WIDTH  # AI가 수정함: FPS 추정 계산
        )  # AI가 수정함: FPS 추정 계산
        # AI가 추가함: total_bands 클램프 — SG radius/gap offset이 원본 밴드 범위 초과 방지
        # AI가 수정함: V2-RRB — total_bands None 시 경고 (mean_spectrum 유추는 위에서 이미 처리됨)
        if total_bands is not None and total_bands > 0:
            required_raw_bands_sorted = [b for b in required_raw_bands_sorted if 0 <= b < total_bands]
        else:
            import warnings
            warnings.warn(
                "export_model: total_bands를 결정할 수 없어 RequiredRawBands 상한 검사를 건너뜁니다. "
                "범위 초과 밴드 인덱스가 포함될 수 있습니다.",
                RuntimeWarning
            )
        
        # AI가 추가함: export 직전 weights/bias 형태 검증 — [L-2]
        # 잘못된 shape으로 model.json 생성 시 C# 런타임 로드 오류 방지
        assert len(weights) > 0, "export_model: Weights must not be empty. Model extraction failed."
        if isinstance(bias, list):
            assert len(bias) > 0, "export_model: Bias must not be empty."
        
        # AI가 추가함: PrepChainOrder — C# HsiPipeline이 전처리 순서를 prep_chain 그대로 재현하기 위한 필드
        # Python prep_chain의 각 step 이름만 순서대로 추출 (파라미터 제외, C#은 Preprocessing 섹션에서 읽음)
        prep_chain_order = []
        if preprocessing_config:
            for step in preprocessing_config:
                step_name = step.get('name')
                if step_name:
                    prep_chain_order.append(step_name)
        
        export_data = {
            "ModelType": "LinearModel", # Unified name
            "OriginalType": model_type_name,
            "ModelName": model_name,    # AI가 추가함
            "Description": description, # AI가 추가함
            "Timestamp": timestamp, # AI가 추가함
            "SelectedBands": [int(b) for b in selected_bands],
            "RequiredRawBands": required_raw_bands_sorted,  # AI가 추가함: 런타임용 원본 밴드 목록
            "EstimatedFPS": round(estimated_fps, 2),  # AI가 수정함: FPS 추정값
            "PrepChainOrder": prep_chain_order,             # AI가 추가함: 전처리 적용 순서 (C# 패리티용)
            "ExcludeBands": exclude_rules if exclude_rules else "",
            "Weights": weights,
            "Bias": bias,
            "IsMultiClass": is_multiclass,
            "Preprocessing": prep_flat,
            "Labels": label_map if label_map else {"0": "Normal", "1": "Defect"},
            "Colors": colors_map if colors_map else {"0": "#00FF00", "1": "#FF0000"},
            "Performance": metrics # AI가 추가함: 모델 성적표
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
            
        print(f"   [LearningService] Model exported to {output_path}")
        
        # Generate Feature Importance Plot
        self._generate_importance_plot(weights, selected_bands, output_path, label_map, mean_spectrum, spa_scores, exclude_rules, band_method=None)

    def _generate_importance_plot(self, weights, bands, output_path, label_map, mean_spectrum=None, spa_scores=None, exclude_rules=None, band_method=None):
        try:
            if plt is None:  # AI가 수정함: matplotlib 미설치 시 플롯 생성을 건너뜀
                print("   [LearningService] Matplotlib unavailable; skipping importance plot.")  # AI가 수정함: 스킵 로그 출력
                return  # AI가 수정함: 플롯 생성 조기 종료
            # Switch to non-interactive backend for thread safety
            plt.switch_backend('Agg')
            
            if sns: sns.reset_orig()
            plt.style.use('default') 
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # --- 1. Background: Excluded Regions (Shading) ---
            # Do this first so it's behind everything
            if exclude_rules:
                import matplotlib.patches as mpatches
                try:
                    for part in exclude_rules.split(','):
                        part = part.strip()
                        if not part: continue
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            # UI 1-based -> Plot 0-based
                            ax1.axvspan(start-1, end, color='#E0E0E0', alpha=0.5, zorder=0)
                        else:
                            idx = int(part) - 1
                            ax1.axvspan(idx-0.5, idx+0.5, color='#E0E0E0', alpha=0.5, zorder=0)
                    
                    # Add dummy handle for legend
                    # We can't easily get handle from axvspan unless saved, so create a Patch
                    # exp_patch = mpatches.Patch(color='#E0E0E0', label='Excluded Region')
                except Exception:  # AI가 수정함: bare except → Exception (코드 품질)
                    pass

            # --- 2. Background: Mean Spectrum (Right Axis) ---
            spec_line = None
            if mean_spectrum is not None:
                ax2 = ax1.twinx()
                spec_line, = ax2.plot(range(len(mean_spectrum)), mean_spectrum, color='gray', linestyle='--', linewidth=1.5, label='Mean Spectrum', alpha=0.8, zorder=1)
                ax2.set_ylabel("Intensity (DN/Ref)", color='gray', fontsize=10)
                ax2.tick_params(axis='y', labelcolor='gray')
            else:
                ax2 = None
                
            # --- 3. Foreground: SPA Scores (Left Axis - Blue/Red Bars) ---
            
            ax1.set_xlabel("Band Index")
            ax1.set_ylabel("Selectivity Score (SPA)", color='blue', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='blue')
            
            bar_candidates = None
            bar_selected = None
            
            if spa_scores is not None and len(spa_scores) > 0:
                x_all = np.arange(len(spa_scores))
                bar_candidates = ax1.bar(x_all, spa_scores, color='skyblue', width=0.8, alpha=0.6, label='Candidate Bands', zorder=2)
                
                sel_scores = []
                for b in bands:
                    if b < len(spa_scores): sel_scores.append(spa_scores[int(b)])
                    else: sel_scores.append(0)
                        
                bar_selected = ax1.bar(bands, sel_scores, color='red', width=0.8, alpha=1.0, label='Selected Bands', zorder=3)
                
                # Labels
                for b, s in zip(bands, sel_scores):
                    ax1.text(b, s, str(int(b)), ha='center', va='bottom', fontsize=9, fontweight='bold', color='red', zorder=4)
                    
            else:
                # Fallback style
                w = np.abs(np.array(weights))
                if w.ndim == 2: w = np.max(w, axis=0) 
                bar_selected = ax1.bar(bands, w, color='red', width=1.5, label='Importance (Coef)')
            
            # AI가 수정함: Full Band 모드 분기 — band_method 또는 spa_scores 유무로 판단
            _is_full_band = (band_method == 'full') or (spa_scores is None or len(spa_scores) == 0)
            if _is_full_band:
                plt.title(f"Band Selection Result (Full Band, Top-{len(bands)})", fontsize=12)
            else:
                plt.title(f"Band Selection Result (SPA Algorithm, Top-{len(bands)})", fontsize=12)
            plt.grid(True, axis='x', alpha=0.3)
            
            # --- Consolidated Legend ---
            handles = []
            labels = []
            
            if exclude_rules:
                import matplotlib.patches as mpatches
                handles.append(mpatches.Patch(color='#E0E0E0', label='Excluded Region'))
            
            if bar_selected: handles.append(bar_selected)
            if bar_candidates: handles.append(bar_candidates)
            if spec_line: handles.append(spec_line)
            
            # Extract labels manually or use helper
            # Since we manually added items, just use their labels if object has one, OR construct
            # Actually bar containers have labels.
            
            final_handles = []
            final_labels = []
            for h in handles:
                if hasattr(h, 'get_label'): 
                    final_labels.append(h.get_label())
                    final_handles.append(h)
                else: 
                    # If it's a Patch
                    final_labels.append(h.get_label())
                    final_handles.append(h)

            plt.legend(handles=final_handles, labels=final_labels, loc='upper left')
            
            plt.tight_layout()
            
            png_path = os.path.join(os.path.dirname(output_path), "band_importance.png")
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"   [LearningService] Importance plot saved to {png_path}")
            
        except Exception as e:
            print(f"   [Error] Failed to generate plot: {e}")
            import traceback
            traceback.print_exc()
