import json
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """
    선형 SVM (Linear Support Vector Machine) 모델을 학습합니다.
    
    [왜 SVM인가?]
    초분광 데이터는 차원(밴드)이 많아도 선형적인 특징이 강해서, 
    무거운 딥러닝 없이 단순한 선 긋기(Linear SVM)만으로도 분류가 아주 잘 됩니다.
    또한, C#에서 계산할 때 단순히 곱하기/더하기만 하면 되어서 속도가 엄청나게 빠릅니다.
    
    Args:
        X: 입력 데이터 (샘플개수, 선택된 밴드개수)
        y: 정답 라벨 (샘플개수, ) -> 0: 정상, 1: 불량
        
    Returns:
        model: 학습이 완료된 Scikit-learn 모델 객체
    """
    print(f"   [Model Trainer] 선형 SVM 학습 시작... (샘플 수: {X.shape[0]}, 특징 수: {X.shape[1]})")
    
    # 1. 학습/검증 데이터 분리 (8:2)
    # 모델이 문제집(Train)만 외우지 않고 실전(Test)에서도 잘하는지 확인하기 위함입니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   [Debug] 학습 데이터 라벨 분포: {np.unique(y_train)}")
    
    # 2. 모델 생성 및 학습
    # dual=False: 샘플 수가 특징 수보다 많을 때 속도가 더 빠르고 안정적입니다.
    model = LinearSVC(dual=False, max_iter=1000)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"   [Error] SVM 학습 실패: {e}")
        # 데이터에 문제가 있는지 확인하기 위해 앞부분을 조금 출력해봅니다.
        print(f"   [Debug] X_train 데이터 일부:\n{X_train[:5]}")
        raise

    # 3. 정확도 검증
    # 학습하지 않은 데이터(Test Set)를 넣어서 얼마나 잘 맞추는지 확인합니다.
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # 정확도는 백분율(%)로 표시합니다. 98% 이상이면 아주 훌륭한 상태입니다.
    print(f"   [Model Trainer] 검증 정확도(Accuracy): {acc*100:.2f}%")
    
    return model

def export_model_for_csharp(model, selected_bands, output_path, preprocessing_config=None):
    """
    학습된 모델의 가중치와 편향 값, 그리고 전처리 설정들을 JSON 파일로 저장합니다.
    이 파일은 C# 프로그램에서 읽혀서 실시간 검사에 사용됩니다.
    
    [저장되는 핵심 정보]
    1. SelectedBands: "카메라야, 이 파장들만 봐!"
    2. Weights: "들어온 값에 이 숫자들을 곱해!"
    3. Bias: "마지막에 이 숫자를 더해!"
    4. Preprocessing: "계산 전에 배경 지우고 필터 걸어!"
    """
    # LinearSVC의 coef_는 (1, n_features) 형태입니다. (이진 분류 시)
    weights = model.coef_[0].tolist()
    bias = float(model.intercept_[0])
    
    export_data = {
        "ModelType": "LinearSVM",
        "SelectedBands": [int(b) for b in selected_bands],
        "Weights": weights,
        "Bias": bias,
        "Preprocessing": preprocessing_config if preprocessing_config else {},
        "Note": "판정 공식: Score = sum(w*x) + b. 만약 Score > 0 이면 '불량(Class 1)' 입니다."
    }
    
    # 저장할 경로의 폴더가 없다면 자동으로 만들어줍니다.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"   [Model Trainer] 모델 내보내기 완료: {output_path}")
    print(f"   [Debug] 저장된 가중치(Weights): {weights}")
    print(f"   [Debug] 저장된 편향(Bias): {bias}")
    if preprocessing_config:
        print(f"   [Debug] 저장된 전처리 설정: {preprocessing_config}")
