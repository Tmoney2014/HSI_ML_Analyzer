import sys
import os
import argparse
import numpy as np

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_hsi_data
from utils.band_selection import select_best_bands
from utils.model_trainer import train_model, export_model_for_csharp

def main():
    # 프로그램 설명 및 옵션 설정
    parser = argparse.ArgumentParser(description="초분광 데이터 분석 & SVM 모델 내보내기 도구")
    
    # 기본 경로 설정 (실습용 데이터 경로)
    default_normal = r"C:\Users\user16g\Desktop\nonbr_br_fx50\0_2_non_br\capture\0_2_non_br.hdr"
    default_defect = r"C:\Users\user16g\Desktop\nonbr_br_fx50\0_2_br_100_200\capture\0_2_br_0001.hdr"
    
    parser.add_argument("--normal_path", type=str, default=default_normal, help="정상 데이터(Class 0)의 .hdr 파일 경로")
    parser.add_argument("--defect_path", type=str, default=default_defect, help="불량 데이터(Class 1)의 .hdr 파일 경로")
    parser.add_argument("--output_path", type=str, default="./output/model_config.json", help="C#용 모델 설정 파일이 저장될 경로")
    
    args = parser.parse_args()

    # 결과 저장 폴더가 없으면 미리 만듭니다.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("🚀 [1단계] 초분광 분석 파이프라인 초기화 중...")
    
    # =========================================================================
    # 1. 데이터 불러오기 (Load Data)
    # =========================================================================
    print(f"   정상(Normal) 데이터 로딩 중: {os.path.basename(args.normal_path)}")
    cube_normal, _ = load_hsi_data(args.normal_path)
    
    # NaN(Not a Number)이나 무한대(Inf) 값이 있으면 0으로 채워줍니다. (에러 방지용)
    cube_normal = np.nan_to_num(cube_normal)
    print(f"   [디버그] 정상 데이터 크기(Shape): {cube_normal.shape}")
    
    print(f"   불량(Defect) 데이터 로딩 중: {os.path.basename(args.defect_path)}")
    cube_defect, wavelengths = load_hsi_data(args.defect_path)
    
    # NaN(Not a Number)이나 무한대(Inf) 값을 처리합니다.
    cube_defect = np.nan_to_num(cube_defect)
    print(f"   [디버그] 불량 데이터 크기(Shape): {cube_defect.shape}")
    
    # 데이터가 비어있으면 함수를 종료합니다.
    if cube_normal.size == 0 or cube_defect.size == 0:
        print("   [오류] 데이터 중 하나가 비어있습니다! 경로를 확인하세요.")
        return

    # 두 데이터의 밴드 개수가 다르면 합칠 수 없으므로 에러를 냅니다.
    if cube_normal.shape[2] != cube_defect.shape[2]:
        print("   [오류] 두 데이터의 밴드(파장) 개수가 서로 다릅니다!")
        return

    # =========================================================================
    # 2. 밴드 선택 (Band Selection) - "Brain" 🧠
    # =========================================================================
    # 원래 224개나 되는 파장을 다 쓰면 너무 느려집니다.
    # 그래서 '정상'과 '불량'을 가장 잘 구분할 수 있는 핵심 파장 n개를 찾습니다.
    # 여기서는 주로 Normal 데이터를 기준으로 데이터의 특징(분산)을 가장 잘 나타내는 파장을 찾습니다.
    # (필요하다면 두 데이터를 섞어서 찾을 수도 있습니다)
    
    n_bands = 5
    
    # 데이터를 분석하기 좋게 1줄로 폅니다. (Flatten)
    # (세로, 가로, 밴드) -> (픽셀수, 밴드)
    h_n, w_n, b = cube_normal.shape
    h_d, w_d, _ = cube_defect.shape
    
    flat_normal = cube_normal.reshape(-1, b)
    flat_defect = cube_defect.reshape(-1, b)
    
    # 분석 속도를 위해 랜덤하게 5,000개씩만 뽑아서 밴드 선택에 사용합니다.
    # 전체 픽셀을 다 쓰면 시간이 너무 오래 걸립니다.
    n_samples = 5000
    idx_n = np.random.choice(flat_normal.shape[0], min(n_samples, flat_normal.shape[0]), replace=False)
    idx_d = np.random.choice(flat_defect.shape[0], min(n_samples, flat_defect.shape[0]), replace=False)
    
    # 정상과 불량 샘플을 합쳐서 분석기에 넣습니다.
    X_band_selection = np.vstack([flat_normal[idx_n], flat_defect[idx_d]])
    
    # 함수 입력을 맞추기 위해 모양을 살짝 바꿉니다. (픽셀수, 1, 밴드)
    dummy_cube = X_band_selection.reshape(-1, 1, b)
    selected_bands = select_best_bands(dummy_cube, n_bands=n_bands)
    
    # =========================================================================
    # 3. 모델 학습 (Training) - "Education" 🎓
    # =========================================================================
    print("🚀 [Step 3] 학습 데이터 준비 및 SVM 모델 학습...")
    
    # 위에서 선택한 중요 밴드(5개)의 데이터만 뽑아냅니다.
    # 이제 데이터는 224칸이 아니라 5칸짜리가 됩니다. (데이터 다이어트 성공!)
    X_normal_subset = flat_normal[:, selected_bands]
    X_defect_subset = flat_defect[:, selected_bands]
    
    # 정답지를 만듭니다 (Labeling)
    # 0: 정상, 1: 불량
    y_normal = np.zeros(X_normal_subset.shape[0])  # Class 0
    y_defect = np.ones(X_defect_subset.shape[0])   # Class 1
    
    # 정상 데이터와 불량 데이터를 하나로 합칩니다.
    X_train = np.vstack([X_normal_subset, X_defect_subset])
    y_train = np.hstack([y_normal, y_defect])
    
    # 디버깅: 데이터가 너무 많으면 학습이 오래 걸리므로 10만 개로 줄여서 테스트합니다.
    # (실제 최종 배포 때는 이 부분을 주석 처리해서 전체 데이터를 다 쓰세요)
    if X_train.shape[0] > 100000:
        print("   [Info] 데이터가 너무 많아 100,000개로 줄여서 빠르게 학습합니다...")
        idx = np.random.choice(X_train.shape[0], 100000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    print(f"   [Info] 최종 학습 샘플 수: {X_train.shape[0]}")
    # print(f"   [Info] 정상 샘플 수: {y_normal.shape[0]}")
    # print(f"   [Info] 불량 샘플 수: {y_defect.shape[0]}")
    
    # 학습 시작! (여기서 w와 b값을 찾아냅니다)
    model = train_model(X_train, y_train)
    
    # =========================================================================
    # 4. 결과 내보내기 (Export) - "Delivery" 🚚
    # =========================================================================
    print(f"🚀 [Step 4] 모델 정보를 C#용으로 내보내는 중: {args.output_path}...")
    export_model_for_csharp(model, selected_bands, args.output_path)
    
    print("✅ 모든 분석 과정이 성공적으로 완료되었습니다.")
    print(f"✅ 결과 파일이 저장되었습니다! 이제 C#에서 '{args.output_path}' 파일을 불러오세요.")

if __name__ == "__main__":
    main()
