# AI가 수정함: Experiment Grid Runner — 신규 파일
import csv  # AI가 수정함: CSV 저장용
import os  # AI가 수정함: 경로 처리용
import time  # AI가 수정함: 학습 시간 측정용
from datetime import datetime  # AI가 수정함: 타임스탬프 생성용

import numpy as np  # AI가 수정함: 배열 연산용
from sklearn.metrics import f1_score, precision_score, recall_score  # AI가 수정함: 평가 지표 계산용
from sklearn.model_selection import train_test_split  # AI가 수정함: stratify split용

from services.processing_service import ProcessingService  # AI가 수정함: 전처리 단일 경유점
from services.band_selection_service import select_best_bands  # AI가 수정함: 밴드 선택 단일 경유점
from services.learning_service import LearningService  # AI가 수정함: 학습 단일 경유점

# AI가 수정함: CSV 컬럼 스키마 (12컬럼 고정)
_CSV_FIELDNAMES = [  # AI가 수정함: 12컬럼 정의
    "timestamp",        # AI가 수정함: 실행 시각
    "band_method",      # AI가 수정함: 밴드 선택 방법
    "n_bands",          # AI가 수정함: 선택 밴드 수 (full이면 "full" 문자열)
    "model_type",       # AI가 수정함: 모델 종류
    "train_acc",        # AI가 수정함: 훈련 정확도 (0~100 scale)
    "test_acc",         # AI가 수정함: 테스트 정확도 (0~100 scale)
    "f1_macro",         # AI가 수정함: macro F1 score
    "precision_macro",  # AI가 수정함: macro precision score
    "recall_macro",     # AI가 수정함: macro recall score
    "gap_pct",          # AI가 수정함: train_acc - test_acc (과적합 지표)
    "train_time_ms",    # AI가 수정함: 학습 소요 시간 (ms)
    "status",           # AI가 수정함: "ok" 또는 "error: {e}"
]  # AI가 수정함: 컬럼 목록 종료


class ExperimentRunner:  # AI가 수정함: QObject 상속 절대 금지 — 순수 Python 클래스
    """
    // AI가 수정함: Experiment Grid Runner
    band_methods × model_types 조합을 순회하며 각 trial 결과를 CSV로 저장합니다.
    UI 레이어에 의존하지 않는 순수 비즈니스 로직 클래스입니다.
    """

    def run_grid(  # AI가 수정함: 메인 그리드 실행 메서드
        self,
        X_base,           # AI가 수정함: ndarray (N, B) — 전처리 전 base data
        y,                # AI가 수정함: ndarray (N,) — 클래스 레이블
        prep_chain,       # AI가 수정함: list — active preprocessing steps
        band_methods,     # AI가 수정함: list[str] — e.g. ['spa', 'anova_f']
        model_types,      # AI가 수정함: list[str] — e.g. ['LDA', 'Ridge Classifier']
        n_bands,          # AI가 수정함: int — 선택할 밴드 수
        test_ratio,       # AI가 수정함: float — 테스트 분리 비율
        output_dir,       # AI가 수정함: str — CSV 저장 디렉토리
        exclude_indices=None,   # AI가 수정함: list[int] | None — 제외 밴드 인덱스
        log_callback=None,      # AI가 수정함: callable(str) → None | None
        stop_flag=None          # AI가 수정함: callable() → bool | None — True이면 즉시 중단
    ):  # AI가 수정함: -> list[dict] — 각 trial 결과 dict 목록
        """
        // AI가 수정함: band_methods × model_types 조합 그리드 실행
        반환값: list[dict] — 각 trial의 12컬럼 결과를 담은 dict 목록
        """
        # AI가 수정함: 내부 로그 헬퍼 — log_callback 없으면 print로 fallback
        def _log(msg):  # AI가 수정함: 로그 헬퍼 정의
            if log_callback is not None:  # AI가 수정함: callback 존재 시 위임
                log_callback(msg)  # AI가 수정함: UI 콜백 호출
            else:  # AI가 수정함: fallback
                print(msg)  # AI가 수정함: 콘솔 출력

        # AI가 수정함: stop_flag 없으면 항상 False 반환하는 기본값
        def _should_stop():  # AI가 수정함: stop 조건 헬퍼
            if stop_flag is not None:  # AI가 수정함: callable이면 호출
                return bool(stop_flag())  # AI가 수정함: bool 강제 변환
            return False  # AI가 수정함: 기본값 — 계속 실행

        # AI가 수정함: 출력 디렉토리 생성 (존재해도 에러 없음)
        os.makedirs(output_dir, exist_ok=True)  # AI가 수정함: 디렉토리 보장

        # AI가 수정함: 결과 누적 리스트
        results = []  # AI가 수정함: trial 결과 dict 목록

        # --- Step 1: 전처리 체인 1회 적용 ---
        # AI가 수정함: ProcessingService 경유 필수 — models/processing.py 직접 임포트 금지
        _log("[ExperimentRunner] Applying preprocessing chain...")  # AI가 수정함: 전처리 시작 로그
        try:  # AI가 수정함: 전처리 실패 시 그리드 전체 중단
            X_prep = ProcessingService.apply_preprocessing_chain(  # AI가 수정함: 단일 경유점
                X_base.copy(), prep_chain  # AI가 수정함: 원본 훼손 방지를 위해 copy()
            )  # AI가 수정함: 반환값: ndarray (N, B')
        except Exception as prep_err:  # AI가 수정함: 전처리 예외 처리
            _log(f"[ExperimentRunner] Preprocessing failed: {prep_err}")  # AI가 수정함: 에러 로그
            raise  # AI가 수정함: 전처리 실패는 치명적 — 재발생

        B = X_prep.shape[1]  # AI가 수정함: 전처리 후 밴드 수
        _log(f"[ExperimentRunner] Preprocessed shape: {X_prep.shape}")  # AI가 수정함: shape 로그

        # --- Step 2: band_methods × model_types 루프 ---
        _log(f"[ExperimentRunner] Grid: {len(band_methods)} band methods × {len(model_types)} model types")  # AI가 수정함: 그리드 크기 로그

        for bm in band_methods:  # AI가 수정함: 밴드 선택 방법 순회
            for mt in model_types:  # AI가 수정함: 모델 종류 순회

                # AI가 수정함: stop_flag 체크 — True이면 즉시 루프 탈출
                if _should_stop():  # AI가 수정함: 중단 조건 확인
                    _log("[ExperimentRunner] Stop flag detected. Breaking grid loop.")  # AI가 수정함: 중단 로그
                    break  # AI가 수정함: 내부 루프 탈출

                # AI가 수정함: trial 타임스탬프
                trial_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # AI가 수정함: ISO 형식 타임스탬프

                _log(f"[ExperimentRunner] Trial: band_method={bm}, model_type={mt}")  # AI가 수정함: trial 시작 로그

                try:  # AI가 수정함: trial별 예외 격리 — 실패해도 루프 계속
                    # --- 2a. 밴드 선택 ---
                    # AI가 수정함: X_prep을 (N, 1, B) 로 reshape 하여 select_best_bands에 전달
                    selected_indices, _, _ = select_best_bands(  # AI가 수정함: 밴드 선택 함수 호출
                        X_prep.reshape(-1, 1, B),  # AI가 수정함: (N, 1, B) 형태로 reshape
                        n_bands,                    # AI가 수정함: 선택할 밴드 수
                        method=bm,                  # AI가 수정함: 밴드 선택 방법
                        labels=y,                   # AI가 수정함: supervised 방법을 위한 레이블
                        exclude_indices=exclude_indices,  # AI가 수정함: 제외 밴드 인덱스
                    )  # AI가 수정함: 반환값: (selected_indices, importance, mean_spectrum)

                    # AI가 수정함: 선택된 밴드로 데이터 슬라이싱
                    X_sub = X_prep[:, selected_indices]  # AI가 수정함: (N, n_selected) 슬라이싱

                    # --- 2b. 학습 ---
                    t_start = time.perf_counter()  # AI가 수정함: 학습 시작 시각
                    model, metrics = LearningService().train_model(  # AI가 수정함: LearningService 경유 필수
                        X_sub, y,  # AI가 수정함: 선택된 밴드 데이터와 레이블
                        model_type=mt,  # AI가 수정함: 모델 종류
                        test_ratio=test_ratio,  # AI가 수정함: 테스트 분리 비율
                    )  # AI가 수정함: 반환값: (model, metrics dict)
                    train_time_ms = (time.perf_counter() - t_start) * 1000  # AI가 수정함: ms 변환

                    # AI가 수정함: train_acc, test_acc 추출 (0~100 scale → 그대로 사용)
                    train_acc = float(metrics.get("TrainAccuracy", 0.0))  # AI가 수정함: 훈련 정확도
                    test_acc = float(metrics.get("TestAccuracy", 0.0))  # AI가 수정함: 테스트 정확도

                    # --- 2c. 추가 메트릭 계산 (precision, recall, f1 macro) ---
                    # AI가 수정함: 동일한 stratify=y split으로 y_pred 구성
                    try:  # AI가 수정함: stratify 실패 시 fallback
                        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(  # AI가 수정함: stratified split
                            X_sub, y,
                            test_size=test_ratio,
                            random_state=42,
                            stratify=y,  # AI가 수정함: stratify=y 우선
                        )
                    except ValueError:  # AI가 수정함: 클래스 샘플 부족 fallback
                        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(  # AI가 수정함: stratify=None fallback
                            X_sub, y,
                            test_size=test_ratio,
                            random_state=42,
                            stratify=None,  # AI가 수정함: fallback — stratify 없이 분리
                        )

                    # AI가 수정함: 학습된 모델로 예측 수행
                    y_pred = model.predict(X_test_m)  # AI가 수정함: 테스트셋 예측

                    # AI가 수정함: macro 메트릭 계산 (zero_division=0 필수)
                    f1_mac = float(f1_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro F1
                    prec_mac = float(precision_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro precision
                    rec_mac = float(recall_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro recall

                    # AI가 수정함: gap_pct = train_acc - test_acc (과적합 지표)
                    gap_pct = round(train_acc - test_acc, 4)  # AI가 수정함: 소수점 4자리 반올림

                    # AI가 수정함: n_bands 컬럼값 — 'full'이면 문자열 "full", 아니면 정수
                    n_bands_val = "full" if bm == "full" else int(n_bands)  # AI가 수정함: 컬럼 값 결정

                    # AI가 수정함: 결과 dict 구성 (12컬럼)
                    row = {  # AI가 수정함: trial 결과 dict
                        "timestamp": trial_ts,  # AI가 수정함: 실행 시각
                        "band_method": bm,  # AI가 수정함: 밴드 선택 방법
                        "n_bands": n_bands_val,  # AI가 수정함: 밴드 수 (full이면 문자열)
                        "model_type": mt,  # AI가 수정함: 모델 종류
                        "train_acc": round(train_acc, 4),  # AI가 수정함: 훈련 정확도
                        "test_acc": round(test_acc, 4),  # AI가 수정함: 테스트 정확도
                        "f1_macro": round(f1_mac, 4),  # AI가 수정함: macro F1
                        "precision_macro": round(prec_mac, 4),  # AI가 수정함: macro precision
                        "recall_macro": round(rec_mac, 4),  # AI가 수정함: macro recall
                        "gap_pct": gap_pct,  # AI가 수정함: 과적합 gap
                        "train_time_ms": round(train_time_ms, 2),  # AI가 수정함: 학습 시간 ms
                        "status": "ok",  # AI가 수정함: 정상 완료
                    }  # AI가 수정함: 결과 dict 종료
                    results.append(row)  # AI가 수정함: 결과 누적

                    _log(  # AI가 수정함: trial 완료 로그
                        f"[ExperimentRunner] OK | {bm}/{mt} | "
                        f"train={train_acc:.2f}% test={test_acc:.2f}% "
                        f"gap={gap_pct:.2f}% f1={f1_mac:.3f} [{train_time_ms:.1f}ms]"
                    )

                except Exception as trial_err:  # AI가 수정함: trial 예외 격리 — 루프 계속
                    # AI가 수정함: n_bands_val 오류 방지 (예외가 n_bands_val 할당 전에 발생 가능)
                    _n_bands_err = "full" if bm == "full" else int(n_bands)  # AI가 수정함: 안전한 n_bands 값

                    err_row = {  # AI가 수정함: 에러 trial 결과 dict
                        "timestamp": trial_ts,  # AI가 수정함: 실행 시각
                        "band_method": bm,  # AI가 수정함: 밴드 선택 방법
                        "n_bands": _n_bands_err,  # AI가 수정함: 밴드 수
                        "model_type": mt,  # AI가 수정함: 모델 종류
                        "train_acc": "",  # AI가 수정함: 에러 시 빈값
                        "test_acc": "",  # AI가 수정함: 에러 시 빈값
                        "f1_macro": "",  # AI가 수정함: 에러 시 빈값
                        "precision_macro": "",  # AI가 수정함: 에러 시 빈값
                        "recall_macro": "",  # AI가 수정함: 에러 시 빈값
                        "gap_pct": "",  # AI가 수정함: 에러 시 빈값
                        "train_time_ms": "",  # AI가 수정함: 에러 시 빈값
                        "status": f"error: {trial_err}",  # AI가 수정함: 에러 메시지 기록
                    }  # AI가 수정함: 에러 dict 종료
                    results.append(err_row)  # AI가 수정함: 에러 결과 누적

                    _log(f"[ExperimentRunner] ERROR | {bm}/{mt} | {trial_err}")  # AI가 수정함: 에러 로그

            else:  # AI가 수정함: 내부 for else — break 없이 완료 시 외부 루프 계속
                continue  # AI가 수정함: 내부 루프 정상 완료 — 외부 루프 계속
            break  # AI가 수정함: 내부 break → 외부 루프도 탈출 (stop_flag)

        # --- Step 3: CSV 저장 ---
        # AI가 수정함: 타임스탬프 기반 파일명 생성
        csv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # AI가 수정함: CSV 파일용 타임스탬프
        csv_filename = f"{csv_timestamp}_experiment_grid.csv"  # AI가 수정함: 파일명 구성
        csv_path = os.path.join(output_dir, csv_filename)  # AI가 수정함: 전체 경로 구성

        # AI가 수정함: CSV 파일 쓰기 (12컬럼 고정 스키마)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:  # AI가 수정함: CSV 파일 열기
            writer = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDNAMES)  # AI가 수정함: DictWriter 생성
            writer.writeheader()  # AI가 수정함: 헤더 행 기록
            writer.writerows(results)  # AI가 수정함: 모든 결과 행 기록

        _log(f"[ExperimentRunner] CSV saved: {csv_path} ({len(results)} trials)")  # AI가 수정함: 저장 완료 로그

        return results  # AI가 수정함: trial 결과 dict 리스트 반환
