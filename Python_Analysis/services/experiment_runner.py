# AI가 수정함: Experiment Grid Runner — 4D search 확장 (gap × band_methods × n_bands_list × model_types)
import csv  # AI가 수정함: CSV 저장용
import os  # AI가 수정함: 경로 처리용
import time  # AI가 수정함: 학습 시간 측정용
from copy import deepcopy  # AI가 수정함: prep_chain 깊은 복사용 (gap 파라미터 주입)
from datetime import datetime  # AI가 수정함: 타임스탬프 생성용

import json  # AI가 수정함: selected_bands JSON 직렬화용

import numpy as np  # AI가 수정함: 배열 연산용
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report  # AI가 수정함: 평가 지표 계산용
from sklearn.model_selection import train_test_split  # AI가 수정함: stratify split용

from services.processing_service import ProcessingService  # AI가 수정함: 전처리 단일 경유점
from services.band_selection_service import select_best_bands  # AI가 수정함: 밴드 선택 단일 경유점
from services.learning_service import LearningService  # AI가 수정함: 학습 단일 경유점

# AI가 수정함: CSV 컬럼 스키마 (15컬럼) — gap 컬럼 index 3에 추가
_CSV_FIELDNAMES = [  # AI가 수정함: 15컬럼 정의
    "timestamp",           # AI가 수정함: 실행 시각
    "band_method",         # AI가 수정함: 밴드 선택 방법
    "n_bands",             # AI가 수정함: 선택 밴드 수 (full이면 "full" 문자열)
    "gap",                 # AI가 수정함: SimpleDeriv gap 값 (SimpleDeriv 없으면 0)
    "model_type",          # AI가 수정함: 모델 종류
    "train_acc",           # AI가 수정함: 훈련 정확도 (0~100 scale)
    "test_acc",            # AI가 수정함: 테스트 정확도 (0~100 scale)
    "f1_macro",            # AI가 수정함: macro F1 score
    "precision_macro",     # AI가 수정함: macro precision score
    "recall_macro",        # AI가 수정함: macro recall score
    "gap_pct",             # AI가 수정함: train_acc - test_acc (과적합 지표)
    "train_time_ms",       # AI가 수정함: 학습 소요 시간 (ms)
    "selected_bands",      # AI가 수정함: 선택된 밴드 인덱스 JSON 문자열 (재현성)
    "confusion_png_path",  # AI가 수정함: best trial confusion matrix PNG 경로 (best trial만 채워짐)
    "per_class_report",    # AI가 수정함: 클래스별 precision/recall/f1 JSON 문자열 (모든 ok trial)
    "status",              # AI가 수정함: "ok" 또는 "error: {e}"
]  # AI가 수정함: 컬럼 목록 종료

_PAPER_MATRIX_METRICS = [
    ("test_acc", "Test Accuracy (%)"),
    ("f1_macro", "Macro F1"),
    ("train_time_ms", "Train Time (ms)"),
]


class ExperimentRunner:  # AI가 수정함: QObject 상속 절대 금지 — 순수 Python 클래스
    """
    // AI가 수정함: Experiment Grid Runner
    band_methods × n_bands_list × gap × model_types 4D 조합을 순회하며 각 trial 결과를 CSV로 저장합니다.
    UI 레이어에 의존하지 않는 순수 비즈니스 로직 클래스입니다.
    """

    def run_grid(  # AI가 수정함: 4D 그리드 실행 메서드
        self,
        X_base,                    # AI가 수정함: ndarray (N, B) — 전처리 전 base data
        y,                         # AI가 수정함: ndarray (N,) — 클래스 레이블
        prep_chain,                # AI가 수정함: list — active preprocessing steps
        band_methods,              # AI가 수정함: list[str] — e.g. ['spa', 'anova_f']
        model_types,               # AI가 수정함: list[str] — e.g. ['LDA', 'Ridge Classifier']
        n_bands_list,              # AI가 수정함: list[int] — 선택할 밴드 수 목록 (구 n_bands 대체)
        test_ratio,                # AI가 수정함: float — 테스트 분리 비율
        output_dir,                # AI가 수정함: str — CSV 저장 디렉토리
        gap_range=(1, 40),         # AI가 수정함: tuple — SimpleDeriv gap 탐색 범위 (inclusive)
        exclude_indices=None,      # AI가 수정함: list[int] | None — 제외 밴드 인덱스
        log_callback=None,         # AI가 수정함: callable(str) → None | None
        stop_flag=None             # AI가 수정함: callable() → bool | None — True이면 즉시 중단
    ):  # AI가 수정함: -> list[dict] — 각 trial 결과 dict 목록
        """
        // AI가 수정함: gap × band_methods × n_bands_list × model_types 4D 그리드 실행
        반환값: list[dict] — 각 trial의 15컬럼 결과를 담은 dict 목록
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
        os.makedirs(output_dir, exist_ok=True)  # AI가 수정함: 최상위 출력 폴더 보장

        # AI가 수정함: 결과 누적 리스트
        results = []  # AI가 수정함: trial 결과 dict 목록

        # --- Step 1: SimpleDeriv 감지 + gap 목록 결정 ---
        _target_keys = ["SimpleDeriv"]  # AI가 수정함: gap 파라미터 주입 대상 step 이름
        _has_simpledv = any(  # AI가 수정함: prep_chain 중 SimpleDeriv 존재 여부
            step.get("name") in _target_keys for step in prep_chain
        )
        _gap_list = list(range(gap_range[0], gap_range[1] + 1)) if _has_simpledv else [0]  # AI가 수정함: gap 탐색 목록

        # AI가 수정함: gap별 전처리 결과 캐시 — 동일 gap 재실행 방지
        _prep_cache: dict = {}  # gap_val -> X_prep

        def _get_X_prep(gap_val):  # AI가 수정함: gap별 전처리 결과 반환 (캐시 적중 시 재사용)
            if gap_val in _prep_cache:  # AI가 수정함: 캐시 적중
                return _prep_cache[gap_val]
            if gap_val > 0 and _has_simpledv:  # AI가 수정함: SimpleDeriv 있고 gap > 0인 경우
                _chain = deepcopy(prep_chain)  # AI가 수정함: 원본 훼손 방지
                for step in _chain:  # AI가 수정함: SimpleDeriv step 찾아 gap 주입
                    if step.get("name") in _target_keys:
                        step["params"]["gap"] = gap_val  # AI가 수정함: gap 값 주입
                        break
            else:  # AI가 수정함: SimpleDeriv 없거나 gap==0인 경우
                _chain = prep_chain  # AI가 수정함: 원본 체인 그대로 사용
            X_p = ProcessingService.apply_preprocessing_chain(  # AI가 수정함: ProcessingService 경유 필수
                X_base.copy(), _chain  # AI가 수정함: 원본 훼손 방지를 위해 copy()
            )
            _prep_cache[gap_val] = X_p  # AI가 수정함: 캐시 저장
            return X_p

        # --- Step 2: 4D 루프 ---
        _log(f"[ExperimentRunner] Grid: {len(band_methods)} band methods × {len(n_bands_list)} n_bands × {len(_gap_list)} gaps × {len(model_types)} model types")  # AI가 수정함: 그리드 크기 로그

        # AI가 수정함: 런 수준 타임스탬프 — 루프 전에 결정하여 모든 산출물 파일명 통일
        csv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # AI가 수정함: CSV + PNG 공용 타임스탬프

        # AI가 수정함: 실행마다 타임스탬프 이름의 전용 폴더 생성 → 파일 뒤섞임 방지
        _exp_dir = os.path.join(output_dir, 'experiments', csv_timestamp)
        os.makedirs(_exp_dir, exist_ok=True)
        _log(f"[ExperimentRunner] Output folder: {_exp_dir}")

        for gap_val in _gap_list:  # AI가 수정함: gap 루프 (Level 1)
            if _should_stop():  # AI가 수정함: stop 체크
                break

            X_prep = _get_X_prep(gap_val)  # AI가 수정함: gap별 전처리 결과 획득
            B = X_prep.shape[1]  # AI가 수정함: 전처리 후 밴드 수
            _log(f"[ExperimentRunner] gap={gap_val} → preprocessed shape: {X_prep.shape}")  # AI가 수정함: shape 로그

            for bm in band_methods:  # AI가 수정함: 밴드 선택 방법 순회 (Level 2)
                if _should_stop():  # AI가 수정함: stop 체크
                    break

                for n_bands_val in n_bands_list:  # AI가 수정함: 밴드 수 순회 (Level 3)
                    if _should_stop():  # AI가 수정함: stop 체크
                        break

                    for mt in model_types:  # AI가 수정함: 모델 종류 순회 (Level 4)
                        if _should_stop():  # AI가 수정함: stop 체크
                            break

                        # AI가 수정함: trial 타임스탬프
                        trial_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # AI가 수정함: ISO 형식 타임스탬프

                        _log(f"[ExperimentRunner] Trial: band_method={bm}, n_bands={n_bands_val}, gap={gap_val}, model_type={mt}")  # AI가 수정함: trial 시작 로그

                        try:  # AI가 수정함: trial별 예외 격리 — 실패해도 루프 계속
                            # AI가 수정함: n_bands > B 검증 — ValueError로 처리
                            if bm != "full" and n_bands_val > B:  # AI가 수정함: full 모드 제외 검증
                                raise ValueError(  # AI가 수정함: n_bands 초과 에러
                                    f"n_bands ({n_bands_val}) exceeds processed band count ({B})"
                                )

                            # --- trial 2a. 밴드 선택 ---
                            # AI가 수정함: X_prep을 (N, 1, B) 로 reshape 하여 select_best_bands에 전달
                            selected_indices, _, _ = select_best_bands(  # AI가 수정함: 밴드 선택 함수 호출
                                X_prep.reshape(-1, 1, B),  # AI가 수정함: (N, 1, B) 형태로 reshape
                                n_bands_val,               # AI가 수정함: 선택할 밴드 수
                                method=bm,                 # AI가 수정함: 밴드 선택 방법
                                labels=y,                  # AI가 수정함: supervised 방법을 위한 레이블
                                exclude_indices=exclude_indices,  # AI가 수정함: 제외 밴드 인덱스
                            )  # AI가 수정함: 반환값: (selected_indices, importance, mean_spectrum)

                            # AI가 수정함: 선택된 밴드로 데이터 슬라이싱
                            X_sub = X_prep[:, selected_indices]  # AI가 수정함: (N, n_selected) 슬라이싱

                            # --- trial 2b. 학습 ---
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

                            # --- trial 2c. 추가 메트릭 계산 (precision, recall, f1 macro) ---
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
                            # AI가 수정함: PLS-DA는 predict()가 회귀 출력 (N, n_classes) 반환 → argmax로 클래스 인덱스 변환
                            y_pred_raw = np.asarray(model.predict(X_test_m))  # AI가 수정함: asarray로 sparse matrix 대응 + 타입 명확화
                            if y_pred_raw.ndim == 2:
                                y_pred = np.argmax(y_pred_raw, axis=1)  # AI가 수정함: PLS-DA 회귀 출력 → 클래스 인덱스
                            else:
                                y_pred = y_pred_raw  # AI가 수정함: 일반 분류 모델 그대로 사용

                            # AI가 수정함: macro 메트릭 계산 (zero_division=0 필수)
                            f1_mac = float(f1_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro F1
                            prec_mac = float(precision_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro precision
                            rec_mac = float(recall_score(y_test_m, y_pred, average='macro', zero_division=0))  # AI가 수정함: macro recall

                            # AI가 수정함: 클래스별 precision/recall/f1 계산 → JSON 직렬화 (CSV 기록용)
                            # output_dict=True → {클래스명: {precision, recall, f1-score, support}, "macro avg": ..., ...}
                            # 집계 키("accuracy", "macro avg", "weighted avg")는 제외하고 클래스 엔트리만 저장
                            _cr_raw = classification_report(y_test_m, y_pred, output_dict=True, zero_division=0)
                            # AI가 수정함: sklearn stub이 classification_report 반환 타입을 str로 추론 → isinstance로 타입 확정
                            _per_class = {}
                            if isinstance(_cr_raw, dict):
                                _per_class = {
                                    str(k): {
                                        "precision": round(float(v["precision"]), 4),
                                        "recall": round(float(v["recall"]), 4),
                                        "f1": round(float(v["f1-score"]), 4),
                                        "support": int(v["support"]),
                                    }
                                    for k, v in _cr_raw.items()
                                    if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
                                }  # AI가 수정함: 클래스별 지표만 추출 (집계 키 제외)
                            per_class_json = json.dumps(_per_class, ensure_ascii=False)  # AI가 수정함: JSON 직렬화

                            # AI가 수정함: gap_pct = train_acc - test_acc (과적합 지표)
                            gap_pct = round(train_acc - test_acc, 4)  # AI가 수정함: 소수점 4자리 반올림

                            # AI가 수정함: n_bands 컬럼값 — 'full'이면 문자열 "full", 아니면 정수
                            n_bands_col = "full" if bm == "full" else int(n_bands_val)  # AI가 수정함: 컬럼 값 결정

                            # AI가 수정함: trial 루프에서 CM PNG 저장하지 않음
                            # y_test_m, y_pred를 임시 보관 → Step 2.5에서 best trial에만 PNG 생성
                            # _y_test/_y_pred는 _CSV_FIELDNAMES에 없으므로 CSV 쓰기 전 반드시 제거됨

                            # AI가 수정함: 결과 dict 구성 (16컬럼 + 임시 필드)
                            row = {  # AI가 수정함: trial 결과 dict
                                "timestamp": trial_ts,  # AI가 수정함: 실행 시각
                                "band_method": bm,  # AI가 수정함: 밴드 선택 방법
                                "n_bands": n_bands_col,  # AI가 수정함: 밴드 수 (full이면 문자열)
                                "gap": gap_val,  # AI가 수정함: SimpleDeriv gap 값
                                "model_type": mt,  # AI가 수정함: 모델 종류
                                "train_acc": round(train_acc, 4),  # AI가 수정함: 훈련 정확도
                                "test_acc": round(test_acc, 4),  # AI가 수정함: 테스트 정확도
                                "f1_macro": round(f1_mac, 4),  # AI가 수정함: macro F1
                                "precision_macro": round(prec_mac, 4),  # AI가 수정함: macro precision
                                "recall_macro": round(rec_mac, 4),  # AI가 수정함: macro recall
                                "gap_pct": gap_pct,  # AI가 수정함: 과적합 gap
                                "train_time_ms": round(train_time_ms, 2),  # AI가 수정함: 학습 시간 ms
                                "selected_bands": json.dumps(list(map(int, selected_indices))),  # AI가 수정함: 밴드 인덱스 JSON 직렬화
                                "confusion_png_path": "",  # AI가 수정함: Step 2.5에서 best trial만 채워짐
                                "per_class_report": per_class_json,  # AI가 수정함: 클래스별 지표 JSON (모든 ok trial)
                                "status": "ok",  # AI가 수정함: 정상 완료
                                "_y_test": y_test_m,  # AI가 수정함: 임시 — best CM 생성용, CSV 쓰기 전 제거
                                "_y_pred": y_pred,   # AI가 수정함: 임시 — best CM 생성용, CSV 쓰기 전 제거
                            }  # AI가 수정함: 결과 dict 종료
                            results.append(row)  # AI가 수정함: 결과 누적

                            _log(  # AI가 수정함: trial 완료 로그
                                f"[ExperimentRunner] OK | {bm}/{n_bands_val}/g{gap_val}/{mt} | "
                                f"train={train_acc:.2f}% test={test_acc:.2f}% "
                                f"gap_pct={gap_pct:.2f}% f1={f1_mac:.3f} [{train_time_ms:.1f}ms]"
                            )

                        except ValueError as e:  # AI가 수정함: n_bands 초과 등 ValueError 격리
                            _n_bands_err = "full" if bm == "full" else int(n_bands_val)  # AI가 수정함: 안전한 n_bands 값
                            err_row = {  # AI가 수정함: ValueError 에러 trial 결과 dict
                                "timestamp": trial_ts,  # AI가 수정함: 실행 시각
                                "band_method": bm,  # AI가 수정함: 밴드 선택 방법
                                "n_bands": _n_bands_err,  # AI가 수정함: 밴드 수
                                "gap": gap_val,  # AI가 수정함: gap 값
                                "model_type": mt,  # AI가 수정함: 모델 종류
                                "train_acc": 0,  # AI가 수정함: 에러 시 0
                                "test_acc": 0,  # AI가 수정함: 에러 시 0
                                "f1_macro": 0,  # AI가 수정함: 에러 시 0
                                "precision_macro": 0,  # AI가 수정함: 에러 시 0
                                "recall_macro": 0,  # AI가 수정함: 에러 시 0
                                "gap_pct": 0,  # AI가 수정함: 에러 시 0
                                "train_time_ms": 0,  # AI가 수정함: 에러 시 0
                                "selected_bands": "",  # AI가 수정함: 에러 시 빈값
                                "confusion_png_path": "",  # AI가 수정함: 에러 시 빈값
                                "status": "error: n_bands exceeds processed band count",  # AI가 수정함: 고정 에러 메시지
                            }  # AI가 수정함: 에러 dict 종료
                            results.append(err_row)  # AI가 수정함: 에러 결과 누적
                            _log(f"[ExperimentRunner] ERROR (ValueError) | {bm}/{n_bands_val}/g{gap_val}/{mt} | {e}")  # AI가 수정함: 에러 로그

                        except Exception as trial_err:  # AI가 수정함: trial 예외 격리 — 루프 계속
                            # AI가 수정함: n_bands_col 오류 방지 (예외가 n_bands_col 할당 전에 발생 가능)
                            _n_bands_err = "full" if bm == "full" else int(n_bands_val)  # AI가 수정함: 안전한 n_bands 값

                            err_row = {  # AI가 수정함: 에러 trial 결과 dict
                                "timestamp": trial_ts,  # AI가 수정함: 실행 시각
                                "band_method": bm,  # AI가 수정함: 밴드 선택 방법
                                "n_bands": _n_bands_err,  # AI가 수정함: 밴드 수
                                "gap": gap_val,  # AI가 수정함: gap 값
                                "model_type": mt,  # AI가 수정함: 모델 종류
                                "train_acc": "",  # AI가 수정함: 에러 시 빈값
                                "test_acc": "",  # AI가 수정함: 에러 시 빈값
                                "f1_macro": "",  # AI가 수정함: 에러 시 빈값
                                "precision_macro": "",  # AI가 수정함: 에러 시 빈값
                                "recall_macro": "",  # AI가 수정함: 에러 시 빈값
                                "gap_pct": "",  # AI가 수정함: 에러 시 빈값
                                "train_time_ms": "",  # AI가 수정함: 에러 시 빈값
                                "selected_bands": "",  # AI가 수정함: 에러 시 빈값
                                "confusion_png_path": "",  # AI가 수정함: 에러 시 빈값
                                "status": f"error: {trial_err}",  # AI가 수정함: 에러 메시지 기록
                            }  # AI가 수정함: 에러 dict 종료
                            results.append(err_row)  # AI가 수정함: 에러 결과 누적

                            _log(f"[ExperimentRunner] ERROR | {bm}/{n_bands_val}/g{gap_val}/{mt} | {trial_err}")  # AI가 수정함: 에러 로그

                        if log_callback:  # AI가 수정함: trial 완료 로그 (Task 9 진행률 추적용)
                            log_callback(f"[Trial] bm={bm} n={n_bands_val} gap={gap_val} mt={mt}")

                    else:  # AI가 수정함: Level 4 (mt) for-else — break 없이 완료 시 상위 루프 계속
                        continue  # AI가 수정함: 내부 루프 정상 완료 — 상위 루프 계속
                    break  # AI가 수정함: 내부 break → 상위 루프도 탈출 (stop_flag)

                else:  # AI가 수정함: Level 3 (n_bands) for-else
                    continue
                break  # AI가 수정함: n_bands 루프 탈출 전파

            else:  # AI가 수정함: Level 2 (bm) for-else
                continue
            break  # AI가 수정함: bm 루프 탈출 전파

        # --- Step 2.5: best trial confusion matrix PNG 생성 ---
        # AI가 수정함: trial 루프에서 PNG 저장 안 함 → 여기서 best (bm, mt) 조합만 생성
        # id() set으로 best trial dict 식별 → 한 번 루프로 생성 + 임시 필드 정리
        _best_for_cm, _ = self._best_per_bm_mt(results)
        _best_id_set = {id(r) for r in _best_for_cm}
        for r in results:
            _yt = r.pop("_y_test", None)
            _yp = r.pop("_y_pred", None)
            if id(r) in _best_id_set and _yt is not None and _yp is not None:
                _safe_bm = r["band_method"].replace(" ", "_").replace("/", "_")
                _safe_mt = r["model_type"].replace(" ", "_").replace("/", "_")
                _cm_stem = f"{csv_timestamp}_best_{_safe_bm}_{_safe_mt}_confusion"
                r["confusion_png_path"] = self._write_confusion_matrix_png(
                    _exp_dir, _yt, _yp, _cm_stem, _log
                )  # AI가 수정함: best trial에만 CM PNG 생성 — disk write 15회 이하

        # --- Step 3: CSV 저장 ---
        # AI가 수정함: csv_timestamp는 루프 시작 전에 결정됨 (confusion PNG 파일명과 동일 prefix)
        csv_filename = f"{csv_timestamp}_experiment_grid.csv"  # AI가 수정함: 파일명 구성
        csv_path = os.path.join(_exp_dir, csv_filename)  # AI가 수정함: experiments 서브디렉토리에 저장

        # AI가 수정함: CSV 파일 쓰기 (15컬럼 스키마)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:  # AI가 수정함: CSV 파일 열기
            writer = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDNAMES)  # AI가 수정함: DictWriter 생성
            writer.writeheader()  # AI가 수정함: 헤더 행 기록
            writer.writerows(results)  # AI가 수정함: 모든 결과 행 기록

        _log(f"[ExperimentRunner] CSV saved: {csv_path} ({len(results)} trials)")  # AI가 수정함: 저장 완료 로그
        self._write_paper_summary(_exp_dir, results, band_methods, model_types, csv_timestamp, _log)

        return results  # AI가 수정함: trial 결과 dict 리스트 반환

    @staticmethod
    def _build_metric_matrix(ok_results, band_methods, model_types, metric_key):
        """Build matrix rows for paper-ready summary tables."""
        rows = []
        for bm in band_methods:
            row = [bm]
            for mt in model_types:
                match = next(
                    (r for r in ok_results if r.get("band_method") == bm and r.get("model_type") == mt),
                    None,
                )
                if match is None:
                    row.append("-")
                else:
                    value = match.get(metric_key, "-")
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}" if metric_key != "train_time_ms" else f"{value:.2f}")
                    else:
                        row.append(str(value))
            rows.append(row)
        return rows

    @staticmethod
    def _write_markdown_table(file_handle, headers, rows):
        file_handle.write("| " + " | ".join(headers) + " |\n")
        file_handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            file_handle.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
        file_handle.write("\n")

    @staticmethod
    def _write_heatmap_png(exp_dir, ok_results, band_methods, model_types, metric_key, title, file_stem, log_callback):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            log_callback(f"[ExperimentRunner] Heatmap skipped ({file_stem}): {exc}")
            return

        matrix = []
        for bm in band_methods:
            row = []
            for mt in model_types:
                match = next(
                    (r for r in ok_results if r.get("band_method") == bm and r.get("model_type") == mt),
                    None,
                )
                row.append(float(match.get(metric_key, np.nan)) if match is not None else np.nan)
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(max(6, len(model_types) * 1.6), max(4, len(band_methods) * 0.8 + 1.5)))
        im = ax.imshow(np.array(matrix, dtype=float), cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(model_types)))
        ax.set_xticklabels(model_types, rotation=25, ha='right')
        ax.set_yticks(range(len(band_methods)))
        ax.set_yticklabels(band_methods)
        ax.set_title(title)

        for i in range(len(band_methods)):
            for j in range(len(model_types)):
                value = matrix[i][j]
                label = "-" if np.isnan(value) else (f"{value:.2f}" if metric_key != "f1_macro" else f"{value:.3f}")
                ax.text(j, i, label, ha='center', va='center', color='white' if not np.isnan(value) else '#cccccc', fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = os.path.join(exp_dir, f"{file_stem}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log_callback(f"[ExperimentRunner] Heatmap saved: {out_path}")

    @staticmethod
    def _write_confusion_matrix_png(exp_dir, y_true, y_pred, file_stem, log_callback):
        """
        // AI가 수정함: per-trial confusion matrix PNG 저장
        matplotlib가 없으면 빈 문자열 반환 (skip).
        반환값: str — 저장된 PNG 절대 경로, 또는 "" (matplotlib 없음/오류 시)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            log_callback(f"[ExperimentRunner] Confusion matrix skipped ({file_stem}): {exc}")
            return ""

        cm = confusion_matrix(y_true, y_pred)  # AI가 수정함: confusion matrix 계산
        labels = sorted(set(y_true) | set(y_pred))  # AI가 수정함: 레이블 정렬 (문자열/정수 모두 가능)
        n = len(labels)

        fig, ax = plt.subplots(figsize=(max(4, n * 0.9 + 1.5), max(3, n * 0.9 + 1.5)))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_xticklabels([str(lb) for lb in labels], rotation=30, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(lb) for lb in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # AI가 수정함: 셀 안에 count 텍스트 표시 (흰색/검정 자동 선택)
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=10,
                )

        fig.tight_layout()
        out_path = os.path.join(exp_dir, f"{file_stem}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log_callback(f"[ExperimentRunner] Confusion matrix saved: {out_path}")
        return out_path  # AI가 수정함: 저장된 PNG 경로 반환

    @staticmethod
    def _build_n_bands_matrix(ok_results, band_methods, model_types, metric_key="test_acc"):
        """
        // AI가 수정함: n_bands별 비교 행렬 생성.
        행 = (band_method, model_type) 조합, 열 = n_bands (숫자 오름차순, "full" 마지막).
        각 셀 = 해당 (band_method, model_type, n_bands) 조합의 metric 최대값.
        반환값:
            n_bands_sorted: list — 정렬된 n_bands 레이블 (숫자 오름차순, "full" 마지막)
            row_labels: list[str] — "(band_method) / (model_type)" 행 레이블
            matrix: list[list[float|nan]] — row_labels × n_bands_sorted 행렬 (acc)
            matrix_gap: list[list[int|None]] — 동일 shape; 각 셀의 best acc를 달성한 gap 값
        """
        # n_bands 값 수집 — 숫자는 int 변환, "full"은 문자열 그대로
        raw_n = set()
        for r in ok_results:
            v = r.get("n_bands", "")
            raw_n.add(v)

        def _sort_key(v):
            try:
                return (0, int(v))
            except (TypeError, ValueError):
                return (1, str(v))  # "full" 등 문자열은 맨 뒤

        n_bands_sorted = sorted(raw_n, key=_sort_key)

        row_labels = [f"{bm} / {mt}" for bm in band_methods for mt in model_types]
        matrix = []
        matrix_gap = []  # AI가 수정함: best acc 달성 시의 gap 행렬 추가
        for bm in band_methods:
            for mt in model_types:
                row = []
                row_gap = []
                for nb in n_bands_sorted:
                    candidates = [
                        r for r in ok_results
                        if r.get("band_method") == bm
                        and r.get("model_type") == mt
                        and r.get("n_bands") == nb
                    ]
                    if candidates:
                        best_r = max(candidates, key=lambda r: float(r.get(metric_key, 0) or 0))
                        row.append(float(best_r.get(metric_key, 0) or 0))
                        # gap: int이면 그대로, 없으면 None
                        gap_val = best_r.get("gap")
                        try:
                            row_gap.append(int(gap_val) if gap_val is not None else None)
                        except (TypeError, ValueError):
                            row_gap.append(None)
                    else:
                        row.append(np.nan)
                        row_gap.append(None)
                matrix.append(row)
                matrix_gap.append(row_gap)

        return n_bands_sorted, row_labels, matrix, matrix_gap

    @staticmethod
    def _write_n_bands_heatmap_png(exp_dir, n_bands_sorted, row_labels, matrix, metric_key, title, file_stem, log_callback):
        """
        // AI가 수정함: n_bands별 비교 heatmap PNG 저장.
        행 = (band_method, model_type), 열 = n_bands.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            log_callback(f"[ExperimentRunner] n_bands heatmap skipped ({file_stem}): {exc}")
            return

        mat = np.array(matrix, dtype=float)
        n_rows = len(row_labels)
        n_cols = len(n_bands_sorted)

        fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2 + 2.0), max(4, n_rows * 0.55 + 1.5)))
        im = ax.imshow(mat, cmap='viridis', aspect='auto')
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([str(nb) for nb in n_bands_sorted], rotation=0)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("n_bands")
        ax.set_title(title)

        for i in range(n_rows):
            for j in range(n_cols):
                value = mat[i][j]
                label = "-" if np.isnan(value) else (f"{value:.2f}" if metric_key != "f1_macro" else f"{value:.3f}")
                ax.text(j, i, label, ha='center', va='center',
                        color='white' if not np.isnan(value) else '#cccccc', fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = os.path.join(exp_dir, f"{file_stem}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        log_callback(f"[ExperimentRunner] n_bands heatmap saved: {out_path}")

    @staticmethod
    def _best_per_bm_mt(results, metric="test_acc"):
        """
        // AI가 수정함: (band_method, model_type) 조합별 최고 성능 trial 선택.
        반환값:
            best_results: list[dict] — 조합별 1개씩, metric 기준 최고 trial
            best_config_map: dict[(band_method, model_type), dict] — 동일 데이터, key 접근용
        """
        from collections import defaultdict  # AI가 수정함: groupby 대용
        groups = defaultdict(list)
        for r in results:
            status = str(r.get("status", ""))
            score = float(r.get(metric, 0) or 0)
            if status.startswith("error") or r.get("error") or score <= 0:  # AI가 수정함: error 필드 + test_acc<=0 제외
                continue
            key = (r["band_method"], r["model_type"])
            groups[key].append((score, r))
        best_results, best_config_map = [], {}
        for key, items in groups.items():
            items.sort(key=lambda x: -x[0])  # AI가 수정함: metric 내림차순 정렬
            best_results.append(items[0][1])
            best_config_map[key] = items[0][1]
        return best_results, best_config_map

    def _write_paper_summary(self, exp_dir, results, band_methods, model_types, csv_timestamp, log_callback):
        """Write paper-ready summary artifacts from aggregate experiment results."""
        ok_results = [r for r in results if r.get("status") == "ok"]
        _best_results, _best_config_map = self._best_per_bm_mt(results)

        summary_path = os.path.join(exp_dir, f"{csv_timestamp}_paper_summary.md")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Matrix Paper Summary\n\n")
            f.write(f"- Generated: {csv_timestamp}\n")
            f.write(f"- Successful trials: {len(ok_results)} / {len(results)}\n")
            f.write(f"- Band methods: {', '.join(band_methods)}\n")
            f.write(f"- Models: {', '.join(model_types)}\n\n")

            if ok_results:
                ranked = sorted(
                    ok_results,
                    key=lambda r: (-float(r.get("test_acc", 0.0)), -float(r.get("f1_macro", 0.0)), float(r.get("gap_pct", 9999.0))),
                )
                f.write("## Best Overall Configuration\n\n")
                self._write_markdown_table(
                    f,
                    ["Rank", "Band Method", "Model", "Test Acc", "F1 Macro", "Precision", "Recall", "Gap %", "Train Time (ms)"],
                    [
                        [
                            idx + 1,
                            row["band_method"],
                            row["model_type"],
                            f"{float(row['test_acc']):.4f}",
                            f"{float(row['f1_macro']):.4f}",
                            f"{float(row['precision_macro']):.4f}",
                            f"{float(row['recall_macro']):.4f}",
                            f"{float(row['gap_pct']):.4f}",
                            f"{float(row['train_time_ms']):.2f}",
                        ]
                        for idx, row in enumerate(ranked)
                    ],
                )

                # AI가 수정함: Best Configuration per (Band Method, Model) 테이블
                if _best_results:
                    f.write("## Best Configuration per (Band Method, Model)\n\n")
                    self._write_markdown_table(
                        f,
                        ["Band Method", "Model", "n_bands", "gap", "Test Acc", "F1 Macro", "Train Time (ms)"],
                        [
                            [
                                r["band_method"],
                                r["model_type"],
                                r.get("n_bands", "-"),
                                r.get("gap", "-"),
                                f"{float(r['test_acc']):.4f}" if r.get("test_acc") not in ("", None) else "-",
                                f"{float(r['f1_macro']):.4f}" if r.get("f1_macro") not in ("", None) else "-",
                                f"{float(r['train_time_ms']):.2f}" if r.get("train_time_ms") not in ("", None) else "-",
                            ]
                            for r in sorted(_best_results, key=lambda x: -float(x.get("test_acc", 0.0)))
                        ],
                    )

                # AI가 수정함: best trial별 클래스별 precision/recall/f1 테이블
                # per_class_report JSON 파싱 — 클래스 이름, 꼴찌 클래스 식별용
                if _best_results:
                    f.write("## Per-Class Metrics (Best Trials)\n\n")
                    for r in sorted(_best_results, key=lambda x: -float(x.get("test_acc", 0.0))):
                        _pcr_raw = r.get("per_class_report", "")
                        if not _pcr_raw:
                            continue
                        try:
                            _pcr = json.loads(_pcr_raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        _bm = r["band_method"]
                        _mt = r["model_type"]
                        _gap = r.get("gap", "-")
                        _nb = r.get("n_bands", "-")
                        _acc = f"{float(r['test_acc']):.2f}%" if r.get("test_acc") not in ("", None) else "-"
                        f.write(f"### {_bm} / {_mt}  (gap={_gap}, n_bands={_nb}, test_acc={_acc})\n\n")
                        self._write_markdown_table(
                            f,
                            ["Class", "Precision", "Recall", "F1", "Support"],
                            [
                                [
                                    cls_name,
                                    f"{vals['precision']:.4f}",
                                    f"{vals['recall']:.4f}",
                                    f"{vals['f1']:.4f}",
                                    vals["support"],
                                ]
                                for cls_name, vals in sorted(_pcr.items())
                            ],
                        )  # AI가 수정함: 클래스명 알파벳순 정렬로 가독성 확보

                f.write("## Matrix Tables\n\n")
                for metric_key, title in _PAPER_MATRIX_METRICS:
                    f.write(f"### {title}\n\n")
                    rows = self._build_metric_matrix(_best_results, band_methods, model_types, metric_key)  # AI가 수정함: first-match → best-per-(bm,mt) 로 수정
                    self._write_markdown_table(f, ["Band Method", *model_types], rows)

                # AI가 수정함: n_bands별 비교 테이블 (행=band_method/model_type, 열=n_bands, 셀=acc%(gap=N))
                n_bands_sorted, row_labels, nb_matrix, nb_matrix_gap = self._build_n_bands_matrix(
                    ok_results, band_methods, model_types, metric_key="test_acc"
                )
                if n_bands_sorted:
                    f.write("## Best Test Accuracy by n_bands\n\n")
                    f.write("> 각 셀은 해당 (밴드선택/모델, n_bands) 조합에서 가장 높은 Test Accuracy와 그 gap 값\n\n")
                    table_headers = ["Band Selection / Model", *[str(nb) for nb in n_bands_sorted]]
                    table_rows = []
                    for i in range(len(row_labels)):
                        row_cells = [row_labels[i]]
                        for j in range(len(n_bands_sorted)):
                            acc = nb_matrix[i][j]
                            gap = nb_matrix_gap[i][j]
                            if np.isnan(acc):
                                row_cells.append("-")
                            else:
                                cell = f"{acc:.2f}%"
                                if gap is not None and gap > 0:
                                    cell += f" (gap={gap})"
                                row_cells.append(cell)
                        table_rows.append(row_cells)
                    self._write_markdown_table(f, table_headers, table_rows)

                # AI가 수정함: per-trial confusion matrix 참조 섹션
                cm_entries = [
                    r for r in ok_results
                    if r.get("confusion_png_path")
                ]
                if cm_entries:
                    f.write("## Confusion Matrices\n\n")
                    for r in cm_entries:
                        png_path = r["confusion_png_path"]
                        png_name = os.path.basename(png_path)
                        f.write(f"### {r['band_method']} / {r['model_type']}\n\n")
                        f.write(f"![Confusion Matrix]({png_name})\n\n")

                error_rows = [r for r in results if r.get("status") != "ok"]
                if error_rows:
                    f.write("## Failed Trials\n\n")
                    self._write_markdown_table(
                        f,
                        ["Band Method", "Model", "Status"],
                        [[r.get("band_method"), r.get("model_type"), r.get("status")] for r in error_rows],
                    )
            else:
                f.write("## No successful trials\n\n")
                f.write("All trials failed. Check the aggregate CSV and logs for error details.\n")

        log_callback(f"[ExperimentRunner] Paper summary saved: {summary_path}")

        if ok_results:
            self._write_heatmap_png(exp_dir, _best_results, band_methods, model_types, "test_acc", "Test Accuracy Matrix", f"{csv_timestamp}_paper_matrix_test_acc", log_callback)
            self._write_heatmap_png(exp_dir, _best_results, band_methods, model_types, "f1_macro", "Macro F1 Matrix", f"{csv_timestamp}_paper_matrix_f1_macro", log_callback)
            self._write_heatmap_png(exp_dir, _best_results, band_methods, model_types, "train_time_ms", "Train Time Matrix (ms)", f"{csv_timestamp}_paper_matrix_train_time_ms", log_callback)
            # AI가 수정함: n_bands별 비교 heatmap (행=band_method/model_type, 열=n_bands)
            n_bands_sorted, row_labels, nb_matrix, _nb_matrix_gap = self._build_n_bands_matrix(
                ok_results, band_methods, model_types, metric_key="test_acc"
            )
            if n_bands_sorted:
                self._write_n_bands_heatmap_png(
                    exp_dir, n_bands_sorted, row_labels, nb_matrix,
                    "test_acc", "Best Test Accuracy by n_bands",
                    f"{csv_timestamp}_paper_matrix_n_bands_test_acc",
                    log_callback,
                )
