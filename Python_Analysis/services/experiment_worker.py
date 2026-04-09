# AI가 수정함: ExperimentWorker — QThread 기반 실험 그리드 워커 (신규)
from PyQt5.QtCore import QObject, pyqtSignal  # AI가 수정함: Qt 기반 시그널 지원
import numpy as np  # AI가 수정함: 배열 연산용

from services.data_loader import load_hsi_data  # AI가 수정함: ENVI 파일 로더
from services.processing_service import ProcessingService  # AI가 수정함: 전처리 단일 경유점
from services.experiment_runner import ExperimentRunner  # AI가 수정함: 그리드 실험 실행기


class ExperimentWorker(QObject):  # AI가 수정함: QObject 상속 (QThread 아님 — moveToThread 패턴)
    """
    // AI가 수정함: ExperimentWorker — 실험 그리드를 백그라운드 스레드에서 실행하는 QThread 워커.
    OptimizationWorker와 동일한 _prepare_base_data 패턴을 따릅니다.
    UI 레이어에 의존하지 않습니다.
    """

    # AI가 수정함: 4개 시그널 정의
    progress_update = pyqtSignal(int)        # AI가 수정함: 0-100
    log_message = pyqtSignal(str)            # AI가 수정함: 로그 문자열
    experiment_finished = pyqtSignal(bool)   # AI가 수정함: 성공 여부
    base_data_ready = pyqtSignal(object, object)  # AI가 수정함: 캐시 handoff

    def __init__(  # AI가 수정함: 생성자 — OptimizationWorker.__init__ 패턴 동일하게 따름
        self,
        file_groups,         # AI가 수정함: {class_name: [file_paths]} dict
        vm_state,            # AI가 수정함: VM 상태 스냅샷 dict (mode, threshold, mask_rules, prep_chain, ...)
        main_vm_cache,       # AI가 수정함: 메인 VM 데이터 캐시 dict {path: (cube, waves)}
        params,              # AI가 수정함: 실험 파라미터 dict
        base_data_cache=None  # AI가 수정함: 기존 base data 캐시 dict {path: (X_base, None)}
    ):
        super().__init__()  # AI가 수정함: QObject 초기화
        self.file_groups = file_groups  # AI가 수정함: 클래스별 파일 그룹

        # AI가 수정함: VM 상태 스냅샷 (Read-Only)
        self.processing_mode = vm_state['mode']  # AI가 수정함: 처리 모드 (Raw/Reflectance)
        self.threshold = vm_state['threshold']   # AI가 수정함: 배경 마스킹 임계값
        self.mask_rules = vm_state['mask_rules']  # AI가 수정함: 마스크 규칙
        self.base_prep_chain = vm_state['prep_chain']  # AI가 수정함: 전처리 체인 (base data에는 미적용)

        # AI가 수정함: 참조 데이터
        self.white_ref = vm_state.get('white_ref')  # AI가 수정함: 화이트 레퍼런스
        self.dark_ref = vm_state.get('dark_ref')    # AI가 수정함: 다크 레퍼런스
        self.exclude_bands_str = vm_state.get('exclude_bands', "")  # AI가 수정함: 제외 밴드 문자열

        self.data_cache = main_vm_cache  # AI가 수정함: 메인 VM 큐브 캐시
        self.base_data_cache = base_data_cache  # AI가 수정함: Base Data 핸드오프 캐시

        # AI가 수정함: params dict에서 실험 파라미터 추출
        self.band_methods = params.get('band_methods', ['spa'])  # AI가 수정함: 밴드 선택 방법 목록
        self.model_types = params.get('model_types', ['LDA'])    # AI가 수정함: 모델 종류 목록
        self.n_bands = params.get('n_bands', 5)                  # AI가 수정함: 선택 밴드 수
        self.test_ratio = params.get('test_ratio', 0.2)          # AI가 수정함: 테스트 분리 비율
        self.output_dir = params.get('output_dir', 'output/experiments')  # AI가 수정함: CSV 저장 디렉토리
        self.excluded_files = params.get('excluded_files', set())  # AI가 수정함: 제외 파일 집합
        self.band_selection_method = params.get('band_selection_method', 'spa')  # AI가 수정함: 단일 밴드 선택 방법 (미사용 — band_methods 우선)

        # AI가 수정함: 스레드 안전 상태
        self.is_running = True  # AI가 수정함: 실행 상태 플래그 — stop() 호출 시 False

        # AI가 수정함: Base Data 캐시 (run() 이전에는 None)
        self.cached_X = None  # AI가 수정함: 로드된 전체 pixel 행렬 (N, B)
        self.cached_y = None  # AI가 수정함: 로드된 전체 레이블 벡터 (N,)

    def run(self):  # AI가 수정함: QThread에서 호출되는 메인 실행 메서드
        """
        // AI가 수정함: 실험 그리드 실행 — _prepare_base_data → ExperimentRunner.run_grid
        """
        try:
            self.log_message.emit("=== Starting Experiment Grid (Background) ===")  # AI가 수정함: 시작 로그

            # AI가 수정함: Step 1 — Base Data 로드 (1회, 캐시 활용)
            self.log_message.emit("Pre-loading and masking data (Full Dataset)...")  # AI가 수정함: 로딩 시작 로그
            if not self._prepare_base_data():  # AI가 수정함: 실패 시 즉시 종료
                self.log_message.emit("Error: No valid data found for experiment.")  # AI가 수정함: 에러 로그
                self.experiment_finished.emit(False)  # AI가 수정함: 실패 시그널
                return

            assert self.cached_X is not None and self.cached_y is not None  # AI가 수정함: 캐시 준비 후 None 방지
            self.log_message.emit(  # AI가 수정함: 데이터 준비 완료 로그
                f"Data Ready: {self.cached_X.shape[0]:,} pixels, {self.cached_X.shape[1]} bands."
            )

            # AI가 수정함: Step 2 — exclude_indices 계산 (OptimizationWorker._evaluate_cached_data 패턴)
            exclude_indices = []  # AI가 수정함: 제외 밴드 인덱스 목록 초기화
            if self.exclude_bands_str:  # AI가 수정함: 제외 밴드 문자열이 있을 때만 파싱
                try:
                    for part in self.exclude_bands_str.split(','):  # AI가 수정함: 콤마 구분 파싱
                        if '-' in part:  # AI가 수정함: 범위 표기 (e.g. "10-20")
                            start, end = map(int, part.split('-'))  # AI가 수정함: 시작-끝 분리
                            exclude_indices.extend(range(start - 1, end))  # AI가 수정함: 0-based 변환
                        else:
                            exclude_indices.append(int(part) - 1)  # AI가 수정함: 단일 인덱스 0-based 변환
                except Exception:  # AI가 수정함: 파싱 실패 시 무시 (방어적)
                    pass

            # AI가 수정함: Step 3 — ExperimentRunner.run_grid 호출
            results = ExperimentRunner().run_grid(  # AI가 수정함: 실험 그리드 실행
                X_base=self.cached_X,                   # AI가 수정함: 전처리 전 base data
                y=self.cached_y,                         # AI가 수정함: 클래스 레이블
                prep_chain=self.base_prep_chain,         # AI가 수정함: vm_state['prep_chain'] 전달
                band_methods=self.band_methods,          # AI가 수정함: 밴드 선택 방법 목록
                model_types=self.model_types,            # AI가 수정함: 모델 종류 목록
                n_bands=self.n_bands,                    # AI가 수정함: 선택 밴드 수
                test_ratio=self.test_ratio,              # AI가 수정함: 테스트 분리 비율
                output_dir=self.output_dir,              # AI가 수정함: CSV 저장 디렉토리
                exclude_indices=exclude_indices,         # AI가 수정함: 제외 밴드 인덱스
                log_callback=self.log_message.emit,      # AI가 수정함: 로그 시그널 콜백
                stop_flag=lambda: not self.is_running,   # AI가 수정함: 중단 조건 람다
            )

            # AI가 수정함: Confusion Matrix PNG 저장 (플랜 line 91)
            if results:  # AI가 수정함: 결과가 있을 때만 저장
                try:
                    import os as _os  # AI가 수정함: os 지연 import
                    from datetime import datetime as _dt  # AI가 수정함: 타임스탬프용
                    import matplotlib  # AI가 수정함: backend 설정 필수 (백그라운드 스레드)
                    matplotlib.use('Agg')  # AI가 수정함: GUI 없는 환경용 backend
                    import matplotlib.pyplot as plt  # AI가 수정함: 플롯용
                    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # AI가 수정함: confusion matrix
                    from sklearn.model_selection import train_test_split as _tts  # AI가 수정함: split
                    from services.processing_service import ProcessingService as _PS  # AI가 수정함: 전처리
                    from services.band_selection_service import select_best_bands as _sbb  # AI가 수정함: 밴드 선택
                    from services.learning_service import LearningService as _LS  # AI가 수정함: 학습

                    ok_results = [r for r in results if r.get('status') == 'ok']  # AI가 수정함: 성공 결과만
                    if ok_results:
                        first_ok = ok_results[0]  # AI가 수정함: 첫 번째 성공 결과 기준
                        bm_cm = first_ok['band_method']  # AI가 수정함: 밴드 방법
                        mt_cm = first_ok['model_type']   # AI가 수정함: 모델 종류

                        X_prep_cm = _PS.apply_preprocessing_chain(  # AI가 수정함: 전처리 적용
                            self.cached_X.copy(), self.base_prep_chain
                        )
                        B_cm = X_prep_cm.shape[1]  # AI가 수정함: 전처리 후 밴드 수
                        sel_idx_cm, _, _ = _sbb(  # AI가 수정함: 밴드 선택
                            X_prep_cm.reshape(-1, 1, B_cm), self.n_bands,
                            method=bm_cm, labels=self.cached_y,
                        )
                        X_sub_cm = X_prep_cm[:, sel_idx_cm]  # AI가 수정함: 선택된 밴드 슬라이싱
                        model_cm, _ = _LS().train_model(  # AI가 수정함: 학습
                            X_sub_cm, self.cached_y, model_type=mt_cm, test_ratio=self.test_ratio
                        )
                        try:  # AI가 수정함: stratify 실패 대비 fallback
                            _, X_te, _, y_te = _tts(
                                X_sub_cm, self.cached_y,
                                test_size=self.test_ratio, random_state=42, stratify=self.cached_y
                            )
                        except ValueError:  # AI가 수정함: 클래스 샘플 부족 fallback
                            _, X_te, _, y_te = _tts(
                                X_sub_cm, self.cached_y,
                                test_size=self.test_ratio, random_state=42
                            )
                        y_pred_cm = model_cm.predict(X_te)  # AI가 수정함: 예측
                        cm_mat = confusion_matrix(y_te, y_pred_cm)  # AI가 수정함: confusion matrix 계산

                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))  # AI가 수정함: figure 생성
                        ConfusionMatrixDisplay(confusion_matrix=cm_mat).plot(ax=ax_cm, colorbar=False)  # AI가 수정함: 플롯
                        ax_cm.set_title(f"Confusion Matrix ({bm_cm} / {mt_cm})")  # AI가 수정함: 제목 설정

                        _exp_dir_cm = _os.path.join(self.output_dir, 'experiments')  # AI가 수정함: experiments 서브디렉토리
                        _os.makedirs(_exp_dir_cm, exist_ok=True)  # AI가 수정함: 디렉토리 보장
                        cm_ts = _dt.now().strftime("%Y%m%d_%H%M%S")  # AI가 수정함: 타임스탬프
                        cm_path = _os.path.join(_exp_dir_cm, f"{cm_ts}_confusion_matrix.png")  # AI가 수정함: PNG 경로
                        fig_cm.savefig(cm_path, dpi=100, bbox_inches='tight')  # AI가 수정함: PNG 저장
                        plt.close(fig_cm)  # AI가 수정함: figure 닫기 (메모리 해제)
                        self.log_message.emit(f"Confusion matrix saved: {cm_path}")  # AI가 수정함: 저장 완료 로그
                except Exception as _cm_err:  # AI가 수정함: confusion matrix 저장 실패 → 비치명적 경고
                    self.log_message.emit(f"[Warning] Confusion matrix save failed: {_cm_err}")  # AI가 수정함: 경고

            # AI가 수정함: Step 4 — 완료 처리
            self.progress_update.emit(100)        # AI가 수정함: 완료 시 100% emit
            self.experiment_finished.emit(True)   # AI가 수정함: 성공 시그널

        except Exception as e:  # AI가 수정함: 예외 처리 — 실패 시그널 emit
            self.log_message.emit(f"Experiment Error: {e}")  # AI가 수정함: 에러 로그
            import traceback  # AI가 수정함: traceback 지연 import
            self.log_message.emit(traceback.format_exc())  # AI가 수정함: 상세 에러 출력
            self.experiment_finished.emit(False)  # AI가 수정함: 실패 시그널

    def _prepare_base_data(self):  # AI가 수정함: OptimizationWorker._prepare_base_data 패턴 동일하게 구현
        """
        // AI가 수정함: 모든 파일을 로드하고 마스킹 후 Base Data를 캐시에 저장.
        NO Subsampling Limit — 전체 데이터 사용.
        ProcessingService.get_base_data() 경유 필수.
        """
        # AI가 수정함: Handoff Cache 초기화 (없으면 빈 dict)
        if self.base_data_cache is None:  # AI가 수정함: None 체크
            self.base_data_cache = {}  # AI가 수정함: 빈 dict로 초기화

        # AI가 수정함: 제외 클래스 이름 목록 (예약어)
        EXCLUDED_NAMES = ["-", "unassigned", "trash", "ignore"]  # AI가 수정함: 예약 제외 이름

        # AI가 수정함: 유효한 클래스 그룹만 추출
        valid_groups = {}  # AI가 수정함: 유효 클래스 그룹 dict
        for name, files in self.file_groups.items():  # AI가 수정함: 전체 그룹 순회
            if len(files) > 0 and name.lower() not in EXCLUDED_NAMES:  # AI가 수정함: 비어있지 않고 예약어 아닌 것만
                valid_groups[name] = files  # AI가 수정함: 유효 그룹 추가

        valid_groups = dict(sorted(valid_groups.items()))  # AI가 수정함: 재현성 보장 (정렬)

        if len(valid_groups) < 2:  # AI가 수정함: 클래스 수 부족 체크
            return False  # AI가 수정함: 최소 2개 클래스 필요

        X_all = []  # AI가 수정함: 전체 픽셀 행렬 누적 리스트
        y_all = []  # AI가 수정함: 전체 레이블 누적 리스트

        # AI가 수정함: 전체 파일 로드 (클래스별)
        total_classes = len(valid_groups)  # AI가 수정함: 전체 클래스 수
        for label_id, (class_name, files) in enumerate(valid_groups.items()):  # AI가 수정함: 클래스 순회
            class_pixels = 0  # AI가 수정함: 클래스별 픽셀 수 추적
            sorted_files = sorted(files)  # AI가 수정함: 파일 로딩 순서 고정 (재현성 보장)

            for f in sorted_files:  # AI가 수정함: 파일별 순회
                if not self.is_running:  # AI가 수정함: 중단 요청 체크
                    return False  # AI가 수정함: 중단 시 즉시 반환

                # AI가 수정함: 제외 파일 체크
                if f in self.excluded_files:  # AI가 수정함: 제외 파일이면 건너뜀
                    continue  # AI가 수정함: 다음 파일로

                try:
                    # AI가 수정함: Base Data 캐시 체크 (Handoff)
                    if f in self.base_data_cache:  # AI가 수정함: 캐시 히트
                        data, _ = self.base_data_cache[f]  # AI가 수정함: (X_base, None) 언팩

                        # AI가 수정함: 유효 픽셀만 누적
                        if data.shape[0] > 0:  # AI가 수정함: 빈 데이터 제외
                            X_all.append(data)  # AI가 수정함: 픽셀 누적
                            y_all.append(np.full(data.shape[0], label_id))  # AI가 수정함: 레이블 누적
                            class_pixels += data.shape[0]  # AI가 수정함: 클래스 픽셀 수 갱신
                    else:
                        # AI가 수정함: 디스크에서 로드
                        if f in self.data_cache:  # AI가 수정함: 큐브 캐시 체크
                            cube, waves = self.data_cache[f]  # AI가 수정함: 캐시에서 로드
                        else:
                            cube, waves = load_hsi_data(f)  # AI가 수정함: 디스크에서 로드
                            cube = np.nan_to_num(cube)       # AI가 수정함: NaN/Inf 제거 (1회만)
                            self.data_cache[f] = (cube, waves)  # AI가 수정함: 큐브 캐시 저장

                        # AI가 수정함: ProcessingService.get_base_data() 경유 (마스킹 + ref 변환, 전처리 없음)
                        data, mask = ProcessingService.get_base_data(  # AI가 수정함: 단일 경유점
                            cube,
                            mode=self.processing_mode,   # AI가 수정함: 처리 모드
                            threshold=self.threshold,     # AI가 수정함: 배경 임계값
                            mask_rules=self.mask_rules,   # AI가 수정함: 마스크 규칙
                            white_ref=self.white_ref,     # AI가 수정함: 화이트 레퍼런스
                            dark_ref=self.dark_ref        # AI가 수정함: 다크 레퍼런스
                        )

                        # AI가 수정함: 유효 픽셀이 있을 때만 누적
                        if data.shape[0] > 0:  # AI가 수정함: 빈 데이터 제외
                            X_all.append(data)  # AI가 수정함: 픽셀 누적
                            y_all.append(np.full(data.shape[0], label_id))  # AI가 수정함: 레이블 누적
                            class_pixels += data.shape[0]  # AI가 수정함: 클래스 픽셀 수 갱신

                            # AI가 수정함: Base Data 캐시 핸드셰이크 emit (파일별)
                            self.base_data_ready.emit(f, (data, None))  # AI가 수정함: 캐시 handoff 시그널

                except Exception as e:  # AI가 수정함: 파일 로드 에러 — Strict Mode (은폐 금지)
                    self.log_message.emit(f"Critical Error loading {f}: {e}")  # AI가 수정함: 에러 로그
                    return False  # AI가 수정함: 치명적 에러 시 즉시 실패 반환

            # AI가 수정함: 클래스별 로딩 완료 로그
            self.log_message.emit(  # AI가 수정함: 클래스 로딩 완료 로그
                f"  [{label_id + 1}/{total_classes}] {class_name}: {class_pixels:,} pixels"
            )

        if not X_all:  # AI가 수정함: 유효 데이터 없음 체크
            return False  # AI가 수정함: 빈 결과 시 실패

        # AI가 수정함: 전체 데이터 스택 (NO LIMIT — 전체 사용)
        X = np.vstack(X_all)          # AI가 수정함: (N, B) 행렬 스택
        y = np.concatenate(y_all)     # AI가 수정함: (N,) 레이블 벡터 결합

        self.cached_X = X  # AI가 수정함: 캐시 저장
        self.cached_y = y  # AI가 수정함: 캐시 저장

        return True  # AI가 수정함: 성공

    def stop(self):  # AI가 수정함: 중단 요청 메서드
        """// AI가 수정함: 실행 중단 요청 — is_running = False 설정"""
        self.is_running = False  # AI가 수정함: 중단 플래그 설정
