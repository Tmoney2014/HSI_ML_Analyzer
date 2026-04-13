from PyQt5.QtCore import QObject, pyqtSignal
import copy

_TARGET_KEYS = ["SimpleDeriv"]

class OptimizationService(QObject):
    """
    Service for Auto-ML Hyperparameter Optimization (Global Search).
    Implements Grid Search for (Band Count x Gap Size) to find Global Optimum.
    Strategies:
    1. Global Grid Search: Band Count (5~40) x Gap Size (1~40)
    """
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def run_optimization(self, initial_params, trial_callback, band_methods=None):  # AI가 수정함: band_methods 파라미터 추가
        """
        Run Global Optimization (Grid Search).
        Outer loop: band_methods × n_bands × gap (3D search space).
        If no Gap-tunable preprocessing is found, runs Band-Only Optimization.
        """
        self.log_message.emit("=== Starting Global Optimization (Grid Search) ===")
        
        # 0. Initialize
        best_params = copy.deepcopy(initial_params)
        best_acc = 0.0
        history = []
        
        # Detect Gap-tunable preprocessing (SimpleDeriv etc.)
        has_simpledv = any(step['name'] in _TARGET_KEYS for step in initial_params.get('prep', []))  # AI가 수정함: has_simpledv로 리팩터

        # 1. Setup Gap Search Space
        if has_simpledv:
            self.log_message.emit("Target: Find best combination of [Band Method] × [Band Count] × [Gap Size]")  # AI가 수정함: 3D 검색 안내
            _gap_range = list(range(1, 41))  # Gap 1~40
        else:
            self.log_message.emit("⚠️ No Gap-tunable preprocessing found. Switching to [Band-Only Optimization].")  # AI가 수정함: 기존 동작 유지
            self.log_message.emit("Target: Find best [Band Method] × [Band Count]")  # AI가 수정함: 2D 검색 안내
            _gap_range = [0]  # Dummy Gap (No change)

        # 2. Resolve band methods list  # AI가 수정함: band_methods 해소
        _band_methods = band_methods if band_methods is not None else [initial_params.get('band_selection_method', 'spa')]  # AI가 수정함: 기본값은 initial_params에서 추출

        if 'spa_lda_greedy' in _band_methods:  # AI가 수정함: spa_lda_greedy 경고
            self.log_message.emit("[Optimize] Warning: spa_lda_greedy included — may be very slow for large grids.")  # AI가 수정함: 느린 method 경고

        self.log_message.emit("-" * 40)

        # 3. Optimization Loop — 3 levels: method × n_bands × gap  # AI가 수정함: 3중 루프
        for method in _band_methods:  # AI가 수정함: 외부 루프: band method
            is_full_band = (method == 'full')  # AI가 수정함: Full Band 감지
            if is_full_band:  # AI가 수정함: Full Band는 단일 trial
                _band_range = [initial_params.get('n_features', initial_params.get('n_bands', 5))]  # AI가 수정함: Full Band 단일 값
            else:  # AI가 수정함: SPA 등 일반 모드
                _band_range = list(range(5, 41, 5))  # AI가 수정함: 5~40 step 5

            self.log_message.emit(f"[Method] band_selection_method={method}")  # AI가 수정함: method 로그

            for n_features in _band_range:  # AI가 수정함: 중간 루프: band count
                for gap in _gap_range:  # AI가 수정함: 내부 루프: gap
                    try:
                        p = copy.deepcopy(initial_params)
                        p['band_selection_method'] = method  # AI가 수정함: method 주입
                        p['n_features'] = n_features  # AI가 수정함: n_features 주입

                        # Apply Gap (only if SimpleDeriv exists)
                        if has_simpledv and gap > 0:  # AI가 수정함: has_simpledv 사용
                            for step in p.get('prep', []):
                                if step.get('name') in _TARGET_KEYS:
                                    step['params']['gap'] = gap
                                    break

                        # Evaluate
                        acc = trial_callback(p)
                        history.append((copy.deepcopy(p), acc))

                        # Update Best
                        if acc > best_acc:
                            diff = acc - best_acc
                            best_acc = acc
                            best_params = copy.deepcopy(p)  # AI가 수정함: deepcopy로 안전하게 보존
                            band_label = "Full Band" if is_full_band else n_features  # AI가 수정함: Full Band 표시
                            msg = f"✨ New Best! {acc:.2f}% (+{diff:.2f}%) | Method={method}, Bands={band_label}"  # AI가 수정함: method 포함
                            if has_simpledv:
                                msg += f", Gap={gap}"
                            self.log_message.emit(msg)

                        # Per-trial log  # AI가 수정함: 필수 trial 로그
                        self.log_message.emit(f"[Trial] method={method} n={n_features} gap={gap} acc={acc:.4f}")  # AI가 수정함: 3D trial 로그

                    except Exception as e:  # AI가 수정함: Stop 예외 처리
                        if "Stopped" in str(e) or "Optimization Stopped" in str(e):  # AI가 수정함: Stop 시그널 감지
                            self.log_message.emit("Optimization stopped by user.")  # AI가 수정함: 중단 로그
                            return best_params, history  # AI가 수정함: 즉시 반환 (raise 없음)
                        raise  # AI가 수정함: 다른 예외는 그대로 전파

        self.log_message.emit("-" * 40)
        self.log_message.emit(f"🏆 Optimization Done. Best: {best_acc:.2f}%")
        
        # 4. Final Report
        self._generate_report(best_params, best_acc, history)  # AI가 수정함: output_dir=None 기본 동작 유지
        
        return best_params, history

    def _generate_report(self, best_params, current_acc, history, output_dir=None):  # AI가 수정함: output_dir 인자 추가
        """Generate final optimization report."""  # AI가 수정함: 파일 저장 옵션 지원
        import csv, json, os  # AI가 수정함: CSV/JSON 저장용 모듈
        report = ["\n🎉 Optimization Completed!", "-" * 40, "[Final Configuration]"]
        
        gap_val = 0
        for step in best_params.get('prep', []):
            if step.get('name') in _TARGET_KEYS:
                gap_val = step.get('params', {}).get('gap', 0)
                break
        report.append(f" • Gap Size: {gap_val}")

        is_full_band = best_params.get('band_selection_method') == 'full'  # AI가 수정함: Full Band 감지
        if is_full_band:  # AI가 수정함: Full Band이면 전체 밴드 표시
            report.append(" • Band Count: Full Band")  # AI가 수정함:
        else:  # AI가 수정함: SPA 등은 기존 표시 유지
            report.append(f" • Band Count: {best_params.get('n_features')}")  # AI가 수정함:
        report.extend(["-" * 40, f"🏆 Final Best Accuracy: {current_acc:.2f}%", "-" * 40, "📜 Top 3 Configurations"])
        
        # Sort by Accuracy
        sorted_history = sorted(history, key=lambda x: x[1], reverse=True)
        
        # Deduplicate (Params can be same)
        seen = set()
        unique_top = []
        for p, acc in sorted_history:
            # Create hashable signature
            _gap = 0
            for _s in p.get('prep', []):
                if _s.get('name') in _TARGET_KEYS:
                    _gap = _s.get('params', {}).get('gap', 0)
                    break
            sig = (p.get('band_selection_method', ''), p['n_features'], _gap)
            
            if sig not in seen:
                seen.add(sig)
                unique_top.append((p, acc))
            if len(unique_top) >= 3: break
            
        for i, (p, acc) in enumerate(unique_top):
            info = ["Bands=Full Band" if is_full_band else f"Bands={p['n_features']}"]  # AI가 수정함: Full Band 표시
            for s in p['prep']:
                if s['name'] in _TARGET_KEYS: info.append(f"Gap={s['params'].get('gap')}")
            medal = ["🥇", "🥈", "🥉"][i]
            report.append(f"{medal} #{i+1}: {acc:.2f}% | {', '.join(info)}")

        self.log_message.emit("\n".join(report))

        if output_dir is not None:  # AI가 수정함: output_dir이 있을 때만 파일 저장
            os.makedirs(output_dir, exist_ok=True)  # AI가 수정함: 저장 폴더 보장
            csv_path = os.path.join(output_dir, "optimization_history.csv")  # AI가 수정함: CSV 경로
            json_path = os.path.join(output_dir, "optimization_best.json")  # AI가 수정함: JSON 경로
            with open(csv_path, "w", newline="", encoding="utf-8") as f:  # AI가 수정함: CSV 기록 시작
                writer = csv.DictWriter(f, fieldnames=["trial_no", "band_method", "n_bands", "gap", "train_acc", "test_acc", "f1_macro", "precision_macro", "recall_macro", "gap_pct", "train_time_ms", "status"])  # AI가 수정함: 12-col schema
                writer.writeheader()  # AI가 수정함: CSV header 출력
                for idx, (params, acc) in enumerate(history, start=1):  # AI가 수정함: trial row 생성
                    band_method = params.get("band_selection_method", "")  # AI가 수정함: band method 추출
                    writer.writerow({  # AI가 수정함: 12개 컬럼 고정 출력
                        "trial_no": idx,  # AI가 수정함: trial 번호
                        "band_method": band_method,  # AI가 수정함: band method
                        "n_bands": "full" if band_method == "full" else params.get("n_features", ""),  # AI가 수정함: full band 처리
                        "gap": params.get("gap", 0),  # AI가 수정함: gap 기록
                        "train_acc": "",  # AI가 수정함: future extension placeholder
                        "test_acc": acc,  # AI가 수정함: 평가 정확도
                        "f1_macro": "",  # AI가 수정함: future extension placeholder
                        "precision_macro": "",  # AI가 수정함: future extension placeholder
                        "recall_macro": "",  # AI가 수정함: future extension placeholder
                        "gap_pct": "",  # AI가 수정함: future extension placeholder
                        "train_time_ms": "",  # AI가 수정함: future extension placeholder
                        "status": "ok",  # AI가 수정함: 상태 고정
                    })  # AI가 수정함: row 종료
            with open(json_path, "w", encoding="utf-8") as f:  # AI가 수정함: JSON 기록 시작
                f.write(json.dumps(best_params, ensure_ascii=False, indent=2))  # AI가 수정함: best params 저장

            # n_bands별 최대 test_acc 요약 테이블 → _summary.md  # AI가 수정함: n_bands 비교 요약 추가
            self._write_n_bands_summary_md(output_dir, history)  # AI가 수정함: MD 요약 저장

    def _write_n_bands_summary_md(self, output_dir, history):  # AI가 수정함: n_bands별 최대 test_acc 요약 MD 생성
        """
        history에서 (band_method, n_bands) 조합별 최대 test_acc를 집계하여
        Markdown 테이블로 저장한다.

        행 = band_method (알파벳 순),
        열 = n_bands (숫자 오름차순, 'full' 맨 뒤),
        셀 = max test_acc (%) — 해당 조합이 없으면 '-'.
        """
        import os  # AI가 수정함: 로컬 import (이미 상위 _generate_report에서 import 됐으나 메서드 독립성 보장)

        # 1. 집계: (band_method, n_bands_label) → max acc  # AI가 수정함: 집계 로직
        agg = {}  # AI가 수정함: {(band_method, n_bands_label): max_acc}
        for params, acc in history:  # AI가 수정함: history 순회
            bm = params.get("band_selection_method", "unknown")  # AI가 수정함: band method 추출
            if bm == "full":  # AI가 수정함: full band → 레이블 "full"
                nb_label = "full"
            else:
                nb_label = str(params.get("n_features", "?"))  # AI가 수정함: 정수 → 문자열
            key = (bm, nb_label)  # AI가 수정함: 집계 키
            if key not in agg or acc > agg[key]:  # AI가 수정함: 최댓값 유지
                agg[key] = acc

        if not agg:  # AI가 수정함: 데이터 없으면 저장 생략
            return

        # 2. 고유 band_methods / n_bands 목록 정렬  # AI가 수정함: 축 정렬
        band_methods = sorted({k[0] for k in agg})  # AI가 수정함: 알파벳 순
        raw_nb = {k[1] for k in agg}  # AI가 수정함: n_bands 레이블 집합
        numeric_nb = sorted([nb for nb in raw_nb if nb != "full"], key=lambda x: int(x))  # AI가 수정함: 숫자 오름차순
        n_bands_cols = numeric_nb + (["full"] if "full" in raw_nb else [])  # AI가 수정함: full 맨 뒤

        # 3. Markdown 테이블 생성  # AI가 수정함: MD 렌더링
        header = "| Band Method | " + " | ".join(n_bands_cols) + " |"  # AI가 수정함: 헤더 행
        separator = "| --- | " + " | ".join(["---"] * len(n_bands_cols)) + " |"  # AI가 수정함: 구분선
        rows = []  # AI가 수정함: 데이터 행 목록
        for bm in band_methods:  # AI가 수정함: band_method 행 순회
            cells = []  # AI가 수정함: 셀 값 목록
            for nb in n_bands_cols:  # AI가 수정함: n_bands 열 순회
                val = agg.get((bm, nb))  # AI가 수정함: 해당 조합의 최대 acc
                cells.append(f"{val:.2f}%" if val is not None else "-")  # AI가 수정함: 포맷 또는 '-'
            rows.append(f"| {bm} | " + " | ".join(cells) + " |")  # AI가 수정함: 행 완성

        lines = [  # AI가 수정함: MD 파일 전체 내용
            "# Optimization Summary — Best Test Accuracy by n_bands",
            "",
            "행: Band Selection Method | 열: n_bands | 셀: 최대 Test Accuracy (gap 무관)",
            "",
            header,
            separator,
        ] + rows + [""]

        md_path = os.path.join(output_dir, "optimization_summary.md")  # AI가 수정함: 저장 경로
        with open(md_path, "w", encoding="utf-8") as f:  # AI가 수정함: MD 파일 저장
            f.write("\n".join(lines))  # AI가 수정함: 줄바꿈 연결 후 쓰기


