from PyQt5.QtCore import QObject, pyqtSignal
import copy

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
        
    def run_optimization(self, initial_params, trial_callback):
        """
        Run Global Optimization (Grid Search).
        If no Gap-tunable preprocessing is found, runs Band-Only Optimization.
        """
        self.log_message.emit("=== Starting Global Optimization (Grid Search) ===")
        
        # 0. Initialize
        best_params = copy.deepcopy(initial_params)
        best_acc = 0.0
        history = []
        
        # Target Preprocessing Steps for Gap Tuning
        target_keys = ["SimpleDeriv"]
        target_prep_name = None
        for step in best_params['prep']:
            if step['name'] in target_keys:
                target_prep_name = step['name']
                break
        
        # 1. Setup Search Space
        band_start, band_end, band_step = 5, 40, 5  # AI가 수정함: Band Count 검색 범위 고정
        is_full_band = initial_params.get('band_selection_method') == 'full'  # AI가 수정함: Full Band 모드 감지
        if is_full_band:  # AI가 수정함: Full Band는 n_features를 무시하므로 단일 값만 사용
            band_range = [initial_params.get('n_features', band_start)]  # AI가 수정함: Band Count 단일 trial로 축소
        else:  # AI가 수정함: SPA 등 일반 모드에서는 기존 그리드 유지
            band_range = range(band_start, band_end + 1, band_step)  # AI가 수정함: 기존 Band Count 그리드 유지

        if target_prep_name:
            self.log_message.emit("Target: Find best combination of [Band Count] and [Gap Size]")  # AI가 수정함: 기존 목적 로그 유지
            gap_range = range(1, 41) # Gap 1~40
            if is_full_band:  # AI가 수정함: Full Band면 Band Count 탐색 생략
                self.log_message.emit("ℹ️ Full Band mode: skipping Band Count search (using all bands)")  # AI가 수정함: Band Count 반복 제거 안내
                self.log_message.emit(f"Search Space: Full Band × Gap 1~40 = {len(gap_range)} trials")  # AI가 수정함: 실제 trial 수 반영
            else:  # AI가 수정함: SPA 경로는 기존 메시지 유지
                self.log_message.emit(f"Search Space: Bands {band_start}~{band_end} x Gap 1~40 = ~320 trials")  # AI가 수정함: 기존 추정 trial 수 유지
        else:
            self.log_message.emit("⚠️ No Gap-tunable preprocessing found. Switching to [Band-Only Optimization].")  # AI가 수정함: 기존 동작 유지
            self.log_message.emit("Target: Find best [Band Count]")  # AI가 수정함: 기존 동작 유지
            gap_range = [0] # Dummy Gap (No change)
            if is_full_band:  # AI가 수정함: Full Band 단일 trial 안내
                self.log_message.emit("ℹ️ Full Band mode: single trial (no variable parameters)")  # AI가 수정함: Full Band는 단일 trial
            else:  # AI가 수정함: SPA 경로는 기존 메시지 유지
                self.log_message.emit(f"Search Space: Bands {band_start}~{band_end}")  # AI가 수정함: 기존 Band Count 검색 안내

        self.log_message.emit("-" * 40)
        
        # 2. Optimization Loop
        for n_features in band_range:  # AI가 수정함: Full Band일 때 단일값, SPA일 때 전체 그리드 반복
            if not is_full_band:  # AI가 수정함: Full Band는 이미 상단에서 안내했으므로 생략
                self.log_message.emit(f"Checking Band Count: {n_features}...")  # AI가 수정함: Band Count trial 로그 유지
            
            for gap in gap_range:
                # Construct Params
                p = copy.deepcopy(initial_params)
                p['n_features'] = n_features
                
                # Apply Gap (only if target exists)
                if target_prep_name and gap > 0:
                    for step in p['prep']:
                        if step['name'] == target_prep_name:
                            step['params']['gap'] = gap
                            break
                
                # Evaluate
                acc = trial_callback(p)
                history.append((copy.deepcopy(p), acc))
                
                # Update Best
                band_label = "Full Band" if is_full_band else n_features  # AI가 수정함: Full Band 표시
                if acc > best_acc:
                    diff = acc - best_acc
                    best_acc = acc
                    best_params = p
                    msg = f"✨ New Best! {acc:.2f}% (+{diff:.2f}%) | Bands={band_label}"  # AI가 수정함: 조건부 표시
                    if target_prep_name: msg += f", Gap={gap}"
                    self.log_message.emit(msg)
                else:
                     if target_prep_name:
                         self.log_message.emit(f"   • Gap={gap}: {acc:.2f}%")
                     else:
                         self.log_message.emit(f"   • Bands={band_label}: {acc:.2f}%")  # AI가 수정함: Full Band 표시
        
        self.log_message.emit("-" * 40)
        self.log_message.emit(f"🏆 Optimization Done. Best: {best_acc:.2f}%")
        
        # 3. Final Report
        self._generate_report(best_params, best_acc, history)  # AI가 수정함: output_dir=None 기본 동작 유지
        
        return best_params, history

    def _generate_report(self, best_params, current_acc, history, output_dir=None):  # AI가 수정함: output_dir 인자 추가
        """Generate final optimization report."""  # AI가 수정함: 파일 저장 옵션 지원
        import csv, json, os  # AI가 수정함: CSV/JSON 저장용 모듈
        report = ["\n🎉 Optimization Completed!", "-" * 40, "[Final Configuration]"]
        
        target_keys = ["SimpleDeriv"]
        for step in best_params['prep']:
            if step['name'] in target_keys:
                report.append(f" • Gap Size: {step['params'].get('gap')}")

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
            sig = (p['n_features'],)
            for s in p['prep']:
                if s['name'] in target_keys: sig += (s['params'].get('gap'),)
            
            if sig not in seen:
                seen.add(sig)
                unique_top.append((p, acc))
            if len(unique_top) >= 3: break
            
        for i, (p, acc) in enumerate(unique_top):
            info = ["Bands=Full Band" if is_full_band else f"Bands={p['n_features']}"]  # AI가 수정함: Full Band 표시
            for s in p['prep']:
                if s['name'] in target_keys: info.append(f"Gap={s['params'].get('gap')}")
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

