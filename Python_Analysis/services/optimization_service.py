from PyQt5.QtCore import QObject, pyqtSignal
import copy # Added deepcopy

class OptimizationService(QObject):
    """
    Service for Auto-ML Hyperparameter Optimization.
    Implements 'Sequential Tuning' with 'Lookahead Hill Climbing' strategy.
    
    Strategies:
    1. Gap Size Tuning (SimpleDeriv / 3PointDepth)
    2. NDI Threshold Tuning (If NDI enabled)
    3. Band Count Tuning (SPA)
    """
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def run_optimization(self, initial_params, trial_callback):
        """
        Run the full optimization sequence.
        
        Args:
            initial_params (dict): Start parameters (from UI).
            trial_callback (func): Function(params) -> accuracy (float 0-100).
            
        Returns:
            dict: Best parameters found.
        """
        # Deep copy to protect initial state
        best_params = copy.deepcopy(initial_params)
        
        current_acc = trial_callback(best_params)
        self.log_message.emit(f"Baseline Accuracy: {current_acc:.2f}%")
        
        history = [] # List of (params, acc)
        history.append((best_params.copy(), current_acc))
        
        # 1. Optimize Gap Size (If applicable)
        # Check active preprocessing for 'SimpleDeriv' or '3PointDepth'
        prep_chain = best_params.get('prep', [])
        target_keys = ["SimpleDeriv", "3PointDepth"]
        target_prep = None
        for step in prep_chain:
            if step['name'] in target_keys:
                target_prep = step
                break
                
        if target_prep:
            self.log_message.emit(f"\n[Phase 1] optimizing Gap Size for {target_prep['name']}...")
            start_gap = target_prep['params'].get('gap', 1)
            self.log_message.emit(f"   Start Gap: {start_gap}")
            
            # Define Tuner Callback for Gap
            def gap_evaluator(val):
                # CRITICAL Fix: Deep Copy to avoid polluting best_params
                p = copy.deepcopy(best_params)
                # Find the step and update
                for s in p['prep']:
                    if s['name'] == target_prep['name']:
                        s['params']['gap'] = val
                return trial_callback(p), p

            best_gap, gap_acc, best_p_gap = self.lookahead_hill_climbing(
                start_val=start_gap, 
                step=2, 
                lookahead=3, 
                max_val=50, 
                evaluator=gap_evaluator,
                initial_acc=current_acc,
                initial_params_obj=best_params
            )
            
            if gap_acc > current_acc:
                self.log_message.emit(f" -> Found Better Gap: {start_gap} -> {best_gap} (+{gap_acc - current_acc:.2f}%)")
                # Update Best Params to carry over to next phase
                best_params = best_p_gap
                current_acc = gap_acc
                history.append((best_params.copy(), current_acc))
            else:
                self.log_message.emit(" -> No improvement on Gap.")

        # 2. Optimize NDI Threshold (If applicable)
        # Check if 'ratio' is True in SimpleDeriv
        ndi_step = None
        for step in best_params.get('prep', []):
            if step['name'] == "SimpleDeriv" and step['params'].get('ratio', False):
                ndi_step = step
                break
        
        if ndi_step:
            self.log_message.emit(f"\n[Phase 2] Optimizing NDI Threshold...")
            start_th = ndi_step['params'].get('ndi_threshold', 1000.0)
            if start_th < 1: start_th = 50 # Force start reasonable
            self.log_message.emit(f"   Start Threshold: {start_th}")
            
            def ndi_evaluator(val):
                # CRITICAL Fix: Deep Copy
                p = copy.deepcopy(best_params)
                for s in p['prep']:
                    if s['name'] == "SimpleDeriv":
                        s['params']['ndi_threshold'] = val
                return trial_callback(p), p

            best_th, th_acc, best_p_th = self.lookahead_hill_climbing(
                start_val=start_th,
                step=100,
                lookahead=3,
                max_val=2000,
                evaluator=ndi_evaluator,
                initial_acc=current_acc,
                initial_params_obj=best_params
            )
            
            if th_acc > current_acc:
                # Update Best Params to carry over to next phase
                self.log_message.emit(f" -> Found Better Threshold: {start_th} -> {best_th} (+{th_acc - current_acc:.2f}%)")
                best_params = best_p_th
                current_acc = th_acc
                history.append((best_params.copy(), current_acc))
            else:
                self.log_message.emit(" -> No improvement on Threshold.")
                
        # 3. Optimize Band Count
        self.log_message.emit(f"\n[Phase 3] Optimizing Feature Count (Bands)...")
        start_features = best_params.get('n_features', 5)
        self.log_message.emit(f"   Start Features: {start_features}")
        
        def band_evaluator(val):
            # CRITICAL Fix: Deep Copy
            p = copy.deepcopy(best_params)
            p['n_features'] = val
            return trial_callback(p), p
            
        best_bands, band_acc, best_p_band = self.lookahead_hill_climbing(
            start_val=start_features,
            step=5,
            lookahead=3,
            max_val=40,
            evaluator=band_evaluator,
            initial_acc=current_acc,
            initial_params_obj=best_params
        )
        
        if band_acc > current_acc:
            self.log_message.emit(f" -> Found Better Band Count: {start_features} -> {best_bands} (+{band_acc - current_acc:.2f}%)")
            best_params = best_p_band
            current_acc = band_acc
            history.append((best_params.copy(), current_acc))
        # Generate Final Report
        report = []
        report.append("\n🎉 Optimization Completed!")
        report.append("-" * 40)
        report.append(f"[Change Log]")
        
        # 1. Gap
        if target_prep:
            initial = initial_params['prep'][0]['params'].get('gap', 5) # Assuming first is target or find again
            # Find initial gap properly
            init_gap = 5
            for s in initial_params['prep']:
                if s['name'] in target_keys: init_gap = s['params'].get('gap', 5)
            
            final_gap = 5
            for s in best_params['prep']:
                if s['name'] in target_keys: final_gap = s['params'].get('gap', 5)
                
            if init_gap != final_gap:
                report.append(f"1. Gap Size: {init_gap} -> {final_gap}")
            else:
                report.append(f"1. Gap Size: {init_gap} (No Change)")

        # 2. Threshold
        if ndi_step:
            init_th = 0
            for s in initial_params['prep']:
                if s['name'] == "SimpleDeriv": init_th = s['params'].get('ndi_threshold', 0)
                
            final_th = 0
            for s in best_params['prep']:
                if s['name'] == "SimpleDeriv": final_th = s['params'].get('ndi_threshold', 0)
                
            if init_th != final_th:
                report.append(f"2. NDI Threshold: {init_th} -> {final_th}")
            else:
                 report.append(f"2. NDI Threshold: {init_th} (No Change)")
                 
        # 3. Bands
        init_bands = initial_params.get('n_features', 5)
        final_bands = best_params.get('n_features', 5)
        if init_bands != final_bands:
             report.append(f"3. Band Count: {init_bands} -> {final_bands}")
        else:
             report.append(f"3. Band Count: {init_bands} (No Change)")
             
        report.append("-" * 40)
        report.append("-" * 40)
        report.append(f"🏆 Final Best Accuracy: {current_acc:.2f}%")
        report.append("-" * 40)
        report.append(f"📜 Top 3 Configurations")
        
        # Sort history by accuracy descending
        sorted_history = sorted(history, key=lambda x: x[1], reverse=True)[:3]
        for i, (p, acc) in enumerate(sorted_history):
            # Extract key info for concise log
            info = []
            info.append(f"Bands={p['n_features']}")
            for s in p['prep']:
                if s['name'] in target_keys: info.append(f"Gap={s['params'].get('gap')}")
                if s['name'] == "SimpleDeriv" and s['params'].get('ratio'): info.append(f"Th={s['params'].get('ndi_threshold')}")
            
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
            report.append(f"{medal} #{i+1}: {acc:.2f}% | {', '.join(info)}")

        self.log_message.emit("\n".join(report))
        return best_params, history

    def lookahead_hill_climbing(self, start_val, step, lookahead, max_val, evaluator, initial_acc=None, initial_params_obj=None):
        """
        Generic Lookahead Walker.
        Args:
            start_val: Initial numeric value
            step: Increment step
            lookahead: How many steps to check ahead
            max_val: Limit
            evaluator: Func(val) -> (accuracy, full_params)
            initial_acc: Optional, accuracy at start_val if already known
            initial_params_obj: Optional, params object at start_val if already known
        
        Returns:
            (best_val, best_acc, best_params_obj)
        """
        current_val = start_val
        
        # Avoid Redundant Calculation if passed
        if initial_acc is not None and initial_params_obj is not None:
             current_acc = initial_acc
             current_full_params = initial_params_obj
        else:
             current_acc, current_full_params = evaluator(current_val)
        
        while True:
            # Lookahead check
            found_better = False
            local_best_val = current_val
            local_best_acc = current_acc
            local_best_params = current_full_params
            
            candidates = []
            for i in range(1, lookahead + 1):
                next_val = current_val + (step * i)
                if next_val > max_val: break
                candidates.append(next_val)
                
            if not candidates: break # Hit max
            
            self.log_message.emit(f"   👀 Lookahead: {candidates} (Baseline: {current_val} @ {current_acc:.2f}%)")
            
            for val in candidates:
                acc, p_obj = evaluator(val)
                # Conciseness: Use Bullet
                self.log_message.emit(f"    • Val={val}: {acc:.2f}%")
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_val = val
                    local_best_params = p_obj
                    found_better = True
            
            if found_better:
                # Move to the new best position
                self.log_message.emit(f"   🚀 Jump to {local_best_val} ({local_best_acc:.2f}%)")
                current_val = local_best_val
                current_acc = local_best_acc
                current_full_params = local_best_params
                # Continue loop to lookahead again from new position
            else:
                # No improvement in lookahead window -> Stop
                break
                
        return current_val, current_acc, current_full_params
