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
        band_start, band_end, band_step = 5, 40, 5
        
        if target_prep_name:
            self.log_message.emit("Target: Find best combination of [Band Count] and [Gap Size]")
            gap_range = range(1, 41) # Gap 1~40
            self.log_message.emit(f"Search Space: Bands {band_start}~{band_end} x Gap 1~40 = ~320 trials")
        else:
            self.log_message.emit("⚠️ No Gap-tunable preprocessing found. Switching to [Band-Only Optimization].")
            self.log_message.emit("Target: Find best [Band Count]")
            gap_range = [0] # Dummy Gap (No change)
            self.log_message.emit(f"Search Space: Bands {band_start}~{band_end}")

        self.log_message.emit("-" * 40)
        
        # 2. Optimization Loop
        for n_features in range(band_start, band_end + 1, band_step):
            self.log_message.emit(f"Checking Band Count: {n_features}...")
            
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
                if acc > best_acc:
                    diff = acc - best_acc
                    best_acc = acc
                    best_params = p
                    msg = f"✨ New Best! {acc:.2f}% (+{diff:.2f}%) | Bands={n_features}"
                    if target_prep_name: msg += f", Gap={gap}"
                    self.log_message.emit(msg)
                else:
                     if target_prep_name:
                         self.log_message.emit(f"   • Gap={gap}: {acc:.2f}%")
                     else:
                         self.log_message.emit(f"   • Bands={n_features}: {acc:.2f}%")
        
        self.log_message.emit("-" * 40)
        self.log_message.emit(f"🏆 Optimization Done. Best: {best_acc:.2f}%")
        
        # 3. Final Report
        self._generate_report(best_params, best_acc, history)
        
        return best_params, history

    def _generate_report(self, best_params, current_acc, history):
        """Generate final optimization report."""
        report = ["\n🎉 Optimization Completed!", "-" * 40, "[Final Configuration]"]
        
        target_keys = ["SimpleDeriv"]
        for step in best_params['prep']:
            if step['name'] in target_keys:
                report.append(f" • Gap Size: {step['params'].get('gap')}")

        report.append(f" • Band Count: {best_params.get('n_features')}")
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
            info = [f"Bands={p['n_features']}"]
            for s in p['prep']:
                if s['name'] in target_keys: info.append(f"Gap={s['params'].get('gap')}")
            medal = ["🥇", "🥈", "🥉"][i]
            report.append(f"{medal} #{i+1}: {acc:.2f}% | {', '.join(info)}")

        self.log_message.emit("\n".join(report))

