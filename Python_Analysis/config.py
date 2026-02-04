"""
/// <ai>AI가 작성함</ai>
설정 파일 관리 모듈 (Read/Write 지원)
- Singleton 패턴 적용
- get/set 메서드로 값 접근 및 수정
- save() 호출 시 settings.json에 영구 저장
"""
import json
import os
import copy

SETTINGS_FILE = 'settings.json'

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config_path = os.path.join(os.path.dirname(__file__), SETTINGS_FILE)
            cls._instance._data = {}
            cls._instance.load()
        return cls._instance

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception as e:
                print(f"[Config] Load failed: {e}. Using empty config.")
                self._data = {}
        else:
            self._data = {}
            
    def save(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)
            print("[Config] Settings saved.")
        except Exception as e:
            print(f"[Config] Save failed: {e}")

    def get(self, section, key=None, default=None):
        """
        Legacy Support: get('section', 'key')
        New Support: get('section.key') if key is None
        """
        if key is None and '.' in section:
            # Nested Key Support (e.g. "ui.training.path")
            keys = section.split('.')
            val = self._data
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                else:
                    return default
            return val if val is not None else default
        
        # Standard Section/Key
        return self._data.get(section, {}).get(key, default)

    def set(self, section, key, value=None):
        """
        Usage: 
          set('section', 'key', value)
          set('section.key', value)
        """
        if value is None:
            # Assume section is "sec.key" and key is value
            value = key
            full_key = section
            keys = full_key.split('.')
            target = self._data
            
            for k in keys[:-1]:
                if k not in target: target[k] = {}
                target = target[k]
                if not isinstance(target, dict): # Overwrite if not dict
                    target = {} 
                    
            target[keys[-1]] = value
        else:
            if section not in self._data:
                self._data[section] = {}
            self._data[section][key] = value

# Singleton Instance
_cfg = ConfigManager()

# --- Public API (Backward Compatibility) ---

def get(section, key=None, default=None):
    return _cfg.get(section, key, default)

def set_value(section, key, value=None):
    """
    section, key, value (Legacy Style) OR "section.key", value (New Style)
    """
    _cfg.set(section, key, value)

def save():
    _cfg.save()

def reload_config():
    _cfg.load()
    return _cfg._data
