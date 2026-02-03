"""
/// <ai>AI가 작성함</ai>
설정 파일 로드 유틸리티
- get(): 설정 값 조회
- reload_config(): 런타임 중 설정 리로드
"""
import json
import os

_config = None

def get_config():
    """설정 전체를 dict로 반환"""
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = json.load(f)
    return _config

def get(section, key, default=None):
    """특정 설정 값 조회"""
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)

def reload_config():
    """런타임 중 설정 리로드 (settings.json 수정 후 호출)"""
    global _config
    _config = None
    return get_config()
