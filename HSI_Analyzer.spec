# -*- mode: python ; coding: utf-8 -*-

import os
block_cipher = None

# AI가 수정함: 상대 경로 사용 (팀 협업용)
project_path = os.path.abspath(os.getcwd())
analysis_path = os.path.join(project_path, 'Python_Analysis')

a = Analysis(
    ['Python_Analysis/main.py'],
    pathex=[analysis_path],
    binaries=[],
    datas=[
        ('Python_Analysis/default_project.json', 'Python_Analysis'),
        ('docs', 'docs'),
        ('Python_Analysis/views', 'views'), 
    ],
    hiddenimports=[
        'scipy.signal', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'sklearn.utils._weight_vector',
        'importlib.resources',
        'matplotlib', 'matplotlib.pyplot'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'ipython', 'notebook', 'jedi', 'pydoc_data', 'curses'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HSI_Professional_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Debugging easier with console, change to False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HSI_Professional_Analyzer',
)
