import os
import spectral.io.envi as envi
import numpy as np

def load_hsi_data(header_path):
    """
    ENVI 포맷의 초분광 데이터(.hdr)를 로드합니다.
    
    이 함수는 Spectral 라이브러리를 사용하여 헤더 파일을 읽고,
    실제 이미지 데이터(Cube)를 메모리로 불러옵니다.
    
    Args:
        header_path (str): .hdr 파일의 전체 경로
        
    Returns:
        tuple: (data_cube, wavelengths)
            - data_cube: (세로, 가로, 밴드수) 형태의 3차원 Numpy 배열
            - wavelengths: 각 밴드에 해당하는 파장 값들의 리스트 (예: [400.1, 402.5, ...])
    """
    if not os.path.exists(header_path):
        print(f"   [Data Loader] 파일을 찾을 수 없습니다: {header_path}")
        print("   [Data Loader] 데모를 위해 가상의 랜덤 데이터를 생성합니다...")
        # 파일이 없을 경우 테스트를 위해 100x100 크기의, 224개 밴드를 가진 랜덤 데이터를 만듭니다.
        return np.random.rand(100, 100, 224), [float(i) for i in range(224)]
        
    print(f"   [Data Loader] 헤더 파일 로딩 중: {header_path}")
    
    try:
        # 1. ENVI 헤더 파일 열기 (메타데이터 로드)
        img = envi.open(header_path)
        
        # 2. 실제 이미지 데이터 로드
        img_obj = img.load()
        # Ensure it's a pure numpy array (not SpyFile or memmap)
        data_cube = np.array(img_obj)
        
        # 3. 파장(Wavelength) 정보 추출
        # 헤더 파일 내에 'wavelength' 정보가 있다면 가져오고, 없으면 0부터 순서대로 번호를 매깁니다.
        metadata = img.metadata
        if 'wavelength' in metadata:
            wavelengths = [float(w) for w in metadata['wavelength']]
        else:
            wavelengths = list(range(data_cube.shape[2]))
            
        print(f"   [Data Loader] 데이터 로드 완료! 크기(Shape): {data_cube.shape}")
        
        return data_cube, wavelengths
        
    except Exception as e:
        print(f"   [Error] 데이터 로딩 실패: {e}")
        raise
