import os

def scan_directory(path):
    print(f"📂 폴더를 재귀적으로 스캔 중입니다: {path}")
    
    if not os.path.exists(path):
        print("❌ 오류: 폴더를 찾을 수 없습니다!")
        return

    hdr_files = []
    # 폴더 내부를 구석구석 뒤져서 파일을 찾습니다 (Walk)
    for root, dirs, files in os.walk(path):
        for file in files:
            # 확장자가 .hdr로 끝나는 파일만 리스트에 담습니다.
            if file.endswith(".hdr"):
                full_path = os.path.join(root, file)
                hdr_files.append(full_path)
                
    if not hdr_files:
        print("⚠️ 폴더 안쪽까지 다 뒤져봤는데 .hdr 파일이 하나도 없습니다.")
        print("   혹시 파일들이 다른 폴더에 있는지 확인해보세요.")
        print("   참고로 이 폴더의 첫 5개 항목은 다음과 같습니다:")
        try:
            items = os.listdir(path)
            for item in items[:5]:
                item_path = os.path.join(path, item)
                print(f"   - {item} (폴더 여부: {os.path.isdir(item_path)})")
        except Exception as e:
            print(f"   목록 조회 중 에러 발생: {e}")
    else:
        print(f"✅ 총 {len(hdr_files)}개의 헤더 파일을 찾았습니다! (상위 5개만 보여드립니다):")
        for f in hdr_files[:5]:
            print(f"   - {f}")
            
if __name__ == "__main__":
    target_path = r"C:\Users\user16g\Desktop\nonbr_br_fx50"
    scan_directory(target_path)
