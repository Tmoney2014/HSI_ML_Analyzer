
import os
import glob

def scan_directory(path):
    print(f"📂 Scanning directory: {path}")
    
    if not os.path.exists(path):
        print("❌ Error: Directory not found!")
        return

    # Find .hdr files
    hdr_files = glob.glob(os.path.join(path, "*.hdr"))
    
    if not hdr_files:
        print("⚠️ No .hdr files found in this directory.")
        print("   Found files:")
        for f in os.listdir(path)[:10]: # List first 10 files
            print(f"   - {f}")
    else:
        print(f"✅ Found {len(hdr_files)} header files:")
        for f in hdr_files:
            print(f"   - {os.path.basename(f)}")
            
if __name__ == "__main__":
    target_path = r"C:\Users\user16g\Desktop\nonbr_br_fx50"
    scan_directory(target_path)
