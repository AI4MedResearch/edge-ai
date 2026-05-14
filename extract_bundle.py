import os

def extract_bundle():
    task_path = 'mri-viewer/public/models/medgemma-1.5-4b-it-int4.task'
    output_dir = 'medgemma_litert/extracted_model'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading {task_path}...")
    with open(task_path, 'rb') as f:
        data = f.read()
    
    # Find ZIP start
    start = data.find(b'PK\x03\x04')
    if start == -1:
        print("❌ Could not find ZIP start in bundle.")
        return
    
    print(f"ZIP start found at {start}. Extracting...")
    zip_path = os.path.join(output_dir, 'bundle.zip')
    with open(zip_path, 'wb') as f:
        f.write(data[start:])
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)
        print(f"✅ Extracted files to {output_dir}:")
        for name in z.namelist():
            print(f"  - {name}")

if __name__ == "__main__":
    extract_bundle()
