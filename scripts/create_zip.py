import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__', 'outputs', '.pytest_cache', 'mlruns', '.dockerignore']]
        
        for file in files:
            # Exclude files
            if file.endswith('.exe') or file.endswith('.zip') or file.endswith('.joblib') or file == 'mlflow.db':
                continue
            
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, start=path)
            ziph.write(file_path, arcname)

if __name__ == '__main__':
    print("Creating assignment deliverables zip...")
    zip_filename = 'Assignment4_Deliverable.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('.', zipf)
    print(f"Successfully created {zip_filename}")
