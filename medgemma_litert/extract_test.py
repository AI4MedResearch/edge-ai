import os
import litert_torch

def extract():
    task_path = 'mri-viewer/public/models/medgemma-1.5-4b-it-int4.task'
    output_dir = 'medgemma_litert/extracted_model'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {task_path}...")
    m = litert_torch.load(task_path)
    
    # In some versions of litert_torch, the model content can be exported
    # For now, let's just try to use the model object directly if it supports it
    print("Model loaded successfully.")
    
    # If we can't extract, we'll just run a simple test using the loaded model
    # but litert_torch.LiteRTModel doesn't have a simple 'generate' method yet.

if __name__ == "__main__":
    extract()
