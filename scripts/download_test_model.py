import os
import sys
from huggingface_hub import snapshot_download

def download_hf_model(repo_id: str, output_folder: str):
    print(f"[*] Connecting to HuggingFace Hub for: {repo_id}")
    
    # We only care about safe weights, tokenizer, and config JSONs.
    # We exclude large legacy PyTorch/TensorFlow bin files to save bandwidth.
    patterns_to_keep = [
        "*.safetensors",
        "*.safetensors.index.json",
        "*.json",
        "tokenizer.model"
    ]
    
    print(f"[*] Downloading into directory: {output_folder}...")
    try:
        # local_dir_use_symlinks=False ensures actual files are downloaded (crucial for Rust reading)
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_folder,
            local_dir_use_symlinks=False,
            allow_patterns=patterns_to_keep
        )
        print(f"\n[SUCCESS] Model downloaded securely to: {local_path}")
        print("[*] Ready for MTF compilation.")
        
    except Exception as e:
        print(f"[-] Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Create the isolated models directory at the root of the workspace
    # os.path.dirname(__file__) is `scripts/`, so `..` goes back to `mtf-code`
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    models_dir = os.path.join(base_dir, "test_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Let's download a small but highly intelligent model for our compiler tests
    # Qwen2-0.5B is less than 1GB and is perfect for rapid local testing!
    model_repo = "Qwen/Qwen2-0.5B"
    save_path = os.path.join(models_dir, "qwen2-0.5b")
    
    download_hf_model(model_repo, save_path)