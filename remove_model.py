import os
import shutil

# Set the path to your Hugging Face cache or models directory
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
SIZE_LIMIT_GB = 3  # models bigger than this will be deleted

def get_dir_size_in_gb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def remove_large_models(cache_dir, size_limit_gb):
    for entry in os.listdir(cache_dir):
        full_path = os.path.join(cache_dir, entry)
        if os.path.isdir(full_path):
            size_gb = get_dir_size_in_gb(full_path)
            if size_gb > size_limit_gb:
                print(f"Deleting {entry} ({size_gb:.2f} GB)...")
                shutil.rmtree(full_path)
            else:
                print(f"Keeping {entry} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    remove_large_models(HF_CACHE_DIR, SIZE_LIMIT_GB)