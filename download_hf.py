from huggingface_hub import snapshot_download
import os


def download_hf_repo(repo_id, local_dir, repo_type="model", token=None, revision=None):
    """
    Download a repository from Hugging Face Hub to a local directory.
    
    Args:
        repo_id (str): Repository ID (e.g., "microsoft/DialoGPT-medium")
        local_dir (str): Local directory path to download to
        repo_type (str): Type of repository ("model", "dataset", or "space")
        token (str, optional): Hugging Face token for private repos
        revision (str, optional): Git revision (branch, tag, or commit hash)
    """
    print(f"Downloading {repo_type} {repo_id} to {local_dir}...")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            local_dir_use_symlinks=False,  # Download actual files instead of symlinks
            # ignore_patterns=["*.mdb"]
            allow_patterns=["merged_anno/"]
        )
        print(f"Successfully downloaded {repo_id} to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading {repo_id}: {str(e)}")
        raise


def main():
    # Configuration - modify these variables as needed
    repo_id = "dorni/SpeakerVid-5M-Dataset"
    local_dir = "./datasets/speaker5M"
    repo_type = "dataset"  # "model", "dataset", or "space"
    token = None  # Set to your HF token if needed for private repos
    revision = None  # Set to specific branch/tag/commit if needed
    
    # Download the repository
    download_hf_repo(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type=repo_type,
    )


if __name__ == "__main__":
    main()
