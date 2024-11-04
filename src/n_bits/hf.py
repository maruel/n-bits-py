import huggingface_hub


def authenticate_hf(token: str):
    """Authenticate with HuggingFace Hub."""
    if token:
        huggingface_hub.login(token=token)
    else:
        # If no token provided, try to use cached token.
        try:
            huggingface_hub.login(new_session=False)
        except Exception:
            print("No valid token found. Some models may not be accessible.")
            print("To authenticate, either:")
            print("  1. Use --token parameter")
            print("  2. Run 'huggingface-cli login' in terminal")
            print("  3. Set HUGGING_FACE_HUB_TOKEN environment variable")


def download_safetensors_from_hf(repo_id: str):
    """Download a safetensors file from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')

    Returns:
        str: Path to the directory containing downloaded files
    """
    # Try local_only first so when it's already cached it doesn't print a
    # progress bar.
    try:
        return huggingface_hub.snapshot_download(
            repo_id=repo_id, allow_patterns=["*.safetensors"], local_files_only=True
        )
    except huggingface_hub.errors.LocalEntryNotFoundError:
        return huggingface_hub.snapshot_download(
            repo_id=repo_id, allow_patterns=["*.safetensors"]
        )
