# Copyright 2024 Marc-Antoine Ruel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
