#!/usr/bin/env python3

"""Inspect a SafeTensors file: count tensors and show first weights"""

import argparse
import os
import sys

import huggingface_hub
import safetensors


def my_function():
    return "Welcome to n-bits!"


def authenticate_hf(token : str):
    """Authenticate with HuggingFace Hub."""
    if token:
        huggingface_hub.login(token=token)
    else:
        # If no token provided, try to use cached token.
        try:
            huggingface_hub.login(new_session=False)
        except Exception as e:
            print("No valid token found. Some models may not be accessible.")
            print("To authenticate, either:")
            print("  1. Use --token parameter")
            print("  2. Run 'huggingface-cli login' in terminal")
            print("  3. Set HUGGING_FACE_HUB_TOKEN environment variable")


def download_from_hf(repo_id : str, filename : str):
    """
    Download a safetensors file from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')
        filename: Name of the safetensors file to download

    Returns:
        str: Path to the downloaded file
    """
    file_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_files_only=False
    )
    return file_path


def inspect_safetensors(file_path: str):
    """Load a safetensors file and print information about its tensors."""
    # Sadly, numpy doesn't support bfloat16 and all recent models use this
    # format!
    with safetensors.safe_open(file_path, framework="pt") as f:
        total = 0
        # Get all tensor names.
        tensor_names = f.keys()
        print(f"Total number of tensors: {len(tensor_names)}\n")
        print("First weight of each tensor:")
        print("-" * 50)
        # Iterate through each tensor.
        for name in tensor_names:
            tensor = f.get_tensor(name)
            flat = tensor.flatten()
            l = len(flat)
            # Get first element as Python scalar.
            first_weight = flat[0].item()
            print(f"{name}({l}): {first_weight}")
            total += l
        print(f"Total number of weights: {total}")


def main_compress(args):
    """Compress a safetensor"""
    print("TODO: implement me")
    return 1


def main_inspect(args):
    """Inspect a safetensor"""
    file_path = args.local_path
    if args.hf_repo:
        authenticate_hf(args.token)
        file_path = download_from_hf(args.hf_repo, args.filename)

    if file_path:
        inspect_safetensors(file_path)
    return 0


def main():
    parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    parser.set_defaults(fn=None)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparser = subparsers.add_parser("inspect", help=main_inspect.__doc__)
    subparser.set_defaults(fn=main_inspect)
    group = subparser.add_mutually_exclusive_group(required=False)
    group.add_argument("--local-path", help="Path to a local SafeTensors model file")
    group.add_argument("--hf-repo", help="HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')")
    subparser.add_argument(
        "--filename",
        default="model.safetensors",
        help="Name of the safetensors file to download (when using --hf-repo)"
    )

    subparser.add_argument("--token", help="HuggingFace API token for accessing private models")
    subparser = subparsers.add_parser("compress", help=main_compress.__doc__)
    subparser.set_defaults(fn=main_compress)
    #group.add_argument("--local-path", help="Path to a local SafeTensors model file")
    #group.add_argument("--hf-repo", help="HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')")
    #subparser.add_argument(
    #    "--filename",
    #    default="model.safetensors",
    #    help="Name of the safetensors file to download (when using --hf-repo)"
    #)
    #subparser.add_argument("--token", help="HuggingFace API token for accessing private models")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
