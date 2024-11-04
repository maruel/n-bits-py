#!/usr/bin/env python3

"""Inspect a SafeTensors file: count tensors and show first weights"""

import argparse
import os
import sys

import gnuplotlib
import huggingface_hub
import safetensors


def graph_histogram(t, name):
    cols, lines = os.get_terminal_size()
    bins = cols - 10
    # if bfloat16:
    t2 = t.dequantize()
    counts, bins = t2.histogram(bins)  # , density=True)
    c = counts.numpy()
    b = bins[:-1].numpy()
    # print(b)
    # print(c)
    terminal = f"dumb {cols} {max(10, lines-10)}"
    try:
        gnuplotlib.plot(b, c, _set="logscale y", terminal=terminal, title=name)
    except OSError:
        print("Please install gnuplot")


def my_function():
    return "Welcome to n-bits!"


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


def download_from_hf(repo_id: str, filename: str):
    """
    Download a safetensors file from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')
        filename: Name of the safetensors file to download

    Returns:
        str: Path to the downloaded file
    """
    file_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename, local_files_only=False
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
        print(f"Total number of tensors: {len(tensor_names)}")
        align = max(len(n) for n in tensor_names)
        for i, name in enumerate(tensor_names):
            # https://pytorch.org/docs/stable/torch.html#tensors
            tensor = f.get_tensor(name)
            flat = tensor.flatten()
            # graph_histogram(flat, name)
            l = len(flat)
            print(
                f"{name:<{align}}: {l:>8} items  avg={flat.mean():+.2f}  min={flat.min():+.2f}  max={flat.max():+.2f}"
            )
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
    group.add_argument(
        "--hf-repo", help="HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')"
    )
    subparser.add_argument(
        "--filename",
        default="model.safetensors",
        help="Name of the safetensors file to download (when using --hf-repo)",
    )

    subparser.add_argument(
        "--token", help="HuggingFace API token for accessing private models"
    )
    subparser = subparsers.add_parser("compress", help=main_compress.__doc__)
    subparser.set_defaults(fn=main_compress)
    # group.add_argument("--local-path", help="Path to a local SafeTensors model file")
    # group.add_argument("--hf-repo", help="HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')")
    # subparser.add_argument(
    #    "--filename",
    #    default="model.safetensors",
    #    help="Name of the safetensors file to download (when using --hf-repo)"
    # )
    # subparser.add_argument("--token", help="HuggingFace API token for accessing private models")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
