#!/usr/bin/env python3

"""Inspect a SafeTensors file: count tensors and show first weights"""

import argparse
import ctypes
import glob
import os
import sys

import gnuplotlib
import huggingface_hub
import safetensors
import safetensors.torch
import torch


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


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given PyTorch dtype."""
    mapping = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1,
        torch.complex64: 8,
        torch.complex128: 16,
    }
    if dtype not in mapping:
        print(f"{dtype} not in mapping", file=sys.stderr)
        return 1
    return mapping[dtype]


def read_tensor_bytes(tensor: torch.Tensor):
    """Read the first n bytes from a PyTorch tensor's memory using ctypes."""
    c = tensor.cpu()
    num_bytes = c.numel() * get_dtype_size(c.dtype)
    byte_array = (ctypes.c_ubyte * num_bytes).from_address(c.data_ptr())
    return bytes(byte_array)


def decode_bfloat16(bfloat16_val: int) -> float:
    """Decode a 16-bit bfloat16 value into its corresponding float value.

    BFloat16 format:
    - 1 bit: sign (bit 15)
    - 8 bits: exponent (bits 14-7)
    - 7 bits: mantissa (bits 6-0)

    Parameters:
        bfloat16_val (int): 16-bit integer representing a bfloat16 value

    Returns:
        float: Decoded floating point value
    """
    sign_bit = (bfloat16_val >> 15) & 0x1
    exponent_bits = (bfloat16_val >> 7) & 0xFF
    mantissa_bits = bfloat16_val & 0x7F
    # Handle special cases.
    if exponent_bits == 0:
        if mantissa_bits == 0:
            return -0.0 if sign_bit else 0.0
        else:
            # Denormalized numbers.
            exponent = -126
            mantissa = mantissa_bits / (1 << 7)
    elif exponent_bits == 0xFF:
        if mantissa_bits == 0:
            return float("-inf") if sign_bit else float("inf")
        else:
            return float("nan")
    else:
        # Normalized numbers.
        exponent = exponent_bits - 127
        mantissa = 1 + (mantissa_bits / (1 << 7))
    # Combine components.
    return (-1 if sign_bit else 1) * mantissa * (2**exponent)


def bfloat16_bytes_to_int(bfloat16_bytes: bytes):
    return int.from_bytes(bfloat16_bytes[:2], byteorder="little")


def print_bfloat16_components(bfloat16_bytes: bytes):
    """Print the components and decoded value of a bfloat16 number"""
    bfloat16_val = bfloat16_bytes_to_int(bfloat16_bytes)
    sign_bit = (bfloat16_val >> 15) & 0x1
    exponent_bits = (bfloat16_val >> 7) & 0xFF
    mantissa_bits = bfloat16_val & 0x7F
    print(f"- Binary:   {bfloat16_val:016b} ({bfloat16_val:2x})")
    print(f"- Sign:     {sign_bit}                ({sign_bit:>4})")
    print(f"- Exponent:  {exponent_bits:08b}        ({exponent_bits:>4})")
    print(f"- Mantissa:          {mantissa_bits:07b} ({mantissa_bits:>4})")
    print(f"- Decoded:  {decode_bfloat16(bfloat16_val)}")


def tensor_stats(t: torch.Tensor):
    std, mean = torch.std_mean(t)
    return (t.numel(), mean, std, t.min(), t.max())


def inspect_tensors(tensors_dict):
    """Inspect and print information of tensors."""
    # Calculate the stats upfront.
    stats = {name: tensor_stats(t) for name, t in tensors_dict.items()}
    name_align = max(len(n) for n in tensors_dict)
    size_align = max(len(str(s[0])) for s in stats.values())
    for name, (length, avg, std, m1, m2) in stats.items():
        print(
            f"{name:<{name_align}}: len={length:>{size_align}}  avg={avg:+.2f}  std={std:+.2f}  min={m1:+.2f}  max={m2:+.2f}"
        )
    print(f"- Total number of weights: {sum(s[0] for s in stats.values())}")
    print(f"- Total number of tensors: {len(tensors_dict)}")
    first_name = next(iter(tensors_dict))
    first = tensors_dict[first_name].flatten()
    b = read_tensor_bytes(first)
    print(f"- {first_name}: {first.numel()} in {first.dtype}; {len(b)}: {b[:10].hex()}")
    for i in range(5):
        print(f"Element #{i}:")
        print_bfloat16_components(b[2 * i :])
        print(f"  Original: {first[i]}")


def main_compress(args):
    """Compress a safetensor"""
    print("TODO: implement me")
    return 1


def main_inspect(args):
    """Inspect a safetensor"""
    local_files = None
    if args.hf_repo:
        authenticate_hf(args.token)
        args.local_path = download_safetensors_from_hf(args.hf_repo)
    if args.local_path:
        local_files = glob.glob(os.path.join(args.local_path, "*.safetensors"))
    if not local_files:
        print("No .safetensors found", file=sys.stderr)
        return 1
    merged_tensors = {}
    for local_path in local_files:
        # Sadly, numpy doesn't support bfloat16 and all recent models use this
        # format! So we need to use pytorch.
        merged_tensors.update(safetensors.torch.load_file(local_path))
    inspect_tensors(merged_tensors)
    return 0


def main():
    parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    parser.set_defaults(fn=None)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparser = subparsers.add_parser("inspect", help=main_inspect.__doc__)
    subparser.set_defaults(fn=main_inspect)
    group = subparser.add_mutually_exclusive_group(required=False)
    group.add_argument("--local-path", help="Path to a local SafeTensors model files")
    group.add_argument(
        "--hf-repo", help="HuggingFace repository ID (e.g., 'meta-llama/Llama-3.2-1B')"
    )

    subparser.add_argument(
        "--token", help="HuggingFace API token for accessing private models"
    )
    subparser = subparsers.add_parser("compress", help=main_compress.__doc__)
    subparser.set_defaults(fn=main_compress)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
