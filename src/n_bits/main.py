#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Tool to analyze deep neural network (DNN) safetensors and look for
optimization opportunities."""

import argparse
import glob
import os
import sys

import safetensors
import safetensors.torch

from .analyze import analyze_tensors
from .hf import authenticate_hf, download_safetensors_from_hf


def main_compress(args):
    """Compress a safetensor"""
    print("TODO: implement me")
    return 1


def main_analyze(args):
    """Analyze a safetensor to see if there's potentials for improvement"""
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
    analyze_tensors(merged_tensors)
    return 0


def main():
    parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    parser.set_defaults(fn=None)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparser = subparsers.add_parser("analyze", help=main_analyze.__doc__)
    subparser.set_defaults(fn=main_analyze)
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
