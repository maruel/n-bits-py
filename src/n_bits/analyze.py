import ctypes
import os
import sys

import gnuplotlib
import torch

from .bits import bfloat16_bytes_to_int, decode_bfloat16


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
        print("Please install gnuplot", file=sys.stderr)
        raise


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


def tensor_stats(t: torch.Tensor):
    std, mean = torch.std_mean(t)
    return (t.numel(), mean, std, t.min(), t.max())


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


def analyze_tensors(tensors_dict):
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