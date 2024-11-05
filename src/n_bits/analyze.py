# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

import ctypes
from dataclasses import dataclass
import math
import os
import sys

import gnuplotlib
import numpy
import torch

from .bits import (
    bfloat16_bytes_to_int,
    decode_bfloat16,
    # unpack_bfloat16,
    # unpack_bfloat16_bytes,
)


@dataclass(frozen=True, slots=True)
class TensorStats:
    name: str
    length: int
    avg: float
    std: float
    min: float
    max: float

    @staticmethod
    def create(name: str, t: torch.Tensor):
        std, avg = torch.std_mean(t)
        return TensorStats(name, t.numel(), avg, std, t.min(), t.max())


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


def read_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Read the first n bytes from a PyTorch tensor's memory using ctypes."""
    c = tensor.cpu()
    num_bytes = c.numel() * get_dtype_size(c.dtype)
    byte_array = (ctypes.c_ubyte * num_bytes).from_address(c.data_ptr())
    return bytes(byte_array)


def calc_histograms_float(t: torch.Tensor):
    if t.dtype != torch.bfloat16:
        raise NotImplementedError(f"implement for {t.dtype}")
    # TODO: Remapping the slice as a uint16 would probably help a lot!
    b = read_tensor_bytes(t)
    signs = [0 for _ in range(1 << 1)]
    exponents = [0 for _ in range(1 << 8)]
    mantissas = [0 for _ in range(1 << 7)]
    i = 0
    end = len(b)
    while i < end:
        # The following is the equivalent of this function call. Inlining has
        # significant performance improvement.
        #   sign, exponent, mantissa = unpack_bfloat16(bfloat16_bytes_to_int(b[i:i+2]))

        # This is quite fast but inlining is faster:
        #   sign, exponent, mantissa = unpack_bfloat16_bytes(b[i], b[i+1])
        #   signs[sign] += 1
        #   exponents[exponent] += 1
        #   mantissas[mantissa] += 1

        # Pretty good but not the fastest implementation:
        #   bf16 = int.from_bytes(b[i:i+2], byteorder="little")
        #   signs[(bf16 >> 15) & 0x1] += 1
        #   exponents[(bf16 >> 7) & 0xFF] += 1
        #   mantissas[bf16 & 0x7F] += 1

        # Pretty good but not the fastest implementation:
        #   signs[(b[i+1] & 0x80) >> 7] += 1
        #   exponents[((b[i+1] & 0x7F) << 1) | ((b[i] & 0x80) >> 7)] += 1
        #   mantissas[b[i] & 0x7F] += 1

        # The "fastest" implementation (as much as python can be fast lol):
        b0, b1 = b[i], b[i + 1]
        signs[(b1 & 0x80) >> 7] += 1
        exponents[((b1 & 0x7F) << 1) | ((b0 & 0x80) >> 7)] += 1
        mantissas[b0 & 0x7F] += 1
        i += 2
    return signs, exponents, mantissas


def print_histograms_float(t: torch.Tensor):
    signs, exponents, mantissas = calc_histograms_float(t)
    cols, lines = os.get_terminal_size()
    terminal = f"dumb {cols} 20"
    print(signs)
    print(exponents)
    print(mantissas)
    try:
        # exponents, mantissas,
        gnuplotlib.plot(
            numpy.ndarray(signs),
            _set="logscale y",
            terminal=terminal,
            title="Bit usage",
        )
    except OSError:
        print("Please install gnuplot", file=sys.stderr)
        raise


def print_graph_histogram(name: str, t: torch.Tensor):
    cols, lines = os.get_terminal_size()
    bins = cols - 10
    try:
        counts, bins = t.histogram(bins)
    except RuntimeError:
        # Necessary for "non-standard" formats like bfloat16.
        t = t.dequantize()
        counts, bins = t.histogram(bins)
    c = counts.numpy()
    b = bins[:-1].numpy()
    terminal = f"dumb {cols} {max(10, lines - 10)}"
    try:
        gnuplotlib.plot(b, c, _set="logscale y", terminal=terminal, title=name)
    except OSError:
        print("Please install gnuplot", file=sys.stderr)
        raise


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


def effective(b) -> int:
    o = 0
    for i in b:
        if i:
            o += 1
    return o


def analyze_tensors(tensors_dict):
    """Inspect and print information of tensors."""
    maxNameLen = max(len(name) for name in tensors_dict)
    maxSizeLen = max(len(str(t.numel())) for t in tensors_dict.values())
    bytesWasted = 0
    totalBytes = 0
    for i, (name, t) in enumerate(tensors_dict.items()):
        signs, exponents, mantissas = calc_histograms_float(t)
        e_signs = effective(signs)
        e_exponents = effective(exponents)
        e_mantissas = effective(mantissas)
        b_signs = math.log2(e_signs)
        b_exponents = math.log2(e_exponents)
        b_mantissas = math.log2(e_mantissas)
        wasted = (
            1
            - int(math.ceil(b_signs))
            + 8
            - int(math.ceil(b_exponents))
            + 7
            - int(math.ceil(b_mantissas))
        )
        num_el = t.numel()
        print(
            f"{name:{maxNameLen}}: {t.numel():{maxSizeLen}}w  "
            + f"avg={t.mean():4.1f} [{t.min():6.1f}, {t.max():6.1f}]  "
            + f"sign={b_signs:1.0f}bit  "
            + f"exponent={b_exponents:3.1f}/8bits  "
            + f"mantissa={b_mantissas:3.1f}/7bits  "
            + f"wasted={wasted}/16bits {100.*wasted/16:.1f}% {wasted*num_el/8:8.0f}bytes"
        )
        sys.stdout.flush()
        bytesWasted += int(wasted * num_el / 8)
        totalBytes += num_el * 2
    print(
        f"{bytesWasted} bytes ({100.*bytesWasted/totalBytes:.1f}%) wasted on {totalBytes} bytes total"
    )


def analyze_tensors_old(tensors_dict):
    """Inspect and print information of tensors."""
    # Calculate the stats upfront.
    stats = {name: TensorStats.create(name, t) for name, t in tensors_dict.items()}
    name_align = max(len(n) for n in tensors_dict)
    size_align = max(len(str(s.length)) for s in stats.values())
    for name, s in stats.items():
        print(
            f"{name:<{name_align}}: len={s.length:>{size_align}}  "
            + f"avg={s.avg:+.2f}  std={s.std:+.2f}  min={s.min:+.2f}  max={s.max:+.2f}"
        )
    print(f"- Total number of weights: {sum(s.length for s in stats.values())}")
    print(f"- Total number of tensors: {len(tensors_dict)}")

    # first_name = next(iter(tensors_dict))
    # first_name = list(tensors_dict)[20]
    first_name = "model.layers.4.self_attn.v_proj.weight"
    first_name = "model.layers.8.input_layernorm.weight"
    first = tensors_dict[first_name].flatten()
    print(f"Analyzing {first_name} ({len(first)} weights):")
    print_histograms_float(first)
    return
    print_graph_histogram(first_name, first)

    b = read_tensor_bytes(first)
    print(f"- {first_name}: {first.numel()} in {first.dtype}; {len(b)}: {b[:10].hex()}")
    for i in range(5):
        print(f"Element #{i}:")
        print_bfloat16_components(b[2 * i :])
        print(f"  Original: {first[i]}")
