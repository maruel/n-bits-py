# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

import math

import torch

from n_bits import analyze, bits


def encode_float_to_bfloat16_int(f: float):
    # Create a tensor and force encoding to be bfloat16.
    t = torch.tensor([f], dtype=torch.bfloat16)
    assert t.numel() == 1, t.numel()
    assert len(t) == 1, len(t)
    b = analyze.read_tensor_bytes(t)
    assert len(b) == 2, len(b)
    return bits.bfloat16_bytes_to_int(b)


def test_bfloat16():
    test_values = [
        (0x3F80, 1.0, 0, 127, 0),
        (0xBF80, -1.0, 1, 127, 0),
        (0x4000, 2.0, 0, 128, 0),
        (0x3F00, 0.5, 0, 126, 0),
        (0xBF00, -0.5, 1, 126, 0),
        (0x0000, 0.0, 0, 0, 0),
        (0xC2F7, -123.5, 1, 133, 119),
        (0x7F80, float("inf"), 0, 255, 0),
        (0xFF80, float("-inf"), 1, 255, 0),
        (0x7FC0, float("nan"), 0, 255, 64),
    ]
    for i, (val, expected_val, e_sign, e_exponent, e_mantissa) in enumerate(
        test_values
    ):
        raw = encode_float_to_bfloat16_int(expected_val)
        assert raw == val, f"#{i}: 0x{val:04x} != 0x{raw:04x}"
        actual = bits.decode_bfloat16(val)
        if math.isnan(expected_val):
            assert math.isnan(actual), actual
        else:
            assert expected_val == actual, f"#{i}: {expected_val} != {actual} for {val}"
        sign, exponent, mantissa = bits.unpack_bfloat16(val)
        assert (
            e_sign == sign and e_exponent == exponent and e_mantissa == mantissa
        ), f"#{i}: {e_sign} == {sign} and {e_exponent} == {exponent} and {e_mantissa} == {mantissa}"
