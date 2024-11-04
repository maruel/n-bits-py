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
        (0x3F80, 1.0),
        (0xBF80, -1.0),
        (0x4000, 2.0),
        (0x3F00, 0.5),
        (0x0000, 0.0),
        (0xC2F7, -123.5),
        (0x7F80, float("inf")),
        (0xFF80, float("-inf")),
        (0x7FC0, float("nan")),
    ]
    for i, (val, expected) in enumerate(test_values):
        raw = encode_float_to_bfloat16_int(expected)
        assert raw == val, f"#{i}: 0x{val:04x} != 0x{raw:04x}"
        actual = bits.decode_bfloat16(val)
        if math.isnan(expected):
            assert math.isnan(actual), actual
        else:
            assert expected == actual, f"#{i}: {expected} != {actual} for {val}"
