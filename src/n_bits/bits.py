

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
