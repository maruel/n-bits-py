# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

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


def unpack_bfloat16(bfloat16_val):
    return (bfloat16_val >> 15) & 0x1, (bfloat16_val >> 7) & 0xFF, bfloat16_val & 0x7F


def unpack_bfloat16_bytes(b0, b1):
    # It's really important to not create temporary variables here otherwise it
    # slows the function down significantly.
    return (b1 & 0x80) >> 7, ((b1 & 0x7F) << 1) | ((b0 & 0x80) >> 7), b0 & 0x7F


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
    return int.from_bytes(bfloat16_bytes, byteorder="little")
