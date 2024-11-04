# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

import numpy as np


class XoshiroStarStar:
    """
    xoshiro256** PRNG - Extremely fast, high-quality generator
    Period: 2^256 - 1
    Returns float32 values in [0, 1)
    """

    def __init__(self, seed):
        import time

        if seed is None:
            seed = int(time.time() * 1000)

        self.state = [0] * 4
        self.state[0] = seed
        self.state[1] = seed << 1
        self.state[2] = seed << 2
        self.state[3] = seed << 3

        # Warm up the state
        for _ in range(10):
            self.next_int()

    def rotl(self, x, k):
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next_int(self):
        result = (self.rotl(self.state[1] * 5, 7) * 9) & 0xFFFFFFFFFFFFFFFF
        t = self.state[1] << 17

        self.state[2] ^= self.state[0]
        self.state[3] ^= self.state[1]
        self.state[1] ^= self.state[2]
        self.state[0] ^= self.state[3]

        self.state[2] ^= t
        self.state[3] = self.rotl(self.state[3], 45)

        return result

    def next(self):
        """Returns a float32 in [0, 1)"""
        return np.float32(self.next_int() >> 32) / np.float32(1 << 32)


class PCG32:
    """
    PCG32 PRNG - Fast, high-quality generator with good statistical properties
    Period: 2^64
    Returns float32 values in [0, 1)
    """

    def __init__(self, seed):
        import time

        if seed is None:
            seed = int(time.time() * 1000)

        self.state = seed
        self.inc = (seed << 1) | 1

        # Warm up the state
        for _ in range(10):
            self.next_int()

    def next_int(self):
        multiplier = 6364136223846793005
        oldstate = self.state

        # Advance internal state
        self.state = (oldstate * multiplier + self.inc) & 0xFFFFFFFFFFFFFFFF

        # Calculate output function
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
        rot = (oldstate >> 59) & 0x1F

        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next(self):
        """Returns a float32 in [0, 1)"""
        return np.float32(self.next_int()) / np.float32(0xFFFFFFFF)


class SplitMix64:
    """
    SplitMix64 PRNG - Simple, fast generator with good statistical properties
    Period: 2^64
    Returns float32 values in [0, 1)
    """

    def __init__(self, seed):
        import time

        if seed is None:
            seed = int(time.time() * 1000)
        self.state = seed

    def next_int(self):
        self.state += 0x9E3779B97F4A7C15
        self.state &= 0xFFFFFFFFFFFFFFFF

        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def next(self):
        """Returns a float32 in [0, 1)"""
        return np.float32(self.next_int() >> 32) / np.float32(1 << 32)


# Usage example
def demo_generators(seed=12345):
    generators = [
        ("Xoshiro256**", XoshiroStarStar(seed)),
        ("PCG32", PCG32(seed)),
        ("SplitMix64", SplitMix64(seed)),
    ]

    print("Generating first 5 float32 numbers from each PRNG:")
    for name, gen in generators:
        print(f"\n{name}:")
        for _ in range(5):
            print(f"{gen.next():.8f}")
