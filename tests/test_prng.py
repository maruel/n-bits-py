# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

import pytest
import numpy as np
from n_bits.prng import XoshiroStarStar, PCG32, SplitMix64


# TODO: XoshiroStarStar is broken.
@pytest.fixture(params=[PCG32, SplitMix64])
def generator(request):
    """Fixture that provides each PRNG class for testing"""
    return request.param(seed=12345)


def test_output_range(generator):
    """Test that all generators produce values in [0, 1)"""
    samples = [generator.next() for _ in range(1000)]
    assert all(0 <= x < 1 for x in samples)
    assert any(x > 0.1 for x in samples), "Poor distribution: no values above 0.1"
    assert any(x < 0.9 for x in samples), "Poor distribution: no values below 0.9"


def test_float32_type(generator):
    """Test that output is actually float32"""
    value = generator.next()
    assert isinstance(value, np.float32)


def test_deterministic(generator):
    """Test that same seed produces same sequence"""
    gen1 = generator.__class__(seed=42)
    gen2 = generator.__class__(seed=42)

    for _ in range(100):
        assert gen1.next() == gen2.next()


def test_different_seeds(generator):
    """Test that different seeds produce different sequences"""
    gen1 = generator.__class__(seed=1)
    gen2 = generator.__class__(seed=2)

    # Check first few numbers are different
    samples1 = [gen1.next() for _ in range(5)]
    samples2 = [gen2.next() for _ in range(5)]
    assert samples1 != samples2


def test_uniformity(generator):
    """Basic test for uniform distribution"""
    samples = [generator.next() for _ in range(10000)]

    # Test basic uniformity by checking distribution across bins
    hist, _ = np.histogram(samples, bins=10, range=(0, 1))
    expected = len(samples) / 10
    chi_squared = np.sum((hist - expected) ** 2 / expected)

    # Chi-squared test with 9 degrees of freedom
    # This should fail less than 1% of the time for a good PRNG
    assert chi_squared < 21.67  # Critical value for p=0.01 with df=9


def test_correlation(generator):
    """Test for obvious correlations between consecutive numbers"""
    samples = [generator.next() for _ in range(1000)]
    correlation = np.corrcoef(samples[:-1], samples[1:])[0, 1]
    assert abs(correlation) < 0.1  # Should be close to 0 for good PRNG


def test_no_obvious_patterns(generator):
    """Test for obvious patterns in the output"""
    samples = [generator.next() for _ in range(1000)]

    # Test that we don't have too many repeated values
    unique_values = len(set(samples))
    assert unique_values > 900  # Should have many unique values

    # Test that differences between consecutive numbers vary
    diffs = np.diff(samples)
    unique_diffs = len(set(diffs))
    assert unique_diffs > 900  # Should have many unique differences


def test_edge_cases():
    """Test edge cases and error handling"""
    for Generator in [XoshiroStarStar, PCG32, SplitMix64]:
        # Test with zero seed
        gen = Generator(seed=0)
        value = gen.next()
        assert 0 <= value < 1

        # Test with large seed
        gen = Generator(seed=2**32 - 1)
        value = gen.next()
        assert 0 <= value < 1

        # Test with negative seed
        gen = Generator(seed=-42)
        value = gen.next()
        assert 0 <= value < 1
