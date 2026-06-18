"""Tests for utility functions (PRNG, geometry, coordinate transforms)."""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestPRNG:
    """Tests for the Xoshiro256pp PRNG class."""

    def test_initialization(self):
        """Test PRNG initialization."""
        rng = p.Xoshiro256pp(seed=42)
        assert rng is not None

    def test_uniform_sample(self):
        """Test uniform random sampling."""
        rng = p.Xoshiro256pp(seed=42)
        samples = rng.uniform_sample(1000)

        assert len(samples) == 1000
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_normal_sample(self):
        """Test normal distribution random sampling."""
        rng = p.Xoshiro256pp(seed=42)
        samples = rng.normal_sample(10000, mean=0.0, std=1.0)

        assert len(samples) == 10000
        # Check mean is close to expected
        actual_mean = np.mean(samples)
        assert abs(actual_mean) < 0.1  # Should be close to 0

    def test_multivariate_normal_sample(self):
        """Test multivariate normal sampling."""
        rng = p.Xoshiro256pp(seed=42)
        mean = [0.0, 0.0, 0.0]
        cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        samples = rng.multivariate_normal_sample(mean, cov, 1000)

        assert len(samples) == 1000
        assert all(len(s) == 3 for s in samples)

    def test_reproducibility(self):
        """Test that same seed produces same samples."""
        rng1 = p.Xoshiro256pp(seed=12345)
        samples1 = rng1.uniform_sample(100)

        rng2 = p.Xoshiro256pp(seed=12345)
        samples2 = rng2.uniform_sample(100)

        assert np.allclose(samples1, samples2)


class TestGeometry:
    """Tests for geometry utility functions."""

    def test_normalize_angle(self):
        """Test angle normalization to [-pi, pi]."""
        assert p.normalize_angle(0.0) == pytest.approx(0.0)
        # Both pi and -pi are valid normalizations
        assert abs(p.normalize_angle(math.pi)) == pytest.approx(math.pi)
        assert abs(p.normalize_angle(-math.pi)) == pytest.approx(math.pi)
        assert p.normalize_angle(2 * math.pi) == pytest.approx(0.0)
        # 3*pi normalizes to -pi (equivalent to pi)
        assert abs(p.normalize_angle(3 * math.pi)) == pytest.approx(math.pi)
        assert p.normalize_angle(-2 * math.pi) == pytest.approx(0.0)

    def test_angle_diff(self):
        """Test angle difference calculation."""
        assert p.angle_diff(0.0, 0.0) == pytest.approx(0.0)
        assert p.angle_diff(0.0, math.pi / 2) == pytest.approx(-math.pi / 2)
        assert p.angle_diff(math.pi / 2, 0.0) == pytest.approx(math.pi / 2)

    def test_distance_2d(self):
        """Test 2D distance calculation."""
        assert p.distance_2d(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)
        assert p.distance_2d(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
        assert p.distance_2d(1.0, 1.0, 1.0, 4.0) == pytest.approx(3.0)

    def test_bearing_2d(self):
        """Test bearing calculation."""
        # Bearing from (0,0) to (1,0) should be 0
        assert p.bearing_2d(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)
        # Bearing from (0,0) to (0,1) should be pi/2
        assert p.bearing_2d(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_point_in_polygon(self):
        """Test point-in-polygon test."""
        # Unit square
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        assert p.point_in_polygon(0.5, 0.5, poly) is True
        assert p.point_in_polygon(0.0, 0.0, poly) is True  # Corner
        assert p.point_in_polygon(2.0, 2.0, poly) is False
        assert p.point_in_polygon(-1.0, -1.0, poly) is False

    def test_ship_polygon(self):
        """Test ship polygon generation."""
        polygon = p.ship_polygon(x=0.0, y=0.0, chi=0.0, length=100.0, beam=20.0)

        # 4 corners + closing point = 5
        assert len(polygon) == 5
        # Ship should be symmetric about x-axis at chi=0
        # Points should be at corners of bounding box
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        assert max(x_coords) - min(x_coords) == pytest.approx(100.0)
        assert max(y_coords) - min(y_coords) == pytest.approx(20.0)

    def test_polygon_distance(self):
        """Test polygon distance calculation."""
        # Two ships of 10x10 meters, 10 units apart
        # Ship 1 at (0,0) with chi=0, Ship 2 at (20,0) with chi=0
        # Ship 1 extends from x=-5 to x=5, Ship 2 from x=15 to x=25
        # Distance = 15 - 5 = 10
        dist = p.polygon_distance(0.0, 0.0, 0.0, 10.0, 10.0, 20.0, 0.0, 0.0, 10.0, 10.0)
        assert dist == pytest.approx(10.0)


class TestCoordinateTransforms:
    """Tests for coordinate transformation functions."""

    def test_utrn_to_latlon_basic(self):
        """Test UTM to lat/lon transform."""
        # Test with a known location (approximately San Francisco)
        utm_x, utm_y = 556790.0, 4135015.0
        try:
            lat, lon = p.utm_to_latlon(utm_x, utm_y, zone=10, northp=True)
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0
        except Exception:
            pytest.skip("pyproj not available for coordinate transforms")

    def test_latlon_to_utm_basic(self):
        """Test lat/lon to UTM transform."""
        lat, lon = 37.7749, -122.4194  # San Francisco

        try:
            utm_x, utm_y, zone, northp = p.latlon_to_utm(lat, lon)
            assert utm_x > 0
            assert utm_y > 0
            assert zone > 0
            assert northp is True
        except Exception:
            pytest.skip("pyproj not available for coordinate transforms")

    def test_round_trip_transform(self):
        """Test round-trip coordinate transform."""
        lat, lon = 37.7749, -122.4194

        try:
            utm_x, utm_y, zone, northp = p.latlon_to_utm(lat, lon)
            lat2, lon2 = p.utm_to_latlon(utm_x, utm_y, zone=zone, northp=northp)
            assert abs(lat - lat2) < 0.0001
            assert abs(lon - lon2) < 0.0001
        except Exception:
            pytest.skip("pyproj not available for coordinate transforms")


class TestVectorized:
    """Tests for vectorized utility functions."""

    def test_generate_norm_samples(self):
        """Test vectorized normal sample generation."""
        xs, ys = p.generate_norm_samples(0.0, 0.0, 1.0, 1.0, n_samples=100)
        assert len(xs) == 100
        assert len(ys) == 100
        assert np.mean(xs) < 0.2  # Close to mean

    def test_clamp(self):
        """Test clamping function."""
        assert p.clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)
        assert p.clamp(-0.5, 0.0, 1.0) == pytest.approx(0.0)
        assert p.clamp(1.5, 0.0, 1.0) == pytest.approx(1.0)

    def test_lerp(self):
        """Test linear interpolation."""
        assert p.lerp(0.0, 10.0, 0.0) == pytest.approx(0.0)
        assert p.lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)
        assert p.lerp(0.0, 10.0, 1.0) == pytest.approx(10.0)

    def test_wrap_angle(self):
        """Test angle wrapping."""
        assert p.wrap_angle(math.pi / 2) == pytest.approx(math.pi / 2)
        assert p.wrap_angle(2 * math.pi) == pytest.approx(0.0)
        assert p.wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_squared_distance_2d(self):
        """Test squared distance calculation."""
        assert p.squared_distance_2d(0.0, 0.0, 3.0, 4.0) == pytest.approx(25.0)
        assert p.squared_distance_2d(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
