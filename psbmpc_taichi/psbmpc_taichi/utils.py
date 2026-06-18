"""Utility functions for PSB-MPC Taichi.

Provides geometry operations, coordinate transformations,
and PRNG functionality needed by ship models, CPE, and cost functions.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ============================================================================
# Geometry Utilities
# ============================================================================


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def angle_diff(a: float, b: float) -> float:
    """Shortest angle difference from b to a, in [-pi, pi]."""
    diff = normalize_angle(a - b)
    return diff


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def bearing_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing from point 1 to point 2, in radians [-pi, pi]."""
    return float(math.atan2(y2 - y1, x2 - x1))


def cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product (z-component) of vectors [ax, ay] and [bx, by]."""
    return ax * by - ay * bx


def dot_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """2D dot product of vectors [ax, ay] and [bx, by]."""
    return ax * bx + ay * by


def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> Tuple[float, float, float]:
    """Distance from point P to line segment AB.

    Returns:
        (distance, t, proj_x, proj_y): distance to segment,
        parametric position t in [0,1], and projection point.
    """
    abx = bx - ax
    aby = by - ay
    len_sq = abx * abx + aby * aby

    if len_sq < 1e-12:
        # A and B are effectively the same point
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy), 0.0, ax, ay

    # Project P onto line AB, clamped to segment
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len_sq))

    # Projection point
    proj_x = ax + t * abx
    proj_y = ay + t * aby

    dx = px - proj_x
    dy = py - proj_y
    dist = math.sqrt(dx * dx + dy * dy)

    return dist, t, proj_x, proj_y


def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray casting algorithm to test if point is inside a polygon.

    Args:
        px, py: point coordinates
        polygon: list of (x, y) tuples forming the polygon

    Returns:
        True if point is inside or on the boundary of the polygon.
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if point is on edge
        if (yi == py and xi == px) or (yj == py and xj == px):
            return True

        # Ray casting test
        if ((yi > py) != (yj > py)):
            # Compute x coordinate of intersection
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i

    return inside


def line_segment_intersection(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> Tuple[bool, float, float]:
    """Test if line segments AB and CD intersect.

    Returns:
        (intersects, tx, ty): whether they intersect and the intersection point.
    """
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)

    if abs(denom) < 1e-12:
        # Parallel lines
        return False, 0.0, 0.0

    tx = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    ty = ((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom

    # Check if intersection is within both segments
    if 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0:
        return True, tx, ty

    return False, 0.0, 0.0


def ship_polygon(
    x: float, y: float, chi: float,
    length: float = 150.0, beam: float = 25.0,
) -> list:
    """Generate the rectangular polygon of a ship in world coordinates.

    The ship is centered at (x, y) with heading chi.
    Returns list of (x, y) tuples for the 4 corners.
    """
    half_l = length / 2.0
    half_b = beam / 2.0

    # Local corners (bow is at +x in local frame)
    # 4 corners of rectangular ship: bow-starboard, bow-port, stern-port, stern-starboard
    corners_local = [
        (half_l, -half_b),  # bow starboard
        (half_l, half_b),   # bow port
        (-half_l, half_b),  # stern port
        (-half_l, -half_b), # stern starboard
    ]

    # Rotate and translate
    cos_chi = math.cos(chi)
    sin_chi = math.sin(chi)

    corners = []
    for lx, ly in corners_local:
        wx = x + lx * cos_chi - ly * sin_chi
        wy = y + lx * sin_chi + ly * cos_chi
        corners.append((wx, wy))

    # Add bow point to close the polygon
    corners.append(corners[0])

    return corners


def polygon_distance(
    x1: float, y1: float, chi1: float,
    l1: float, b1: float,
    x2: float, y2: float, chi2: float,
    l2: float, b2: float,
) -> float:
    """Minimum distance between two ship polygons.

    Computes the minimum distance between any two line segments
    of the two rectangular ship polygons.
    """
    poly1 = ship_polygon(x1, y1, chi1, l1, b1)
    poly2 = ship_polygon(x2, y2, chi2, l2, b2)

    min_dist = float("inf")
    for i in range(len(poly1) - 1):
        ax, ay = poly1[i]
        bx, by = poly1[i + 1]
        for j in range(len(poly2) - 1):
            cx, cy = poly2[j]
            dx, dy = poly2[j + 1]
            dist, _, _, _ = point_to_segment_distance(cx, cy, ax, ay, bx, by)
            min_dist = min(min_dist, dist)
            dist, _, _, _ = point_to_segment_distance(ax, ay, cx, cy, dx, dy)
            min_dist = min(min_dist, dist)

    return min_dist


# ============================================================================
# Coordinate Transformations
# ============================================================================


def utm_to_latlon(utm_e: float, utm_n: float, zone: int, south: bool = False) -> Tuple[float, float]:
    """Convert UTM coordinates to latitude/longitude.

    Uses pyproj if available, otherwise returns placeholder values.
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(
            f"EPSG:{32600 + zone if not south else 32700 + zone}",
            "EPSG:4326",
            always_xy=True,
        )
        lat, lon = transformer.transform(utm_e, utm_n)
        return float(lat), float(lon)
    except ImportError:
        # Fallback: return UTM coordinates as-is (caller should handle)
        return utm_e, utm_n


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, bool]:
    """Convert latitude/longitude to UTM coordinates.

    Returns (easting, northing, zone, is_south).
    """
    try:
        from pyproj import Transformer
        zone = int((lon + 180) / 6) + 1
        south = lat < 0

        crs_src = f"EPSG:4326"
        crs_dst = f"EPSG:{32600 + zone if not south else 32700 + zone}"

        transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        easting, northing = transformer.transform(lat, lon)
        return float(easting), float(northing), zone, south
    except ImportError:
        # Fallback
        return lat, lon, 0, False


# ============================================================================
# PRNG Utilities
# ============================================================================


class Xoshiro256pp:
    """xoshiro256++ pseudo-random number generator.

    A high-quality PRNG suitable for Monte Carlo simulations.
    Ported from the original C++ implementation.
    Uses Python ints to avoid numpy uint64 overflow issues.
    """

    MASK64 = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: int = 0):
        """Initialize with a seed value."""
        self.state = [0, 0, 0, 0]  # Python ints, not numpy
        self._seed(seed)

    def _seed(self, seed: int):
        """Seed the generator using splitmix64."""
        self.state = [0, 0, 0, 0]

        s = seed & self.MASK64
        for _ in range(4):
            s = (s + 0x9E3779B97F4A7C15) & self.MASK64
            z = s
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK64
            z = z ^ (z >> 31)
            self.state[(s >> 48) & 3] = z

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        """Rotate left for 64-bit integers."""
        x &= Xoshiro256pp.MASK64
        return ((x << k) | (x >> (64 - k))) & Xoshiro256pp.MASK64

    def next_u64(self) -> int:
        """Generate the next 64-bit unsigned integer."""
        result = (
            self._rotl((self.state[1] * 5) & self.MASK64, 7) * 9
            + self.state[3]
        ) & self.MASK64
        t = (self.state[1] << 17) & self.MASK64

        # Rotate state
        self.state[2] ^= self.state[0]
        self.state[3] ^= self.state[1]
        self.state[1] ^= self.state[2]
        self.state[0] ^= self.state[3]

        self.state[2] ^= t
        self.state[3] = self._rotl(self.state[3], 45)

        return result

    def next_f32(self) -> float:
        """Generate a float32 in [0, 1)."""
        return float((self.next_u64() >> 11) & 0x7FFFFFFF) / float(0x80000000)

    def next_normal(self) -> float:
        """Generate a standard normal random variable using Box-Muller transform."""
        u1 = max(self.next_f32(), 1e-15)
        u2 = self.next_f32()
        return float(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))

    def next_normal_sigma(self, sigma: float) -> float:
        """Generate a normal random variable with given standard deviation."""
        return self.next_normal() * sigma

    def next_normal_mu_sigma(self, mu: float, sigma: float) -> float:
        """Generate a normal random variable with given mean and std dev."""
        return mu + self.next_normal() * sigma

    def next_multivariate_normal_2d(
        self,
        mu_x: float, mu_y: float,
        cov_xx: float, cov_yy: float, cov_xy: float,
    ) -> Tuple[float, float]:
        """Generate a 2D multivariate normal sample.

        Uses Cholesky decomposition of the covariance matrix.
        """
        # Cholesky decomposition of [[cov_xx, cov_xy], [cov_xy, cov_yy]]
        L11 = math.sqrt(max(cov_xx, 1e-12))
        L21 = cov_xy / L11 if L11 > 1e-12 else 0.0
        L22 = math.sqrt(max(cov_yy - L21 * L21, 1e-12))

        z1 = self.next_normal()
        z2 = self.next_normal()

        sx = mu_x + L11 * z1
        sy = mu_y + L21 * z1 + L22 * z2

        return float(sx), float(sy)

    def next_multivariate_normal_sigma(
        self,
        mu_x: float, mu_y: float,
        sigma: float,
    ) -> Tuple[float, float]:
        """Generate a 2D normal sample with isotropic covariance sigma^2 * I."""
        z1 = self.next_normal()
        z2 = self.next_normal()
        return mu_x + sigma * z1, mu_y + sigma * z2

    def uniform_sample(self, n: int) -> np.ndarray:
        """Generate n uniform random samples in [0, 1).

        Args:
            n: number of samples

        Returns:
            NumPy array of n uniform random samples
        """
        return np.array([self.next_f32() for _ in range(n)])

    def normal_sample(self, n: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Generate n normal random samples.

        Args:
            n: number of samples
            mean: mean of the distribution
            std: standard deviation of the distribution

        Returns:
            NumPy array of n normal random samples
        """
        return np.array([self.next_normal_mu_sigma(mean, std) for _ in range(n)])

    def multivariate_normal_sample(
        self,
        mean: list,
        cov: list,
        n: int,
    ) -> np.ndarray:
        """Generate n samples from a multivariate normal distribution.

        Args:
            mean: list of means (e.g., [0.0, 0.0, 0.0])
            cov: covariance matrix (list of lists)
            n: number of samples

        Returns:
            NumPy array of shape (n, len(mean)) with samples
        """
        dim = len(mean)
        samples = []
        for _ in range(n):
            # Cholesky decomposition
            L = self._cholesky(cov)
            z = [self.next_normal() for _ in range(dim)]
            sample = [0.0] * dim
            for i in range(dim):
                for j in range(i + 1):
                    sample[i] += L[i][j] * z[j]
                sample[i] += mean[i]
            samples.append(sample)
        return np.array(samples)

    @staticmethod
    def _cholesky(matrix: list) -> list:
        """Compute Cholesky decomposition of a positive-definite matrix.

        Args:
            matrix: square covariance matrix

        Returns:
            Lower triangular matrix L such that A = L @ L.T
        """
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = matrix[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-12))
                else:
                    if abs(L[j][j]) < 1e-12:
                        L[i][j] = 0.0
                    else:
                        L[i][j] = (matrix[i][j] - s) / L[j][j]
        return L


# ============================================================================
# NumPy-based vectorized utilities
# ============================================================================


def generate_norm_samples(
    mu_x: float, mu_y: float,
    sigma_x: float, sigma_y: float,
    rho: float = 0.0,
    n_samples: int = 1000,
    rng: Xoshiro256pp | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D normal samples using NumPy.

    Args:
        mu_x, mu_y: mean coordinates
        sigma_x, sigma_y: standard deviations
        rho: correlation coefficient
        n_samples: number of samples
        rng: optional PRNG (uses NumPy default if not provided)

    Returns:
        (xs, ys): arrays of sampled coordinates
    """
    if rng is not None:
        # Use xoshiro256++ for reproducibility with C++
        xs = np.zeros(n_samples, dtype=np.float32)
        ys = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            sx, sy = rng.next_multivariate_normal_2d(
                mu_x, mu_y,
                sigma_x ** 2, sigma_y ** 2,
                rho * sigma_x * sigma_y,
            )
            xs[i] = sx
            ys[i] = sy
        return xs, ys
    else:
        # Use NumPy's built-in for speed
        cov = np.array([
            [sigma_x ** 2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y ** 2],
        ])
        samples = np.random.multivariate_normal([mu_x, mu_y], cov, n_samples)
        return samples[:, 0].astype(np.float32), samples[:, 1].astype(np.float32)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]. Same as normalize_angle but different name."""
    return normalize_angle(angle)


def squared_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Squared Euclidean distance (avoids sqrt for efficiency)."""
    return (x2 - x1) ** 2 + (y2 - y1) ** 2
