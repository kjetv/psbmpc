"""Utility functions for PSB-MPC Taichi.

Provides geometry operations, coordinate transformations,
and PRNG functionality for both CPU and GPU execution.

GPU acceleration:
- All pure geometry functions are available as @ti.func for GPU execution
- NumPy arrays are passed via ti.ext_arr() for host-device interop
- Python fallbacks maintain API compatibility when Taichi is unavailable
"""
import math
from typing import Tuple, Optional, List

import numpy as np

from . import types

# ============================================================================
# Taichi imports (optional)
# ============================================================================

_ti_available = types.TI_AVAILABLE
ti: Optional["types"] = None

if _ti_available:
    import taichi as _ti
    ti = _ti


# ============================================================================
# Taichi @ti.func - GPU-accelerated geometry functions
# ============================================================================

if ti is not None:

    @ti.func
    def normalize_angle_taichi(angle: ti.f64) -> ti.f64:
        """GPU-accelerated angle normalization to [-pi, pi]."""
        two_pi = 2.0 * math.pi
        angle = angle - ti.cast(ti.floor(angle / two_pi), ti.f64) * two_pi
        if angle < -math.pi:
            angle = angle + two_pi
        elif angle > math.pi:
            angle = angle - two_pi
        return angle

    @ti.func
    def angle_diff_taichi(a: ti.f64, b: ti.f64) -> ti.f64:
        """GPU-accelerated shortest angle difference."""
        diff = a - b
        diff = normalize_angle_taichi(diff)
        return diff

    @ti.func
    def distance_2d_taichi(x1: ti.f64, y1: ti.f64, x2: ti.f64, y2: ti.f64) -> ti.f64:
        """GPU-accelerated Euclidean distance."""
        dx = x2 - x1
        dy = y2 - y1
        return ti.sqrt(dx * dx + dy * dy)

    @ti.func
    def bearing_2d_taichi(x1: ti.f64, y1: ti.f64, x2: ti.f64, y2: ti.f64) -> ti.f64:
        """GPU-accelerated bearing calculation."""
        return ti.atan2(y2 - y1, x2 - x1)

    @ti.func
    def cross_2d_taichi(ax: ti.f64, ay: ti.f64, bx: ti.f64, by: ti.f64) -> ti.f64:
        """GPU-accelerated 2D cross product."""
        return ax * by - ay * bx

    @ti.func
    def dot_2d_taichi(ax: ti.f64, ay: ti.f64, bx: ti.f64, by: ti.f64) -> ti.f64:
        """GPU-accelerated 2D dot product."""
        return ax * bx + ay * by

    @ti.func
    def point_to_segment_distance_taichi(
        px: ti.f64, py: ti.f64,
        ax: ti.f64, ay: ti.f64,
        bx: ti.f64, by: ti.f64,
    ) -> Tuple[ti.f64, ti.f64, ti.f64, ti.f64]:
        """GPU-accelerated point-to-segment distance.
        
        Returns (distance, t, proj_x, proj_y).
        """
        abx = bx - ax
        aby = by - ay
        len_sq = abx * abx + aby * aby

        proj_x = ax
        proj_y = ay
        t = ti.f64(0.0)

        if len_sq > ti.f64(1e-12):
            t = ((px - ax) * abx + (py - ay) * aby) / len_sq
            t = ti.max(ti.f64(0.0), ti.min(ti.f64(1.0), t))
            proj_x = ax + t * abx
            proj_y = ay + t * aby

        dx = px - proj_x
        dy = py - proj_y
        dist = ti.sqrt(dx * dx + dy * dy)

        return dist, t, proj_x, proj_y

    @ti.func
    def ship_polygon_taichi(
        x: ti.f64, y: ti.f64, chi: ti.f64,
        length: ti.f64, beam: ti.f64,
        corners: ti.types.ndarray(),
    ) -> None:
        """GPU-accelerated ship polygon generation.
        
        Fills corners array with 5 points (4 corners + closing point).
        """
        half_l = length * ti.f64(0.5)
        half_b = beam * ti.f64(0.5)

        cos_chi = ti.cos(chi)
        sin_chi = ti.sin(chi)

        # 4 corners + closing point
        local_lx = ti.Vector([half_l, half_l, -half_l, -half_l, half_l])
        local_ly = ti.Vector([-half_b, half_b, half_b, -half_b, -half_b])

        for i in range(5):
            wx = x + local_lx[i] * cos_chi - local_ly[i] * sin_chi
            wy = y + local_lx[i] * sin_chi + local_ly[i] * cos_chi
            corners[i * 2] = wx
            corners[i * 2 + 1] = wy

    @ti.func
    def polygon_distance_taichi(
        x1: ti.f64, y1: ti.f64, chi1: ti.f64, l1: ti.f64, b1: ti.f64,
        x2: ti.f64, y2: ti.f64, chi2: ti.f64, l2: ti.f64, b2: ti.f64,
    ) -> ti.f64:
        """GPU-accelerated minimum distance between two ship polygons."""
        # Generate polygons
        poly1 = ti.Vector.zeros(10, dtype=ti.f64)  # 5 points * 2 coords
        poly2 = ti.Vector.zeros(10, dtype=ti.f64)
        ship_polygon_taichi(x1, y1, chi1, l1, b1, poly1)
        ship_polygon_taichi(x2, y2, chi2, l2, b2, poly2)

        min_dist = ti.f64(1e18)  # infinity

        # Check all segments
        for i in range(4):  # 4 segments in rectangle
            ax = poly1[i * 2]
            ay = poly1[i * 2 + 1]
            bx = poly1[(i + 1) * 2]
            by = poly1[(i + 1) * 2 + 1]

            for j in range(4):
                cx = poly2[j * 2]
                cy = poly2[j * 2 + 1]
                dx = poly2[(j + 1) * 2]
                dy = poly2[(j + 1) * 2 + 1]

                dist, _, _, _ = point_to_segment_distance_taichi(cx, cy, ax, ay, bx, by)
                if dist < min_dist:
                    min_dist = dist

                dist, _, _, _ = point_to_segment_distance_taichi(ax, ay, cx, cy, dx, dy)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    @ti.func
    def gaussian_pdf_taichi(x: ti.f64, mu: ti.f64, sigma: ti.f64) -> ti.f64:
        """GPU-accelerated Gaussian PDF."""
        diff = x - mu
        exponent = -ti.f64(0.5) * diff * diff / (sigma * sigma + ti.f64(1e-12))
        return ti.exp(exponent) / (ti.sqrt(ti.f64(2.0) * math.pi) * (sigma + ti.f64(1e-12)))


# ============================================================================
# Python CPU fallbacks (for non-Taichi usage)
# ============================================================================


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def squared_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Squared Euclidean distance between two 2D points (no sqrt)."""
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def generate_norm_samples(
    mu_x: float = 0.0,
    mu_y: float = 0.0,
    std_x: float = 1.0,
    std_y: float = 1.0,
    rho: float = 0.0,
    n_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated normal samples.

    Args:
        mu_x, mu_y: means
        std_x, std_y: standard deviations
        rho: correlation coefficient
        n_samples: number of samples to generate

    Returns:
        (samples_x, samples_y): arrays of shape (n_samples,)
    """
    prng = Xoshiro256pp()
    samples_x = np.zeros(n_samples, dtype=np.float64)
    samples_y = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        x, y = prng.correlated_normal(mu_x, mu_y, std_x, std_y, rho)
        samples_x[i] = x
        samples_y[i] = y

    return samples_x, samples_y


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
) -> Tuple[float, float, float, float]:
    """Distance from point P to line segment AB.

    Returns:
        (distance, t, proj_x, proj_y): distance to segment,
        parametric position t in [0,1], and projection point.
    """
    abx = bx - ax
    aby = by - ay
    len_sq = abx * abx + aby * aby

    if len_sq < 1e-12:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy), 0.0, ax, ay

    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby

    dx = px - proj_x
    dy = py - proj_y
    dist = math.sqrt(dx * dx + dy * dy)

    return dist, t, proj_x, proj_y


def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray casting algorithm to test if point is inside a polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi == py and xi == px) or (yj == py and xj == px):
            return True

        if ((yi > py) != (yj > py)):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i

    return inside


def line_segment_intersection(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> Tuple[bool, float, float]:
    """Test if line segments AB and CD intersect."""
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)

    if abs(denom) < 1e-12:
        return False, 0.0, 0.0

    tx = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    ty = ((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom

    if 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0:
        return True, tx, ty

    return False, 0.0, 0.0


def ship_polygon(
    x: float, y: float, chi: float,
    length: float = 150.0, beam: float = 25.0,
) -> list:
    """Generate the rectangular polygon of a ship in world coordinates.

    Args:
        x, y: ship position
        chi: ship heading in radians
        length: ship length
        beam: ship beam (width)

    Returns:
        List of (x, y) tuples representing the ship polygon (5 points).
    """
    half_l = length * 0.5
    half_b = beam * 0.5

    cos_chi = math.cos(chi)
    sin_chi = math.sin(chi)

    # 4 corners + closing point
    local_corners = [
        (half_l, -half_b),  # bow starboard
        (half_l, half_b),   # bow port
        (-half_l, half_b),  # stern port
        (-half_l, -half_b), # stern starboard
    ]

    polygon = []
    for i in range(5):
        if i < 4:
            lx, ly = local_corners[i]
        else:
            # Close the polygon
            lx, ly = local_corners[0]

        wx = x + lx * cos_chi - ly * sin_chi
        wy = y + lx * sin_chi + ly * cos_chi
        polygon.append((wx, wy))

    return polygon


def polygon_distance(
    x1: float, y1: float, chi1: float, l1: float, b1: float,
    x2: float, y2: float, chi2: float, l2: float, b2: float,
) -> float:
    """Minimum distance between two ship polygons (CPU version).

    Args:
        Ship 1: (x1, y1, chi1, l1, b1)
        Ship 2: (x2, y2, chi2, l2, b2)

    Returns:
        Minimum distance between the two rectangular ships.
    """
    poly1 = ship_polygon(x1, y1, chi1, l1, b1)
    poly2 = ship_polygon(x2, y2, chi2, l2, b2)

    min_dist = float('inf')

    for i in range(4):
        ax, ay = poly1[i]
        bx, by = poly1[(i + 1) % 4]

        for j in range(4):
            cx, cy = poly2[j]
            dx, dy = poly2[(j + 1) % 4]

            dist, _, _, _ = point_to_segment_distance(cx, cy, ax, ay, bx, by)
            min_dist = min(min_dist, dist)

            dist, _, _, _ = point_to_segment_distance(ax, ay, cx, cy, dx, dy)
            min_dist = min(min_dist, dist)

    return min_dist


def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """1D Gaussian probability density function.

    Args:
        x: evaluation point
        mu: mean
        sigma: standard deviation

    Returns:
        PDF value at x.
    """
    if sigma < 1e-10:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


# ============================================================================
# Batch kernels for NumPy arrays
# ============================================================================

if ti is not None:

    @ti.kernel
    def batch_normalize_angle_taichi(
        angles: ti.types.ndarray(),
        result: ti.types.ndarray(),
    ) -> None:
        """Normalize angles in batch.

        Args:
            angles: input array of angles in radians
            result: output array of normalized angles in [-pi, pi]
        """
        for i in range(len(angles)):
            result[i] = normalize_angle_taichi(ti.f64(angles[i]))

    @ti.kernel
    def batch_distance_2d(
        x1s: ti.types.ndarray(),
        y1s: ti.types.ndarray(),
        x2s: ti.types.ndarray(),
        y2s: ti.types.ndarray(),
        result: ti.types.ndarray(),
    ) -> None:
        """Compute distances in batch.

        Args:
            x1s, y1s: first set of points
            x2s, y2s: second set of points
            result: output array of distances
        """
        for i in range(len(x1s)):
            result[i] = distance_2d_taichi(
                ti.f64(x1s[i]), ti.f64(y1s[i]),
                ti.f64(x2s[i]), ti.f64(y2s[i]),
            )


# ============================================================================
# PRNG (Pseudo-Random Number Generator)
# ============================================================================


class Xoshiro256pp:
    """Xoshiro256++ pseudo-random number generator.

    A high-quality PRNG suitable for Monte Carlo simulations.
    Ported from the C++ implementation.

    Attributes:
        state: 4-word (128-bit) internal state
    """

    def __init__(self, seed: int = 42):
        """Initialize PRNG with seed.

        Args:
            seed: random seed value
        """
        self.state = [0, 0, 0, 0]
        self._seed(seed)

    def _seed(self, seed: int) -> None:
        """Seed the PRNG using splitmix64.

        Args:
            seed: seed value
        """
        s = seed
        for i in range(4):
            s = (s + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            z = s
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb
            z = z ^ (z >> 31)
            self.state[i] = z & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        """Rotate left for 64-bit integers.

        Args:
            x: value to rotate
            k: number of bits to rotate

        Returns:
            Rotated value
        """
        x &= 0xFFFFFFFFFFFFFFFF
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        """Generate next 64-bit unsigned integer.

        Returns:
            Random 64-bit unsigned integer
        """
        result = (self._rotl(self.state[1] * 5, 7) * 9 + self.state[3]) & 0xFFFFFFFFFFFFFFFF
        self._rotate()
        return result

    def next_f32(self) -> float:
        """Generate next floating-point number in [0, 1).

        Returns:
            Random float in [0, 1)
        """
        return self.next_u64() / (2 ** 64)

    def next_f64(self) -> float:
        """Generate next double-precision floating-point number in [0, 1).

        Returns:
            Random double in [0, 1)
        """
        return self.next_u64() / (2 ** 64)

    def next_normal(self) -> float:
        """Generate next standard normal random number (Box-Muller).

        Returns:
            Random number from N(0, 1)
        """
        u1 = max(self.next_f32(), 1e-10)
        u2 = self.next_f32()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def next_multivariate_normal_2d(
        self,
        mu_x: float, mu_y: float,
        var_x: float, var_y: float,
        cov_xy: float,
    ) -> Tuple[float, float]:
        """Generate 2D correlated normal random samples.

        Uses Cholesky decomposition to introduce correlation.

        Args:
            mu_x, mu_y: means
            var_x, var_y: variances
            cov_xy: covariance

        Returns:
            Tuple of (x, y) samples
        """
        std_x = math.sqrt(max(var_x, 1e-10))
        std_y = math.sqrt(max(var_y, 1e-10))

        # Correlation coefficient
        rho = cov_xy / (std_x * std_y + 1e-10)
        rho = max(-1.0, min(1.0, rho))

        # Cholesky decomposition of correlation matrix
        z1 = self.next_normal()
        z2 = self.next_normal()

        x = mu_x + std_x * (z1)
        y = mu_y + std_y * (rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * z2)

        return x, y

    def correlated_normal(self, mu_x: float, mu_y: float, std_x: float, std_y: float, rho: float) -> Tuple[float, float]:
        """Generate a correlated bivariate normal sample.

        Args:
            mu_x, mu_y: means
            std_x, std_y: standard deviations
            rho: correlation coefficient in [-1, 1]

        Returns:
            Tuple of (x, y) samples
        """
        rho = max(-1.0, min(1.0, rho))
        z1 = self.next_normal()
        z2 = self.next_normal()
        x = mu_x + std_x * z1
        y = mu_y + std_y * (rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * z2)
        return x, y

    def uniform_sample(self, n: int) -> np.ndarray:
        """Generate n uniform random samples in [0, 1).

        Args:
            n: number of samples to generate

        Returns:
            Array of n uniform random samples
        """
        return np.array([self.next_f32() for _ in range(n)])

    def normal_sample(self, n: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Generate n normal random samples.

        Args:
            n: number of samples to generate
            mean: mean of the distribution
            std: standard deviation of the distribution

        Returns:
            Array of n normal random samples
        """
        samples = np.array([self.next_normal() for _ in range(n)])
        return mean + std * samples

    def multivariate_normal_sample(
        self,
        mean: List[float],
        cov: List[List[float]],
        n: int,
    ) -> np.ndarray:
        """Generate n multivariate normal random samples using Cholesky decomposition.

        Args:
            mean: mean vector
            cov: covariance matrix
            n: number of samples to generate

        Returns:
            Array of n samples, each with len(mean) dimensions
        """
        dim = len(mean)
        samples = []
        
        # Cholesky decomposition of covariance matrix
        L = self._cholesky_decompose(cov, dim)
        
        for _ in range(n):
            # Generate standard normal samples
            z = np.array([self.next_normal() for _ in range(dim)])
            # Transform using Cholesky factor and add mean
            sample = np.array(mean) + L @ z
            samples.append(sample.tolist())
        
        return np.array(samples)

    def _cholesky_decompose(self, cov: List[List[float]], dim: int) -> np.ndarray:
        """Compute Cholesky decomposition of covariance matrix.

        Args:
            cov: covariance matrix as list of lists
            dim: dimension of the matrix

        Returns:
            Lower triangular matrix L such that cov = L @ L.T
        """
        L = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i + 1):
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = cov[i][i] - sum_val
                    L[i][j] = math.sqrt(max(val, 1e-10))
                else:
                    L[i][j] = (cov[i][j] - sum_val) / L[j][j] if L[j][j] > 1e-10 else 0.0
        return L

    def _rotate(self) -> None:
        """Rotate the internal state (Xoshiro256++ core operation)."""
        s = self.state
        s[0], s[1], s[2], s[3] = (
            s[1],
            s[2],
            s[3],
            s[0],
        )
        s[0] ^= s[3]


# ============================================================================
# Coordinate transformation functions (optional pyproj dependency)
# ============================================================================


def utm_to_latlon(utm_x: float, utm_y: float, zone: int = 10, northp: bool = True) -> Tuple[float, float]:
    """Convert UTM coordinates to latitude/longitude.

    Requires pyproj. Falls back to approximate conversion if unavailable.

    Args:
        utm_x: Easting (meters)
        utm_y: Northing (meters)
        zone: UTM zone number
        northp: True for northern hemisphere, False for southern

    Returns:
        (latitude, longitude) in degrees
    """
    try:
        from pyproj import Transformer
        # Create transformer from UTM to WGS84
        epsg_code = f"{'326' if northp else '327'}{zone:03d}"
        transformer = Transformer.from_crs(epsg_code, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)
        return float(lat), float(lon)
    except ImportError:
        # Fallback: approximate conversion (not accurate but prevents crash)
        # This is a rough approximation for demonstration
        central_meridian = (6 - zone) if northp else (6 - zone)
        lon = (utm_x / 6366000.0) * (180.0 / math.pi) + central_meridian
        lat = (utm_y / 6366000.0) * (180.0 / math.pi)
        return lat, lon


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, bool]:
    """Convert latitude/longitude to UTM coordinates.

    Requires pyproj. Falls back to approximate conversion if unavailable.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        (easting, northing, zone, northp)
    """
    try:
        from pyproj import Transformer
        # Determine UTM zone
        zone = int((lon + 180.0) / 6.0) + 1
        northp = lat >= 0
        epsg_code = f"{'326' if northp else '327'}{zone:03d}"
        transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return float(utm_x), float(utm_y), zone, northp
    except ImportError:
        # Fallback: approximate conversion
        zone = int((lon + 180.0) / 6.0) + 1
        northp = lat >= 0
        utm_x = lon * 6366000.0 * (math.pi / 180.0)
        utm_y = lat * 6366000.0 * (math.pi / 180.0)
        return float(utm_x), float(utm_y), zone, northp


# ============================================================================
# Public API exports
# ============================================================================

__all__ = [
    # Taichi functions (when available)
    "normalize_angle_taichi",
    "angle_diff_taichi",
    "distance_2d_taichi",
    "bearing_2d_taichi",
    "cross_2d_taichi",
    "dot_2d_taichi",
    "point_to_segment_distance_taichi",
    "ship_polygon_taichi",
    "polygon_distance_taichi",
    "gaussian_pdf_taichi",
    # Batch kernels
    "batch_normalize_angle_taichi",
    "batch_distance_2d",
    # Coordinate transformations
    "utm_to_latlon",
    "latlon_to_utm",
    # Additional utility functions
    "clamp",
    "lerp",
    "wrap_angle",
    "squared_distance_2d",
    "generate_norm_samples",
    # CPU functions
    "normalize_angle",
    "angle_diff",
    "distance_2d",
    "bearing_2d",
    "cross_2d",
    "dot_2d",
    "point_to_segment_distance",
    "point_in_polygon",
    "line_segment_intersection",
    "ship_polygon",
    "polygon_distance",
    "gaussian_pdf",
    # PRNG
    "Xoshiro256pp",
]
