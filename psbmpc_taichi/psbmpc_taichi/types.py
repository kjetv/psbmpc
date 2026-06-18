"""Type definitions for PSB-MPC Taichi.

Replaces C++ structs with Python dataclasses and Taichi-compatible types.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import taichi as ti

    TI_AVAILABLE = True
    # Initialize Taichi before using @ti.dataclass
    ti.init(arch=ti.cpu)
except ImportError:
    TI_AVAILABLE = False


# ============================================================================
# Taichi dataclasses (when running on GPU)
# ============================================================================

if TI_AVAILABLE:

    @ti.dataclass
    class ShipState4TI:
        """4D ship state: [x, y, chi, U]."""
        x: ti.f32
        y: ti.f32
        chi: ti.f32
        U: ti.f32

    @ti.dataclass
    class ShipState6TI:
        """6D ship state: [x, y, psi, u, v, r]."""
        x: ti.f32
        y: ti.f32
        psi: ti.f32
        u: ti.f32
        v: ti.f32
        r: ti.f32

    @ti.dataclass
    class WaypointTI:
        """Navigation waypoint."""
        x: ti.f32
        y: ti.f32
        id: ti.i32

    @ti.dataclass
    class ObstacleDataTI:
        """Obstacle ship data for collision avoidance."""
        # Position and velocity
        x: ti.f32
        y: ti.f32
        chi: ti.f32
        U: ti.f32
        # Dimensions (length, beam)
        length: ti.f32
        beam: ti.f32
        # COLREGs role: 0=unknown, 1=giving-way, 2=stand-on
        colregs_role: ti.i32
        # Collision detection parameters
        d_safe: ti.f32
        # Covariance for probabilistic methods
        cov_00: ti.f32
        cov_01: ti.f32
        cov_10: ti.f32
        cov_11: ti.f32

    @ti.dataclass
    class CPEResultTI:
        """Collision probability estimation result."""
        probability: ti.f32
        converged: ti.i32
        iterations: ti.i32
        n_samples: ti.i32

    @ti.dataclass
    class MPCResultTI:
        """MPC solver result."""
        # Optimal offsets
        offset_chi: ti.f32
        offset_U: ti.f32
        # Predicted trajectory
        traj_x: ti.types.vector(101, ti.f32)
        traj_y: ti.types.vector(101, ti.f32)
        traj_chi: ti.types.vector(101, ti.f32)
        traj_U: ti.types.vector(101, ti.f32)
        # Cost breakdown
        total_cost: ti.f32
        path_cost: ti.f32
        collision_cost: ti.f32
        colregs_cost: ti.f32


# ============================================================================
# Python dataclasses (CPU / host-side)
# ============================================================================


@dataclass
class ShipState4:
    """4D ship state: [x, y, chi (heading), U (surge speed)]."""
    x: float
    y: float
    chi: float
    U: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.chi, self.U], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ShipState4":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


@dataclass
class ShipState6:
    """6D ship state: [x, y, psi, u, v, r]."""
    x: float
    y: float
    psi: float
    u: float
    v: float
    r: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ShipState6":
        return cls(
            float(arr[0]), float(arr[1]), float(arr[2]),
            float(arr[3]), float(arr[4]), float(arr[5]),
        )


@dataclass
class Waypoint:
    """Navigation waypoint."""
    x: float
    y: float
    id: int = -1

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class ObstacleData:
    """Data for an obstacle ship in collision avoidance."""
    x: float = 0.0
    y: float = 0.0
    chi: float = 0.0
    U: float = 0.0
    length: float = 150.0
    beam: float = 25.0
    colregs_role: int = 0  # 0=unknown, 1=giving-way, 2=stand-on
    d_safe: float = 600.0  # safe distance in meters
    cov_xx: float = 10.0  # position covariance
    cov_yy: float = 10.0
    cov_xy: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.chi, self.U,
            self.length, self.beam, self.colregs_role, self.d_safe,
            self.cov_xx, self.cov_yy, self.cov_xy,
        ], dtype=np.float32)


@dataclass
class CPEResult:
    """Result from collision probability estimation."""
    probability: float = 0.0
    converged: bool = False
    iterations: int = 0
    n_samples: int = 0


@dataclass
class MPCResult:
    """Result from the MPC solver."""
    offset_chi: float = 0.0  # optimal heading offset (rad)
    offset_U: float = 0.0  # optimal surge offset
    traj_x: List[float] = field(default_factory=list)
    traj_y: List[float] = field(default_factory=list)
    traj_chi: List[float] = field(default_factory=list)
    traj_U: List[float] = field(default_factory=list)
    total_cost: float = 0.0
    path_cost: float = 0.0
    collision_cost: float = 0.0
    colregs_cost: float = 0.0


@dataclass
class PSBMPCParameters:
    """Parameters for Probabilistic/Safety-Bounded MPC."""
    n_M: int = 10  # number of MPC candidates
    n_cbs: int = 5  # number of candidate headings
    T: float = 300.0  # prediction horizon (seconds)
    dt: float = 1.0  # time step (seconds)
    d_safe: float = 600.0  # safe distance (meters)
    kappa_GW: float = 100.0  # giving-way cost weight
    kappa_SO: float = 50.0  # stand-on cost weight
    kappa_RA: float = 75.0  # readily apparent cost weight
    kappa_GN: float = 200.0  # grounding cost weight
    n_steps: int = 100  # number of prediction steps (T/dt)

    # CPE parameters
    cpe_max_iter: int = 50
    cpe_tolerance: float = 1e-4
    cpe_n_samples: int = 1000

    # Cost weights
    w_path: float = 1.0
    w_collision: float = 100.0
    w_colregs: float = 50.0
    w_deviation: float = 10.0

    def n_steps_total(self) -> int:
        return int(self.T / self.dt)


@dataclass
class SBMPCParameters:
    """Simplified Safety-Bounded MPC parameters (deterministic)."""
    n_M: int = 10
    n_cbs: int = 5
    T: float = 300.0
    dt: float = 1.0
    d_safe: float = 600.0
    kappa_GW: float = 100.0
    kappa_SO: float = 50.0
    kappa_RA: float = 75.0
    kappa_GN: float = 200.0
    safety_margin: float = 50.0  # additional safety margin in meters
    n_steps: int = 100
    use_probability: bool = False  # False = deterministic safety bounds
