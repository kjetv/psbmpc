"""Type definitions for PSB-MPC Taichi.

Replaces C++ structs with Python dataclasses and Taichi-compatible types.
Supports both CPU (pure Python) and GPU (Taichi) execution modes.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import taichi as ti

# ============================================================================
# Taichi initialization (lazy, with GPU fallback)
# ============================================================================

TI_AVAILABLE = False
TI_GPU_AVAILABLE = False
ti: Optional["ti"] = None

def _init_taichi() -> None:
    """Initialize Taichi with GPU if available, CPU otherwise."""
    global TI_AVAILABLE, TI_GPU_AVAILABLE, ti
    try:
        import taichi as _ti
        TI_AVAILABLE = True

        # Try CUDA first, fall back to CPU
        # In Taichi 1.7+, ti.cuda is an Arch enum value, not a module
        # We try to initialize with CUDA and catch any errors
        gpu_available = False
        for arch in [_ti.cuda, _ti.vulkan, _ti.metal]:
            try:
                _ti.init(arch=arch, device_memory_ratio=0.5)
                # Check if the init actually used GPU
                from taichi import config
                if config.arch_name not in ('cpu',):
                    gpu_available = True
                    TI_GPU_AVAILABLE = True
                    break
            except Exception:
                continue

        if not gpu_available:
            _ti.init(arch=_ti.cpu)
            TI_GPU_AVAILABLE = False

        ti = _ti
    except ImportError:
        TI_AVAILABLE = False
        TI_GPU_AVAILABLE = False


# Initialize on import
_init_taichi()


# ============================================================================
# Taichi struct and field definitions (when Taichi is available)
# ============================================================================

if TI_AVAILABLE and ti is not None:

    # --- Taichi structs ---

    @ti.dataclass
    class ShipState4TI:
        """4D ship state: [x, y, chi, U]."""
        x: ti.f64
        y: ti.f64
        chi: ti.f64
        U: ti.f64

    @ti.dataclass
    class ShipState6TI:
        """6D ship state: [x, y, psi, u, v, r]."""
        x: ti.f64
        y: ti.f64
        psi: ti.f64
        u: ti.f64
        v: ti.f64
        r: ti.f64

    @ti.dataclass
    class WaypointTI:
        """Navigation waypoint."""
        x: ti.f64
        y: ti.f64
        id: ti.i32

    @ti.dataclass
    class ObstacleDataTI:
        """Obstacle ship data for collision avoidance."""
        x: ti.f64
        y: ti.f64
        chi: ti.f64
        U: ti.f64
        length: ti.f64
        beam: ti.f64
        colregs_role: ti.i32
        d_safe: ti.f64
        cov_xx: ti.f64
        cov_yy: ti.f64
        cov_xy: ti.f64

    @ti.dataclass
    class CPEResultTI:
        """Collision probability estimation result."""
        probability: ti.f64
        converged: ti.i32
        iterations: ti.i32
        n_samples: ti.i32

    @ti.dataclass
    class MPCResultTI:
        """MPC solver result."""
        offset_chi: ti.f64
        offset_U: ti.f64
        traj_x: ti.types.vector(101, ti.f64)
        traj_y: ti.types.vector(101, ti.f64)
        traj_chi: ti.types.vector(101, ti.f64)
        traj_U: ti.types.vector(101, ti.f64)
        total_cost: ti.f64
        path_cost: ti.f64
        collision_cost: ti.f64
        colregs_cost: ti.f64

    # --- Field layouts for batched GPU data ---

    MAX_OBSTACLES = 64
    MAX_WAYPOINTS = 64
    MAX_CANDIDATES = 16
    MAX_TRAJ_STEPS = 301  # T=300, dt=1 => 301 points

    # Obstacle fields (dense, one row per obstacle)
    obs_x_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_y_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_chi_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_U_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_length_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_beam_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)
    obs_colregs_f = ti.field(dtype=ti.i32, shape=MAX_OBSTACLES)
    obs_dsafe_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)

    # Waypoint fields
    wp_x_f = ti.field(dtype=ti.f64, shape=MAX_WAYPOINTS)
    wp_y_f = ti.field(dtype=ti.f64, shape=MAX_WAYPOINTS)

    # Ship state fields
    ship_x_f = ti.field(dtype=ti.f64, shape=())
    ship_y_f = ti.field(dtype=ti.f64, shape=())
    ship_chi_f = ti.field(dtype=ti.f64, shape=())
    ship_U_f = ti.field(dtype=ti.f64, shape=())

    # Candidate bearing fields (one per candidate)
    cand_chi_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)
    cand_U_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)

    # Trajectory output fields: [candidate_idx, timestep]
    traj_x_f = ti.field(dtype=ti.f64, shape=(MAX_CANDIDATES, MAX_TRAJ_STEPS))
    traj_y_f = ti.field(dtype=ti.f64, shape=(MAX_CANDIDATES, MAX_TRAJ_STEPS))
    traj_chi_f = ti.field(dtype=ti.f64, shape=(MAX_CANDIDATES, MAX_TRAJ_STEPS))
    traj_U_f = ti.field(dtype=ti.f64, shape=(MAX_CANDIDATES, MAX_TRAJ_STEPS))

    # Cost output fields: [candidate_idx]
    cost_total_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)
    cost_path_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)
    cost_collision_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)
    cost_colregs_f = ti.field(dtype=ti.f64, shape=MAX_CANDIDATES)

    # CPE fields: [obstacle_idx, sample_idx]
    MAX_CPE_SAMPLES = 2048
    cpe_samples_x_f = ti.field(dtype=ti.f64, shape=(MAX_OBSTACLES, MAX_CPE_SAMPLES))
    cpe_samples_y_f = ti.field(dtype=ti.f64, shape=(MAX_OBSTACLES, MAX_CPE_SAMPLES))
    cpe_weights_f = ti.field(dtype=ti.f64, shape=(MAX_OBSTACLES, MAX_CPE_SAMPLES))
    cpe_collisions_f = ti.field(dtype=ti.i32, shape=MAX_OBSTACLES)
    cpe_prob_f = ti.field(dtype=ti.f64, shape=MAX_OBSTACLES)

    # --- TaichiBuffers class ---

    class TaichiBuffers:
        """Manages GPU buffer allocation/deallocation.

        Mirrors C++ cudaMalloc/cudaFree pattern. Taichi fields are
        automatically managed, but this class provides a clean API
        for resetting and tracking buffer state.
        """

        def __init__(self, n_obstacles: int = 0, n_candidates: int = 0,
                     n_traj_steps: int = 0):
            self.n_obstacles = n_obstacles
            self.n_candidates = n_candidates
            self.n_traj_steps = n_traj_steps

        def reset(self, n_obstacles: int = 0, n_candidates: int = 0,
                  n_traj_steps: int = 0) -> None:
            """Reset buffer dimensions."""
            self.n_obstacles = n_obstacles or self.n_obstacles
            self.n_candidates = n_candidates or self.n_candidates
            self.n_traj_steps = n_traj_steps or self.n_traj_steps

        def clear(self) -> None:
            """Deactivate all dynamic fields and reset counts."""
            for f in [
                obs_x_f, obs_y_f, obs_chi_f, obs_U_f,
                obs_length_f, obs_beam_f, obs_colregs_f, obs_dsafe_f,
                wp_x_f, wp_y_f,
                ship_x_f, ship_y_f, ship_chi_f, ship_U_f,
                cand_chi_f, cand_U_f,
                traj_x_f, traj_y_f, traj_chi_f, traj_U_f,
                cost_total_f, cost_path_f, cost_collision_f, cost_colregs_f,
                cpe_samples_x_f, cpe_samples_y_f, cpe_weights_f,
                cpe_collisions_f, cpe_prob_f,
            ]:
                if hasattr(f, 'activate'):
                    try:
                        f.deactivate_all()
                    except Exception:
                        pass

        @property
        def is_gpu(self) -> bool:
            """Check if running on GPU."""
            return TI_GPU_AVAILABLE


# ============================================================================
# Python dataclasses (CPU / host-side API compatibility)
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
