# Plan: PSB-MPC Taichi Port (Python)

**TL;DR**: Port the GPU-accelerated PSB-MPC (Probabilistic/Safety-Bounded MPC) C++/CUDA library to Python with Taichi for cross-platform GPU execution. Focus on the 4 most GPU-intensive kernels: trajectory prediction, collision probability estimation (CPE), cost evaluation, and COLREGS violation checking.

**Goal**: Create a maintainable Python/Taichi implementation that matches the C++/CUDA numerical results while being more accessible and cross-platform.

---

## Architecture Overview

### Current C++/CUDA Structure
- **Core MPC**: `psbmpc_gpu.cu` — orchestrates the full MPC loop
- **Trajectory Prediction**: `kinematic_ship_models_gpu.cu` — kinematic/kinetic ship models
- **Collision Probability**: `cpe_gpu.cu` — CE and MCSKF4D methods
- **Cost Evaluation**: `cb_cost_functor.cu` — path, grounding, dynamic obstacle, COLREGS costs
- **Cost Calculation**: `mpc_cost_gpu.cu` — distance to polygon, collision cost functions
- **Matrix Library**: `tml/` — TML (Template Matrix Library) with PDMatrix for GPU

### Target Taichi Structure
```
psbmpc_taichi/
├── psbmpc_taichi/                  # Python package
│   ├── __init__.py
│   ├── types.py                    # Data classes (replacing TML structs)
│   ├── ship_models.py              # Kinematic/kinetic ship prediction
│   ├── cpe.py                      # Collision probability estimation
│   ├── cost.py                     # Cost functions (path, grounding, COLREGS)
│   ├── mpc.py                      # Main MPC solver (orchestration)
│   └── utils.py                    # Utilities (UTM, geometry, PRNG)
├── tests/                          # Test suite
│   ├── test_ship_models.py
│   ├── test_cpe.py
│   ├── test_cost.py
│   ├── test_mpc.py
│   └── test_comparison.py          # C++ vs Taichi comparison
├── examples/                       # Usage examples
│   ├── basic_mpc.py
│   ├── colregs_scenarios.py
│   └── grounding_hazards.py
├── benchmarks/                     # Performance benchmarks
│   ├── benchmark_cpu_vs_gpu.py
│   └── benchmark_vs_cpp.py
├── conftest.py                     # pytest fixtures
├── pyproject.toml                  # Package config
└── README.md
```

---

## Phase 1: Foundation & Types (Week 1)

### 1.1 Package Setup
- Create `psbmpc_taichi/` package structure with `pyproject.toml`
- Dependencies: `taichi >= 1.7.0`, `numpy`, `pytest`, `scipy`
- Configure Taichi for CUDA backend (with CPU fallback for testing)
- Set up CI with pytest

### 1.2 Type Definitions (`types.py`)
Replace C++ structs with Python dataclasses/Taichi `ti.dataclass`:

```python
@ti.dataclass
class PSBMPC_Parameters:
    n_M: int
    n_cbs: int
    T: float
    dt: float
    d_safe: float
    kappa_GW: float
    kappa_SO: float
    kappa_RA: float
    kappa_GN: float
    # ... all parameters from psbmpc_parameters.hpp
```

Key types to define:
- `PSBMPC_Parameters` / `SBMPC_Parameters`
- `ShipState4` (x, y, chi, U) and `ShipState6` (x, y, psi, u, v, r)
- `ObstacleData` (position, covariance, dimensions, COLREGS role)
- `CPE_Result` (probability, convergence status)
- `MPC_Result` (optimal offsets, predicted trajectory)

### 1.3 Utility Functions (`utils.py`)
- **PRNG**: Replace xoshiro256++ with Taichi-compatible PRNG or NumPy
- **Geometry**: Distance to line segment, point-in-polygon, line intersection
- **UTM Projection**: Use `pyproj` for coordinate transformations
- **Matrix ops**: Use `numpy` for CPU, `taichi_lang` for GPU kernels

---

## Phase 2: Ship Models & Trajectory Prediction (Week 2)

### 2.1 Kinematic Ship Model (`ship_models.py`)
Port `Kinematic_Ship` class:
- State: `[x, y, chi, U]` (position, heading, surge speed)
- Guidance: LOS waypoint following with cross-track error
- Prediction: Linear or ERK1 (Runge-Kutta) integration
- Key methods:
  - `predict(xs, U_d, chi_d, dt, method)` — single step
  - `predict_trajectory(xs, offsets, times, waypoints, T, dt)` — full horizon
  - `update_guidance_references(u_d, chi_d, waypoints, xs, dt, method)`

### 2.2 Kinetic Ship Model (3DOF) (`ship_models.py`)
Port `Kinetic_Ship` class (optional, lower priority):
- State: `[x, y, psi, u, v, r]`
- Includes: Coriolis forces, damping, inertia matrix
- Controller: Force inputs for surge, sway, yaw

### 2.3 Obstacle Ship Model
- Simple LOS follower for obstacle prediction
- State: `[x, y, chi, U]` with configurable parameters

**Verification**: Compare trajectories against C++ `test_ownship.cpp` results (max error < 1e-6)

---

## Phase 3: Collision Probability Estimation (Week 3)

### 3.1 CE Method (Cross-Entropy) (`cpe.py`)
Port `CPE::CE_estimate()`:
- Generate samples from normal distribution (2D)
- Mark samples inside collision region
- Update importance sampling distribution (mean, covariance)
- Iterate until convergence (max 50 iterations)
- Key Taichi kernel: `generate_norm_dist_samples()` using `ti.math.normal`

### 3.2 MCSKF4D Method (Monte Carlo Sequential KF) (`cpe.py`)
Port `CPE::MCSKF4D_estimate()`:
- 4D state: relative position + velocity
- Sequential Kalman filtering with collision probability measurement
- Key kernel: `determine_sample_validity_4D()`

### 3.3 Verification
- Compare against `test_cpe_cpu.cpp` and `test_cpe_gpu.cu` results
- Validate collision probabilities match within 1%
- Test convergence behavior

---

## Phase 4: Cost Functions (Week 4)

### 4.1 Path & Grounding Cost (`cost.py`)
Port `MPC_Cost::calculate_grounding_cost()`:
- Distance from ship polygon to static obstacle polygons
- Point-in-polygon test (ray casting)
- Line segment intersection
- Control deviation cost (penalize deviation from last optimal)

### 4.2 Dynamic Obstacle Cost (`cost.py`)
Port `MPC_Cost::calculate_dynamic_obstacle_cost()`:
- Collision probability × cost weight
- Distance-based hazard function
- Time-to-CPA (closest point of approach) calculation

### 4.3 COLREGS Violation Cost (`cost.py`)
Port `COLREGS_Violation_Evaluator`:
- Detect situations: head-on, crossing (port/starboard), overtaking
- Evaluate giving-way (GW) and stand-on (SO) violations
- Readily apparent violation detection
- Cost based on situation type and timing

### 4.4 Cost Functor (Parallel Evaluation)
- Taichi `@ti.kernel` for parallel CB × obstacle evaluation
- Similar to Thrust `transform` pattern but Taichi-native

**Verification**: Compare against C++ cost calculations (max error < 1e-4)

---

## Phase 5: Main MPC Solver (Week 5)

### 5.1 MPC Orchestration (`mpc.py`)
Port `GPU::PSBMPC::calculate_optimal_offsets()`:
1. Predict nominal trajectory with default offsets (1.0, 0.0)
2. Determine if COLAV (collision avoidance) is active
3. Setup data for all obstacles
4. **Cost Functor 1**: Predict trajectory for each CB, calculate path/grounding costs
5. **Cost Functor 2**: Evaluate dynamic obstacle + COLREGS costs for all (CB, obstacle, scenario)
6. Aggregate costs, find optimal CB
7. Return optimal surge/course offsets and predicted trajectory

### 5.2 Memory Management
- Use Taichi `ti.ndarray` for fixed-size arrays (replaces PDMatrix)
- Pre-allocate all working buffers
- Efficient data transfer between host and device

### 5.3 Configuration & Parameters
- Load parameters from YAML or Python dicts
- Support both PSBMPC and SBMPC modes
- Configurable CPE method (CE vs MCSKF4D)

**Verification**: End-to-end test matching C++ `run_psbmpc.cpp` output

---

## Phase 6: Tests & Benchmarks (Week 6)

### 6.1 Unit Tests
- `test_ship_models.py`: Trajectory prediction accuracy
- `test_cpe.py`: Collision probability estimation
- `test_cost.py`: Cost function values
- `test_mpc.py`: Full MPC solver

### 6.2 Comparison Tests
- `test_comparison.py`: Run same scenario in C++ and Taichi, compare outputs
- Numerical tolerance: max 0.1% relative error
- Focus on: optimal offsets, collision probabilities, total cost

### 6.3 Benchmarks
- `benchmark_cpu_vs_gpu.py`: CPU (NumPy) vs GPU (Taichi CUDA) speedup
- `benchmark_vs_cpp.py`: Taichi vs C++/CUDA performance
- Expected: 10-50x speedup over pure NumPy, ~50-80% of C++/CUDA performance

---

## Key Design Decisions

### Decision 1: Taichi `@ti.dataclass` vs `ti.Struct`
**Recommendation**: Use `@ti.dataclass` (Taichi 1.7+). More Pythonic, supports inheritance, and compiles to both CPU and GPU.

### Decision 2: Matrix Representation
**Recommendation**: Use `ti.ndarray` for dynamic-sized arrays and `ti.Matrix` for fixed-size (2x2, 3x3, 4x4). Avoid custom matrix class — use Taichi's built-in types.

### Decision 3: PRNG Strategy
**Recommendation**: Use NumPy for host-side PRNG (sampling, initialization) and Taichi's `ti.random()` or custom LCG for device-side PRNG. Avoid `curand` dependency.

### Decision 4: Parallelism Model
**Recommendation**: Use `@ti.kernel` with `ti.grouped` for embarrassingly parallel operations (CB × obstacle evaluation). Use `@ti.func` for reusable device-side functions.

### Decision 5: Shapefile Support
**Decision**: Defer shapefile reading to a later phase. Use CSV/JSON for static obstacle polygons in initial implementation. Can add `pyshp` or `fiona` dependency later.

### Decision 6: COLREGS Complexity
**Decision**: Start with basic COLREGS detection (head-on, crossing, overtaking). Add detailed violation evaluation (readily apparent, specific crossing scenarios) in a follow-up.

### Decision 7: Development Environment
**Decision**: All implementation, testing, and benchmarking happens **inside the Docker container** (`psbmpc-dev:latest`), not locally. The devcontainer setup mounts the workspace into the container and runs everything there.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Taichi `ti.random()` quality insufficient for CPE | High | Use custom LCG or import xoshiro implementation |
| Taichi CUDA kernel complexity limits | Medium | Start with simple kernels, iterate |
| Numerical differences between C++ float and Python float | Medium | Use `float32` consistently, validate with tolerance |
| Performance gap vs C++/CUDA | Medium | Profile early, optimize hot paths |
| Shapefile dependency complexity | Low | Defer to CSV/JSON initially |

---

## Files to Reference (C++/CUDA)

| Taichi Module | C++ Source Files |
|---------------|------------------|
| `types.py` | `include/psbmpc_parameters.hpp`, `include/psbmpc_defines.hpp` |
| `ship_models.py` | `src/gpu/kinematic_ship_models_gpu.cu`, `include/gpu/kinematic_ship_models_gpu.cuh` |
| `cpe.py` | `src/gpu/cpe_gpu.cu`, `include/gpu/cpe_gpu.cuh` |
| `cost.py` | `src/gpu/cb_cost_functor.cu`, `include/gpu/mpc_cost_gpu.cuh`, `include/gpu/colregs_violation_evaluator.cuh` |
| `mpc.py` | `src/gpu/psbmpc_gpu.cu`, `include/gpu/psbmpc_gpu.cuh` |
| `utils.py` | `src/gpu/utilities_gpu.cuh`, `include/cpu/geometry.hpp` |

---

## Next Steps

1. **Immediate**: Create package structure and `pyproject.toml` **inside the Docker container**
2. **Priority order**: Types → Ship Models → CPE → Cost → MPC → Tests
3. **Validation strategy**: Port one kernel at a time, validate against C++ before moving to next
4. **Iterative refinement**: Start with CPU-only Taichi kernels for correctness, then enable CUDA backend
5. **All development inside container**: Python/Taichi installation, testing, and benchmarking all happen inside the Docker container (`psbmpc-dev:latest`), not locally
