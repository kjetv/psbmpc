# Plan: Full Taichi GPU Acceleration for Python PSB-MPC

## Status: Planning Phase
- Created: 2026-06-18
- Target: Convert 100% CPU Python PSB-MPC to full Taichi GPU acceleration

---

## Current State Analysis

### C++/CUDA Implementation
- **Full GPU acceleration** with Thrust functors
- CUDA kernels for: CPE (CE + MCSKF4D), ship dynamics, cost functions, COLREGS
- Parallelism: `thrust::transform` over candidate bearings × obstacles
- Pre-allocated device buffers via `cudaMalloc`
- Custom `TML::PDMatrix` pinned dynamic matrices for device access

### Python/Taichi Implementation (Before Plan)
- **100% CPU** — zero `@ti.kernel` in core package
- All computation is pure Python loops with `math` module and NumPy
- Taichi dataclasses (`ShipState4TI`, `ObstacleDataTI`, etc.) defined in `types.py` but **never instantiated or used at runtime**
- `ti.init(arch=ti.cpu)` — initialized on CPU, not GPU
- `examples/` directory is empty
- Benchmarks exist but have no captured timing results

---

## Computation Graph (What Needs GPU Conversion)

```
Input: ShipState4, ObstacleData[], Waypoint[]
│
├── Kinematic_Ship.predict_trajectory()          → O(T/dt) sequential Euler/RK1 integration
├── CPE.ce_estimate()                            → 50 iterations × 1000 samples (importance sampling)
├── CPE.mcskf4d_estimate()                       → 300 steps × 1000 particles (Monte Carlo KF)
├── Path_Grounding_Cost.calculate_path_cost()    → O(T × n_waypoints) cross-track error
├── Path_Grounding_Cost.calculate_grounding_cost() → O(T × n_hazards × n_vertices²) polygon distance
├── Dynamic_Obstacle_Cost.calculate_dynamic_obstacle_cost() → O(T) × n_obstacles
├── COLREGS_Evaluator.detect_situation()         → O(1) per obstacle
│
└── PSBMPC_Solver.calculate_optimal_offsets()    → Grid search: n_cbs candidates → predict → cost → argmin
```

---

## Parallelism Opportunities (from C++/CUDA Design)

| Level | Parallelism | C++ Pattern | Taichi Equivalent |
|-------|------------|-------------|-------------------|
| Candidate bearings (n_cbs=5) | Independent | `thrust::transform` | `@ti.kernel` with `ti.ndrange(n_cbs)` |
| Monte Carlo samples (n=1000) | Independent | `curand_kernel` per thread | `ti.grouped(ti.ndrange(n_samples))` |
| CPE iterations (50) | Sequential (CE) | Loop on GPU | `@ti.kernel` inside Python loop |
| Obstacles | Independent | Separate threads per obstacle | Loop over obstacles in kernel |
| Polygon vertex pairs | Independent | Nested loops | `ti.grouped(ti.ndrange(n_v1, n_v2))` |

---

## Implementation Steps

### Phase 1: Foundation — Taichi GPU Initialization & Data Layout
**File:** `psbmpc_taichi/psbmpc_taichi/types.py`

1. Change `ti.init(arch=ti.cpu)` to `ti.init(arch=ti.cuda, device_memory_size=1e9)` (or let Taichi auto-detect)
2. Convert Python `@dataclass` types to use `ti.field` for batched GPU data:
   - Create `ti.field` layouts for batched ship states, obstacles, waypoints, and results
   - Keep Python `@dataclass` as a thin wrapper for host-side API compatibility
   - Define `ti.types.struct` equivalents for `ShipState4TI`, `ObstacleDataTI`, `WaypointTI`, `CPEResultTI`, `MPCResultTI`
3. Add a `TaichiBuffers` class that allocates/deallocates all `ti.field` memory (mirrors C++ `cudaMalloc`/`cudaFree` pattern)

### Phase 2: Utility Functions to `@ti.func`
**File:** `psbmpc_taichi/psbmpc_taichi/utils.py`

Convert all pure math geometry functions to `@ti.func`:
- `normalize_angle`, `angle_diff`, `distance_2d`, `bearing_2d`, `cross_2d`, `dot_2d`
- `point_to_segment_distance` — critical for grounding cost
- `point_in_polygon` — ray casting (note: `point_in_polygon` has a `for` loop over polygon vertices; keep as `@ti.func` with bounded loop)
- `ship_polygon` — generate 4-corner rectangle (returns fixed-size array, use `ti.Vector`)
- `polygon_distance` — compose `ship_polygon` + `point_to_segment_distance`
- `line_segment_intersection`

**Taichi PRNG:** Replace `Xoshiro256pp` with Taichi's built-in `ti.math` or a `curand`-equivalent kernel. Taichi 1.7+ supports `ti.random()` and `ti.math.snorm` for Gaussian sampling. Alternatively, implement a simple LCG/Xoshiro as `@ti.func` seeded per-thread.

### Phase 3: Ship Dynamics Kernel
**File:** `psbmpc_taichi/psbmpc_taichi/ship_models.py`

1. Convert `Kinematic_Ship.predict_trajectory()` to a `@ti.kernel`:
   - Input: `ti.ext_arr()` for initial state, waypoints, offsets
   - Output: `ti.ext_arr()` for trajectory arrays (traj_x, traj_y, traj_chi, traj_U)
   - The entire `for i in range(n_steps)` loop becomes a Taichi parallel loop (or sequential if state-dependent — Euler integration is inherently sequential per step, but **parallel across candidate bearings**)
2. Add `predict_trajectory_batch()` kernel that predicts **all n_cbs candidate bearings in parallel** (mirrors C++ `CB_Cost_Functor_1` thrust transform)
3. Keep Python wrapper `Kinematic_Ship.predict_trajectory()` that allocates fields, calls kernel, returns NumPy arrays

### Phase 4: CPE Kernels — The Big Win
**File:** `psbmpc_taichi/psbmpc_taichi/cpe.py`

This is the **highest compute-density** part of the code. Currently runs 50 iterations × 1000 samples = 50,000 iterations of importance sampling per obstacle, all in Python loops.

4a. **CE (Cross-Entropy) Kernel:**
   - `@ti.kernel` for the inner loop: generate `n_samples` Gaussian samples in parallel (one thread per sample), check collision, compute weights
   - Use `ti.math.normal()` or Box-Muller transform via `@ti.func` for Gaussian sampling (mirrors C++ `curand_normal()` + Cholesky)
   - The outer iteration loop (50 iters) remains in Python but calls the kernel each time
   - Collision check: `@ti.func` for circular/rectangular ship overlap
   - Importance weight computation: vectorized over all samples in one kernel launch

4b. **MCSKF4D Kernel:**
   - `@ti.kernel` for the Monte Carlo particle filter: `n_particles` threads, each tracking one particle
   - Kalman filter update runs on host (or as a separate small kernel) since it's O(1) per step
   - Mirror C++ `determine_sample_validity_4D()` — quadratic root finding for CPA crossing

### Phase 5: Cost Function Kernels
**File:** `psbmpc_taichi/psbmpc_taichi/cost.py`

5a. **Path & Grounding Cost Kernel:**
   - `@ti.kernel` for `calculate_path_cost()`: parallel over timesteps, each thread computes cross-track error for one timestep
   - `@ti.kernel` for `calculate_grounding_cost()`: parallel over (timestep, hazard) pairs
   - Use `polygon_distance()` as `@ti.func` (it's O(n_vertices²) but bounded by small constants ~8-20 vertices)

5b. **Dynamic Obstacle & COLREGS Cost Kernels:**
   - `@ti.kernel` for `calculate_dynamic_obstacle_cost()`: parallel over obstacles
   - `COLREGS_Evaluator.detect_situation()` → `@ti.func` (already O(1))
   - `calculate_colregs_cost()` → `@ti.kernel` parallel over obstacles

### Phase 6: MPC Solver Integration
**Files:** `psbmpc_taichi/psbmpc_taichi/mpc.py`, `psbmpc_taichi/psbmpc_taichi/__init__.py`

6. Refactor `PSBMPC_Solver.calculate_optimal_offsets()` to use the GPU pipeline:
   - Allocate `TaichiBuffers` once in `__init__`
   - `calculate_optimal_offsets()` becomes a Python orchestration function that:
     1. Copies host data → `ti.ext_arr` / `ti.field`
     2. Launches `predict_trajectory_batch()` kernel (all n_cbs candidates in parallel)
     3. Launches cost kernels in parallel for all candidates
     4. Launches `ti.atomic_min` or reduces to find argmin
     5. Copies results back to host
7. Add `calculate_optimal_offsets_gpu()` as the primary method, keep CPU version as fallback
8. Update `__init__.py` exports

### Phase 7: Benchmarking & Verification
**Files:** `benchmarks/benchmark_vs_cpp.py`, `psbmpc_taichi/tests/`

9. Update `benchmark_vs_cpp.py` to add GPU timing measurements:
   - Compare: Python-CPU vs Python-Taichi-GPU vs C++-CPU (already exists)
   - Add scaling tests: n_obstacles, n_samples, n_cbs, prediction horizon
10. Add GPU-specific tests:
    - `test_cpe_gpu.py`: Verify CE and MCSKF4D match CPU within float32 tolerance (rtol=1e-3)
    - `test_dynamics_gpu.py`: Verify trajectory prediction matches
    - `test_cost_gpu.py`: Verify cost components match
    - `test_mpc_gpu.py`: Full solver accuracy test

---

## Relevant Files Summary

| File | What to Modify |
|------|---------------|
| `psbmpc_taichi/psbmpc_taichi/types.py` | `ti.init(arch=ti.cuda)`, add `ti.field` layouts, `TaichiBuffers` class |
| `psbmpc_taichi/psbmpc_taichi/utils.py` | Convert all geometry functions to `@ti.func` |
| `psbmpc_taichi/psbmpc_taichi/ship_models.py` | Add `@ti.kernel` for batched trajectory prediction |
| `psbmpc_taichi/psbmpc_taichi/cpe.py` | Add `@ti.kernel` for CE importance sampling + MCSKF4D particle filter |
| `psbmpc_taichi/psbmpc_taichi/cost.py` | Add `@ti.kernel` for path, grounding, collision, COLREGS costs |
| `psbmpc_taichi/psbmpc_taichi/mpc.py` | Refactor solver to orchestrate GPU kernels |
| `psbmpc_taichi/psbmpc_taichi/__init__.py` | Update exports |
| `benchmarks/benchmark_vs_cpp.py` | Add GPU timing comparisons |
| `psbmpc_taichi/tests/test_cpe_gpu.py` | New: GPU CPE accuracy tests |
| `psbmpc_taichi/tests/test_mpc_gpu.py` | New: Full solver GPU accuracy test |

---

## Verification

1. **Accuracy**: Run `pytest psbmpc_taichi/tests/` — all existing tests pass, new GPU tests match CPU within `rtol=1e-3`
2. **GPU kernel launch**: Verify `ti.kernel` functions compile without errors (`ti.init(arch=ti.cuda)` succeeds)
3. **Benchmark**: Run `python benchmarks/benchmark_vs_cpp.py` — verify Python-GPU is ≥10× faster than Python-CPU for large obstacle counts (n_obstacles ≥ 10, n_samples ≥ 1000)
4. **Memory**: Verify no `ti.field` leaks by checking `ti.field` allocation/deallocation counts

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use `ti.ext_arr()` for NumPy interop | Minimal data copying; mirrors C++ device pointer pattern |
| Keep Python loop for CPE iterations | CE method requires sequential distribution updates; only inner sample loop parallelizes |
| Batch all n_cbs candidates in one kernel launch | Matches C++ `thrust::transform` pattern; avoids Python loop overhead |
| Use `float32` for all GPU computation | Matches C++ CUDA code; sufficient for MPC decision quality |
| Fallback to CPU if no GPU available | Check `ti.cuda` availability; gracefully degrade |
| Bounded loops in `@ti.func` | Taichi requires statically unrollable loops; polygon vertex counts are small constants |

---

## Architecture: Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Host (Python/NumPy)                         │
│                                                                 │
│  ShipState4 ──cudaHostAlloc──► ti.ext_arr ──► ti.cuda graph    │
│  ObstacleData[] ──► ti.field (dense) ──────────────────────────┤
│  Waypoint[] ──► ti.field (dense) ──────────────────────────────┤
│                                                                 │
│                              ◄── ti.ext_arr ──► NumPy arrays   │
│                              ◄── copied back to host           │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ CUDA
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Device (Taichi GPU)                          │
│                                                                 │
│  @ti.kernel predict_trajectory_batch()                         │
│  @ti.kernel ce_estimate_kernel()                               │
│  @ti.kernel mcskf4d_kernel()                                   │
│  @ti.kernel path_grounding_cost()                              │
│  @ti.kernel collision_cost()                                   │
│  @ti.kernel colregs_cost()                                     │
│                                                                 │
│  @ti.func normalize_angle, distance_2d, point_to_segment...    │
│  @ti.func ship_polygon, polygon_distance, check_collision...   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Further Considerations

1. **Kinetic ship model**: Currently low-priority in Python. If needed, apply same GPU pattern to `Kinetic_Ship` (Murray 3-DOF integration).
2. **Dynamic obstacle count**: If n_obstacles varies widely per MPC step, use `ti.root.dynamic()` SNodes instead of fixed-size fields to avoid over-allocation.
3. **COLREGS `point_in_polygon`**: The ray-casting loop is O(n_vertices) and not trivially vectorizable. Consider approximating with circular collision checks on GPU if this becomes a bottleneck (C++ CUDA uses polygon distance with bbox pre-filter).
4. **Memory management**: Taichi fields are automatically managed by Taichi's memory system; no explicit `cudaFree` needed. Use `ti.root.deactivate()` for dynamic arrays.
5. **Error handling**: Wrap kernel launches in try/except; log detailed Taichi error messages for debugging.
6. **Profiling**: Use `ti.kernel_profile_enabled(True)` and `ti.kernel_profiler_print()` to identify bottlenecks.
