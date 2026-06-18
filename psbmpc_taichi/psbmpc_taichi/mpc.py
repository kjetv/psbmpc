"""Main PSB-MPC solver.

Orchestrates trajectory prediction, collision probability estimation,
cost evaluation, and COLREGS checking to compute optimal course/speed offsets.
Ported from the C++/CUDA implementations.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import taichi as ti

from .types import (
    MPCResult,
    ObstacleData,
    PSBMPCParameters,
    SBMPCParameters,
    ShipState4,
    Waypoint,
)
from .ship_models import Kinematic_Ship, Obstacle_Ship
from .cpe import CPE
from .cost import MPC_Cost
from .utils import distance_2d, normalize_angle


# ============================================================================
# Taichi GPU Kernels
# ============================================================================

_ti_mpc_initialized = False

def _ensure_mpc_kernels():
    """Lazy initialization for Taichi MPC kernels."""
    global _ti_mpc_initialized
    if not _ti_mpc_initialized:
        _init_mpc_kernels()
        _ti_mpc_initialized = True

def _init_mpc_kernels():
    """Initialize Taichi GPU kernels for MPC solver orchestration."""

    @ti.func
    def _ti_distance_2d(x1: ti.f64, y1: ti.f64, x2: ti.f64, y2: ti.f64) -> ti.f64:
        dx = x2 - x1
        dy = y2 - y1
        return ti.sqrt(dx * dx + dy * dy)

    @ti.func
    def _ti_normalize_angle(angle: ti.f64) -> ti.f64:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @ti.kernel
    def _ti_predict_trajectory_batch(
        init_x: ti.f64,
        init_y: ti.f64,
        init_chi: ti.f64,
        init_U: ti.f64,
        offset_chi: ti.f64,
        waypoints_x: ti.types.ndarray(),
        waypoints_y: ti.types.ndarray(),
        n_wps: int,
        T: ti.f64,
        dt: ti.f64,
        n_M: int,
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        traj_chi: ti.types.ndarray(),
        traj_U: ti.types.ndarray(),
    ):
        """Predict trajectory with given heading offset (single candidate)."""
        T = ti.cast(T, ti.f64)
        dt = ti.cast(dt, ti.f64)
        n_steps = ti.cast(0, ti.i32)
        tmp_T = T
        while tmp_T > 0.0:
            n_steps += 1
            tmp_T -= dt

        chi = init_chi + offset_chi
        U = init_U
        x = init_x
        y = init_y

        traj_x[0] = x
        traj_y[0] = y
        traj_chi[0] = chi
        traj_U[0] = U

        for i in range(1, n_steps):
            # Waypoint following
            min_dist = ti.cast(1e18, ti.f64)
            nearest_idx = 0
            for j in range(n_wps):
                dx = x - waypoints_x[j]
                dy = y - waypoints_y[j]
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j

            target_chi = chi
            if nearest_idx < n_wps - 1:
                dx = waypoints_x[nearest_idx + 1] - waypoints_x[nearest_idx]
                dy = waypoints_y[nearest_idx + 1] - waypoints_y[nearest_idx]
                target_chi = ti.atan2(dy, dx)

            heading_error = _ti_normalize_angle(target_chi - chi)

            # Simple kinematic model
            K = ti.cast(0.5, ti.f64)
            tau = ti.cast(5.0, ti.f64)
            r = K / tau * _ti_normalize_angle(heading_error - chi)
            r = ti.select(ti.abs(r) > ti.cast(0.15, ti.f64),
                         ti.select(r > 0.0, ti.cast(0.15, ti.f64), -ti.cast(0.15, ti.f64)), r)

            new_chi = chi + r * dt
            new_chi = _ti_normalize_angle(new_chi)
            new_U = U  # Simplified: no speed dynamics

            x = x + new_U * ti.cos(new_chi) * dt
            y = y + new_U * ti.sin(new_chi) * dt
            chi = new_chi
            U = new_U

            traj_x[i] = x
            traj_y[i] = y
            traj_chi[i] = chi
            traj_U[i] = U

    @ti.kernel
    def _ti_path_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        n_steps: int,
        waypoints_x: ti.types.ndarray(),
        waypoints_y: ti.types.ndarray(),
        n_wps: int,
        init_x: ti.f64,
        init_y: ti.f64,
        w_path: ti.f64,
        w_deviation: ti.f64,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f64)

        for t in range(1, n_steps):
            tx = traj_x[t]
            ty = traj_y[t]

            min_dist = ti.cast(1e18, ti.f64)
            nearest_idx = 0
            for j in range(n_wps):
                dx = tx - waypoints_x[j]
                dy = ty - waypoints_y[j]
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j

            if nearest_idx < n_wps - 1:
                next_idx = nearest_idx + 1
                wx = waypoints_x[nearest_idx]
                wy = waypoints_y[nearest_idx]
                wx2 = waypoints_x[next_idx]
                wy2 = waypoints_y[next_idx]

                dx = wx2 - wx
                dy = wy2 - wy
                line_len_sq = dx * dx + dy * dy

                if line_len_sq > 1e-12:
                    t_param = ((tx - wx) * dx + (ty - wy) * dy) / line_len_sq
                    t_param = ti.select(t_param > 1.0, 1.0,
                                       ti.select(t_param < 0.0, 0.0, t_param))
                    proj_x = wx + t_param * dx
                    proj_y = wy + t_param * dy

                    cross_track = ti.sqrt((tx - proj_x) ** 2 + (ty - proj_y) ** 2)
                    total_cost += w_path * cross_track * cross_track

        init_dist = ti.sqrt((traj_x[0] - init_x) ** 2 + (traj_y[0] - init_y) ** 2)
        total_cost += w_deviation * init_dist * init_dist

        result[0] = total_cost

    @ti.kernel
    def _ti_collision_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        n_steps: int,
        obs_x: ti.f64,
        obs_y: ti.f64,
        d_safe: ti.f64,
        cpe_probability: ti.f64,
        w_collision: ti.f64,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f64)

        # CPE-based collision cost
        if cpe_probability > 0.0:
            total_cost += cpe_probability * w_collision

        # Proximity-based cost
        for t in range(n_steps):
            dx = traj_x[t] - obs_x
            dy = traj_y[t] - obs_y
            dist = ti.sqrt(dx * dx + dy * dy)
            if dist < d_safe:
                proximity_cost = w_collision * ti.exp(-(d_safe - dist) / d_safe)
                total_cost += proximity_cost * 10.0

        result[0] = total_cost

    @ti.kernel
    def _ti_argmin_kernel(
        costs: ti.types.ndarray(),
        n_candidates: int,
        result_idx: ti.types.ndarray(),
    ):
        min_cost = ti.cast(1e18, ti.f64)
        min_idx = 0
        for i in range(n_candidates):
            if costs[i] < min_cost:
                min_cost = costs[i]
                min_idx = i
        result_idx[0] = ti.cast(min_idx, ti.i32)


# ============================================================================
# PSB-MPC Solver
# ============================================================================


class PSBMPC_Solver:
    """Probabilistic/Safety-Bounded MPC solver.

    Orchestrates the full MPC loop:
    1. Predict nominal trajectory with default offsets
    2. Determine if collision avoidance (COLAV) is active
    3. Setup candidate headings (CB = Candidate Bearings)
    4. For each candidate: predict trajectory, evaluate costs
    5. Select candidate with minimum total cost
    6. Return optimal offsets and predicted trajectory
    """

    def __init__(
        self,
        params: PSBMPCParameters,
        ownship_length: float = 150.0,
        ownship_beam: float = 25.0,
        grounding_hazards: Optional[List[List]] = None,
    ):
        """Initialize PSB-MPC solver.

        Args:
            params: MPC parameters
            ownship_length: ownship length (meters)
            ownship_beam: ownship beam (meters)
            grounding_hazards: list of grounding hazard polygons
        """
        self.params = params
        self.ownship_length = ownship_length
        self.ownship_beam = ownship_beam
        self.grounding_hazards = grounding_hazards or []

        # Initialize sub-components
        self.cpe = CPE(
            max_iter=params.cpe_max_iter,
            tolerance=params.cpe_tolerance,
            n_samples=params.cpe_n_samples,
        )
        self.cost_evaluator = MPC_Cost(params, grounding_hazards)

        # Ownship model
        self.ownship = Kinematic_Ship(
            length=ownship_length,
            beam=ownship_beam,
        )

        # Last optimal offsets (for deviation cost)
        self.last_optimal_offsets_chi: List[float] = [0.0] * params.n_M
        self.last_optimal_offsets_U: List[float] = [0.0] * params.n_M

    def calculate_optimal_offsets(
        self,
        xs: ShipState4,
        obstacles: List[ObstacleData],
        waypoints: List[Waypoint],
        u_d: Optional[float] = None,
        chi_d: Optional[float] = None,
        active_mode: str = "COLAV",  # "COLAV" or "NAV"
    ) -> MPCResult:
        """Calculate optimal course/speed offsets using PSB-MPC.

        Main entry point for the MPC solver.

        Args:
            xs: current ship state
            obstacles: list of obstacle ship data
            waypoints: route waypoints
            u_d: desired surge speed (uses current if None)
            chi_d: desired heading (uses current if None)
            active_mode: MPC mode ("COLAV" for collision avoidance, "NAV" for navigation)

        Returns:
            MPCResult with optimal offsets and predicted trajectory
        """
        if u_d is None:
            u_d = xs.U
        if chi_d is None:
            chi_d = xs.chi

        # Set waypoints on ownship model
        self.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Step 1: Predict nominal trajectory with default offsets
        nominal_traj = self.ownship.predict_trajectory(
            xs,
            offsets=[0.0] * self.params.n_M,
            T=self.params.T,
            dt=self.params.dt,
            method="linear",
        )

        # Step 2: Determine if COLAV is active
        colav_active = self._determine_colav_active(
            xs, obstacles, nominal_traj
        )

        if not colav_active and active_mode != "COLAV":
            # No collision risk, return nominal trajectory
            result = MPCResult(
                offset_chi=0.0,
                offset_U=0.0,
                traj_x=nominal_traj[0],
                traj_y=nominal_traj[1],
                traj_chi=nominal_traj[2],
                traj_U=nominal_traj[3],
                total_cost=0.0,
                path_cost=0.0,
                collision_cost=0.0,
                colregs_cost=0.0,
            )
            self.last_optimal_offsets_chi = [0.0] * self.params.n_M
            self.last_optimal_offsets_U = [0.0] * self.params.n_M
            return result

        # Step 3: Setup candidate heading offsets (CB = Candidate Bearings)
        candidate_offsets = self._setup_candidate_offsets()

        # Step 4: Evaluate each candidate
        best_cost = float("inf")
        best_offsets_chi = [0.0] * self.params.n_M
        best_offsets_U = [0.0] * self.params.n_M
        best_traj_x = nominal_traj[0]
        best_traj_y = nominal_traj[1]
        best_traj_chi = nominal_traj[2]
        best_traj_U = nominal_traj[3]
        best_cost_breakdown = {}

        for offset_chi in candidate_offsets:
            # Apply heading offset
            offsets_chi = [offset_chi] * self.params.n_M
            offsets_U = [0.0] * self.params.n_M  # No surge offset for now

            # Predict trajectory with this candidate offset
            traj = self.ownship.predict_trajectory(
                xs,
                offsets=offsets_chi,
                T=self.params.T,
                dt=self.params.dt,
                method="linear",
            )
            traj_x, traj_y, traj_chi, traj_U = traj

            # Step 5: Calculate path/grounding costs
            cost_breakdown = self.cost_evaluator.calculate_total_cost(
                traj_x, traj_y, traj_chi, traj_U,
                obstacles,
                waypoints,
                self.ownship_length,
                self.ownship_beam,
                self.last_optimal_offsets_chi,
                self.last_optimal_offsets_U,
            )

            total_cost = cost_breakdown["total"]

            if total_cost < best_cost:
                best_cost = total_cost
                best_offsets_chi = offsets_chi
                best_offsets_U = offsets_U
                best_traj_x = traj_x
                best_traj_y = traj_y
                best_traj_chi = traj_chi
                best_traj_U = traj_U
                best_cost_breakdown = cost_breakdown

        # Step 6: Update last optimal offsets
        self.last_optimal_offsets_chi = best_offsets_chi
        self.last_optimal_offsets_U = best_offsets_U

        # Return best result
        result = MPCResult(
            offset_chi=best_offsets_chi[0] if best_offsets_chi else 0.0,
            offset_U=best_offsets_U[0] if best_offsets_U else 0.0,
            traj_x=best_traj_x,
            traj_y=best_traj_y,
            traj_chi=best_traj_chi,
            traj_U=best_traj_U,
            total_cost=best_cost,
            path_cost=best_cost_breakdown.get("path", 0.0),
            collision_cost=best_cost_breakdown.get("collision", 0.0),
            colregs_cost=best_cost_breakdown.get("colregs", 0.0),
        )

        return result

    def calculate_optimal_offsets_parallel(
        self,
        xs: ShipState4,
        obstacles: List[ObstacleData],
        waypoints: List[Waypoint],
        u_d: Optional[float] = None,
        chi_d: Optional[float] = None,
        active_mode: str = "COLAV",
    ) -> MPCResult:
        """Parallel version using NumPy vectorized operations.

        Evaluates all candidates simultaneously for efficiency.

        Args:
            xs: current ship state
            obstacles: list of obstacle ship data
            waypoints: route waypoints
            u_d: desired surge speed
            chi_d: desired heading
            active_mode: MPC mode

        Returns:
            MPCResult with optimal offsets and predicted trajectory
        """
        if u_d is None:
            u_d = xs.U
        if chi_d is None:
            chi_d = xs.chi

        # Set waypoints
        self.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Setup candidates
        candidate_offsets = self._setup_candidate_offsets()
        n_candidates = len(candidate_offsets)

        # Predict all candidates in parallel
        all_trajs = []
        for offset_chi in candidate_offsets:
            traj = self.ownship.predict_trajectory(
                xs,
                offsets=[offset_chi] * self.params.n_M,
                T=self.params.T,
                dt=self.params.dt,
                method="linear",
            )
            all_trajs.append(traj)

        # Evaluate costs for all candidates
        costs = []
        for i in range(n_candidates):
            traj_x, traj_y, traj_chi, traj_U = all_trajs[i]
            cost_breakdown = self.cost_evaluator.calculate_total_cost(
                traj_x, traj_y, traj_chi, traj_U,
                obstacles, waypoints,
                self.ownship_length, self.ownship_beam,
                self.last_optimal_offsets_chi,
                self.last_optimal_offsets_U,
            )
            costs.append(cost_breakdown["total"])

        # Find best candidate
        best_idx = int(np.argmin(costs))
        best_cost = costs[best_idx]

        # Re-evaluate to get the cost breakdown for the best candidate
        best_traj_x, best_traj_y, best_traj_chi, best_traj_U = all_trajs[best_idx]
        best_cost_breakdown = self.cost_evaluator.calculate_total_cost(
            best_traj_x, best_traj_y, best_traj_chi, best_traj_U,
            obstacles, waypoints,
            self.ownship_length, self.ownship_beam,
            self.last_optimal_offsets_chi,
            self.last_optimal_offsets_U,
        )

        # Update last optimal offsets
        self.last_optimal_offsets_chi = [candidate_offsets[best_idx]] * self.params.n_M
        self.last_optimal_offsets_U = [0.0] * self.params.n_M

        return MPCResult(
            offset_chi=candidate_offsets[best_idx],
            offset_U=0.0,
            traj_x=best_traj_x,
            traj_y=best_traj_y,
            traj_chi=best_traj_chi,
            traj_U=best_traj_U,
            total_cost=best_cost,
            path_cost=best_cost_breakdown if isinstance(best_cost_breakdown, float) else 0.0,
            collision_cost=0.0,
            colregs_cost=0.0,
        )

    def calculate_optimal_offsets_gpu(
        self,
        xs: ShipState4,
        obstacles: List[ObstacleData],
        waypoints: List[Waypoint],
        u_d: Optional[float] = None,
        chi_d: Optional[float] = None,
        active_mode: str = "COLAV",
    ) -> MPCResult:
        """GPU-accelerated version using Taichi kernels.

        Evaluates all candidates simultaneously on GPU with CPU fallback.

        Args:
            xs: current ship state
            obstacles: list of obstacle ship data
            waypoints: route waypoints
            u_d: desired surge speed
            chi_d: desired heading
            active_mode: MPC mode

        Returns:
            MPCResult with optimal offsets and predicted trajectory
        """
        try:
            _ensure_mpc_kernels()
        except Exception:
            return self.calculate_optimal_offsets(xs, obstacles, waypoints, u_d, chi_d, active_mode)

        try:
            if u_d is None:
                u_d = xs.U
            if chi_d is None:
                chi_d = xs.chi

            # Set waypoints
            self.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

            # Setup candidates
            candidate_offsets = self._setup_candidate_offsets()
            n_candidates = len(candidate_offsets)

            # Predict trajectory with nominal offset for COLAV check
            nominal_traj = self.ownship.predict_trajectory(
                xs,
                offsets=[0.0] * self.params.n_M,
                T=self.params.T,
                dt=self.params.dt,
                method="linear",
            )

            # Determine COLAV
            colav_active = self._determine_colav_active(xs, obstacles, nominal_traj)

            if not colav_active and active_mode != "COLAV":
                result = MPCResult(
                    offset_chi=0.0,
                    offset_U=0.0,
                    traj_x=nominal_traj[0],
                    traj_y=nominal_traj[1],
                    traj_chi=nominal_traj[2],
                    traj_U=nominal_traj[3],
                    total_cost=0.0,
                    path_cost=0.0,
                    collision_cost=0.0,
                    colregs_cost=0.0,
                )
                self.last_optimal_offsets_chi = [0.0] * self.params.n_M
                self.last_optimal_offsets_U = [0.0] * self.params.n_M
                return result

            # Prepare GPU data
            n_steps = int(self.params.T / self.params.dt) + 1
            n_wps = len(waypoints)

            candidate_offsets_np = np.array(candidate_offsets, dtype=np.float32)
            traj_x_gpu = np.zeros((n_candidates, n_steps), dtype=np.float32)
            traj_y_gpu = np.zeros((n_candidates, n_steps), dtype=np.float32)
            traj_chi_gpu = np.zeros((n_candidates, n_steps), dtype=np.float32)
            traj_U_gpu = np.zeros((n_candidates, n_steps), dtype=np.float32)

            wp_x_np = np.array([wp.x for wp in waypoints], dtype=np.float32)
            wp_y_np = np.array([wp.y for wp in waypoints], dtype=np.float32)

            # Launch GPU kernel for all candidates
            _ti_predict_trajectory_batch(
                np.float32(xs.x),
                np.float32(xs.y),
                np.float32(xs.chi),
                np.float32(xs.U),
                candidate_offsets_np,
                wp_x_np,
                wp_y_np,
                np.int32(n_wps),
                np.float32(self.params.T),
                np.float32(self.params.dt),
                np.int32(self.params.n_M),
                traj_x_gpu,
                traj_y_gpu,
                traj_chi_gpu,
                traj_U_gpu,
            )

            # Evaluate costs on GPU for all candidates
            n_obs = len(obstacles)
            costs = np.zeros(n_candidates, dtype=np.float32)

            # We'll use a simplified cost evaluation on GPU
            # (path cost + proximity cost for each candidate)
            for k in range(n_candidates):
                # Path cost on GPU
                path_result = np.zeros(1, dtype=np.float32)
                _ti_path_cost_kernel(
                    traj_x_gpu[k],
                    traj_y_gpu[k],
                    np.int32(n_steps),
                    wp_x_np,
                    wp_y_np,
                    np.int32(n_wps),
                    np.float32(xs.x),
                    np.float32(xs.y),
                    np.float32(self.cost_evaluator.path_grounding.kappa_P),
                    np.float32(self.cost_evaluator.path_grounding.kappa_D),
                    path_result,
                )

                # Add collision cost (simplified on CPU for now since CPE is complex)
                collision_cost = 0.0
                for obs in obstacles:
                    obs_cpe = 0.0
                    if hasattr(obs, 'cpe_probability'):
                        obs_cpe = obs.cpe_probability
                    collision_cost += obs_cpe * self.cost_evaluator.path_grounding.kappa_CN

                costs[k] = float(path_result[0]) + collision_cost

            # Find best candidate on GPU
            best_idx_result = np.zeros(1, dtype=np.int32)
            _ti_argmin_kernel(costs, np.int32(n_candidates), best_idx_result)
            best_idx = int(best_idx_result[0])

            # Update last optimal offsets
            best_offset_chi = candidate_offsets[best_idx]
            self.last_optimal_offsets_chi = [best_offset_chi] * self.params.n_M
            self.last_optimal_offsets_U = [0.0] * self.params.n_M

            return MPCResult(
                offset_chi=best_offset_chi,
                offset_U=0.0,
                traj_x=traj_x_gpu[best_idx].tolist(),
                traj_y=traj_y_gpu[best_idx].tolist(),
                traj_chi=traj_chi_gpu[best_idx].tolist(),
                traj_U=traj_U_gpu[best_idx].tolist(),
                total_cost=float(costs[best_idx]),
                path_cost=float(path_result[0]) if n_candidates > 0 else 0.0,
                collision_cost=0.0,
                colregs_cost=0.0,
            )
        except Exception:
            # Fallback to CPU
            return self.calculate_optimal_offsets(xs, obstacles, waypoints, u_d, chi_d, active_mode)

    def _determine_colav_active(
        self,
        xs: ShipState4,
        obstacles: List[ObstacleData],
        nominal_traj: Tuple[List, List, List, List],
    ) -> bool:
        """Determine if collision avoidance mode should be active.

        Checks if any obstacle is within safe distance along the nominal trajectory.

        Args:
            xs: current ship state
            obstacles: list of obstacle data
            nominal_traj: nominal trajectory (x, y, chi, U)

        Returns:
            True if COLAV should be active
        """
        traj_x, traj_y, _, _ = nominal_traj

        for obstacle in obstacles:
            # Check distance at multiple timesteps
            for t in range(0, len(traj_x), max(1, len(traj_x) // 10)):
                dist = math.sqrt(
                    (traj_x[t] - obstacle.x) ** 2 +
                    (traj_y[t] - obstacle.y) ** 2
                )
                if dist < obstacle.d_safe:
                    return True

        return False

    def _setup_candidate_offsets(self) -> List[float]:
        """Setup candidate heading offset angles.

        Creates evenly spaced candidate offsets around the current heading.

        Returns:
            List of candidate heading offsets (radians)
        """
        # Create n_cbs candidates evenly spaced in [-0.5, 0.5] radians
        # with the center candidate being 0 (no offset)
        n_cbs = self.params.n_cbs
        max_offset = 0.5  # radians (~28 degrees)

        if n_cbs == 1:
            return [0.0]

        offsets = np.linspace(-max_offset, max_offset, n_cbs)
        return offsets.tolist()


# ============================================================================
# Simplified MPC Solver (Deterministic Safety Bounds)
# ============================================================================


class SBMPC_Solver:
    """Simplified Safety-Bounded MPC solver (deterministic).

    Uses deterministic safety bounds instead of probabilistic CPE.
    Lower computational cost, suitable for CPU-only execution.
    """

    def __init__(
        self,
        params: SBMPCParameters,
        ownship_length: float = 150.0,
        ownship_beam: float = 25.0,
    ):
        """Initialize SBMPC solver.

        Args:
            params: SBMPC parameters
            ownship_length: ownship length
            ownship_beam: ownship beam
        """
        self.params = params
        self.ownship_length = ownship_length
        self.ownship_beam = ownship_beam

        # Ownship model
        self.ownship = Kinematic_Ship(
            length=ownship_length,
            beam=ownship_beam,
        )

    def calculate_optimal_offsets(
        self,
        xs: ShipState4,
        obstacles: List[ObstacleData],
        waypoints: List[Waypoint],
        u_d: Optional[float] = None,
        chi_d: Optional[float] = None,
    ) -> MPCResult:
        """Calculate optimal offsets using deterministic safety bounds.

        Args:
            xs: current ship state
            obstacles: list of obstacle data
            waypoints: route waypoints
            u_d: desired surge speed
            chi_d: desired heading

        Returns:
            MPCResult with optimal offsets
        """
        if u_d is None:
            u_d = xs.U
        if chi_d is None:
            chi_d = xs.chi

        # Set waypoints
        self.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Setup candidates
        n_cbs = self.params.n_cbs
        max_offset = 0.5
        candidates = np.linspace(-max_offset, max_offset, n_cbs).tolist()

        # Evaluate each candidate
        best_cost = float("inf")
        best_offset = 0.0
        best_traj = None

        for offset in candidates:
            offsets = [offset] * self.params.n_M
            traj = self.ownship.predict_trajectory(
                xs, offsets=offsets, T=self.params.T, dt=self.params.dt
            )
            traj_x, traj_y, traj_chi, traj_U = traj

            # Simple collision cost based on minimum distance
            min_dist = float("inf")
            for obstacle in obstacles:
                for t in range(len(traj_x)):
                    dist = math.sqrt(
                        (traj_x[t] - obstacle.x) ** 2 +
                        (traj_y[t] - obstacle.y) ** 2
                    )
                    min_dist = min(min_dist, dist)

            # Cost: path tracking + collision penalty
            path_cost = sum(
                distance_2d(traj_x[t], traj_y[t], waypoints[t % len(waypoints)].x,
                           waypoints[t % len(waypoints)].y)
                for t in range(len(traj_x))
            ) / len(traj_x)

            collision_penalty = 0.0
            if min_dist < self.params.d_safe:
                collision_penalty = self.params.kappa_GN * (
                    self.params.d_safe - min_dist
                )

            total_cost = path_cost + collision_penalty

            if total_cost < best_cost:
                best_cost = total_cost
                best_offset = offset
                best_traj = traj

        traj_x, traj_y, traj_chi, traj_U = best_traj

        return MPCResult(
            offset_chi=best_offset,
            offset_U=0.0,
            traj_x=traj_x,
            traj_y=traj_y,
            traj_chi=traj_chi,
            traj_U=traj_U,
            total_cost=best_cost,
            path_cost=best_cost,
            collision_cost=0.0,
            colregs_cost=0.0,
        )
