"""Main PSB-MPC solver.

Orchestrates trajectory prediction, collision probability estimation,
cost evaluation, and COLREGS checking to compute optimal course/speed offsets.
Ported from the C++/CUDA implementations.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

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

        best_traj_x, best_traj_y, best_traj_chi, best_traj_U = all_trajs[best_idx]
        best_cost_breakdown = costs[best_idx]

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
