"""Cost functions for PSB-MPC.

Implements path cost, grounding cost, dynamic obstacle cost,
and COLREGS violation cost evaluation.
Ported from the C++/CUDA implementations.
"""

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
from .utils import (
    distance_2d,
    line_segment_intersection,
    normalize_angle,
    point_in_polygon,
    point_to_segment_distance,
    polygon_distance,
    ship_polygon,
)

# ============================================================================
# Taichi GPU Kernels
# ============================================================================

# Taichi kernel initialization flag
_ti_cost_initialized = False

def _ensure_cost_kernels():
    """Lazy initialization for Taichi cost kernels."""
    global _ti_cost_initialized
    if not _ti_cost_initialized:
        _init_cost_kernels()
        _ti_cost_initialized = True

def _init_cost_kernels():
    """Initialize Taichi GPU kernels for cost calculations."""

    @ti.func
    def _ti_distance_2d(x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32) -> ti.f32:
        dx = x2 - x1
        dy = y2 - y1
        return ti.sqrt(dx * dx + dy * dy)

    @ti.func
    def _ti_normalize_angle(angle: ti.f32) -> ti.f32:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @ti.func
    def _ti_point_to_segment_distance(
        px: ti.f32, py: ti.f32,
        ax: ti.f32, ay: ti.f32,
        bx: ti.f32, by: ti.f32,
    ) -> ti.f32:
        """Compute squared distance from point P to segment AB."""
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab_sq = abx * abx + aby * aby

        if ab_sq < 1e-12:
            return ti.sqrt(apx * apx + apy * apy)

        t = (apx * abx + apy * aby) / ab_sq
        t = ti.select(t > 1.0, 1.0, ti.select(t < 0.0, 0.0, t))

        proj_x = ax + t * abx
        proj_y = ay + t * aby

        dx = px - proj_x
        dy = py - proj_y
        return ti.sqrt(dx * dx + dy * dy)

    @ti.func
    def _ti_ship_polygon_vertex(
        cx: ti.f32, cy: ti.f32, chi: ti.f32,
        length: ti.f32, beam: ti.f32,
        idx: int,
        out_x: ti.types.ndarray(),
        out_y: ti.types.ndarray(),
    ):
        """Compute vertex of ship polygon given index 0..3."""
        # Ship polygon: 4 corners (bow, starboard, stern, port)
        half_l = length * 0.5
        half_b = beam * 0.5

        # Local coordinates
        local_x = ti.cast(0.0, ti.f32)
        local_y = ti.cast(0.0, ti.f32)

        if idx == 0:
            local_x = half_l
            local_y = ti.cast(0.0, ti.f32)
        elif idx == 1:
            local_x = ti.cast(-0.3, ti.f32) * half_l
            local_y = half_b
        elif idx == 2:
            local_x = -half_l
            local_y = ti.cast(0.0, ti.f32)
        elif idx == 3:
            local_x = ti.cast(-0.3, ti.f32) * half_l
            local_y = -half_b

        # Rotate
        cos_chi = ti.cos(chi)
        sin_chi = ti.sin(chi)
        out_x[0] = cx + local_x * cos_chi - local_y * sin_chi
        out_y[0] = cy + local_x * sin_chi + local_y * cos_chi

    @ti.kernel
    def _ti_compute_ship_polygon(
        cx: ti.f32, cy: ti.f32, chi: ti.f32,
        length: ti.f32, beam: ti.f32,
        vertices_x: ti.types.ndarray(),
        vertices_y: ti.types.ndarray(),
    ):
        for i in range(4):
            _ti_ship_polygon_vertex(cx, cy, chi, length, beam, i, vertices_x, vertices_y)

    @ti.kernel
    def _ti_path_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        n_steps: int,
        waypoints_x: ti.types.ndarray(),
        waypoints_y: ti.types.ndarray(),
        n_wps: int,
        init_x: ti.f32,
        init_y: ti.f32,
        w_path: ti.f32,
        w_deviation: ti.f32,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f32)

        for t in range(1, n_steps):
            tx = traj_x[t]
            ty = traj_y[t]

            # Find nearest waypoint
            min_dist = ti.cast(1e18, ti.f32)
            nearest_idx = 0

            for j in range(n_wps):
                dx = tx - waypoints_x[j]
                dy = ty - waypoints_y[j]
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j

            if nearest_idx < n_wps - 1:
                # Cross-track error
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
                    t_param = ti.select(t_param > 1.0, 1.0, ti.select(t_param < 0.0, 0.0, t_param))

                    proj_x = wx + t_param * dx
                    proj_y = wy + t_param * dy

                    cross_track = _ti_distance_2d(tx, ty, proj_x, proj_y)
                    total_cost += w_path * cross_track * cross_track

        # Initial position deviation
        init_dist = _ti_distance_2d(traj_x[0], traj_y[0], init_x, init_y)
        total_cost += w_deviation * init_dist * init_dist

        result[0] = total_cost

    @ti.kernel
    def _ti_grounding_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.f32,
        traj_chi: ti.types.ndarray(),
        n_steps: int,
        hazard_x: ti.types.ndarray(),
        hazard_y: ti.types.ndarray(),
        hazard_chi: ti.types.ndarray(),
        hazard_length: ti.types.ndarray(),
        hazard_beam: ti.types.ndarray(),
        n_hazards: int,
        ship_length: ti.f32,
        ship_beam: ti.f32,
        kappa_GN: ti.f32,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f32)
        d_safe = ti.cast(100.0, ti.f32)

        for t in range(n_steps):
            tx = traj_x[t]
            ty = traj_y[t]
            chi = traj_chi[t]

            # Compute ship polygon vertices
            ship_vx = ti.Vector.zeros(4, ti.f32)
            ship_vy = ti.Vector.zeros(4, ti.f32)
            _ti_compute_ship_polygon(tx, ty, chi, ship_length, ship_beam, ship_vx, ship_vy)

            for h in range(n_hazards):
                hx = hazard_x[h]
                hy = hazard_y[h]
                h_chi = hazard_chi[h]
                h_len = hazard_length[h]
                h_beam = hazard_beam[h]

                # Compute hazard polygon vertices
                h_vx = ti.Vector.zeros(4, ti.f32)
                h_vy = ti.Vector.zeros(4, ti.f32)
                _ti_compute_ship_polygon(hx, hy, h_chi, h_len, h_beam, h_vx, h_vy)

                # Check if ship center inside hazard (simplified: check against hazard center)
                dist_to_center = _ti_distance_2d(tx, ty, hx, hy)
                if dist_to_center < h_beam:
                    total_cost += kappa_GN * 10.0
                    continue

                # Minimum distance between ship and hazard polygons
                min_dist = ti.cast(1e18, ti.f32)

                for i in range(4):
                    for j in range(4):
                        dist = _ti_point_to_segment_distance(
                            h_vx[j], h_vy[j],
                            ship_vx[i], ship_vy[i],
                            ship_vx[(i + 1) % 4], ship_vy[(i + 1) % 4],
                        )
                        if dist < min_dist:
                            min_dist = dist

                if min_dist < d_safe:
                    cost = kappa_GN * ti.exp(-min_dist / (d_safe * 0.5))
                    total_cost += cost

        result[0] = total_cost

    @ti.kernel
    def _ti_deviation_cost_kernel(
        offsets_chi: ti.types.ndarray(),
        offsets_U: ti.types.ndarray(),
        last_chi: ti.types.ndarray(),
        last_U: ti.types.ndarray(),
        n_M: int,
        w_deviation: ti.f32,
        result: ti.types.ndarray(),
    ):
        cost = ti.cast(0.0, ti.f32)

        for i in range(n_M):
            chi_diff = offsets_chi[i] - last_chi[i]
            U_diff = offsets_U[i] - last_U[i]
            cost += w_deviation * (chi_diff * chi_diff + U_diff * U_diff)

        result[0] = cost

    @ti.kernel
    def _ti_dynamic_obstacle_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        n_steps: int,
        obs_x: ti.f32,
        obs_y: ti.f32,
        obs_U: ti.f32,
        d_safe: ti.f32,
        cpe_probability: ti.f32,
        w_collision: ti.f32,
        kappa_SO: ti.f32,
        colregs_role: int,
        time_horizon: ti.f32,
        dt: ti.f32,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f32)

        # Base collision cost
        collision_cost = cpe_probability * w_collision

        # Adjust by COLREGs role
        if colregs_role == 2:  # stand-on
            collision_cost *= kappa_SO
        elif colregs_role == 1:  # giving-way
            collision_cost *= kappa_SO * 0.5

        # Time-discounted cost
        n_steps_actual = ti.min(ti.cast(int(time_horizon / dt), ti.i32), n_steps)
        for t in range(n_steps_actual):
            discount = ti.cast(1.0, ti.f32) - ti.cast(t, ti.f32) / ti.cast(n_steps_actual, ti.f32)
            total_cost += collision_cost * discount

        # Proximity-based cost
        for t in range(1, n_steps):
            dx = traj_x[t] - obs_x
            dy = traj_y[t] - obs_y
            dist = ti.sqrt(dx * dx + dy * dy)

            if dist < d_safe:
                obs_U_safe = ti.select(obs_U > 0.0, obs_U, 1.0)
                ttc = dist / obs_U_safe
                if ttc < time_horizon:
                    proximity_cost = w_collision * ti.exp(-ttc / ti.cast(30.0, ti.f32))
                    total_cost += proximity_cost

        result[0] = total_cost

    @ti.kernel
    def _ti_colregs_cost_kernel(
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        n_steps: int,
        obs_x: ti.f32,
        obs_y: ti.f32,
        d_safe: ti.f32,
        situation: ti.i32,  # 0=none, 1=crossing_port, 2=head-on, 3=crossing_stb, 4=overtaking
        kappa_GW: ti.f32,
        kappa_SO: ti.f32,
        result: ti.types.ndarray(),
    ):
        total_cost = ti.cast(0.0, ti.f32)

        # Check if crossing_port or head-on (giving-way)
        if situation == 1 or situation == 2:
            min_dist = ti.cast(1e18, ti.f32)
            for t in range(n_steps):
                dist = _ti_distance_2d(traj_x[t], traj_y[t], obs_x, obs_y)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < d_safe:
                total_cost += kappa_GW * ti.exp(-min_dist / (d_safe * 0.3))

        # Check if crossing_stb (stand-on)
        if situation == 3:
            if n_steps > 10:
                deviation = _ti_distance_2d(traj_x[10], traj_y[10], traj_x[0], traj_y[0])
                if deviation > 20.0:  # expected_dist * 2 where expected_dist = 10
                    total_cost += kappa_SO * 0.5

        result[0] = total_cost


class Path_Grounding_Cost:
    """Path and grounding hazard cost evaluation.

    Computes costs for:
    - Deviation from reference path/waypoints
    - Proximity to grounding hazards (static obstacles)
    - Control deviation (penalize large changes from last optimal)
    """

    def __init__(
        self,
        kappa_GN: float = 200.0,  # grounding cost weight
        w_path: float = 1.0,  # path cost weight
        w_deviation: float = 10.0,  # control deviation weight
    ):
        """Initialize path/grounding cost evaluator.

        Args:
            kappa_GN: grounding hazard cost weight
            w_path: path tracking cost weight
            w_deviation: control deviation cost weight
        """
        self.kappa_GN = kappa_GN
        self.w_path = w_path
        self.w_deviation = w_deviation

    def calculate_grounding_cost(
        self,
        traj_x: List[float],
        traj_y: List[float],
        traj_chi: List[float],
        grounding_hazards: List[ObstacleData],
        ship_length: float = 150.0,
        ship_beam: float = 25.0,
    ) -> float:
        """Calculate grounding hazard cost for a trajectory.

        Computes distance from ship polygon to each grounding hazard polygon
        and applies cost based on proximity.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            traj_chi: trajectory headings
            grounding_hazards: list of grounding hazard polygons
            ship_length: ownship length
            ship_beam: ownship beam

        Returns:
            Total grounding cost
        """
        total_cost = 0.0

        for t in range(len(traj_x)):
            # Ship polygon at this timestep
            ship_poly = ship_polygon(
                traj_x[t], traj_y[t], traj_chi[t],
                ship_length, ship_beam,
            )

            for hazard in grounding_hazards:
                # Generate hazard polygon from ObstacleData
                # Use hazard's chi and (length, beam) to generate polygon
                hazard_chi = getattr(hazard, 'chi', 0.0)
                hazard_length = getattr(hazard, 'length', 100.0)
                hazard_beam = getattr(hazard, 'beam', 50.0)
                hazard_x = getattr(hazard, 'x', 0.0)
                hazard_y = getattr(hazard, 'y', 0.0)
                hazard_poly = ship_polygon(hazard_x, hazard_y, hazard_chi, hazard_length, hazard_beam)

                # Check if ship center is inside hazard
                if point_in_polygon(traj_x[t], traj_y[t], hazard_poly):
                    # Direct collision - very high cost
                    total_cost += self.kappa_GN * 10.0
                    continue

                # Compute minimum distance between ship and hazard
                min_dist = float("inf")

                # Check distance from ship polygon vertices to hazard polygon
                for i in range(len(ship_poly) - 1):
                    ax, ay = ship_poly[i]
                    bx, by = ship_poly[i + 1]
                    for j in range(len(hazard_poly) - 1):
                        cx, cy = hazard_poly[j]
                        dx, dy = hazard_poly[j + 1]

                        dist, _, _, _ = point_to_segment_distance(cx, cy, ax, ay, bx, by)
                        min_dist = min(min_dist, dist)

                        dist, _, _, _ = point_to_segment_distance(ax, ay, cx, cy, dx, dy)
                        min_dist = min(min_dist, dist)

                # Apply cost based on distance
                if min_dist < float("inf"):
                    # Exponential cost as distance decreases
                    # Cost becomes significant within safe distance
                    d_safe = 100.0  # minimum safe distance
                    if min_dist < d_safe:
                        cost = self.kappa_GN * math.exp(-min_dist / (d_safe * 0.5))
                        total_cost += cost

        return total_cost

    def calculate_path_cost(
        self,
        traj_x: List[float],
        traj_y: List[float],
        waypoints: List[Waypoint],
        ship_current_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Calculate path tracking cost.

        Measures deviation from the reference path defined by waypoints.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            waypoints: reference waypoints
            ship_current_pos: current ship position (for initial deviation)

        Returns:
            Path tracking cost
        """
        total_cost = 0.0

        for t in range(1, len(traj_x)):
            # Find nearest waypoint
            min_dist = float("inf")
            nearest_wp = None

            for wp in waypoints:
                dist = distance_2d(traj_x[t], traj_y[t], wp.x, wp.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_wp = wp

            if nearest_wp:
                # Cross-track error (perpendicular distance to waypoint line)
                if t > 0:
                    # Direction to next waypoint
                    next_wp_idx = min(waypoints.index(nearest_wp) + 1, len(waypoints) - 1)
                    next_wp = waypoints[next_wp_idx]

                    # Line direction
                    dx = next_wp.x - nearest_wp.x
                    dy = next_wp.y - nearest_wp.y
                    line_len = math.sqrt(dx * dx + dy * dy)

                    if line_len > 1e-6:
                        # Project ship position onto line
                        t_param = ((traj_x[t] - nearest_wp.x) * dx +
                                   (traj_y[t] - nearest_wp.y) * dy) / (line_len * line_len)
                        t_param = max(0.0, min(1.0, t_param))

                        # Projection point
                        proj_x = nearest_wp.x + t_param * dx
                        proj_y = nearest_wp.y + t_param * dy

                        # Cross-track error
                        cross_track = distance_2d(traj_x[t], traj_y[t], proj_x, proj_y)
                        total_cost += self.w_path * cross_track ** 2

        # Initial position deviation
        init_dist = distance_2d(
            traj_x[0], traj_y[0],
            ship_current_pos[0], ship_current_pos[1],
        )
        total_cost += self.w_deviation * init_dist ** 2

        return total_cost

    def calculate_deviation_cost(
        self,
        offsets_chi: List[float],
        offsets_U: List[float],
        last_optimal_offsets_chi: List[float] = None,
        last_optimal_offsets_U: List[float] = None,
    ) -> float:
        """Calculate control deviation cost.

        Penalizes large changes from the last optimal offsets.

        Args:
            offsets_chi: current heading offsets
            offsets_U: current surge offsets
            last_optimal_offsets_chi: previous optimal heading offsets
            last_optimal_offsets_U: previous optimal surge offsets

        Returns:
            Deviation cost
        """
        if last_optimal_offsets_chi is None:
            last_optimal_offsets_chi = [0.0] * len(offsets_chi)
        if last_optimal_offsets_U is None:
            last_optimal_offsets_U = [0.0] * len(offsets_U)

        cost = 0.0
        for i in range(len(offsets_chi)):
            chi_diff = offsets_chi[i] - last_optimal_offsets_chi[i]
            U_diff = offsets_U[i] - last_optimal_offsets_U[i]
            cost += self.w_deviation * (chi_diff ** 2 + U_diff ** 2)

        return cost

    def _calculate_grounding_cost_gpu(
        self,
        traj_x: List[float],
        traj_y: List[float],
        traj_chi: List[float],
        grounding_hazards: List[ObstacleData],
        ship_length: float = 150.0,
        ship_beam: float = 25.0,
    ) -> float:
        """GPU-accelerated grounding cost calculation using Taichi.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            traj_chi: trajectory headings
            grounding_hazards: list of grounding hazard polygons
            ship_length: ownship length
            ship_beam: ownship beam

        Returns:
            Total grounding cost
        """
        try:
            _ensure_cost_kernels()
        except Exception:
            # Fall back to CPU if Taichi not available
            return self.calculate_grounding_cost(traj_x, traj_y, traj_chi, grounding_hazards, ship_length, ship_beam)

        n_steps = len(traj_x)
        n_hazards = len(grounding_hazards)

        traj_x_np = np.array(traj_x, dtype=np.float64)
        traj_y_np = np.array(traj_y, dtype=np.float64)
        traj_chi_np = np.array(traj_chi, dtype=np.float64)

        hazard_x_np = np.array([getattr(h, 'x', 0.0) for h in grounding_hazards], dtype=np.float64)
        hazard_y_np = np.array([getattr(h, 'y', 0.0) for h in grounding_hazards], dtype=np.float64)
        hazard_chi_np = np.array([getattr(h, 'chi', 0.0) for h in grounding_hazards], dtype=np.float64)
        hazard_length_np = np.array([getattr(h, 'length', 100.0) for h in grounding_hazards], dtype=np.float64)
        hazard_beam_np = np.array([getattr(h, 'beam', 50.0) for h in grounding_hazards], dtype=np.float64)

        # Create Taichi NDArrays
        traj_x_taichi = ti.ndarray(np.float64, n_steps)
        traj_y_taichi = ti.ndarray(np.float64, n_steps)
        traj_chi_taichi = ti.ndarray(np.float64, n_steps)
        hazard_x_taichi = ti.ndarray(np.float64, n_hazards)
        hazard_y_taichi = ti.ndarray(np.float64, n_hazards)
        hazard_chi_taichi = ti.ndarray(np.float64, n_hazards)
        hazard_length_taichi = ti.ndarray(np.float64, n_hazards)
        hazard_beam_taichi = ti.ndarray(np.float64, n_hazards)
        result_taichi = ti.ndarray(np.float64, 1)

        traj_x_taichi.from_numpy(traj_x_np)
        traj_y_taichi.from_numpy(traj_y_np)
        traj_chi_taichi.from_numpy(traj_chi_np)
        hazard_x_taichi.from_numpy(hazard_x_np)
        hazard_y_taichi.from_numpy(hazard_y_np)
        hazard_chi_taichi.from_numpy(hazard_chi_np)
        hazard_length_taichi.from_numpy(hazard_length_np)
        hazard_beam_taichi.from_numpy(hazard_beam_np)

        _ti_grounding_cost_kernel(
            traj_x_taichi, traj_y_taichi, traj_chi_taichi,
            n_steps,
            hazard_x_taichi, hazard_y_taichi, hazard_chi_taichi,
            hazard_length_taichi, hazard_beam_taichi,
            n_hazards,
            np.float32(ship_length), np.float32(ship_beam),
            np.float32(self.kappa_GN),
            result_taichi,
        )

        return float(result_taichi.to_numpy()[0])

    def _calculate_path_cost_gpu(
        self,
        traj_x: List[float],
        traj_y: List[float],
        waypoints: List[Waypoint],
        ship_current_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """GPU-accelerated path cost calculation using Taichi.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            waypoints: reference waypoints
            ship_current_pos: current ship position

        Returns:
            Path tracking cost
        """
        try:
            _ensure_cost_kernels()
        except Exception:
            return self.calculate_path_cost(traj_x, traj_y, waypoints, ship_current_pos)

        n_steps = len(traj_x)
        n_wps = len(waypoints)

        traj_x_np = np.array(traj_x, dtype=np.float64)
        traj_y_np = np.array(traj_y, dtype=np.float64)
        waypoints_x_np = np.array([wp.x for wp in waypoints], dtype=np.float64)
        waypoints_y_np = np.array([wp.y for wp in waypoints], dtype=np.float64)

        traj_x_taichi = ti.ndarray(np.float64, n_steps)
        traj_y_taichi = ti.ndarray(np.float64, n_steps)
        waypoints_x_taichi = ti.ndarray(np.float64, n_wps)
        waypoints_y_taichi = ti.ndarray(np.float64, n_wps)
        result_taichi = ti.ndarray(np.float64, 1)

        traj_x_taichi.from_numpy(traj_x_np)
        traj_y_taichi.from_numpy(traj_y_np)
        waypoints_x_taichi.from_numpy(waypoints_x_np)
        waypoints_y_taichi.from_numpy(waypoints_y_np)

        _ti_path_cost_kernel(
            traj_x_taichi, traj_y_taichi,
            n_steps,
            waypoints_x_taichi, waypoints_y_taichi,
            n_wps,
            np.float32(ship_current_pos[0]), np.float32(ship_current_pos[1]),
            np.float32(self.w_path), np.float32(self.w_deviation),
            result_taichi,
        )

        return float(result_taichi.to_numpy()[0])

    def _calculate_deviation_cost_gpu(
        self,
        offsets_chi: List[float],
        offsets_U: List[float],
        last_optimal_offsets_chi: List[float] = None,
        last_optimal_offsets_U: List[float] = None,
    ) -> float:
        """GPU-accelerated deviation cost calculation using Taichi.

        Args:
            offsets_chi: current heading offsets
            offsets_U: current surge offsets
            last_optimal_offsets_chi: previous optimal heading offsets
            last_optimal_offsets_U: previous optimal surge offsets

        Returns:
            Deviation cost
        """
        try:
            _ensure_cost_kernels()
        except Exception:
            return self.calculate_deviation_cost(offsets_chi, offsets_U, last_optimal_offsets_chi, last_optimal_offsets_U)

        if last_optimal_offsets_chi is None:
            last_optimal_offsets_chi = [0.0] * len(offsets_chi)
        if last_optimal_offsets_U is None:
            last_optimal_offsets_U = [0.0] * len(offsets_U)

        n_M = len(offsets_chi)

        offsets_chi_np = np.array(offsets_chi, dtype=np.float64)
        offsets_U_np = np.array(offsets_U, dtype=np.float64)
        last_chi_np = np.array(last_optimal_offsets_chi, dtype=np.float64)
        last_U_np = np.array(last_optimal_offsets_U, dtype=np.float64)

        offsets_chi_taichi = ti.ndarray(np.float64, n_M)
        offsets_U_taichi = ti.ndarray(np.float64, n_M)
        last_chi_taichi = ti.ndarray(np.float64, n_M)
        last_U_taichi = ti.ndarray(np.float64, n_M)
        result_taichi = ti.ndarray(np.float64, 1)

        offsets_chi_taichi.from_numpy(offsets_chi_np)
        offsets_U_taichi.from_numpy(offsets_U_np)
        last_chi_taichi.from_numpy(last_chi_np)
        last_U_taichi.from_numpy(last_U_np)

        _ti_deviation_cost_kernel(
            offsets_chi_taichi, offsets_U_taichi,
            last_chi_taichi, last_U_taichi,
            n_M, np.float32(self.w_deviation),
            result_taichi,
        )

        return float(result_taichi.to_numpy()[0])


# ============================================================================
# Dynamic Obstacle Cost
# ============================================================================


class Dynamic_Obstacle_Cost:
    """Dynamic obstacle collision cost evaluation.

    Computes collision probability-based costs for moving obstacles.
    """

    def __init__(
        self,
        kappa_SO: float = 50.0,  # stand-on cost weight
        kappa_RA: float = 75.0,  # readily apparent cost weight
        w_collision: float = 100.0,  # collision cost weight
    ):
        """Initialize dynamic obstacle cost evaluator.

        Args:
            kappa_SO: stand-on vessel cost weight
            kappa_RA: readily apparent situation cost weight
            w_collision: base collision cost weight
        """
        self.kappa_SO = kappa_SO
        self.kappa_RA = kappa_RA
        self.w_collision = w_collision

    def calculate_dynamic_obstacle_cost(
        self,
        traj_x: List[float],
        traj_y: List[float],
        traj_chi: List[float],
        obstacle: ObstacleData,
        cpe_probability: float = 0.0,
        time_horizon: float = 300.0,
        dt: float = 1.0,
    ) -> float:
        """Calculate dynamic obstacle collision cost.

        Combines collision probability with cost weights based on
        COLREGs situation type.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            traj_chi: trajectory headings
            obstacle: obstacle ship data
            cpe_probability: pre-computed collision probability from CPE
            time_horizon: prediction horizon
            dt: time step

        Returns:
            Dynamic obstacle cost
        """
        total_cost = 0.0

        # Base collision cost
        collision_cost = cpe_probability * self.w_collision

        # Adjust by COLREGs role
        if obstacle.colregs_role == 2:  # stand-on
            collision_cost *= self.kappa_SO
        elif obstacle.colregs_role == 1:  # giving-way
            collision_cost *= self.kappa_SO * 0.5

        # Time-discounted cost (penalize later collisions less)
        n_steps = int(time_horizon / dt)
        for t in range(min(n_steps, len(traj_x))):
            discount = 1.0 - t / n_steps  # linear discount
            total_cost += collision_cost * discount

        # Proximity-based cost (time-to-CPA approximation)
        if len(traj_x) > 1:
            # Estimate time to CPA
            for t in range(1, len(traj_x)):
                dx = traj_x[t] - obstacle.x
                dy = traj_y[t] - obstacle.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < obstacle.d_safe:
                    # Time-based proximity cost
                    ttc = dist / max(obstacle.U, 1.0) if obstacle.U > 0 else float("inf")
                    if ttc < time_horizon:
                        proximity_cost = self.w_collision * math.exp(-ttc / 30.0)
                        total_cost += proximity_cost

        return total_cost

    def _calculate_dynamic_obstacle_cost_gpu(
        self,
        traj_x: List[float],
        traj_y: List[float],
        traj_chi: List[float],
        obstacle: ObstacleData,
        cpe_probability: float = 0.0,
        time_horizon: float = 300.0,
        dt: float = 1.0,
    ) -> float:
        """GPU-accelerated dynamic obstacle cost calculation using Taichi.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            traj_chi: trajectory headings
            obstacle: obstacle ship data
            cpe_probability: pre-computed collision probability from CPE
            time_horizon: prediction horizon
            dt: time step

        Returns:
            Dynamic obstacle cost
        """
        try:
            _ensure_cost_kernels()
        except Exception:
            return self.calculate_dynamic_obstacle_cost(
                traj_x, traj_y, traj_chi, obstacle, cpe_probability, time_horizon, dt
            )

        n_steps = len(traj_x)

        traj_x_np = np.array(traj_x, dtype=np.float64)
        traj_y_np = np.array(traj_y, dtype=np.float64)

        traj_x_taichi = ti.ndarray(np.float64, n_steps)
        traj_y_taichi = ti.ndarray(np.float64, n_steps)
        result_taichi = ti.ndarray(np.float64, 1)

        traj_x_taichi.from_numpy(traj_x_np)
        traj_y_taichi.from_numpy(traj_y_np)

        _ti_dynamic_obstacle_cost_kernel(
            traj_x_taichi, traj_y_taichi,
            n_steps,
            np.float32(getattr(obstacle, 'x', 0.0)),
            np.float32(getattr(obstacle, 'y', 0.0)),
            np.float32(getattr(obstacle, 'U', 0.0)),
            np.float32(getattr(obstacle, 'd_safe', 100.0)),
            np.float32(cpe_probability),
            np.float32(self.w_collision),
            np.float32(self.kappa_SO),
            getattr(obstacle, 'colregs_role', 0),
            np.float32(time_horizon),
            np.float32(dt),
            result_taichi,
        )

        return float(result_taichi.to_numpy()[0])


# ============================================================================
# COLREGS Violation Evaluator
# ============================================================================


class COLREGS_Evaluator:
    """COLREGs situation detection and violation evaluation.

    Detects COLREGs situations and evaluates giving-way / stand-on violations.
    """

    def __init__(
        self,
        kappa_GW: float = 100.0,  # giving-way cost weight
        kappa_SO: float = 50.0,  # stand-on cost weight
        kappa_RA: float = 75.0,  # readily apparent cost weight
    ):
        """Initialize COLREGs evaluator.

        Args:
            kappa_GW: giving-way vessel cost weight
            kappa_SO: stand-on vessel cost weight
            kappa_RA: readily apparent violation cost weight
        """
        self.kappa_GW = kappa_GW
        self.kappa_SO = kappa_SO
        self.kappa_RA = kappa_RA

    def detect_situation(
        self,
        ownship_chi: float,
        ownship_U: float,
        obstacle: ObstacleData,
        ownship_x: float = 0.0,
        ownship_y: float = 0.0,
    ) -> str:
        """Detect COLREGs situation between ownship and obstacle.

        Args:
            ownship_chi: ownship heading (radians)
            ownship_U: ownship speed
            obstacle: obstacle ship data
            ownship_x: ownship x position (default 0.0 for backward compatibility)
            ownship_y: ownship y position (default 0.0 for backward compatibility)

        Returns:
            Situation type: "head-on", "crossing_port", "crossing_stb",
                          "overtaking", "none"
        """
        # Handle None obstacle
        if obstacle is None:
            return "none"

        # Relative bearing from ownship to obstacle
        bearing = math.atan2(
            obstacle.y - ownship_y,
            obstacle.x - ownship_x,
        )
        rel_bearing = normalize_angle(bearing - ownship_chi)

        # Relative bearing in [0, 2pi)
        rel_bearing_pos = rel_bearing if rel_bearing >= 0 else rel_bearing + 2 * math.pi

        # Overtaking detection
        if rel_bearing_pos > math.radians(112.5) and rel_bearing_pos < math.radians(247.5):
            return "overtaking"

        # Head-on detection (nearly opposite headings, within 6 degrees)
        if abs(rel_bearing) < math.radians(5):
            return "head-on"

        # Crossing detection
        if math.radians(10) < abs(rel_bearing) < math.radians(112.5):
            if rel_bearing > 0:
                return "crossing_port"  # Obstacle is on port side
            else:
                return "crossing_stb"  # Obstacle is on starboard side

        return "none"

    def evaluate_giving_way(
        self,
        situation: str,
        traj_x: List[float],
        traj_y: List[float],
        obstacle: ObstacleData,
    ) -> float:
        """Evaluate giving-way vessel violation cost.

        Args:
            situation: COLREGs situation type
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            obstacle: obstacle ship data

        Returns:
            Giving-way violation cost
        """
        if situation not in ["crossing_port", "head-on"]:
            return 0.0

        # Check if trajectory approaches the obstacle
        min_dist = float("inf")
        for t in range(len(traj_x)):
            dist = distance_2d(traj_x[t], traj_y[t], obstacle.x, obstacle.y)
            min_dist = min(min_dist, dist)

        # Cost based on proximity
        if min_dist < obstacle.d_safe:
            return self.kappa_GW * math.exp(-min_dist / (obstacle.d_safe * 0.3))

        return 0.0

    def evaluate_stand_on(
        self,
        situation: str,
        traj_x: List[float],
        traj_y: List[float],
        obstacle: ObstacleData,
    ) -> float:
        """Evaluate stand-on vessel violation cost.

        Args:
            situation: COLREGs situation type
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            obstacle: obstacle ship data

        Returns:
            Stand-on violation cost
        """
        if situation != "crossing_stb":
            return 0.0

        # Stand-on vessel should maintain course and speed
        # Penalize large deviations
        if len(traj_x) > 10:
            # Check course change
            initial_heading = normalize_angle(
                math.atan2(traj_y[10] - traj_y[0], traj_x[10] - traj_x[0])
            )
            course_change = abs(normalize_angle(initial_heading - traj_x[0] if False else 0))
            # Simplified: check position deviation
            deviation = distance_2d(traj_x[10], traj_y[10], traj_x[0], traj_y[0])
            expected_dist = 10.0  # Expected distance in 10 steps at nominal speed
            if deviation > expected_dist * 2:
                return self.kappa_SO * 0.5

        return 0.0

    def evaluate_readily_apparent(
        self,
        situation: str,
        traj_x: List[float],
        traj_y: List[float],
        obstacle: ObstacleData,
    ) -> float:
        """Evaluate readily apparent collision situation cost.

        High cost when collision is imminent and obvious.

        Args:
            situation: COLREGs situation type
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            obstacle: obstacle ship data

        Returns:
            Readily apparent violation cost
        """
        if situation == "none":
            return 0.0

        # Find minimum distance
        min_dist = float("inf")
        for t in range(len(traj_x)):
            dist = distance_2d(traj_x[t], traj_y[t], obstacle.x, obstacle.y)
            min_dist = min(min_dist, dist)

        # Readily apparent when very close
        if min_dist < obstacle.d_safe * 0.5:
            return self.kappa_RA * math.exp(-min_dist / (obstacle.d_safe * 0.2))

        return 0.0

    def calculate_colregs_cost(
        self,
        situation: str,
        traj_x: List[float],
        traj_y: List[float],
        obstacle: ObstacleData,
    ) -> float:
        """Calculate total COLREGs violation cost.

        Args:
            situation: detected COLREGs situation
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            obstacle: obstacle ship data

        Returns:
            Total COLREGs cost
        """
        cost_gw = self.evaluate_giving_way(situation, traj_x, traj_y, obstacle)
        cost_so = self.evaluate_stand_on(situation, traj_x, traj_y, obstacle)
        cost_ra = self.evaluate_readily_apparent(situation, traj_x, traj_y, obstacle)

        return cost_gw + cost_so + cost_ra


# ============================================================================
# Combined Cost Functors
# ============================================================================


class MPC_Cost:
    """Combined MPC cost evaluation.

    Orchestrates all cost components for the MPC solver.
    """

    def __init__(
        self,
        params: PSBMPCParameters,
        grounding_hazards: Optional[List[List]] = None,
    ):
        """Initialize MPC cost evaluator.

        Args:
            params: MPC parameters
            grounding_hazards: list of grounding hazard polygons
        """
        self.params = params
        self.grounding_hazards = grounding_hazards or []

        # Initialize sub-evaluators
        self.path_grounding = Path_Grounding_Cost(
            kappa_GN=params.kappa_GN,
            w_path=params.w_path,
            w_deviation=params.w_deviation,
        )
        self.dynamic_obstacle = Dynamic_Obstacle_Cost(
            kappa_SO=params.kappa_SO,
            kappa_RA=params.kappa_RA,
            w_collision=params.w_collision,
        )
        self.colregs = COLREGS_Evaluator(
            kappa_GW=params.kappa_GW,
            kappa_SO=params.kappa_SO,
            kappa_RA=params.kappa_RA,
        )

    def calculate_total_cost(
        self,
        traj_x: List[float],
        traj_y: List[float],
        traj_chi: List[float],
        traj_U: List[float],
        obstacles: List[ObstacleData],
        waypoints: List[Waypoint],
        ship_length: float = 150.0,
        ship_beam: float = 25.0,
        last_optimal_offsets_chi: Optional[List[float]] = None,
        last_optimal_offsets_U: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Calculate total MPC cost for a trajectory.

        Args:
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            traj_chi: trajectory headings
            traj_U: trajectory surge speeds
            obstacles: list of obstacle ships
            waypoints: reference waypoints
            ship_length: ownship length
            ship_beam: ownship beam
            last_optimal_offsets_chi: previous optimal heading offsets
            last_optimal_offsets_U: previous optimal surge offsets

        Returns:
            Dictionary with cost breakdown
        """
        # Path cost
        path_cost = self.path_grounding.calculate_path_cost(
            traj_x, traj_y, waypoints,
            ship_current_pos=(traj_x[0], traj_y[0]),
        )

        # Grounding cost
        grounding_cost = self.path_grounding.calculate_grounding_cost(
            traj_x, traj_y, traj_chi,
            self.grounding_hazards,
            ship_length, ship_beam,
        )

        # Dynamic obstacle costs
        collision_cost = 0.0
        colregs_costs = {}

        for obstacle in obstacles:
            # Calculate dynamic obstacle cost (simplified - no CPE in this method)
            dyn_cost = self.dynamic_obstacle.calculate_dynamic_obstacle_cost(
                traj_x, traj_y, traj_chi,
                obstacle,
                cpe_probability=0.0,  # Pre-computed CPE would go here
                time_horizon=self.params.T,
                dt=self.params.dt,
            )
            collision_cost += dyn_cost

            # COLREGs cost
            situation = self.colregs.detect_situation(
                traj_chi[0] if traj_chi else 0.0,
                traj_U[0] if traj_U else 0.0,
                obstacle,
            )

            colregs_cost = self.colregs.calculate_colregs_cost(
                situation, traj_x, traj_y, obstacle,
            )
            colregs_costs[id(obstacle)] = colregs_cost

        # Deviation cost - use n_M sized offsets, not trajectory length
        n_M = self.params.n_M if self.params else 10
        if last_optimal_offsets_chi is None:
            last_optimal_offsets_chi = [0.0] * n_M
        if last_optimal_offsets_U is None:
            last_optimal_offsets_U = [0.0] * n_M
        deviation_cost = self.path_grounding.calculate_deviation_cost(
            [0.0] * n_M,
            [0.0] * n_M,
            last_optimal_offsets_chi,
            last_optimal_offsets_U,
        )

        total_cost = path_cost + grounding_cost + collision_cost + deviation_cost + sum(colregs_costs.values())

        return {
            "total": total_cost,
            "path": path_cost,
            "grounding": grounding_cost,
            "collision": collision_cost,
            "colregs": sum(colregs_costs.values()),
            "deviation": deviation_cost,
        }

    def _calculate_colregs_cost_gpu(
        self,
        situation: str,
        traj_x: List[float],
        traj_y: List[float],
        obstacle: ObstacleData,
    ) -> float:
        """GPU-accelerated COLREGs cost calculation using Taichi.

        Args:
            situation: detected COLREGs situation
            traj_x: trajectory x coordinates
            traj_y: trajectory y coordinates
            obstacle: obstacle ship data

        Returns:
            Total COLREGs cost
        """
        try:
            _ensure_cost_kernels()
        except Exception:
            return self.calculate_colregs_cost(situation, traj_x, traj_y, obstacle)

        n_steps = len(traj_x)

        # Map situation string to integer code
        situation_map = {
            "none": 0,
            "crossing_port": 1,
            "head-on": 2,
            "crossing_stb": 3,
            "overtaking": 4,
        }
        situation_code = situation_map.get(situation, 0)

        traj_x_np = np.array(traj_x, dtype=np.float64)
        traj_y_np = np.array(traj_y, dtype=np.float64)

        traj_x_taichi = ti.ndarray(np.float64, n_steps)
        traj_y_taichi = ti.ndarray(np.float64, n_steps)
        result_taichi = ti.ndarray(np.float64, 1)

        traj_x_taichi.from_numpy(traj_x_np)
        traj_y_taichi.from_numpy(traj_y_np)

        _ti_colregs_cost_kernel(
            traj_x_taichi, traj_y_taichi,
            n_steps,
            np.float32(getattr(obstacle, 'x', 0.0)),
            np.float32(getattr(obstacle, 'y', 0.0)),
            np.float32(getattr(obstacle, 'd_safe', 100.0)),
            np.int32(situation_code),
            np.float32(self.kappa_GW),
            np.float32(self.kappa_SO),
            result_taichi,
        )

        return float(result_taichi.to_numpy()[0])
