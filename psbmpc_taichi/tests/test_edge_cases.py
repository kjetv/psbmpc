"""Edge case tests for pspbmpc_taichi.

Tests boundary conditions, empty inputs, and extreme values.
"""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestEdgeCasePRNG:
    """Edge case tests for PRNG."""

    def test_zero_samples(self):
        """Test PRNG with zero samples."""
        rng = p.Xoshiro256pp(seed=42)
        samples = rng.uniform_sample(0)
        assert len(samples) == 0

    def test_single_sample(self):
        """Test PRNG with single sample."""
        rng = p.Xoshiro256pp(seed=42)
        samples = rng.uniform_sample(1)
        assert len(samples) == 1
        assert 0.0 <= samples[0] <= 1.0

    def test_very_small_std(self):
        """Test normal sample with very small std."""
        rng = p.Xoshiro256pp(seed=42)
        samples = rng.normal_sample(100, mean=5.0, std=1e-10)
        assert all(abs(s - 5.0) < 1e-8 for s in samples)

    def test_very_large_covariance(self):
        """Test multivariate normal with large covariance."""
        rng = p.Xoshiro256pp(seed=42)
        cov = [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]]
        samples = rng.multivariate_normal_sample([0.0, 0.0, 0.0], cov, 100)
        assert len(samples) == 100


class TestEdgeCaseGeometry:
    """Edge case tests for geometry utilities."""

    def test_zero_length_ship(self):
        """Test ship polygon with zero dimensions."""
        polygon = p.ship_polygon(x=0.0, y=0.0, chi=0.0, length=0.0, beam=0.0)
        assert len(polygon) == 5  # 4 corners + closing point
        assert all(p == (0.0, 0.0) for p in polygon)

    def test_very_large_ship(self):
        """Test ship polygon with very large dimensions."""
        polygon = p.ship_polygon(x=0.0, y=0.0, chi=0.0, length=10000.0, beam=1000.0)
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        assert max(x_coords) - min(x_coords) == pytest.approx(10000.0, rel=0.01)
        assert max(y_coords) - min(y_coords) == pytest.approx(1000.0, rel=0.01)

    def test_pi_heading(self):
        """Test ship polygon rotated by pi radians."""
        polygon = p.ship_polygon(x=0.0, y=0.0, chi=math.pi, length=100.0, beam=20.0)
        # At chi=pi, bow should point in -x direction
        x_coords = [p[0] for p in polygon]
        assert min(x_coords) < 0  # Some points should have negative x

    def test_2pi_heading(self):
        """Test ship polygon rotated by 2pi (full rotation)."""
        polygon_0 = p.ship_polygon(x=0.0, y=0.0, chi=0.0, length=100.0, beam=20.0)
        polygon_2pi = p.ship_polygon(x=0.0, y=0.0, chi=2*math.pi, length=100.0, beam=20.0)
        # Should be identical after full rotation
        for i in range(4):
            assert polygon_0[i] == pytest.approx(polygon_2pi[i], abs=1e-10)

    def test_point_on_polygon_boundary(self):
        """Test point_in_polygon with point on boundary."""
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        # Point on edge
        assert p.point_in_polygon(5.0, 0.0, poly) is True
        # Corner point
        assert p.point_in_polygon(0.0, 0.0, poly) is True

    def test_degenerate_polygon(self):
        """Test point_in_polygon with degenerate polygon."""
        poly = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        # All points at origin - should not crash
        result = p.point_in_polygon(0.0, 0.0, poly)
        assert isinstance(result, bool)

    def test_polygon_distance_overlapping(self):
        """Test polygon_distance with overlapping ships."""
        # Two ships at same position
        dist = p.polygon_distance(0.0, 0.0, 0.0, 100.0, 20.0, 0.0, 0.0, 0.0, 100.0, 20.0)
        # Should be zero or negative (overlapping)
        assert dist <= 0.0

    def test_polygon_distance_tangent(self):
        """Test polygon_distance with tangent ships."""
        # Two ships touching at edges
        dist = p.polygon_distance(0.0, 0.0, 0.0, 100.0, 20.0, 50.0, 0.0, 0.0, 100.0, 20.0)
        assert dist == pytest.approx(0.0, abs=0.01)


class TestEdgeCaseShipModels:
    """Edge case tests for ship models."""

    def test_zero_speed_prediction(self, kinematic_ship, ship_state4):
        """Test trajectory prediction with zero speed."""
        state_zero = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=0.0)
        traj = kinematic_ship.predict_trajectory(
            state_zero,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )
        # Ship should stay at origin
        assert traj[0][0] == pytest.approx(0.0, abs=1.0)
        assert traj[1][0] == pytest.approx(0.0, abs=1.0)

    def test_very_long_horizon(self, kinematic_ship, ship_state4):
        """Test prediction with very long time horizon."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 3600,  # 1 hour
            T=3600.0,
            dt=1.0,
        )
        assert len(traj[0]) == 3601

    def test_single_timestep(self, kinematic_ship, ship_state4):
        """Test prediction with single timestep."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0],
            T=1.0,
            dt=1.0,
        )
        assert len(traj[0]) == 2  # Start + 1 step

    def test_negative_offset(self, kinematic_ship, ship_state4):
        """Test prediction with negative heading offset."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[-0.5] * 100,
            T=100.0,
            dt=1.0,
        )
        assert traj[1][-1] < traj[1][0]  # Should curve downward

    def test_very_large_offset(self, kinematic_ship, ship_state4):
        """Test prediction with very large heading offset."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[math.pi] * 100,  # 180 degree turn
            T=100.0,
            dt=1.0,
        )
        assert traj is not None  # Should not crash


class TestEdgeCaseCOLREGS:
    """Edge case tests for COLREGS evaluator."""

    def test_none_obstacle(self, colregs_evaluator, ship_state4):
        """Test COLREGS detection with None obstacle."""
        situation = colregs_evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, None,
        )
        assert situation == "none"

    def test_obstacle_at_origin(self, colregs_evaluator, ship_state4):
        """Test COLREGS detection with obstacle at origin."""
        obs = p.ObstacleData(
            x=0.0, y=0.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        situation = colregs_evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obs,
        )
        assert situation is not None

    def test_obstacle_at_infinity(self, colregs_evaluator, ship_state4):
        """Test COLREGS detection with very distant obstacle."""
        obs = p.ObstacleData(
            x=1e10, y=1e10, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        situation = colregs_evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obs,
        )
        assert situation is not None

    def test_multiple_obstacles(self, colregs_evaluator, ship_state4):
        """Test with multiple obstacles at different positions."""
        obs1 = p.ObstacleData(
            x=100.0, y=100.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        obs2 = p.ObstacleData(
            x=-100.0, y=100.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        obs3 = p.ObstacleData(
            x=0.0, y=-200.0, chi=0.0, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        for obs in [obs1, obs2, obs3]:
            situation = colregs_evaluator.detect_situation(
                ship_state4.chi, ship_state4.U, obs,
            )
            assert situation in ["head-on", "crossing_port", "crossing_stb", "overtaking", "none"]


class TestEdgeCaseCosts:
    """Edge case tests for cost functions."""

    def test_empty_trajectory_path_cost(self):
        """Test path cost with minimal trajectory."""
        cost = p.Path_Grounding_Cost()
        # Ship at (0, 0) with single-point trajectory going to waypoint at (1000, 0)
        traj_x, traj_y = [0.0, 100.0], [0.0, 100.0]
        waypoints = [p.Waypoint(x=1000.0, y=0.0)]
        path_cost = cost.calculate_path_cost(
            traj_x, traj_y, waypoints,
            ship_current_pos=(0.0, 0.0),
        )
        # Should be positive since trajectory deviates from direct path to waypoint
        assert path_cost >= 0.0

    def test_single_point_trajectory(self):
        """Test path cost with single point trajectory."""
        cost = p.Path_Grounding_Cost()
        traj_x, traj_y = [0.0], [0.0]
        waypoints = [p.Waypoint(x=1000.0, y=0.0)]
        path_cost = cost.calculate_path_cost(
            traj_x, traj_y, waypoints,
            ship_current_pos=(0.0, 0.0),
        )
        assert path_cost >= 0.0

    def test_grounding_cost_empty_hazards(self, ship_state4, kinematic_ship):
        """Test grounding cost with no hazards."""
        cost = p.Path_Grounding_Cost()
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )
        grounding_cost = cost.calculate_grounding_cost(
            traj[0], traj[1], traj[2], [],
        )
        assert grounding_cost == 0.0

    def test_dynamic_cost_no_obstacle(self):
        """Test dynamic obstacle cost with zero probability."""
        cost = p.Dynamic_Obstacle_Cost()
        traj_x = [i * 100 for i in range(10)]
        traj_y = [0.0] * 10
        traj_chi = [0.0] * 10
        dyn_cost = cost.calculate_dynamic_obstacle_cost(
            traj_x, traj_y, traj_chi,
            p.ObstacleData(x=1000.0, y=0.0, chi=0.0, U=0.0,
                          length=150.0, beam=25.0, d_safe=300.0),
            cpe_probability=0.0,
        )
        assert dyn_cost == 0.0

    def test_colregs_cost_zero_offsets(self, colregs_evaluator):
        """Test COLREGS cost with zero offsets."""
        traj_x, traj_y = [0.0, 100.0, 200.0], [0.0, 0.0, 0.0]
        obstacle = p.ObstacleData(x=500.0, y=500.0, chi=0.0, U=0.0,
                                  length=150.0, beam=25.0, d_safe=300.0)
        colregs_cost = colregs_evaluator.calculate_colregs_cost(
            "stand-on", traj_x, traj_y, obstacle,
        )
        assert colregs_cost >= 0.0


class TestEdgeCaseTypes:
    """Edge case tests for type dataclasses."""

    def test_ship_state4_zeros(self):
        """Test ShipState4 with zero values."""
        state = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=0.0)
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.chi == 0.0
        assert state.U == 0.0

    def test_ship_state6_zeros(self):
        """Test ShipState6 with zero values."""
        state = p.ShipState6(x=0.0, y=0.0, psi=0.0, u=0.0, v=0.0, r=0.0)
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.psi == 0.0
        assert state.u == 0.0
        assert state.v == 0.0
        assert state.r == 0.0

    def test_obstacle_data_minimal(self):
        """Test ObstacleData with minimal values."""
        obs = p.ObstacleData(x=0.0, y=0.0, chi=0.0, U=0.0,
                            length=150.0, beam=25.0, colregs_role=0,
                            d_safe=600.0, cov_xx=10.0, cov_yy=10.0, cov_xy=0.0)
        assert obs.x == 0.0
        assert obs.y == 0.0
        assert obs.chi == 0.0
        assert obs.U == 0.0
        # Default values from dataclass
        assert obs.length == 150.0
        assert obs.beam == 25.0
        assert obs.colregs_role == 0
        assert obs.d_safe == 600.0
        assert obs.cov_xx == 10.0

    def test_waypoint_minimal(self):
        """Test Waypoint with minimal values."""
        wp = p.Waypoint(x=1000.0, y=0.0)
        assert wp.x == 1000.0
        assert wp.y == 0.0

    def test_mpc_result_default(self):
        """Test MPCResult default values."""
        result = p.MPCResult()
        assert result.total_cost == 0.0
        assert result.path_cost == 0.0
        assert result.collision_cost == 0.0
        assert result.colregs_cost == 0.0

    def test_psbmpc_params_values(self):
        """Test PSBMPCParameters actual default values."""
        params = p.PSBMPCParameters()
        assert params.T == 300.0
        assert params.dt == 1.0
        assert params.cpe_n_samples == 1000
        assert params.n_M == 10
        assert params.n_cbs == 5

    def test_sbmpc_params_values(self):
        """Test SBMPCParameters actual default values."""
        params = p.SBMPCParameters()
        assert params.T == 300.0
        assert params.dt == 1.0
        assert params.n_M == 10


class TestEdgeCaseUtils:
    """Edge case tests for utility functions."""

    def test_normalize_angle_extreme(self):
        """Test normalize_angle with extreme values."""
        assert p.normalize_angle(100 * math.pi) == pytest.approx(0.0)
        assert p.normalize_angle(-100 * math.pi) == pytest.approx(0.0)
        # 1000 mod 2pi ≈ 0.974
        assert p.normalize_angle(1000.0) == pytest.approx(0.974, abs=0.01)

    def test_angle_diff_pi(self):
        """Test angle_diff at pi boundary."""
        diff1 = p.angle_diff(math.pi, -math.pi)
        assert diff1 == pytest.approx(0.0)

    def test_distance_identical_points(self):
        """Test distance_2d with identical points."""
        assert p.distance_2d(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)

    def test_bearing_same_point(self):
        """Test bearing_2d with same points."""
        # atan2(0, 0) = 0 in Python
        assert p.bearing_2d(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)

    def test_clamp_negative_range(self):
        """Test clamp with negative range."""
        assert p.clamp(-5.0, -10.0, 0.0) == pytest.approx(-5.0)
        assert p.clamp(-15.0, -10.0, 0.0) == pytest.approx(-10.0)
        assert p.clamp(5.0, -10.0, 0.0) == pytest.approx(0.0)

    def test_lerp_same_values(self):
        """Test lerp with same start and end."""
        assert p.lerp(5.0, 5.0, 0.5) == pytest.approx(5.0)

    def test_wrap_angle_large_positive(self):
        """Test wrap_angle with large positive value."""
        assert p.wrap_angle(10 * math.pi) == pytest.approx(0.0)

    def test_wrap_angle_large_negative(self):
        """Test wrap_angle with large negative value."""
        assert p.wrap_angle(-10 * math.pi) == pytest.approx(0.0)

    def test_squared_distance_zero(self):
        """Test squared_distance_2d with zero distance."""
        assert p.squared_distance_2d(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_generate_norm_samples_zero_std(self):
        """Test generate_norm_samples with zero std."""
        xs, ys = p.generate_norm_samples(0.0, 0.0, 0.0, 0.0, n_samples=100)
        assert all(x == 0.0 for x in xs)
        assert all(y == 0.0 for y in ys)

    def test_line_segment_intersection_parallel(self):
        """Test line_segment_intersection with parallel lines."""
        intersects, _, _ = p.line_segment_intersection(
            0.0, 0.0, 10.0, 0.0,
            0.0, 1.0, 10.0, 1.0,
        )
        assert intersects is False

    def test_line_segment_intersection_identical_points(self):
        """Test line_segment_intersection with identical points."""
        intersects, _, _ = p.line_segment_intersection(
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
        )
        assert intersects is False
