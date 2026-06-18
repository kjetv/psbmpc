"""Tests for MPC solvers (PSBMPC_Solver, SBMPC_Solver)."""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestPSBMPCSolver:
    """Tests for the PSBMPC_Solver class."""

    def test_initialization(self, psbmpc_params):
        """Test PSBMPC solver initialization."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        assert solver.params is psbmpc_params
        assert solver.ownship_length == 150.0
        assert solver.ownship_beam == 25.0
        assert solver.last_optimal_offsets_chi == [0.0] * psbmpc_params.n_M
        assert solver.last_optimal_offsets_U == [0.0] * psbmpc_params.n_M

    def test_initialization_with_grounding(self, psbmpc_params, grounding_hazards):
        """Test PSBMPC solver initialization with grounding hazards."""
        solver = p.PSBMPC_Solver(
            psbmpc_params,
            ownship_length=150.0,
            ownship_beam=25.0,
            grounding_hazards=grounding_hazards,
        )

        assert len(solver.grounding_hazards) == 2

    def test_calculate_optimal_offsets_no_obstacle(self, psbmpc_params, kinematic_ship, ship_state4):
        """Test optimal offset calculation with no obstacles."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0), p.Waypoint(x=2000.0, y=1000.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[],
            waypoints=waypoints,
        )

        assert result is not None
        assert isinstance(result, p.MPCResult)
        assert isinstance(result.offset_chi, float)
        assert isinstance(result.offset_U, float)
        assert len(result.traj_x) == 301
        assert len(result.traj_y) == 301
        assert len(result.traj_chi) == 301
        assert len(result.traj_U) == 301

    def test_calculate_optimal_offsets_with_obstacle(self, psbmpc_params, ship_state4, obstacle_near):
        """Test optimal offset calculation with nearby obstacle."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0), p.Waypoint(x=2000.0, y=1000.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[obstacle_near],
            waypoints=waypoints,
        )

        assert result.total_cost >= 0.0
        # Should have some collision and/or COLREGS cost
        assert result.collision_cost >= 0.0
        assert result.colregs_cost >= 0.0

    def test_calculate_optimal_offsets_returns_valid_trajectory(self, psbmpc_params, ship_state4):
        """Test that returned trajectory is physically reasonable."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[],
            waypoints=waypoints,
        )

        # Trajectory should start at current state
        assert result.traj_x[0] == pytest.approx(ship_state4.x, abs=1.0)
        assert result.traj_y[0] == pytest.approx(ship_state4.y, abs=1.0)

        # Trajectory should move forward
        assert result.traj_x[-1] > result.traj_x[0]

    def test_calculate_optimal_offsets_parallel(self, psbmpc_params, ship_state4):
        """Test parallel offset calculation."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result = solver.calculate_optimal_offsets_parallel(
            xs=ship_state4,
            obstacles=[],
            waypoints=waypoints,
        )

        assert result is not None
        assert len(result.traj_x) == 301

    def test_candidate_offsets_setup(self, psbmpc_params):
        """Test candidate offset generation."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        offsets = solver._setup_candidate_offsets()

        assert len(offsets) == psbmpc_params.n_cbs
        assert -0.5 <= min(offsets) <= 0.0
        assert 0.0 <= max(offsets) <= 0.5

    def test_colav_detection_active(self, psbmpc_params, ship_state4, obstacle_near):
        """Test COLAV active detection."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]
        solver.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Get nominal trajectory
        nominal_traj = solver.ownship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )

        # Obstacle should be close enough to activate COLAV
        colav_active = solver._determine_colav_active(
            ship_state4, [obstacle_near], nominal_traj,
        )

        assert colav_active is True

    def test_colav_detection_inactive(self, psbmpc_params, ship_state4):
        """Test COLAV inactive detection."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]
        solver.ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Distant obstacle
        far_obs = p.ObstacleData(
            x=10000.0, y=10000.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=100.0,
        )

        nominal_traj = solver.ownship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )

        colav_active = solver._determine_colav_active(
            ship_state4, [far_obs], nominal_traj,
        )

        assert colav_active is False

    def test_multiple_iterations(self, psbmpc_params, ship_state4, obstacle_near):
        """Test that multiple MPC iterations produce consistent results."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result1 = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[obstacle_near],
            waypoints=waypoints,
        )

        result2 = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[obstacle_near],
            waypoints=waypoints,
        )

        # Both should produce valid results
        assert result1.total_cost >= 0.0
        assert result2.total_cost >= 0.0

    def test_result_structure(self, psbmpc_params, ship_state4):
        """Test that MPCResult has all required fields."""
        solver = p.PSBMPC_Solver(psbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[],
            waypoints=waypoints,
        )

        assert hasattr(result, 'offset_chi')
        assert hasattr(result, 'offset_U')
        assert hasattr(result, 'traj_x')
        assert hasattr(result, 'traj_y')
        assert hasattr(result, 'traj_chi')
        assert hasattr(result, 'traj_U')
        assert hasattr(result, 'total_cost')
        assert hasattr(result, 'path_cost')
        assert hasattr(result, 'collision_cost')
        assert hasattr(result, 'colregs_cost')


class TestSBMPCSolver:
    """Tests for the SBMPC_Solver class."""

    def test_initialization(self, sbmpc_params):
        """Test SBMPC solver initialization."""
        solver = p.SBMPC_Solver(sbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        assert solver.params is sbmpc_params
        assert solver.ownship_length == 150.0
        assert solver.ownship_beam == 25.0

    def test_calculate_optimal_offsets(self, sbmpc_params, ship_state4):
        """Test optimal offset calculation for SBMPC."""
        solver = p.SBMPC_Solver(sbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[],
            waypoints=waypoints,
        )

        assert result is not None
        assert isinstance(result, p.MPCResult)
        assert len(result.traj_x) == 301

    def test_calculate_optimal_offsets_with_obstacle(self, sbmpc_params, ship_state4, obstacle_near):
        """Test SBMPC with obstacle."""
        solver = p.SBMPC_Solver(sbmpc_params, ownship_length=150.0, ownship_beam=25.0)

        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        result = solver.calculate_optimal_offsets(
            xs=ship_state4,
            obstacles=[obstacle_near],
            waypoints=waypoints,
        )

        assert result.total_cost >= 0.0
        assert len(result.traj_x) == 301
