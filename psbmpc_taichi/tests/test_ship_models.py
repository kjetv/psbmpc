"""Tests for ship models (Kinematic_Ship, Obstacle_Ship)."""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestKinematicShip:
    """Tests for the Kinematic_Ship model."""

    def test_initialization(self, kinematic_ship):
        """Test ship model initialization."""
        assert kinematic_ship.length == 150.0
        assert kinematic_ship.beam == 25.0
        assert kinematic_ship.los_range == 500.0
        assert kinematic_ship.max_rudder == math.pi / 6
        assert kinematic_ship.time_constant == 20.0

    def test_set_waypoints_tuple(self, kinematic_ship, ship_state4):
        """Test setting waypoints as tuples."""
        kinematic_ship.set_waypoints([(1000.0, 0.0), (2000.0, 1000.0)])
        assert len(kinematic_ship.waypoints) == 2
        assert kinematic_ship.waypoints[0].x == pytest.approx(1000.0)
        assert kinematic_ship.waypoints[0].y == pytest.approx(0.0)
        assert kinematic_ship.waypoints[1].x == pytest.approx(2000.0)
        assert kinematic_ship.waypoints[1].y == pytest.approx(1000.0)

    def test_set_waypoint_objects(self, kinematic_ship):
        """Test setting waypoints as Waypoint objects."""
        kinematic_ship.set_waypoints([
            p.Waypoint(x=1000.0, y=0.0),
            p.Waypoint(x=2000.0, y=1000.0),
        ])
        assert len(kinematic_ship.waypoints) == 2

    def test_linear_prediction_straight(self, kinematic_ship, ship_state4):
        """Test linear prediction with zero heading offset (straight line)."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
            method="linear",
        )
        traj_x, traj_y, traj_chi, traj_U = traj

        assert len(traj_x) == 301  # 0 to 300 inclusive
        assert len(traj_y) == 301
        assert len(traj_chi) == 301
        assert len(traj_U) == 301

        # With zero offset, ship should travel in straight line at initial heading
        # chi=0 means traveling along x-axis
        assert traj_chi[0] == pytest.approx(0.0, abs=0.1)
        assert traj_U[0] == pytest.approx(5.0, abs=0.1)

        # Final position should be approximately (1500, 0) for U=5, T=300
        assert traj_x[-1] == pytest.approx(1500.0, abs=50.0)
        assert abs(traj_y[-1]) < 50.0  # Small y deviation is ok

    def test_erks_prediction(self, kinematic_ship, ship_state4):
        """Test ERK1 (Runge-Kutta) prediction."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
            method="erks",
        )
        traj_x, traj_y, traj_chi, traj_U = traj

        assert len(traj_x) == 301
        # ERK1 should give similar results to linear for small time steps
        assert traj_x[-1] > 0  # Ship should move forward

    def test_trajectory_with_offset(self, kinematic_ship, ship_state4):
        """Test trajectory prediction with heading offset."""
        # Apply a constant positive heading offset
        offset = 0.1  # ~5.7 degrees
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[offset] * 300,
            T=300.0,
            dt=1.0,
            method="linear",
        )
        traj_x, traj_y, traj_chi, traj_U = traj

        # Ship should curve upward (positive y) with positive heading offset
        assert traj_y[-1] > traj_y[0]

    def test_los_waypoint_following(self, kinematic_ship, ship_state4):
        """Test that ship follows waypoints via LOS guidance."""
        kinematic_ship.set_waypoints([(1000.0, 0.0), (2000.0, 0.0)])
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
            method="linear",
        )
        traj_x, traj_y, traj_chi, traj_U = traj

        # Ship should move toward first waypoint
        assert traj_x[-1] > traj_x[0]

    def test_predict_trajectory_api(self, kinematic_ship, ship_state4):
        """Test the convenience predict_trajectory method."""
        traj = kinematic_ship.predict_trajectory(
            xs=ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )
        assert traj is not None
        assert len(traj[0]) == 301


class TestKineticShip:
    """Tests for the Kinetic_Ship model."""

    def test_initialization(self):
        """Test kinetic ship initialization."""
        ship = p.Kinetic_Ship(length=150.0, beam=25.0)
        assert ship.length == 150.0
        assert ship.beam == 25.0

    def test_state_update(self):
        """Test state update with 6DOF dynamics."""
        ship = p.Kinetic_Ship(length=150.0, beam=25.0)
        state6 = p.ShipState6(x=0.0, y=0.0, psi=0.0, u=5.0, v=0.0, r=0.0)
        # Just verify it doesn't crash
        ship.predict(state6, dt=1.0)


class TestObstacleShip:
    """Tests for the Obstacle_Ship model."""

    def test_initialization(self):
        """Test obstacle ship initialization."""
        obs = p.Obstacle_Ship(length=100.0, beam=20.0)
        assert obs.length == 100.0
        assert obs.beam == 20.0

    def test_prediction(self):
        """Test obstacle ship trajectory prediction."""
        obs = p.Obstacle_Ship(length=100.0, beam=20.0)
        state = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=3.0)
        obs.set_waypoints([(1000.0, 0.0)])
        traj = obs.predict(state, T=100.0, dt=1.0)
        assert traj is not None
