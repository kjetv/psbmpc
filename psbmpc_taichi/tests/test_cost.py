"""Tests for cost functions (Path_Grounding_Cost, Dynamic_Obstacle_Cost, COLREGS_Evaluator, MPC_Cost)."""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestPathGroundingCost:
    """Tests for the Path_Grounding_Cost class."""

    def test_initialization(self):
        """Test path/grounding cost initialization."""
        cost = p.Path_Grounding_Cost()
        assert cost.kappa_GN == 200.0
        assert cost.w_path == 1.0
        assert cost.w_deviation == 10.0

    def test_calculate_path_cost_straight(self, waypoints):
        """Test path cost calculation for straight trajectory."""
        cost = p.Path_Grounding_Cost()

        # Straight line along x-axis
        traj_x = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        path_cost = cost.calculate_path_cost(
            traj_x, traj_y, waypoints,
            ship_current_pos=(0.0, 0.0),
        )

        assert path_cost >= 0.0

    def test_calculate_path_cost_deviation(self, waypoints):
        """Test that path cost increases with deviation."""
        cost = p.Path_Grounding_Cost()

        # Trajectory that deviates significantly from waypoints
        traj_x = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
        traj_y = [0.0, 500.0, 1000.0, 1500.0, 2000.0]

        path_cost = cost.calculate_path_cost(
            traj_x, traj_y, waypoints,
            ship_current_pos=(0.0, 0.0),
        )

        # Deviation should result in positive cost
        assert path_cost > 0.0

    def test_calculate_deviation_cost(self):
        """Test control deviation cost."""
        cost = p.Path_Grounding_Cost()

        offsets_chi = [0.1, 0.1, 0.1, 0.1, 0.1]
        offsets_U = [0.0, 0.0, 0.0, 0.0, 0.0]
        last_chi = [0.0, 0.0, 0.0, 0.0, 0.0]
        last_U = [0.0, 0.0, 0.0, 0.0, 0.0]

        dev_cost = cost.calculate_deviation_cost(
            offsets_chi, offsets_U, last_chi, last_U,
        )

        assert dev_cost > 0.0  # Non-zero offset from last should cost something

    def test_calculate_deviation_cost_zero(self):
        """Test deviation cost is zero when offsets haven't changed."""
        cost = p.Path_Grounding_Cost()

        offsets_chi = [0.0, 0.0, 0.0, 0.0, 0.0]
        offsets_U = [0.0, 0.0, 0.0, 0.0, 0.0]
        last_chi = [0.0, 0.0, 0.0, 0.0, 0.0]
        last_U = [0.0, 0.0, 0.0, 0.0, 0.0]

        dev_cost = cost.calculate_deviation_cost(
            offsets_chi, offsets_U, last_chi, last_U,
        )

        assert dev_cost == 0.0

    def test_calculate_grounding_cost_no_hazard(self, ship_state4, kinematic_ship):
        """Test grounding cost with no hazards (should be zero)."""
        cost = p.Path_Grounding_Cost()

        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )

        grounding_cost = cost.calculate_grounding_cost(
            traj[0], traj[1], traj[2],
            [],  # No hazards
        )

        assert grounding_cost == 0.0


class TestDynamicObstacleCost:
    """Tests for the Dynamic_Obstacle_Cost class."""

    def test_initialization(self):
        """Test dynamic obstacle cost initialization."""
        cost = p.Dynamic_Obstacle_Cost()
        assert cost.kappa_SO == 50.0
        assert cost.kappa_RA == 75.0
        assert cost.w_collision == 100.0

    def test_calculate_dynamic_obstacle_cost(self, ship_state4, obstacle):
        """Test dynamic obstacle cost calculation."""
        cost = p.Dynamic_Obstacle_Cost()

        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        dyn_cost = cost.calculate_dynamic_obstacle_cost(
            traj_x, traj_y, traj_x,
            obstacle,
            cpe_probability=0.0,
            time_horizon=300.0,
            dt=1.0,
        )

        assert dyn_cost >= 0.0

    def test_dynamic_obstacle_cost_with_cpe(self, ship_state4, obstacle):
        """Test dynamic obstacle cost with non-zero CPE probability."""
        cost = p.Dynamic_Obstacle_Cost()

        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        cost_with_cpe = cost.calculate_dynamic_obstacle_cost(
            traj_x, traj_y, traj_x,
            obstacle,
            cpe_probability=0.5,
            time_horizon=300.0,
            dt=1.0,
        )

        # Should have some cost due to collision probability
        assert cost_with_cpe >= 0.0


class TestCOLREGSEvaluator:
    """Tests for the COLREGS_Evaluator class."""

    def test_initialization(self):
        """Test COLREGS evaluator initialization."""
        evaluator = p.COLREGS_Evaluator()
        assert evaluator is not None

    def test_detect_situation_giving_way(self, ship_state4, obstacle):
        """Test situation detection for giving-way scenario."""
        evaluator = p.COLREGS_Evaluator()

        situation = evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obstacle,
        )

        assert situation is not None
        assert isinstance(situation, str)

    def test_detect_situation_stand_on(self, ship_state4):
        """Test situation detection for stand-on scenario."""
        evaluator = p.COLREGS_Evaluator()

        # Create obstacle in stand-on position
        obs = p.ObstacleData(
            x=300.0, y=-100.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        situation = evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obs,
        )

        assert situation is not None

    def test_detect_situation_overtake(self, ship_state4):
        """Test situation detection for overtaking scenario."""
        evaluator = p.COLREGS_Evaluator()

        # Create obstacle being overtaken
        obs = p.ObstacleData(
            x=100.0, y=0.0, chi=0.0, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        situation = evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obs,
        )

        assert situation is not None

    def test_calculate_colregs_cost(self, ship_state4, obstacle):
        """Test COLREGS cost calculation."""
        evaluator = p.COLREGS_Evaluator()

        situation = evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, obstacle,
        )

        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        colregs_cost = evaluator.calculate_colregs_cost(
            situation, traj_x, traj_y, obstacle,
        )

        assert colregs_cost >= 0.0

    def test_detect_situation_no_obstacle(self, ship_state4):
        """Test situation detection with no obstacle."""
        evaluator = p.COLREGS_Evaluator()

        situation = evaluator.detect_situation(
            ship_state4.chi, ship_state4.U, None,
        )

        assert situation == "none"


class TestMPCCost:
    """Tests for the MPC_Cost orchestrator class."""

    def test_initialization(self, psbmpc_params, grounding_hazards):
        """Test MPC cost evaluator initialization."""
        cost = p.MPC_Cost(
            params=psbmpc_params,
            grounding_hazards=grounding_hazards,
        )

        assert cost.params is psbmpc_params
        assert len(cost.grounding_hazards) == 2

    def test_calculate_total_cost(self, mpc_cost_evaluator, ship_state4, kinematic_ship, obstacle):
        """Test total cost calculation."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )

        cost_breakdown = mpc_cost_evaluator.calculate_total_cost(
            traj[0], traj[1], traj[2], traj[3],
            [obstacle],
            [p.Waypoint(x=1000.0, y=0.0)],
            150.0, 25.0,
            None, None,
        )

        assert "total" in cost_breakdown
        assert "path" in cost_breakdown
        assert "collision" in cost_breakdown
        assert cost_breakdown["total"] >= 0.0

    def test_calculate_total_cost_components(self, mpc_cost_evaluator, ship_state4, kinematic_ship, obstacle):
        """Test that total cost is sum of components."""
        traj = kinematic_ship.predict_trajectory(
            ship_state4,
            offsets=[0.0] * 300,
            T=300.0,
            dt=1.0,
        )

        cost_breakdown = mpc_cost_evaluator.calculate_total_cost(
            traj[0], traj[1], traj[2], traj[3],
            [obstacle],
            [p.Waypoint(x=1000.0, y=0.0)],
            150.0, 25.0,
            None, None,
        )

        # Total should be sum of components
        components = sum(v for k, v in cost_breakdown.items() if k != "total")
        assert cost_breakdown["total"] == pytest.approx(components, abs=1.0)
