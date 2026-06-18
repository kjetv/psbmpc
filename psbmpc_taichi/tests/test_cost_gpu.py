"""GPU vs CPU cost function accuracy tests.

Verifies that GPU-accelerated cost functions match CPU results
within float32 tolerance (rtol=1e-3).
"""
import math

import pytest

import psbmpc_taichi as p


class TestCostGPUAccuracy:
    """Compare GPU cost results against CPU baseline."""

    def test_path_cost_gpu_vs_cpu(self):
        """Path cost GPU vs CPU comparison."""
        waypoints = [
            p.Waypoint(x=0.0, y=0.0),
            p.Waypoint(x=1000.0, y=0.0),
            p.Waypoint(x=2000.0, y=0.0),
        ]
        # Trajectory along the path
        traj_x = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_path_cost(traj_x, traj_y, waypoints)
        result_gpu = cost_eval._calculate_path_cost_gpu(traj_x, traj_y, waypoints)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_path_cost_gpu_vs_cpu_offset_trajectory(self):
        """Path cost with offset trajectory."""
        waypoints = [
            p.Waypoint(x=0.0, y=0.0),
            p.Waypoint(x=1000.0, y=0.0),
        ]
        # Trajectory offset from path
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [100.0, 100.0, 100.0]  # 100m offset

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_path_cost(traj_x, traj_y, waypoints)
        result_gpu = cost_eval._calculate_path_cost_gpu(traj_x, traj_y, waypoints)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_grounding_cost_gpu_vs_cpu(self):
        """Grounding cost GPU vs CPU comparison."""
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [0.0, 0.0, 0.0]
        traj_chi = [0.0, 0.0, 0.0]

        # Grounding hazard near the trajectory (as lists of [x, y, chi, length, beam])
        hazards = [
            [0.0, 0.0, 0.0, 100.0, 100.0],  # x, y, chi, length, beam
        ]

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_grounding_cost(traj_x, traj_y, traj_chi, hazards, 150.0, 25.0)
        result_gpu = cost_eval._calculate_grounding_cost_gpu(traj_x, traj_y, traj_chi, hazards, 150.0, 25.0)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_grounding_cost_gpu_vs_cpu_no_hazard(self):
        """Grounding cost with no hazards."""
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [0.0, 0.0, 0.0]
        traj_chi = [0.0, 0.0, 0.0]
        hazards = []

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_grounding_cost(traj_x, traj_y, traj_chi, hazards, 150.0, 25.0)
        result_gpu = cost_eval._calculate_grounding_cost_gpu(traj_x, traj_y, traj_chi, hazards, 150.0, 25.0)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_deviation_cost_gpu_vs_cpu(self):
        """Deviation cost GPU vs CPU comparison."""
        # Deviation cost uses heading and surge offsets, not waypoints
        n_M = 5
        offsets_chi = [0.0, 0.1, -0.1, 0.05, -0.05]
        offsets_U = [0.0, 0.5, -0.5, 0.3, -0.3]
        last_optimal_chi = [0.0] * n_M
        last_optimal_U = [0.0] * n_M

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_deviation_cost(offsets_chi, offsets_U, last_optimal_chi, last_optimal_U)
        result_gpu = cost_eval._calculate_deviation_cost_gpu(offsets_chi, offsets_U, last_optimal_chi, last_optimal_U)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_dynamic_obstacle_cost_gpu_vs_cpu(self):
        """Dynamic obstacle cost GPU vs CPU comparison."""
        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        traj_chi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        obstacle = p.ObstacleData(
            x=300.0, y=0.0, chi=math.pi, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cost_eval = p.Dynamic_Obstacle_Cost()
        result_cpu = cost_eval.calculate_dynamic_obstacle_cost(traj_x, traj_y, traj_chi, obstacle)
        result_gpu = cost_eval._calculate_dynamic_obstacle_cost_gpu(traj_x, traj_y, traj_chi, obstacle)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_colregs_cost_gpu_vs_cpu(self):
        """COLREGS cost GPU vs CPU comparison."""
        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        obstacle = p.ObstacleData(
            x=300.0, y=0.0, chi=math.pi, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cost_eval = p.COLREGS_Evaluator()
        # situation: 0=none, 1=crossing_port, 2=head-on, 3=crossing_stb, 4=overtaking
        situation = "head-on"
        result_cpu = cost_eval.calculate_colregs_cost(situation, traj_x, traj_y, obstacle)
        result_gpu = cost_eval._calculate_colregs_cost_gpu(situation, traj_x, traj_y, obstacle)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_colregs_cost_gpu_vs_cpu_multiple(self):
        """COLREGS cost with multiple obstacles - test each separately."""
        traj_x = [0.0, 100.0, 200.0, 300.0, 400.0]
        traj_y = [0.0, 0.0, 0.0, 0.0, 0.0]

        obstacles = [
            p.ObstacleData(
                x=300.0, y=0.0, chi=math.pi, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
            p.ObstacleData(
                x=0.0, y=300.0, chi=math.pi * 0.5, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
            p.ObstacleData(
                x=-300.0, y=0.0, chi=0.0, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
        ]

        cost_eval = p.COLREGS_Evaluator()
        situation = "head-on"
        total_cpu = 0.0
        total_gpu = 0.0
        for obs in obstacles:
            total_cpu += cost_eval.calculate_colregs_cost(situation, traj_x, traj_y, obs)
            total_gpu += cost_eval._calculate_colregs_cost_gpu(situation, traj_x, traj_y, obs)

        assert total_gpu == pytest.approx(total_cpu, rel=1e-3, abs=1e-6)
