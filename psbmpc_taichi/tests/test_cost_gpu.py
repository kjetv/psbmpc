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
        result_gpu = cost_eval.calculate_path_cost_gpu(traj_x, traj_y, waypoints)

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
        result_gpu = cost_eval.calculate_path_cost_gpu(traj_x, traj_y, waypoints)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_grounding_cost_gpu_vs_cpu(self):
        """Grounding cost GPU vs CPU comparison."""
        waypoints = [
            p.Waypoint(x=0.0, y=0.0),
            p.Waypoint(x=1000.0, y=0.0),
        ]
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [0.0, 0.0, 0.0]

        # Grounding hazard near the trajectory
        hazards = [
            p.GroundingHazard(
                polygon=[(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)],
            )
        ]

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_grounding_cost(traj_x, traj_y, hazards)
        result_gpu = cost_eval.calculate_grounding_cost_gpu(traj_x, traj_y, hazards)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_grounding_cost_gpu_vs_cpu_no_hazard(self):
        """Grounding cost with no hazards."""
        waypoints = [
            p.Waypoint(x=0.0, y=0.0),
            p.Waypoint(x=1000.0, y=0.0),
        ]
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [0.0, 0.0, 0.0]
        hazards = []

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_grounding_cost(traj_x, traj_y, hazards)
        result_gpu = cost_eval.calculate_grounding_cost_gpu(traj_x, traj_y, hazards)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_deviation_cost_gpu_vs_cpu(self):
        """Deviation cost GPU vs CPU comparison."""
        waypoints = [
            p.Waypoint(x=0.0, y=0.0),
            p.Waypoint(x=1000.0, y=0.0),
        ]
        traj_x = [0.0, 500.0, 1000.0]
        traj_y = [0.0, 0.0, 0.0]

        cost_eval = p.Path_Grounding_Cost()
        result_cpu = cost_eval.calculate_deviation_cost(traj_x, traj_y, waypoints)
        result_gpu = cost_eval.calculate_deviation_cost_gpu(traj_x, traj_y, waypoints)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_dynamic_obstacle_cost_gpu_vs_cpu(self):
        """Dynamic obstacle cost GPU vs CPU comparison."""
        obstacles = [
            p.ObstacleData(
                x=300.0, y=0.0, chi=math.pi, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
            p.ObstacleData(
                x=500.0, y=100.0, chi=math.pi * 0.5, U=3.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
        ]

        cost_eval = p.Dynamic_Obstacle_Cost()
        result_cpu = cost_eval.calculate_dynamic_obstacle_cost(obstacles)
        result_gpu = cost_eval.calculate_dynamic_obstacle_cost_gpu(obstacles)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_colregs_cost_gpu_vs_cpu(self):
        """COLREGS cost GPU vs CPU comparison."""
        obstacles = [
            p.ObstacleData(
                x=300.0, y=0.0, chi=math.pi, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            ),
        ]

        cost_eval = p.COLREGS_Evaluator()
        result_cpu = cost_eval.calculate_colregs_cost(obstacles)
        result_gpu = cost_eval.calculate_colregs_cost_gpu(obstacles)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)

    def test_colregs_cost_gpu_vs_cpu_multiple(self):
        """COLREGS cost with multiple obstacles."""
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
        result_cpu = cost_eval.calculate_colregs_cost(obstacles)
        result_gpu = cost_eval.calculate_colregs_cost_gpu(obstacles)

        assert result_gpu == pytest.approx(result_cpu, rel=1e-3, abs=1e-6)
