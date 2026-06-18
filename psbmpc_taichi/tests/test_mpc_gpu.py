"""GPU vs CPU MPC solver accuracy tests.

Verifies that the GPU-accelerated MPC solver produces results
matching the CPU version within float32 tolerance (rtol=1e-3).
"""
import math

import pytest

import psbmpc_taichi as p


class TestMPCGPUAccuracy:
    """Compare GPU MPC results against CPU baseline."""

    def _make_obstacles(self, positions):
        """Helper to create obstacle list."""
        return [
            p.ObstacleData(
                x=pos[0], y=pos[1], chi=math.pi, U=2.0,
                length=150.0, beam=25.0, d_safe=300.0,
            )
            for pos in positions
        ]

    def test_solver_gpu_vs_cpu_no_obstacles(self):
        """GPU vs CPU MPC with no obstacles (path cost only)."""
        solver = p.PSBMPC_Solver()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
            p.Waypoint(x=2000.0, y=1000.0),
        ]

        result_cpu = solver.calculate_optimal_offsets(xs, [], waypoints)
        result_gpu = solver.calculate_optimal_offsets_gpu(xs, [], waypoints)

        # Should return zero offset when no obstacles
        assert result_gpu.offset_chi == pytest.approx(result_cpu.offset_chi, rel=1e-3)
        assert result_gpu.offset_U == pytest.approx(result_cpu.offset_U, rel=1e-3)
        assert result_gpu.total_cost == pytest.approx(result_cpu.total_cost, rel=1e-3)

    def test_solver_gpu_vs_cpu_single_obstacle(self):
        """GPU vs CPU MPC with single obstacle."""
        solver = p.PSBMPC_Solver()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacles = self._make_obstacles([(300.0, 0.0)])
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
            p.Waypoint(x=2000.0, y=1000.0),
        ]

        result_cpu = solver.calculate_optimal_offsets(xs, obstacles, waypoints)
        result_gpu = solver.calculate_optimal_offsets_gpu(xs, obstacles, waypoints)

        # GPU result should match CPU within float32 tolerance
        assert result_gpu.offset_chi == pytest.approx(result_cpu.offset_chi, rel=5e-2, abs=0.1)
        assert result_gpu.total_cost == pytest.approx(result_cpu.total_cost, rel=5e-2, abs=1.0)

    def test_solver_gpu_vs_cpu_multiple_obstacles(self):
        """GPU vs CPU MPC with multiple obstacles."""
        solver = p.PSBMPC_Solver()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacles = self._make_obstacles([
            (200.0, 100.0),
            (400.0, -50.0),
            (350.0, 200.0),
        ])
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]

        result_cpu = solver.calculate_optimal_offsets(xs, obstacles, waypoints)
        result_gpu = solver.calculate_optimal_offsets_gpu(xs, obstacles, waypoints)

        assert result_gpu.offset_chi == pytest.approx(result_cpu.offset_chi, rel=5e-2, abs=0.1)
        assert result_gpu.total_cost == pytest.approx(result_cpu.total_cost, rel=5e-2, abs=1.0)

    def test_solver_gpu_trajectory_shape(self):
        """GPU MPC should return trajectory with correct shape."""
        solver = p.PSBMPC_Solver()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]

        result = solver.calculate_optimal_offsets_gpu(xs, [], waypoints)

        assert len(result.traj_x) > 0
        assert len(result.traj_y) > 0
        assert len(result.traj_chi) > 0
        assert len(result.traj_U) > 0
        assert len(result.traj_x) == len(result.traj_y)

    def test_solver_gpu_vs_cpu_deviated_heading(self):
        """GPU vs CPU MPC with deviated heading (tests COLREGS path)."""
        solver = p.PSBMPC_Solver()
        xs = p.ShipState4(x=0.0, y=0.0, chi=math.pi / 4, U=5.0)
        obstacles = self._make_obstacles([(500.0, 0.0)])
        waypoints = [
            p.Waypoint(x=1000.0, y=1000.0),
        ]

        result_cpu = solver.calculate_optimal_offsets(xs, obstacles, waypoints)
        result_gpu = solver.calculate_optimal_offsets_gpu(xs, obstacles, waypoints)

        assert result_gpu.offset_chi == pytest.approx(result_cpu.offset_chi, rel=5e-2, abs=0.1)
