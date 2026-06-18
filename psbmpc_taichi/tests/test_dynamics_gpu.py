"""GPU vs CPU trajectory prediction accuracy tests.

Verifies that GPU-accelerated trajectory prediction matches CPU results
within float32 tolerance (rtol=1e-3).

Uses predict_trajectory_batch_gpu() which accepts a flat list of constant
offsets (one per candidate) and returns arrays of shape (n_candidates, n_steps+1).
"""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestDynamicsGPUAccuracy:
    """Compare GPU trajectory prediction against CPU baseline."""

    def _cpu_single_candidate(self, ownship, xs, offset, n_steps):
        """Run CPU prediction for a single candidate with constant offset."""
        # predict_trajectory expects a flat list of offsets (one per time step)
        offsets_flat = [offset] * n_steps
        result = ownship.predict_trajectory(xs, offsets_flat, T=float(n_steps), dt=1.0, method="linear")
        return result[0], result[1]  # (traj_x, traj_y) for first candidate

    def test_trajectory_prediction_gpu_vs_cpu_straight(self):
        """GPU vs CPU trajectory prediction for straight path using batch method."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
            p.Waypoint(x=2000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        n_candidates = 2
        n_steps = 100
        offsets = [0.0, 0.0]

        # CPU: predict each candidate individually
        cpu_results = []
        for off in offsets:
            cpu_results.append(self._cpu_single_candidate(ownship, xs, off, n_steps))

        # GPU: batch prediction (constant offsets)
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=100.0, dt=1.0, method="linear"
        )

        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_prediction_gpu_vs_cpu_turning(self):
        """GPU vs CPU trajectory prediction for turning path using batch method."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=500.0, y=500.0),
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        offsets = [0.0, 0.0]
        n_candidates = 2
        n_steps = 100

        # CPU
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in offsets]

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=100.0, dt=1.0, method="linear"
        )

        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_prediction_gpu_vs_cpu_with_offset(self):
        """GPU vs CPU trajectory prediction with lateral offset using batch method."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Two candidates: one straight, one with heading offset
        offsets = [0.0, math.pi / 8]  # 0 and 22.5 degrees
        n_candidates = 2
        n_steps = 100

        # CPU
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in offsets]

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=100.0, dt=1.0, method="linear"
        )

        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_batch_gpu_vs_cpu(self):
        """GPU batch trajectory prediction vs CPU loop."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Multiple candidate offsets (constant per candidate)
        candidate_offsets = [0.0, math.pi / 16, -math.pi / 16]
        n_candidates = len(candidate_offsets)
        n_steps = 100

        # CPU: predict each candidate
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in candidate_offsets]

        # GPU: predict all candidates at once
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, candidate_offsets, T=100.0, dt=1.0, method="linear"
        )

        # Verify shapes
        assert traj_x_gpu.shape == (n_candidates, n_steps + 1)
        assert traj_y_gpu.shape == (n_candidates, n_steps + 1)

        # Verify each candidate matches
        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_prediction_gpu_short_horizon(self):
        """GPU vs CPU with short prediction horizon."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        offsets = [0.0, 0.0]
        n_candidates = 2
        n_steps = 10  # T=10, dt=1

        # CPU
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in offsets]

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=10.0, dt=1.0, method="linear"
        )

        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_prediction_gpu_long_horizon(self):
        """GPU vs CPU with long prediction horizon."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=5000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        offsets = [0.0, 0.0]
        n_candidates = 2
        n_steps = 500  # T=500, dt=1

        # CPU
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in offsets]

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=500.0, dt=1.0, method="linear"
        )

        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_batch_single_candidate(self):
        """GPU batch prediction with single candidate."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        offsets = [0.0]  # Single candidate
        n_steps = 100

        # CPU
        traj_x_cpu, traj_y_cpu = self._cpu_single_candidate(ownship, xs, 0.0, n_steps)

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=100.0, dt=1.0, method="linear"
        )

        assert traj_x_gpu.shape == (1, n_steps + 1)
        assert traj_x_gpu[0] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
        assert traj_y_gpu[0] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)

    def test_trajectory_batch_different_offsets(self):
        """GPU batch prediction with varied offsets."""
        ownship = p.Kinematic_Ship()
        xs = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        waypoints = [
            p.Waypoint(x=1000.0, y=0.0),
        ]
        ownship.set_waypoints([(wp.x, wp.y) for wp in waypoints])

        # Diverse set of offsets
        offsets = [0.0, math.pi / 4, -math.pi / 4, math.pi / 8, -math.pi / 8]
        n_candidates = len(offsets)
        n_steps = 50

        # CPU
        cpu_results = [self._cpu_single_candidate(ownship, xs, off, n_steps) for off in offsets]

        # GPU
        traj_x_gpu, traj_y_gpu, _, _ = ownship.predict_trajectory_batch_gpu(
            xs, offsets, T=50.0, dt=1.0, method="linear"
        )

        # Verify shapes
        assert traj_x_gpu.shape == (n_candidates, n_steps + 1)
        assert traj_y_gpu.shape == (n_candidates, n_steps + 1)

        # Verify each candidate
        for i in range(n_candidates):
            traj_x_cpu, traj_y_cpu = cpu_results[i]
            assert traj_x_gpu[i] == pytest.approx(traj_x_cpu, rel=1e-3, abs=1e-6)
            assert traj_y_gpu[i] == pytest.approx(traj_y_cpu, rel=1e-3, abs=1e-6)
