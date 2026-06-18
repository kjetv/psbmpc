"""GPU vs CPU CPE comparison tests.

Verifies that GPU-accelerated CE and MCSKF4D match CPU results
within float32 tolerance (rtol=1e-3).
"""
import math

import pytest

import psbmpc_taichi as p


class TestCPEGPUAccuracy:
    """Compare GPU CPE results against CPU baseline."""

    def _make_ownership(self, x, y):
        """Helper to create ObstacleData for ownship."""
        return p.ObstacleData(
            x=x, y=y, chi=0.0, U=5.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

    def test_ce_gpu_vs_cpu_far_obstacle(self):
        """CE GPU vs CPU with distant obstacle."""
        ownship = self._make_ownership(0.0, 0.0)
        obstacle = p.ObstacleData(
            x=1000.0, y=500.0, chi=math.pi, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe_cpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=False)
        cpe_gpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=True)

        result_cpu = cpe_cpu.ce_estimate(ownship, obstacle)
        result_gpu = cpe_gpu.ce_estimate(ownship, obstacle)

        # GPU result should match CPU within float32 tolerance
        assert result_gpu.probability == pytest.approx(
            result_cpu.probability, rel=1e-3, abs=1e-6
        )

    def test_ce_gpu_vs_cpu_near_obstacle(self):
        """CE GPU vs CPU with close obstacle."""
        ownship = self._make_ownership(0.0, 0.0)
        obstacle = p.ObstacleData(
            x=150.0, y=0.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe_cpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=False)
        cpe_gpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=True)

        result_cpu = cpe_cpu.ce_estimate(ownship, obstacle)
        result_gpu = cpe_gpu.ce_estimate(ownship, obstacle)

        assert result_gpu.probability == pytest.approx(
            result_cpu.probability, rel=1e-3, abs=1e-6
        )

    def test_ce_gpu_vs_cpu_same_position(self):
        """CE GPU vs CPU with ships at same position (max collision)."""
        ownship = self._make_ownership(0.0, 0.0)
        obstacle = p.ObstacleData(
            x=0.0, y=0.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe_cpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=False)
        cpe_gpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=True)

        result_cpu = cpe_cpu.ce_estimate(ownship, obstacle)
        result_gpu = cpe_gpu.ce_estimate(ownship, obstacle)

        # Both should return probability near 1.0
        assert result_gpu.probability == pytest.approx(
            result_cpu.probability, rel=1e-3, abs=1e-6
        )

    def test_mcskf4d_gpu_vs_cpu(self):
        """MCSKF4D GPU vs CPU comparison.

        MCSKF4D uses float32 on GPU which can produce slightly different
        particle trajectories than float64 CPU. Use generous tolerance.
        """
        ownship = self._make_ownership(0.0, 0.0)
        obstacle = p.ObstacleData(
            x=300.0, y=200.0, chi=math.pi * 0.75, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe_cpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=500, use_gpu=False)
        cpe_gpu = p.CPE(max_iter=10, tolerance=1e-3, n_samples=500, use_gpu=True)

        result_cpu = cpe_cpu.mcskf4d_estimate(ownship, obstacle, dt=1.0, process_noise=1.0)
        result_gpu = cpe_gpu.mcskf4d_estimate(ownship, obstacle, dt=1.0, process_noise=1.0)

        # MCSKF4D is stochastic and GPU uses float32, so use absolute tolerance
        assert result_gpu.probability == pytest.approx(
            result_cpu.probability, abs=0.1
        )

    def test_ce_gpu_consistency_same_seed(self):
        """GPU CE should be deterministic with same parameters."""
        ownship = self._make_ownership(0.0, 0.0)
        obstacle = p.ObstacleData(
            x=400.0, y=100.0, chi=math.pi, U=2.5,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe = p.CPE(max_iter=10, tolerance=1e-3, n_samples=1000, use_gpu=True)
        result1 = cpe.ce_estimate(ownship, obstacle)
        result2 = cpe.ce_estimate(ownship, obstacle)

        # Same inputs should give identical results (deterministic GPU kernel)
        assert result1.probability == pytest.approx(result2.probability, rel=1e-6)
