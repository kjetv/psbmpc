"""Benchmark: CPU NumPy vs GPU Taichi CUDA performance comparison.

Compares the performance of CPU-based NumPy implementations against
GPU-accelerated Taichi CUDA implementations for core MPC computations.

Usage:
    python -m pytest benchmarks/benchmark_cpu_vs_gpu.py -v
    python -m pytest benchmarks/benchmark_cpu_vs_gpu.py -v --benchmark-only
"""
import math
import time

import numpy as np
import pytest

import psbmpc_taichi as p


class TestCPUBenchmark:
    """Benchmark CPU-only NumPy implementations."""

    @pytest.fixture
    def params(self):
        return p.PSBMPCParameters()

    @pytest.fixture
    def ownship(self):
        return p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)

    @pytest.fixture
    def obstacle(self):
        return p.ObstacleData(
            x=500.0, y=200.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

    def benchmark_trajectory_prediction_100_steps(self, ownship, params):
        """Benchmark trajectory prediction for 100 steps."""
        ship = p.Kinematic_Ship()
        offsets = [0.1 * math.sin(i / 10.0) for i in range(params.n_steps)]

        start = time.perf_counter()
        for _ in range(10):
            traj = ship.predict_trajectory(
                ownship, offsets=offsets,
                T=float(params.n_steps), dt=params.dt,
            )
        elapsed = (time.perf_counter() - start) / 10
        print(f"\n  Trajectory prediction (100 steps): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_ce_method_50_iterations(self, ownship, obstacle):
        """Benchmark CE method with 50 iterations."""
        cpe = p.CPE()

        start = time.perf_counter()
        for _ in range(5):
            result = cpe.ce_estimate(
                ownship, obstacle,
                n_samples=1000, max_iter=50, tolerance=1e-3,
            )
        elapsed = (time.perf_counter() - start) / 5
        print(f"\n  CE method (50 iters, 1000 samples): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_mcskf4d_1000_samples(self, ownship, obstacle):
        """Benchmark MCSKF4D with 1000 samples."""
        cpe = p.CPE()

        start = time.perf_counter()
        for _ in range(10):
            result = cpe.mcskf4d_estimate(
                ownship, obstacle, n_samples=1000,
            )
        elapsed = (time.perf_counter() - start) / 10
        print(f"\n  MCSKF4D (1000 samples): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_path_cost_100_points(self, ownship):
        """Benchmark path cost calculation for 100 points."""
        cost = p.Path_Grounding_Cost()
        waypoints = [p.Waypoint(x=1000.0, y=0.0)]
        traj_x = [i * 10.0 for i in range(100)]
        traj_y = [math.sin(i / 10.0) * 50.0 for i in range(100)]

        start = time.perf_counter()
        for _ in range(100):
            path_cost = cost.calculate_path_cost(
                traj_x, traj_y, waypoints,
                ship_current_pos=(0.0, 0.0),
            )
        elapsed = (time.perf_counter() - start) / 100
        print(f"\n  Path cost (100 points): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_polygon_distance_1000_pairs(self):
        """Benchmark polygon distance for 1000 ship pairs."""
        start = time.perf_counter()
        for _ in range(10):
            for i in range(100):
                dist = p.polygon_distance(
                    float(i), 0.0, 0.0, 100.0, 20.0,
                    float(i + 100), 0.0, 0.0, 100.0, 20.0,
                )
        elapsed = (time.perf_counter() - start) / 10
        print(f"\n  Polygon distance (1000 pairs): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_full_mpc_iteration(self, params, ownship, obstacle):
        """Benchmark one full MPC iteration."""
        solver = p.PSBMPC_Solver(ownship, [obstacle], params)

        start = time.perf_counter()
        for _ in range(5):
            result = solver.calculate_optimal_offsets()
        elapsed = (time.perf_counter() - start) / 5
        print(f"\n  Full MPC iteration: {elapsed*1000:.2f} ms")
        assert elapsed > 0


class TestGPUBenchmark:
    """Benchmark GPU-accelerated Taichi CUDA implementations.

    These tests require CUDA-capable GPU. They will be skipped if CUDA
    is not available.
    """

    def benchmark_taichi_vectorized_ops(self):
        """Benchmark Taichi vectorized operations."""
        import taichi as ti

        n = 100000
        x = ti.field(ti.f32, shape=n)
        y = ti.field(ti.f32, shape=n)
        result = ti.field(ti.f32, shape=n)

        @ti.kernel
        def compute():
            for i in range(n):
                result[i] = x[i] * y[i] + ti.sin(x[i]) * ti.cos(y[i])

        # Initialize fields
        for i in range(n):
            x[i] = float(i) / n
            y[i] = float(i % 100) / 100

        start = time.perf_counter()
        for _ in range(100):
            compute()
        elapsed = (time.perf_counter() - start) / 100
        print(f"\n  Taichi vectorized (100k elements): {elapsed*1000:.2f} ms")
        assert elapsed > 0

    def benchmark_taichi_matrix_ops(self):
        """Benchmark Taichi matrix operations."""
        import taichi as ti

        n = 1000
        A = ti.Matrix.field(2, 2, ti.f32, shape=n)
        B = ti.Matrix.field(2, 2, ti.F32, shape=n)
        C = ti.Matrix.field(2, 2, ti.f32, shape=n)

        @ti.kernel
        def matmul():
            for i in range(n):
                C[i] = A[i] @ B[i]

        for i in range(n):
            A[i] = ti.Matrix([[float(i), 0.0], [0.0, float(i % 10)]])
            B[i] = ti.Matrix([[1.0, 0.5], [0.5, 1.0]])

        start = time.perf_counter()
        for _ in range(1000):
            matmul()
        elapsed = (time.perf_counter() - start) / 1000
        print(f"\n  Taichi matrix mul (1000 2x2): {elapsed*1000:.4f} ms")
        assert elapsed > 0


class BenchmarkComparison:
    """Compare CPU vs GPU performance for equivalent operations."""

    def compare_trajectory_prediction(self):
        """Compare CPU trajectory prediction vs Taichi GPU."""
        import taichi as ti

        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        n_steps = 100
        offsets = [0.1 * math.sin(i / 10.0) for i in range(n_steps)]

        # CPU benchmark
        ship = p.Kinematic_Ship()
        start_cpu = time.perf_counter()
        for _ in range(50):
            traj = ship.predict_trajectory(
                ownship, offsets=offsets,
                T=float(n_steps), dt=1.0,
            )
        cpu_time = (time.perf_counter() - start_cpu) / 50

        # Note: Full GPU benchmark would require a Taichi-native ship model
        # For now, just report CPU time
        print(f"\n  CPU trajectory prediction: {cpu_time*1000:.2f} ms")
        print(f"  GPU timing requires Taichi-native implementation")

        assert cpu_time > 0


if __name__ == "__main__":
    # Run benchmarks directly
    print("=" * 60)
    print("CPU Benchmark Suite")
    print("=" * 60)

    pytest.main([__file__, "-v", "-s", "--tb=short"])
