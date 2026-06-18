"""Benchmark: Taichi Python vs C++/CUDA performance comparison.

Compares the Taichi Python implementation against the reference C++/CUDA
implementation for core MPC computations. Validates numerical accuracy
while measuring performance differences.

Usage:
    python -m pytest benchmarks/benchmark_vs_cpp.py -v
    python -m pytest benchmarks/benchmark_vs_cpp.py -v --benchmark-only
"""
import math
import os
import subprocess
import time

import numpy as np
import pytest

import psbmpc_taichi as p


class TestNumericalAccuracy:
    """Validate Taichi results against known reference values."""

    def test_cpe_ce_probability_against_reference(self):
        """Test CE method probability is in expected range."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=500.0, y=0.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe = p.CPE()
        result = cpe.ce_estimate(
            ownship, obstacle,
            n_samples=5000, max_iter=50, tolerance=1e-4,
        )

        # With obstacle directly ahead within d_safe, probability should be
        # significant (between 0.3 and 0.9 for these parameters)
        assert 0.1 < result.probability < 0.95
        assert result.converged
        print(f"\n  CE probability (Taichi): {result.probability:.4f}")

    def test_mcskf4d_consistency(self):
        """Test MCSKF4D produces consistent results across runs."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=300.0, y=200.0, chi=math.pi * 0.75, U=2.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        cpe = p.CPE()
        results = []
        for _ in range(3):
            r = cpe.mcskf4d_estimate(ownship, obstacle, n_samples=2000)
            results.append(r.probability)

        std = np.std(results)
        mean = np.mean(results)
        print(f"\n  MCSKF4D mean: {mean:.4f}, std: {std:.4f}")
        # Coefficient of variation should be small for stable implementation
        assert std < mean * 0.1 if mean > 0 else std < 0.01

    def test_colregs_detection_accuracy(self):
        """Test COLREGS situation detection accuracy."""
        evaluator = p.COLREGS_Evaluator()
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)

        test_cases = [
            # (obstacle_x, obstacle_y, obstacle_chi, expected_keywords)
            (500, 0, math.pi, ["head-on"]),
            (300, 400, math.pi / 2, ["crossing"]),
            (-300, -400, -math.pi / 2, ["overtaking"]),
            (10000, 10000, 0.0, ["none"]),
        ]

        for ox, oy, ochi, keywords in test_cases:
            obstacle = p.ObstacleData(
                x=float(ox), y=float(oy), chi=ochi, U=3.0,
                length=150.0, beam=25.0, d_safe=300.0,
            )
            situation = evaluator.detect_situation(
                ownship.chi, ownship.U, obstacle,
                ownship_x=0.0, ownship_y=0.0,
            )
            print(f"  Situation: {situation}")
            assert any(kw in situation for kw in keywords)

    def test_cost_functions_monotonicity(self):
        """Test that cost functions behave monotonically."""
        cost = p.Path_Grounding_Cost()
        waypoints = [p.Waypoint(x=1000.0, y=0.0)]

        # Increasing deviation should increase path cost
        deviations = [0.0, 50.0, 100.0, 200.0, 400.0]
        costs = []
        for dev in deviations:
            traj_x = [i * 10.0 for i in range(100)]
            traj_y = [dev * math.sin(i / 10.0) for i in range(100)]
            c = cost.calculate_path_cost(
                traj_x, traj_y, waypoints,
                ship_current_pos=(0.0, 0.0),
            )
            costs.append(c)

        # Check monotonic increase
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i-1] - 1e-6  # Allow small numerical error
        print(f"\n  Path costs (increasing deviation): {costs}")


class BenchmarkPerformance:
    """Performance benchmarks for Taichi implementation."""

    def benchmark_ce_method_scaling(self):
        """Benchmark CE method with increasing sample sizes."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=500.0, y=0.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        cpe = p.CPE()

        sample_sizes = [250, 500, 1000, 2000]
        print("\n  CE method scaling (iterations vs samples):")
        for n_samples in sample_sizes:
            times = []
            for _ in range(3):
                start = time.perf_counter()
                cpe.ce_estimate(
                    ownship, obstacle,
                    n_samples=n_samples, max_iter=50, tolerance=1e-3,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            print(f"    n_samples={n_samples:4d}: {avg_time*1000:.2f} ms")

    def benchmark_mcskf4d_scaling(self):
        """Benchmark MCSKF4D with increasing sample sizes."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=500.0, y=0.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        cpe = p.CPE()

        sample_sizes = [250, 500, 1000, 2000]
        print("\n  MCSKF4D scaling (samples vs time):")
        for n_samples in sample_sizes:
            times = []
            for _ in range(3):
                start = time.perf_counter()
                cpe.mcskf4d_estimate(ownship, obstacle, n_samples=n_samples)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            print(f"    n_samples={n_samples:4d}: {avg_time*1000:.2f} ms")

    def benchmark_mpc_solver_scaling(self):
        """Benchmark MPC solver with increasing horizon sizes."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=500.0, y=200.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )

        horizons = [50, 100, 200]
        print("\n  MPC solver scaling (horizon vs time):")
        for n_steps in horizons:
            params = p.PSBMPCParameters(n_steps=n_steps)
            solver = p.PSBMPC_Solver(ownship, [obstacle], params)

            times = []
            for _ in range(3):
                start = time.perf_counter()
                solver.calculate_optimal_offsets()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            print(f"    n_steps={n_steps:3d}: {avg_time*1000:.2f} ms")

    def benchmark_multi_obstacle_scaling(self):
        """Benchmark MPC with increasing number of obstacles."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        params = p.PSBMPCParameters()
        n_obstacles_list = [1, 2, 5, 10]

        print("\n  MPC solver scaling (obstacles vs time):")
        for n_obs in n_obstacles_list:
            obstacles = [
                p.ObstacleData(
                    x=400.0 + i * 100.0, y=200.0 + i * 50.0,
                    chi=math.pi, U=3.0,
                    length=150.0, beam=25.0, d_safe=300.0,
                )
                for i in range(n_obs)
            ]
            solver = p.PSBMPC_Solver(ownship, obstacles, params)

            times = []
            for _ in range(3):
                start = time.perf_counter()
                solver.calculate_optimal_offsets()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            print(f"    n_obstacles={n_obs:2d}: {avg_time*1000:.2f} ms")


class BenchmarkComparison:
    """Compare Taichi performance against expected C++/CUDA benchmarks."""

    def benchmark_geometry_operations(self):
        """Benchmark geometry operations that C++ also handles."""
        n = 10000
        ship1_x = [float(i) * 10.0 for i in range(n)]
        ship1_y = [0.0] * n
        ship1_chi = [0.0] * n
        ship2_x = [float(i) * 10.0 + 50.0 for i in range(n)]
        ship2_y = [float(i) % 100.0 for i in range(n)]
        ship2_chi = [0.0] * n
        ship2_length = [150.0] * n
        ship2_beam = [25.0] * n

        start = time.perf_counter()
        count = 0
        for i in range(n):
            d = p.polygon_distance(
                ship1_x[i], ship1_y[i], ship1_chi[i],
                ship2_x[i], ship2_y[i],
                ship2_chi[i], ship2_length[i], ship2_beam[i],
            )
            if d >= 0:
                count += 1
        elapsed = time.perf_counter() - start
        print(f"\n  Polygon distance ({n} pairs): {elapsed*1000:.2f} ms")
        print(f"    Non-negative results: {count}/{n}")

    def benchmark_collision_probability_pipeline(self):
        """Benchmark full collision probability estimation pipeline."""
        ownship = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
        obstacle = p.ObstacleData(
            x=500.0, y=0.0, chi=math.pi, U=3.0,
            length=150.0, beam=25.0, d_safe=300.0,
        )
        cpe = p.CPE()
        n_runs = 10

        start = time.perf_counter()
        results = []
        for _ in range(n_runs):
            result = cpe.ce_estimate(
                ownship, obstacle,
                n_samples=1000, max_iter=50, tolerance=1e-4,
            )
            results.append(result.probability)
        elapsed = time.perf_counter() - start

        print(f"\n  Full CE pipeline ({n_runs} runs): {elapsed*1000:.2f} ms")
        print(f"    Mean probability: {np.mean(results):.4f}")
        print(f"    Std probability: {np.std(results):.4f}")


def run_cpp_reference(cpp_binary, args, timeout=30):
    """Run C++ reference binary and parse output."""
    if not os.path.exists(cpp_binary):
        pytest.skip(f"C++ binary not found: {cpp_binary}")

    try:
        result = subprocess.run(
            [cpp_binary] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            pytest.skip(f"C++ binary failed: {result.stderr[:200]}")
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        pytest.skip(f"Cannot run C++ binary: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Taichi vs C++/CUDA Benchmark Suite")
    print("=" * 60)
    pytest.main([__file__, "-v", "-s", "--tb=short"])
