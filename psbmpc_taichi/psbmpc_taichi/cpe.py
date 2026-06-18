"""Collision Probability Estimation (CPE) methods.

Implements Cross-Entropy (CE) and Monte Carlo Sequential Kalman Filter
(MCSKF4D) methods for collision probability estimation.
Ported from the C++/CUDA implementations.

Supports both CPU (pure Python) and GPU (Taichi) execution modes.
"""
import math
from typing import Tuple

import numpy as np

from .types import CPEResult, ObstacleData, TI_AVAILABLE, ti
from .utils import (
    Xoshiro256pp,
    distance_2d,
    point_in_polygon,
    ship_polygon,
    normalize_angle,
)


# ============================================================================
# GPU Kernels (Taichi)
# ============================================================================

if TI_AVAILABLE and ti is not None:
    # ========================================================================
    # Gaussian sampling using Box-Muller transform
    # ========================================================================
    @ti.func
    def _box_muller_normal(mu: ti.f64, sigma: ti.f64, out0: ti.types.ndarray(), out1: ti.types.ndarray()) -> None:
        """Generate two standard normal samples using Box-Muller transform.

        Args:
            mu: mean
            sigma: standard deviation
            out0: output array for first sample
            out1: output array for second sample
        """
        u1 = ti.max(ti.random(ti.f64), 1e-10)
        u2 = ti.random(ti.f64)
        z0 = ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(6.283185307179586 * u2)
        z1 = ti.sqrt(-2.0 * ti.log(u1)) * ti.sin(6.283185307179586 * u2)
        out0[0] = mu + sigma * z0
        out1[0] = mu + sigma * z1

    # ========================================================================
    # 2D Collision checking (kernel)
    # ========================================================================
    @ti.func
    def _check_collision_2d_taichi(
        sample_x: ti.f64, sample_y: ti.f64,
        ownship_length: ti.f64, obstacle_length: ti.f64,
        d_safe: ti.f64, collision_margin: ti.f64,
    ) -> ti.i32:
        """Check if a sample results in collision (2D circular approximation).

        Args:
            sample_x, sample_y: sampled relative position
            ownship_length: ownship length
            obstacle_length: obstacle length
            d_safe: safety distance
            collision_margin: additional margin

        Returns:
            1 if collision, 0 otherwise
        """
        dist = ti.sqrt(sample_x * sample_x + sample_y * sample_y)
        collision_distance = (ownship_length + obstacle_length) * 0.5 + d_safe + collision_margin
        return 1 if dist < collision_distance else 0

    # ========================================================================
    # 4D Collision checking (kernel) — position + velocity
    # ========================================================================
    @ti.func
    def _check_collision_4d_taichi(
        rel_x: ti.f64, rel_y: ti.f64,
        rel_vx: ti.f64, rel_vy: ti.f64,
        ownship_length: ti.f64, obstacle_length: ti.f64,
        d_safe: ti.f64, collision_margin: ti.f64,
    ) -> ti.i32:
        """Check collision using 4D state (position + velocity).

        Args:
            rel_x, rel_y: relative position
            rel_vx, rel_vy: relative velocity
            ownship_length: ownship length
            obstacle_length: obstacle length
            d_safe: safety distance
            collision_margin: additional margin

        Returns:
            1 if collision, 0 otherwise
        """
        dist = ti.sqrt(rel_x * rel_x + rel_y * rel_y)
        collision_distance = (ownship_length + obstacle_length) * 0.5 + d_safe + collision_margin

        if dist < collision_distance:
            return 1

        # Check time-to-CPA (closest point of approach)
        relative_v_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if relative_v_sq > 1e-6:
            # Time to CPA
            t_cpa = -(rel_x * rel_vx + rel_y * rel_vy) / relative_v_sq

            if t_cpa > 0:
                # Position at CPA
                x_cpa = rel_x + rel_vx * t_cpa
                y_cpa = rel_y + rel_vy * t_cpa
                dist_cpa = ti.sqrt(x_cpa * x_cpa + y_cpa * y_cpa)

                if dist_cpa < collision_distance:
                    return 1

        return 0

    # ========================================================================
    # 1D Gaussian PDF (kernel)
    # ========================================================================
    @ti.func
    def _gaussian_pdf_taichi(x: ti.f64, mu: ti.f64, sigma: ti.f64) -> ti.f64:
        """1D Gaussian probability density function (Taichi func)."""
        z: ti.f64 = (x - mu) / sigma
        result: ti.f64 = ti.exp(-0.5 * z * z) / (sigma * 2.5066282746310002)  # sqrt(2*pi) ≈ 2.5066
        result = ti.select(sigma >= 1e-10, result, 0.0)
        return result

    # ========================================================================
    # CE (Cross-Entropy) batch sampling kernel
    # ========================================================================
    @ti.kernel
    def _ce_sample_batch_taichi(
        n_samples: ti.i32,
        dist_mu_x: ti.f64, dist_mu_y: ti.f64,
        dist_sigma_x: ti.f64, dist_sigma_y: ti.f64,
        true_mu_x: ti.f64, true_mu_y: ti.f64,
        true_sigma_x: ti.f64, true_sigma_y: ti.f64,
        ownship_length: ti.f64, obstacle_length: ti.f64,
        d_safe: ti.f64, collision_margin: ti.f64,
        samples_x: ti.types.ndarray(),
        samples_y: ti.types.ndarray(),
        collision_flags: ti.types.ndarray(),
        weights: ti.types.ndarray(),
    ):
        """Sample n_samples in parallel and compute collision flags + importance weights.

        Args:
            n_samples: number of samples
            dist_mu_x, dist_mu_y: importance distribution mean
            dist_sigma_x, dist_sigma_y: importance distribution std
            true_mu_x, true_mu_y: true distribution mean
            true_sigma_x, true_sigma_y: true distribution std
            Collision parameters
            samples_x, samples_y: output sampled positions
            collision_flags: output collision detection (0 or 1)
            weights: output importance weights (0 if no collision)
        """
        for i in range(n_samples):
            # Generate two samples via Box-Muller
            sx1, sx2 = _box_muller_normal(dist_mu_x, dist_sigma_x)
            sy1, sy2 = _box_muller_normal(dist_mu_y, dist_sigma_y)

            # Store samples (use odd indices for second samples)
            idx1 = i * 2
            idx2 = i * 2 + 1

            samples_x[idx1] = sx1
            samples_y[idx1] = sy1
            if idx2 < n_samples * 2:
                samples_x[idx2] = sx2
                samples_y[idx2] = sy2

            # Check collision for first sample
            collision_flags[i] = _check_collision_2d_taichi(
                sx1, sy1,
                ownship_length, obstacle_length,
                d_safe, collision_margin,
            )

            # Compute importance weight if collision
            if collision_flags[i] == 1:
                wx = _gaussian_pdf_taichi(sx1, true_mu_x, true_sigma_x)
                wy = _gaussian_pdf_taichi(sy1, true_mu_y, true_sigma_y)
                qw_x = _gaussian_pdf_taichi(sx1, dist_mu_x, dist_sigma_x)
                qw_y = _gaussian_pdf_taichi(sy1, dist_mu_y, dist_sigma_y)
                weights[i] = (wx * wy) / (qw_x * qw_y + 1e-30)
            else:
                weights[i] = 0.0

    # ========================================================================
    # CE batch with statistics reduction (kernel)
    # Uses ti.atomic_add with an ndarray counter parameter
    # ========================================================================
    @ti.kernel
    def _ce_sample_and_reduce_taichi(
        n_samples: ti.i32,
        dist_mu_x: ti.f64, dist_mu_y: ti.f64,
        dist_sigma_x: ti.f64, dist_sigma_y: ti.f64,
        true_mu_x: ti.f64, true_mu_y: ti.f64,
        true_sigma_x: ti.f64, true_sigma_y: ti.f64,
        ownship_length: ti.f64, obstacle_length: ti.f64,
        d_safe: ti.f64, collision_margin: ti.f64,
        collision_indices: ti.types.ndarray(),
        n_collision_arr: ti.types.ndarray(),
        max_collisions: ti.i32,
        out_sum_coll_x: ti.types.ndarray(),
        out_sum_coll_y: ti.types.ndarray(),
        out_sum_coll_x_sq: ti.types.ndarray(),
        out_sum_coll_y_sq: ti.types.ndarray(),
        out_total_weight: ti.types.ndarray(),
    ):
        """Sample and compute collision statistics in parallel.

        Uses an ndarray as an atomic counter for collision count.
        Results are written to output arrays.
        """
        # Reset counter
        n_collision_arr[0] = 0

        sum_coll_x: ti.f64 = 0.0
        sum_coll_y: ti.f64 = 0.0
        sum_coll_x_sq: ti.f64 = 0.0
        sum_coll_y_sq: ti.f64 = 0.0
        total_weight: ti.f64 = 0.0

        for i in range(n_samples):
            # Generate sample using Box-Muller (inline to avoid tuple return issues in Taichi 1.7)
            u1: ti.f64 = ti.max(ti.random(ti.f64), 1e-10)
            u2: ti.f64 = ti.random(ti.f64)
            sx: ti.f64 = dist_mu_x + dist_sigma_x * ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(6.283185307179586 * u2)
            u1 = ti.max(ti.random(ti.f64), 1e-10)
            u2 = ti.random(ti.f64)
            sy: ti.f64 = dist_mu_y + dist_sigma_y * ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(6.283185307179586 * u2)

            # Check collision
            is_collision = _check_collision_2d_taichi(
                sx, sy,
                ownship_length, obstacle_length,
                d_safe, collision_margin,
            )

            if is_collision == 1:
                # Atomic increment of counter array
                idx = int(ti.atomic_add(n_collision_arr[0], 1))
                if idx < max_collisions:
                    collision_indices[idx] = i  # store index for reference
                    sum_coll_x += sx
                    sum_coll_y += sy
                    sum_coll_x_sq += sx * sx
                    sum_coll_y_sq += sy * sy

                    # Importance weight
                    wx = _gaussian_pdf_taichi(sx, true_mu_x, true_sigma_x)
                    wy = _gaussian_pdf_taichi(sy, true_mu_y, true_sigma_y)
                    qw_x = _gaussian_pdf_taichi(sx, dist_mu_x, dist_sigma_x)
                    qw_y = _gaussian_pdf_taichi(sy, dist_mu_y, dist_sigma_y)
                    weight = (wx * wy) / (qw_x * qw_y + 1e-30)
                    total_weight += weight

        # Write results to output arrays
        out_sum_coll_x[0] = sum_coll_x
        out_sum_coll_y[0] = sum_coll_y
        out_sum_coll_x_sq[0] = sum_coll_x_sq
        out_sum_coll_y_sq[0] = sum_coll_y_sq
        out_total_weight[0] = total_weight

    # ========================================================================
    # MCSKF4D particle filter kernel
    # Uses ti.atomic_add with an ndarray counter parameter
    # ========================================================================
    @ti.kernel
    def _mcskf4d_kernel_taichi(
        n_particles: ti.i32,
        n_steps: ti.i32,
        dt: ti.f64,
        process_noise: ti.f64,
        ownship_length: ti.f64, obstacle_length: ti.f64,
        d_safe: ti.f64, collision_margin: ti.f64,
        init_x_rel: ti.f64, init_y_rel: ti.f64,
        init_vx_rel: ti.f64, init_vy_rel: ti.f64,
        meas_x: ti.f64, meas_y: ti.f64,
        meas_noise: ti.f64,
        collision_count: ti.types.ndarray(),
    ):
        """Monte Carlo Sequential Kalman Filter (MCSKF4D) kernel.

        Simulates n_particles × n_steps and counts collisions.

        Args:
            n_particles: number of particles
            n_steps: number of simulation steps
            dt: time step
            process_noise: process noise std
            Collision parameters
            init_x_rel, init_y_rel: initial relative position
            init_vx_rel, init_vy_rel: initial relative velocity
            meas_x, meas_y: measurement of relative position
            meas_noise: measurement noise std
            collision_count: output counter (ndarray, updated atomically)
        """
        # Reset counter
        collision_count[0] = 0

        for p in range(n_particles):
            # Initialize particle state
            px: ti.f64 = init_x_rel
            py: ti.f64 = init_y_rel
            pvx: ti.f64 = init_vx_rel
            pvy: ti.f64 = init_vy_rel

            for s in range(n_steps):
                # Add process noise to position
                noise_x = ti.sqrt(process_noise) * (ti.random(ti.f64) * 2.0 - 1.0)
                noise_y = ti.sqrt(process_noise) * (ti.random(ti.f64) * 2.0 - 1.0)
                px += noise_x * dt
                py += noise_y * dt

                # Add process noise to velocity
                pvx += ti.sqrt(process_noise) * (ti.random(ti.f64) * 2.0 - 1.0) * dt
                pvy += ti.sqrt(process_noise) * (ti.random(ti.f64) * 2.0 - 1.0) * dt

                # Kalman-like measurement update
                meas_x_noisy = meas_x + ti.sqrt(meas_noise) * (ti.random(ti.f64) * 2.0 - 1.0)
                meas_y_noisy = meas_y + ti.sqrt(meas_noise) * (ti.random(ti.f64) * 2.0 - 1.0)

                # Simple Kalman gain (fixed for simplicity)
                K_x: ti.f64 = 0.1
                K_y: ti.f64 = 0.1
                px += K_x * (meas_x_noisy - px)
                py += K_y * (meas_y_noisy - py)

                # Update position based on velocity
                px += pvx * dt
                py += pvy * dt

                # Check collision
                dist = ti.sqrt(px * px + py * py)
                collision_distance = (ownship_length + obstacle_length) * 0.5 + d_safe + collision_margin

                if dist < collision_distance:
                    ti.atomic_add(collision_count[0], 1)


# ============================================================================
# Collision Probability Estimation
# ============================================================================


class CPE:
    """Collision Probability Estimation methods.

    Implements:
    - Cross-Entropy (CE) method with importance sampling
    - Monte Carlo Sequential Kalman Filter (MCSKF4D)
    """

    def __init__(
        self,
        max_iter: int = 50,
        tolerance: float = 1e-4,
        n_samples: int = 1000,
        collision_margin: float = 5.0,  # meters additional margin
        use_gpu: bool = True,
    ):
        """Initialize CPE estimator.

        Args:
            max_iter: maximum iterations for CE method
            tolerance: convergence tolerance
            n_samples: number of samples per iteration
            collision_margin: additional safety margin for collision detection
            use_gpu: whether to use GPU acceleration (if available)
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_samples = n_samples
        self.collision_margin = collision_margin
        self.use_gpu = use_gpu and TI_AVAILABLE and ti is not None

    def ce_estimate(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        relative_cov_xx: float = 10.0,
        relative_cov_yy: float = 10.0,
        relative_cov_xy: float = 0.0,
    ) -> CPEResult:
        """Cross-Entropy method for collision probability estimation.

        Uses importance sampling with iterative distribution refinement.

        Args:
            ownship: ownship obstacle data
            obstacle: obstacle ship data
            relative_cov_xx: relative position covariance x
            relative_cov_yy: relative position covariance y
            relative_cov_xy: relative position covariance xy

        Returns:
            CPEResult with estimated collision probability
        """
        if self.use_gpu:
            return self._ce_estimate_gpu(
                ownship, obstacle,
                relative_cov_xx, relative_cov_yy, relative_cov_xy,
            )
        return self._ce_estimate_cpu(
            ownship, obstacle,
            relative_cov_xx, relative_cov_yy, relative_cov_xy,
        )

    def _ce_estimate_cpu(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        relative_cov_xx: float = 10.0,
        relative_cov_yy: float = 10.0,
        relative_cov_xy: float = 0.0,
    ) -> CPEResult:
        """CPU implementation of Cross-Entropy method (original)."""
        # Initial distribution parameters (relative position)
        mu_x = obstacle.x - ownship.x
        mu_y = obstacle.y - ownship.y
        sigma_x = math.sqrt(max(relative_cov_xx, 1e-6))
        sigma_y = math.sqrt(max(relative_cov_yy, 1e-6))

        # Initialize importance sampling distribution
        dist_mu_x = mu_x
        dist_mu_y = mu_y
        dist_sigma_x = sigma_x
        dist_sigma_y = sigma_y

        best_prob = 0.0
        prev_prob = 0.0

        # Generate PRNG for reproducibility
        rng = Xoshiro256pp(seed=42)

        for iteration in range(self.max_iter):
            # Generate samples from importance distribution
            samples_x = []
            samples_y = []
            for _ in range(self.n_samples):
                sx, sy = rng.next_multivariate_normal_2d(
                    dist_mu_x, dist_mu_y,
                    dist_sigma_x ** 2, dist_sigma_y ** 2,
                    0.0,  # uncorrelated in importance distribution
                )
                samples_x.append(sx)
                samples_y.append(sy)

            # Determine which samples result in collision
            collision_indices = []
            for i in range(self.n_samples):
                if self._check_collision_2d(
                    mu_x, mu_y,
                    samples_x[i], samples_y[i],
                    ownship, obstacle,
                ):
                    collision_indices.append(i)

            n_collision = len(collision_indices)
            collision_rate = n_collision / self.n_samples

            # Compute importance weights
            total_weight = 0.0
            for idx in collision_indices:
                # Weight = p(sample | true_dist) / q(sample | importance_dist)
                wx = self._gaussian_pdf(
                    samples_x[idx], mu_x, math.sqrt(max(relative_cov_xx, 1e-6))
                )
                wy = self._gaussian_pdf(
                    samples_y[idx], mu_y, math.sqrt(max(relative_cov_yy, 1e-6))
                )
                qw_x = self._gaussian_pdf(
                    samples_x[idx], dist_mu_x, dist_sigma_x
                )
                qw_y = self._gaussian_pdf(
                    samples_y[idx], dist_mu_y, dist_sigma_y
                )

                weight = (wx * wy) / (qw_x * qw_y + 1e-30)
                total_weight += weight

            # Estimated probability
            estimated_prob = total_weight / self.n_samples

            # Update importance sampling distribution using top-k samples
            if n_collision > 10:  # Need enough collisions to update
                # Use all collision samples to update distribution
                coll_x = [samples_x[i] for i in collision_indices]
                coll_y = [samples_y[i] for i in collision_indices]

                new_mu_x = sum(coll_x) / len(coll_x)
                new_mu_y = sum(coll_y) / len(coll_y)
                new_sigma_x = max(math.sqrt(sum((x - new_mu_x) ** 2 for x in coll_x) / len(coll_x)), 1.0)
                new_sigma_y = max(math.sqrt(sum((y - new_mu_y) ** 2 for y in coll_y) / len(coll_y)), 1.0)

                dist_mu_x = new_mu_x
                dist_mu_y = new_mu_y
                dist_sigma_x = new_sigma_x
                dist_sigma_y = new_sigma_y

            # Check convergence
            if iteration > 5 and abs(estimated_prob - prev_prob) < self.tolerance:
                return CPEResult(
                    probability=estimated_prob,
                    converged=True,
                    iterations=iteration + 1,
                    n_samples=self.n_samples * (iteration + 1),
                )

            prev_prob = estimated_prob
            best_prob = max(best_prob, estimated_prob)

        # Clamp probability to valid range [0, 1]
        clamped_prob = min(max(best_prob, 0.0), 1.0)

        return CPEResult(
            probability=clamped_prob,
            converged=False,
            iterations=self.max_iter,
            n_samples=self.n_samples * self.max_iter,
        )

    def _ce_estimate_gpu(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        relative_cov_xx: float = 10.0,
        relative_cov_yy: float = 10.0,
        relative_cov_xy: float = 0.0,
    ) -> CPEResult:
        """GPU-accelerated Cross-Entropy method.

        Uses Taichi kernels for parallel sampling and collision detection.
        """
        # Initial distribution parameters (relative position)
        mu_x = obstacle.x - ownship.x
        mu_y = obstacle.y - ownship.y
        sigma_x = math.sqrt(max(relative_cov_xx, 1e-6))
        sigma_y = math.sqrt(max(relative_cov_yy, 1e-6))

        # Initialize importance sampling distribution
        dist_mu_x = mu_x
        dist_mu_y = mu_y
        dist_sigma_x = sigma_x
        dist_sigma_y = sigma_y

        best_prob = 0.0
        prev_prob = 0.0

        # Pre-allocate arrays for GPU
        n_samples = self.n_samples
        max_collisions = n_samples  # upper bound
        collision_indices = np.zeros(max_collisions, dtype=np.int32)
        n_collision_arr = np.array([0], dtype=np.float32)
        out_sum_coll_x = np.array([0.0], dtype=np.float32)
        out_sum_coll_y = np.array([0.0], dtype=np.float32)
        out_sum_coll_x_sq = np.array([0.0], dtype=np.float32)
        out_sum_coll_y_sq = np.array([0.0], dtype=np.float32)
        out_total_weight = np.array([0.0], dtype=np.float32)

        for iteration in range(self.max_iter):
            # Launch GPU kernel for sampling and reduction
            _ce_sample_and_reduce_taichi(
                n_samples,
                float(dist_mu_x), float(dist_mu_y),
                float(dist_sigma_x), float(dist_sigma_y),
                float(mu_x), float(mu_y),
                float(sigma_x), float(sigma_y),
                float(ownship.length), float(obstacle.length),
                float(ownship.d_safe), float(self.collision_margin),
                collision_indices,
                n_collision_arr,
                max_collisions,
                out_sum_coll_x,
                out_sum_coll_y,
                out_sum_coll_x_sq,
                out_sum_coll_y_sq,
                out_total_weight,
            )

            sum_coll_x = float(out_sum_coll_x[0])
            sum_coll_y = float(out_sum_coll_y[0])
            sum_coll_x_sq = float(out_sum_coll_x_sq[0])
            sum_coll_y_sq = float(out_sum_coll_y_sq[0])
            total_weight = float(out_total_weight[0])
            n_collision = int(n_collision_arr[0])

            # Estimated probability
            estimated_prob = total_weight / n_samples

            # Update importance sampling distribution using collision samples
            if n_collision > 10:
                new_mu_x = sum_coll_x / n_collision
                new_mu_y = sum_coll_y / n_collision
                var_x = sum_coll_x_sq / n_collision - new_mu_x * new_mu_x
                var_y = sum_coll_y_sq / n_collision - new_mu_y * new_mu_y
                new_sigma_x = max(math.sqrt(max(var_x, 0.0)), 1.0)
                new_sigma_y = max(math.sqrt(max(var_y, 0.0)), 1.0)

                dist_mu_x = new_mu_x
                dist_mu_y = new_mu_y
                dist_sigma_x = new_sigma_x
                dist_sigma_y = new_sigma_y

            # Check convergence
            if iteration > 5 and abs(estimated_prob - prev_prob) < self.tolerance:
                return CPEResult(
                    probability=estimated_prob,
                    converged=True,
                    iterations=iteration + 1,
                    n_samples=n_samples * (iteration + 1),
                )

            prev_prob = estimated_prob
            best_prob = max(best_prob, estimated_prob)

        # Clamp probability to valid range [0, 1]
        clamped_prob = min(max(best_prob, 0.0), 1.0)

        return CPEResult(
            probability=clamped_prob,
            converged=False,
            iterations=self.max_iter,
            n_samples=n_samples * self.max_iter,
        )

    def mcskf4d_estimate(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        dt: float = 1.0,
        process_noise: float = 1.0,
    ) -> CPEResult:
        """Monte Carlo Sequential Kalman Filter (MCSKF4D) method.

        Uses 4D state: [relative_x, relative_y, relative_vx, relative_vy]
        with sequential Kalman filtering and collision probability measurement.

        Args:
            ownship: ownship obstacle data
            obstacle: obstacle ship data
            dt: time step
            process_noise: process noise standard deviation

        Returns:
            CPEResult with estimated collision probability
        """
        if self.use_gpu:
            return self._mcskf4d_estimate_gpu(
                ownship, obstacle, dt, process_noise,
            )
        return self._mcskf4d_estimate_cpu(
            ownship, obstacle, dt, process_noise,
        )

    def _mcskf4d_estimate_cpu(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        dt: float = 1.0,
        process_noise: float = 1.0,
    ) -> CPEResult:
        """CPU implementation of MCSKF4D (original)."""
        # Initial relative state
        x_rel = obstacle.x - ownship.x
        y_rel = obstacle.y - ownship.y
        vx_rel = 0.0  # Relative velocity (simplified)
        vy_rel = 0.0

        # State: [x_rel, y_rel, vx_rel, vy_rel]
        state = np.array([x_rel, y_rel, vx_rel, vy_rel], dtype=np.float64)

        # Covariance matrix (4x4)
        P = np.eye(4, dtype=np.float64) * 100.0

        # Process noise covariance
        Q = np.eye(4, dtype=np.float64) * (process_noise ** 2)

        # Measurement noise covariance (2x2)
        R = np.eye(2, dtype=np.float64) * 10.0

        # Generate samples for Monte Carlo
        rng = Xoshiro256pp(seed=123)
        n_particles = 1000

        # Particle weights
        weights = np.ones(n_particles) / n_particles

        # Predict using Kalman filter for each particle
        collision_count = 0

        for step in range(int(300 / dt)):  # 300 second horizon
            for p in range(n_particles):
                # Add process noise (only to position components)
                noise_2d = rng.next_multivariate_normal_2d(
                    0.0, 0.0,
                    process_noise ** 2, process_noise ** 2, 0.0,
                )
                state_noisy = state.copy()
                state_noisy[0] += noise_2d[0]  # x_rel
                state_noisy[1] += noise_2d[1]  # y_rel

                # Check collision
                if self._check_collision_4d(
                    state_noisy[0], state_noisy[1],
                    state_noisy[2], state_noisy[3],
                    ownship, obstacle,
                ):
                    collision_count += 1

            # Update state (simplified Kalman update)
            # Measurement: relative position from obstacle data
            z = np.array([obstacle.x - ownship.x, obstacle.y - ownship.y], dtype=np.float64)

            # Predicted measurement
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
            z_pred = H @ state

            # Innovation
            y_innov = z - z_pred

            # Innovation covariance
            S = H @ P @ H.T + R

            # Kalman gain
            S_inv = np.linalg.inv(S)
            K = P @ H.T @ S_inv

            # State update
            state = state + K @ y_innov

            # Covariance update
            I = np.eye(4)
            P = (I - K @ H) @ P

            # Process noise addition
            P = P + Q

        # Collision probability
        probability = collision_count / (n_particles * int(300 / dt))

        return CPEResult(
            probability=float(probability),
            converged=True,
            iterations=int(300 / dt),
            n_samples=n_particles * int(300 / dt),
        )

    def _mcskf4d_estimate_gpu(
        self,
        ownship: ObstacleData,
        obstacle: ObstacleData,
        dt: float = 1.0,
        process_noise: float = 1.0,
    ) -> CPEResult:
        """GPU-accelerated MCSKF4D.

        Uses Taichi kernel for parallel particle simulation.
        """
        # Initial relative state
        x_rel = obstacle.x - ownship.x
        y_rel = obstacle.y - ownship.y
        vx_rel = 0.0
        vy_rel = 0.0

        n_particles = 1000
        n_steps = int(300 / dt)

        # Collision count (on CPU, will be updated by kernel)
        collision_count = np.array([0], dtype=np.float32)

        # Measurement noise (simplified)
        meas_noise = 10.0

        # Launch GPU kernel
        _mcskf4d_kernel_taichi(
            n_particles, n_steps,
            float(dt), float(process_noise),
            float(ownship.length), float(obstacle.length),
            float(ownship.d_safe), float(self.collision_margin),
            float(x_rel), float(y_rel),
            float(vx_rel), float(vy_rel),
            float(x_rel), float(y_rel),  # measurement = initial relative position
            float(meas_noise),
            collision_count,
        )

        # Collision probability
        probability = float(collision_count[0]) / (n_particles * n_steps)

        return CPEResult(
            probability=float(probability),
            converged=True,
            iterations=n_steps,
            n_samples=n_particles * n_steps,
        )

    def _check_collision_2d(
        self,
        mu_x: float, mu_y: float,
        sample_x: float, sample_y: float,
        ownship: ObstacleData,
        obstacle: ObstacleData,
    ) -> bool:
        """Check if a sample results in collision.

        Uses simplified rectangular ship approximation.

        Args:
            mu_x, mu_y: mean relative position (used for reference)
            sample_x, sample_y: sampled relative position
            ownship: ownship data
            obstacle: obstacle data

        Returns:
            True if collision detected
        """
        # Distance from origin (collision region is centered at origin)
        dist = math.sqrt(sample_x ** 2 + sample_y ** 2)

        # Simple circular collision check
        collision_distance = (ownship.length + obstacle.length) / 2.0 + ownship.d_safe + self.collision_margin

        return dist < collision_distance

    def _check_collision_4d(
        self,
        rel_x: float, rel_y: float,
        rel_vx: float, rel_vy: float,
        ownship: ObstacleData,
        obstacle: ObstacleData,
    ) -> bool:
        """Check collision using 4D state (position + velocity).

        Args:
            rel_x, rel_y: relative position
            rel_vx, rel_vy: relative velocity
            ownship: ownship data
            obstacle: obstacle data

        Returns:
            True if collision detected
        """
        dist = math.sqrt(rel_x ** 2 + rel_y ** 2)

        # Simple circular collision check
        collision_distance = (ownship.length + obstacle.length) / 2.0 + ownship.d_safe + self.collision_margin

        if dist < collision_distance:
            return True

        # Check time-to-CPA (closest point of approach)
        relative_v_sq = rel_vx ** 2 + rel_vy ** 2
        if relative_v_sq > 1e-6:
            # Time to CPA
            t_cpa = -(rel_x * rel_vx + rel_y * rel_vy) / relative_v_sq

            if t_cpa > 0:
                # Position at CPA
                x_cpa = rel_x + rel_vx * t_cpa
                y_cpa = rel_y + rel_vy * t_cpa
                dist_cpa = math.sqrt(x_cpa ** 2 + y_cpa ** 2)

                if dist_cpa < collision_distance:
                    return True

        return False

    @staticmethod
    def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
        """1D Gaussian probability density function."""
        if sigma < 1e-10:
            return 0.0
        z = (x - mu) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))
