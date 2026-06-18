"""Collision Probability Estimation (CPE) methods.

Implements Cross-Entropy (CE) and Monte Carlo Sequential Kalman Filter
(MCSKF4D) methods for collision probability estimation.
Ported from the C++/CUDA implementations.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .types import CPEResult, ObstacleData
from .utils import (
    Xoshiro256pp,
    distance_2d,
    point_in_polygon,
    ship_polygon,
)


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
    ):
        """Initialize CPE estimator.

        Args:
            max_iter: maximum iterations for CE method
            tolerance: convergence tolerance
            n_samples: number of samples per iteration
            collision_margin: additional safety margin for collision detection
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_samples = n_samples
        self.collision_margin = collision_margin

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
        # Initial distribution parameters (relative position)
        mu_x = obstacle.x - ownship.x
        mu_y = obstacle.y - ownship.y
        sigma_x = math.sqrt(max(relative_cov_xx, 1e-6))
        sigma_y = math.sqrt(max(relative_cov_yy, 1e-6))

        # Covariance matrix
        cov_xx = relative_cov_xx
        cov_yy = relative_cov_yy
        cov_xy = relative_cov_xy

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
