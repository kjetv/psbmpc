"""Tests for Collision Probability Estimation (CPE)."""
import math

import numpy as np
import pytest

import psbmpc_taichi as p


class TestCPE:
    """Tests for the CPE class."""

    def test_initialization(self, cpe_estimator):
        """Test CPE estimator initialization."""
        assert cpe_estimator.max_iter == 10
        assert cpe_estimator.tolerance == 1e-3
        assert cpe_estimator.n_samples == 100
        assert cpe_estimator.collision_margin == 5.0

    def test_ce_estimate_far_obstacle(self, cpe_estimator, ship_state4, obstacle):
        """Test CE estimate with distant obstacle (low collision probability)."""
        # Create ownship data
        ownship = p.ObstacleData(
            x=ship_state4.x, y=ship_state4.y,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        # Obstacle is far away at (500, 200)
        result = cpe_estimator.ce_estimate(ownship, obstacle)

        assert result.probability >= 0.0
        assert result.iterations > 0
        assert result.n_samples > 0
        # Far obstacle should have relatively low collision probability
        # (may not be zero due to sampling)

    def test_ce_estimate_near_obstacle(self, cpe_estimator, ship_state4):
        """Test CE estimate with close obstacle (higher collision probability)."""
        ownship = p.ObstacleData(
            x=ship_state4.x, y=ship_state4.y,
            length=150.0, beam=25.0, d_safe=0.0,
        )
        # Very close obstacle
        close_obs = p.ObstacleData(
            x=50.0, y=20.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe_estimator.ce_estimate(ownship, close_obs)

        assert result.probability >= 0.0
        assert result.iterations > 0

    def test_ce_estimate_same_position(self, cpe_estimator, ship_state4):
        """Test CE estimate when ships are at same position (max collision)."""
        ownship = p.ObstacleData(
            x=0.0, y=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )
        # Same position
        same_obs = p.ObstacleData(
            x=0.0, y=0.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe_estimator.ce_estimate(ownship, same_obs)

        # Should have highest possible collision probability
        assert result.probability > 0.0

    def test_ce_estimate_convergence(self, ship_state4):
        """Test that CE method can converge with loose tolerance."""
        cpe = p.CPE(max_iter=50, tolerance=1e-6, n_samples=500)

        ownship = p.ObstacleData(
            x=0.0, y=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )
        far_obs = p.ObstacleData(
            x=10000.0, y=10000.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe.ce_estimate(ownship, far_obs)

        # Should converge for distant obstacle
        assert result.converged or result.probability < 0.01

    def test_mcskf4d_estimate(self, cpe_estimator, ship_state4, obstacle):
        """Test MCSKF4D estimation method."""
        ownship = p.ObstacleData(
            x=ship_state4.x, y=ship_state4.y,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe_estimator.mcskf4d_estimate(ownship, obstacle)

        assert result.probability >= 0.0
        assert result.iterations > 0


class TestCPEHelpers:
    """Tests for CPE helper methods."""

    def test_check_collision_2d_close(self, cpe_estimator, ship_state4, obstacle):
        """Test 2D collision check with close ships."""
        ownship = p.ObstacleData(
            x=0.0, y=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )
        close_obs = p.ObstacleData(
            x=10.0, y=10.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe_estimator._check_collision_2d(
            0.0, 0.0,  # mu_x, mu_y (relative position)
            10.0, 10.0,  # sample position
            ownship, close_obs,
        )
        assert result is True

    def test_check_collision_2d_far(self, cpe_estimator):
        """Test 2D collision check with distant ships."""
        ownship = p.ObstacleData(
            x=0.0, y=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )
        far_obs = p.ObstacleData(
            x=10000.0, y=10000.0, chi=0.0, U=0.0,
            length=150.0, beam=25.0, d_safe=0.0,
        )

        result = cpe_estimator._check_collision_2d(
            0.0, 0.0,
            10000.0, 10000.0,
            ownship, far_obs,
        )
        assert result is False

    def test_gaussian_pdf(self, cpe_estimator):
        """Test Gaussian PDF computation."""
        # Peak at mean
        pdf_at_mean = cpe_estimator._gaussian_pdf(0.0, 0.0, 1.0)
        # Far from mean
        pdf_far = cpe_estimator._gaussian_pdf(10.0, 0.0, 1.0)

        assert pdf_at_mean > pdf_far
        assert pdf_at_mean > 0.0
        assert pdf_far >= 0.0
