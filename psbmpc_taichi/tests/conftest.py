"""Pytest fixtures for psbmpc_taichi tests."""
import pytest
import numpy as np

import psbmpc_taichi as p


@pytest.fixture
def ship_state4():
    """Create a standard ShipState4 for tests."""
    return p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)


@pytest.fixture
def ship_state6():
    """Create a standard ShipState6 for tests."""
    return p.ShipState6(x=0.0, y=0.0, psi=0.0, u=5.0, v=0.0, r=0.0)


@pytest.fixture
def waypoints():
    """Create standard waypoints for tests."""
    return [
        p.Waypoint(x=1000.0, y=0.0),
        p.Waypoint(x=2000.0, y=1000.0),
        p.Waypoint(x=3000.0, y=2000.0),
    ]


@pytest.fixture
def obstacle():
    """Create a standard obstacle for tests."""
    return p.ObstacleData(
        x=500.0, y=200.0, chi=3.14, U=3.0,
        length=150.0, beam=25.0,
        d_safe=300.0,
    )


@pytest.fixture
def obstacle_near():
    """Create a nearby obstacle for collision tests."""
    return p.ObstacleData(
        x=200.0, y=100.0, chi=3.14, U=2.0,
        length=150.0, beam=25.0,
        d_safe=400.0,
    )


@pytest.fixture
def psbmpc_params():
    """Create standard PSBMPC parameters."""
    return p.PSBMPCParameters(
        cpe_max_iter=10,
        cpe_tolerance=1e-3,
        cpe_n_samples=100,
        n_M=10,
        n_cbs=5,
        T=300.0,
        dt=1.0,
    )


@pytest.fixture
def sbmpc_params():
    """Create standard SBMPC parameters."""
    return p.SBMPCParameters(
        n_M=10,
        n_cbs=5,
        T=300.0,
        dt=1.0,
        safety_margin=50.0,
    )


@pytest.fixture
def grounding_hazards():
    """Create grounding hazard polygons for tests."""
    return [
        p.ObstacleData(x=800.0, y=50.0, length=100.0, beam=50.0, d_safe=0.0),
        p.ObstacleData(x=1500.0, y=800.0, length=100.0, beam=50.0, d_safe=0.0),
    ]


@pytest.fixture
def kinematic_ship():
    """Create a Kinematic_Ship instance."""
    return p.Kinematic_Ship(length=150.0, beam=25.0)


@pytest.fixture
def cpe_estimator():
    """Create a CPE estimator with reduced parameters for fast tests."""
    return p.CPE(max_iter=10, tolerance=1e-3, n_samples=100)


@pytest.fixture
def colregs_evaluator():
    """Create a COLREGS evaluator."""
    return p.COLREGS_Evaluator()


@pytest.fixture
def mpc_cost_evaluator(psbmpc_params, grounding_hazards):
    """Create a full MPC cost evaluator."""
    return p.MPC_Cost(
        params=psbmpc_params,
        grounding_hazards=grounding_hazards,
    )
