#!/usr/bin/env python3
"""Quick verification script for the PSB-MPC solver."""
import sys

print("=== Importing psbmpc_taichi ===")
import psbmpc_taichi as p
print(f"Package imported successfully")
print(f"PSBMPC_Solver: {p.PSBMPC_Solver}")
print(f"SBMPC_Solver: {p.SBMPC_Solver}")
print(f"PSBMPCParameters: {p.PSBMPCParameters}")
print(f"SBMPCParameters: {p.SBMPCParameters}")
print(f"ShipState4: {p.ShipState4}")
print(f"ShipState6: {p.ShipState6}")
print(f"ObstacleData: {p.ObstacleData}")
print(f"Waypoint: {p.Waypoint}")
print(f"CPEResult: {p.CPEResult}")
print(f"MPCResult: {p.MPCResult}")
print(f"Kinematic_Ship: {p.Kinematic_Ship}")
print(f"Obstacle_Ship: {p.Obstacle_Ship}")
print(f"CPE: {p.CPE}")
print(f"MPC_Cost: {p.MPC_Cost}")
print(f"COLREGS_Evaluator: {p.COLREGS_Evaluator}")

# Create a ship state
print("\n=== Creating ShipState4 ===")
state4 = p.ShipState4(x=0.0, y=0.0, chi=0.0, U=5.0)
print(f"State: x={state4.x}, y={state4.y}, chi={state4.chi}, U={state4.U}")

# Create waypoints
print("\n=== Creating Waypoints ===")
wp1 = p.Waypoint(x=1000.0, y=0.0)
wp2 = p.Waypoint(x=2000.0, y=1000.0)
print(f"WP1: ({wp1.x}, {wp1.y})")
print(f"WP2: ({wp2.x}, {wp2.y})")

# Create obstacle
print("\n=== Creating Obstacle ===")
obs = p.ObstacleData(x=500.0, y=200.0, chi=3.14, U=3.0,
                     d_safe=300.0, length=150.0, beam=25.0)
print(f"Obs: ({obs.x}, {obs.y}), d_safe={obs.d_safe}")

# Create parameters
print("\n=== Creating PSBMPCParameters ===")
params = p.PSBMPCParameters()
print(f"cpe_max_iter={params.cpe_max_iter}")
print(f"n_M={params.n_M}")
print(f"n_cbs={params.n_cbs}")
print(f"T={params.T}, dt={params.dt}")

# Test Kinematic_Ship
print("\n=== Testing Kinematic_Ship ===")
ship = p.Kinematic_Ship(length=150.0, beam=25.0)
ship.set_waypoints([(1000.0, 0.0), (2000.0, 1000.0)])
print(f"LOS range: {ship.los_range}")
traj = ship.predict_trajectory(state4, offsets=[0.0]*300, T=300.0, dt=1.0)
print(f"Trajectory length: {len(traj[0])}")
print(f"Final x: {traj[0][-1]:.2f}, y: {traj[1][-1]:.2f}")

# Test CPE
print("\n=== Testing CPE ===")
cpe = p.CPE(max_iter=10, tolerance=1e-3, n_samples=100)
# Create ownship ObstacleData for CPE (CPE uses ObstacleData for both ships)
ownship_obs = p.ObstacleData(x=state4.x, y=state4.y, 
                              length=150.0, beam=25.0, d_safe=0.0)
result = cpe.ce_estimate(ownship_obs, obs)
print(f"CE collision probability: {result.probability:.6f}")
print(f"CE iterations: {result.iterations}")
print(f"CE converged: {result.converged}")

# Test COLREGS_Evaluator
print("\n=== Testing COLREGS_Evaluator ===")
colregs = p.COLREGS_Evaluator()
# detect_situation expects (ownship_chi, ownship_U, obstacle)
situation = colregs.detect_situation(state4.chi, state4.U, obs)
print(f"Situation: {situation}")

# Test Path_Grounding_Cost
print("\n=== Testing Path_Grounding_Cost ===")
path_cost = p.Path_Grounding_Cost()
wp_list = [wp1, wp2]
pc = path_cost.calculate_path_cost(traj[0], traj[1], wp_list, (0.0, 0.0))
print(f"Path cost: {pc:.2f}")

# Test PSBMPC_Solver
print("\n=== Testing PSBMPC_Solver ===")
solver = p.PSBMPC_Solver(params, ownship_length=150.0, ownship_beam=25.0)
result = solver.calculate_optimal_offsets(
    xs=state4,
    obstacles=[obs],
    waypoints=[wp1, wp2]
)
print(f"Optimal offset_chi: {result.offset_chi:.4f}")
print(f"Optimal offset_U: {result.offset_U:.4f}")
print(f"Total cost: {result.total_cost:.2f}")
print(f"Path cost: {result.path_cost:.2f}")
print(f"Collision cost: {result.collision_cost:.2f}")
print(f"COLREGS cost: {result.colregs_cost:.2f}")
print(f"Trajectory length: {len(result.traj_x)}")

print("\n=== ALL TESTS PASSED ===")
