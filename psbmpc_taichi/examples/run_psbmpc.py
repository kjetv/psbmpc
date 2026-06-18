#!/usr/bin/env python3
"""
PSB-MPC Taichi Example - Basic Maritime Trajectory Planning

Demonstrates the PSB-MPC solver navigating a ship through waypoints
while avoiding dynamic obstacles. Similar to run_psbmpc.cpp.

Usage:
    python examples/run_psbmpc.py
"""
import math
import time

import numpy as np

import psbmpc_taichi as p


def main():
    # ========================================================================
    # Simulation Setup
    # ========================================================================
    T_sim = 100.0          # Total simulation time (seconds)
    dt = 0.5               # Time step (seconds)
    N = int(round(T_sim / dt))  # Number of MPC steps

    print("=" * 70)
    print("PSB-MPC Taichi - Basic Example")
    print("=" * 70)
    print(f"Simulation time: {T_sim}s | Time step: {dt}s | Steps: {N}")

    # ========================================================================
    # Own-ship Setup
    # ========================================================================
    offset = 300.0
    xs_os_0 = p.ShipState4(
        x=-248.0 + offset,
        y=-380.0 + offset,
        chi=math.radians(48),
        U=2.0
    )
    u_d = 2.0       # Desired speed (m/s)
    chi_d = 0.0     # Desired heading (rad)

    ownship_length = 150.0
    ownship_beam = 25.0

    # Waypoints (relative to offset)
    waypoints = [
        p.Waypoint(x=-248.0 + offset, y=-380.0 + offset),
        p.Waypoint(x=-80.0 + offset, y=-180.0 + offset),
        p.Waypoint(x=-36.0 + offset, y=-138.0 + offset),
    ]

    # ========================================================================
    # Obstacle Setup
    # ========================================================================
    # Create a dynamic obstacle (another ship)
    obstacles = [
        p.ObstacleData(
            x=400.0,
            y=100.0,
            chi=math.pi,          # Heading toward own-ship
            U=2.5,                # Speed
            length=150.0,
            beam=25.0,
            d_safe=300.0,         # Safety distance
        ),
    ]

    # ========================================================================
    # MPC Parameters
    # ========================================================================
    params = p.PSBMPCParameters(
        T=T_sim,            # Prediction horizon
        dt=dt,              # Time step
        n_M=5,              # Number of control intervals
        n_cbs=11,           # Number of candidate headings
        cpe_max_iter=50,
        cpe_tolerance=1e-4,
        cpe_n_samples=1000,
    )

    # ========================================================================
    # Initialize Solver
    # ========================================================================
    solver = p.PSBMPC_Solver(
        params=params,
        ownship_length=ownship_length,
        ownship_beam=ownship_beam,
    )

    # ========================================================================
    # Simulation Loop
    # ========================================================================
    # Storage for trajectory history
    traj_history_x = [xs_os_0.x]
    traj_history_y = [xs_os_0.y]
    traj_history_chi = [xs_os_0.chi]
    traj_history_U = [xs_os_0.U]

    colav_active_count = 0
    start_time = time.time()

    for step in range(N):
        current_time = step * dt

        # Calculate optimal offsets
        result = solver.calculate_optimal_offsets(
            xs=xs_os_0,
            obstacles=obstacles,
            waypoints=waypoints,
            u_d=u_d,
            chi_d=chi_d,
            active_mode="COLAV",  # Collision avoidance mode
        )

        # Track COLAV activations
        if result.total_cost > 1.0:
            colav_active_count += 1

        # Store trajectory
        traj_history_x.append(result.traj_x[-1])
        traj_history_y.append(result.traj_y[-1])
        traj_history_chi.append(result.traj_chi[-1])
        traj_history_U.append(result.traj_U[-1])

        # Update own-ship state (apply the optimal control)
        xs_os_0 = p.ShipState4(
            x=result.traj_x[-1],
            y=result.traj_y[-1],
            chi=result.traj_chi[-1],
            U=result.traj_U[-1],
        )

        # Print progress
        if step % 10 == 0:
            dist_to_next = math.sqrt(
                (waypoints[1].x - traj_history_x[-1]) ** 2 +
                (waypoints[1].y - traj_history_y[-1]) ** 2
            )
            print(f"t={current_time:6.1f}s | "
                  f"pos=({traj_history_x[-1]:8.1f}, {traj_history_y[-1]:8.1f}) | "
                  f"chi={math.degrees(result.traj_chi[-1]):6.1f}° | "
                  f"U={result.traj_U[-1]:4.2f} m/s | "
                  f"cost={result.total_cost:8.4f} | "
                  f"dist_to_wp2={dist_to_next:8.1f}")

    elapsed = time.time() - start_time

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print(f"Final position:    ({traj_history_x[-1]:.2f}, {traj_history_y[-1]:.2f})")
    print(f"Final heading:     {math.degrees(traj_history_chi[-1]):.2f}°")
    print(f"Final speed:       {traj_history_U[-1]:.2f} m/s")
    print(f"COLAV activations: {colav_active_count}/{N} steps ({100*colav_active_count/N:.1f}%)")
    print(f"Wall time:         {elapsed:.2f}s")

    # ========================================================================
    # Export trajectory data
    # ========================================================================
    trajectory = np.column_stack([
        traj_history_x, traj_history_y, traj_history_chi, traj_history_U
    ])
    np.save("examples/trajectory_basic.npy", trajectory)
    print(f"\nTrajectory saved to: examples/trajectory_basic.npy")

    # ========================================================================
    # Simple ASCII Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("Trajectory Overview (ASCII)")
    print("=" * 70)

    # Create a 2D grid representation
    x_min, x_max = min(traj_history_x) - 50, max(traj_history_x) + 50
    y_min, y_max = min(traj_history_y) - 50, max(traj_history_y) + 50
    x_range = x_max - x_min
    y_range = y_max - y_min

    grid_width = 60
    grid_height = 30
    grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]

    # Plot waypoints
    for wp in waypoints:
        gx = int((wp.x - x_min) / x_range * grid_width)
        gy = int((wp.y - y_min) / y_range * grid_height)
        gx = max(0, min(grid_width - 1, gx))
        gy = max(0, min(grid_height - 1, gy))
        grid[grid_height - 1 - gy][gx] = 'W'

    # Plot obstacle
    obs = obstacles[0]
    gx = int((obs.x - x_min) / x_range * grid_width)
    gy = int((obs.y - y_min) / y_range * grid_height)
    gx = max(0, min(grid_width - 1, gx))
    gy = max(0, min(grid_height - 1, gy))
    grid[grid_height - 1 - gy][gx] = 'O'

    # Plot trajectory
    for i in range(0, len(traj_history_x), max(1, len(traj_history_x) // 200)):
        gx = int((traj_history_x[i] - x_min) / x_range * grid_width)
        gy = int((traj_history_y[i] - y_min) / y_range * grid_height)
        gx = max(0, min(grid_width - 1, gx))
        gy = max(0, min(grid_height - 1, gy))
        if grid[grid_height - 1 - gy][gx] == '.':
            grid[grid_height - 1 - gy][gx] = '*'

    # Plot start and end
    sx = int((traj_history_x[0] - x_min) / x_range * grid_width)
    sy = int((traj_history_y[0] - y_min) / y_range * grid_height)
    grid[grid_height - 1 - sy][sx] = 'S'

    ex = int((traj_history_x[-1] - x_min) / x_range * grid_width)
    ey = int((traj_history_y[-1] - y_min) / y_range * grid_height)
    grid[grid_height - 1 - ey][ex] = 'E'

    # Print grid
    print("  " + " " * (grid_width + 2))
    for row in grid:
        print("  " + "".join(row))
    print("  " + "-" * (grid_width + 2))
    print(f"  {x_min:.0f}" + " " * (grid_width - 10) + f"{x_max:.0f}")
    print(f"  S=Start, W=Waypoint, O=Obstacle, *=Trajectory, E=End")
    print(f"  Legend: O = obstacle ship at ({obs.x:.0f}, {obs.y:.0f})")


if __name__ == "__main__":
    main()
