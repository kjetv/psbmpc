#!/usr/bin/env python3
"""
PSB-MPC Taichi Example - Grounding Hazard Avoidance

Demonstrates the PSB-MPC solver navigating around grounding hazards
(shoals, rocks, shallow water areas) while following waypoints.
Similar to run_psbmpc_with_grounding_hazards.cpp.

Usage:
    python examples/run_psbmpc_grounding.py
"""
import math
import time

import numpy as np

import psbmpc_taichi as p


def create_shapefile_polygon(x_coords, y_coords):
    """Create a grounding hazard polygon from coordinate lists.

    Args:
        x_coords: list of x coordinates (meters, local frame)
        y_coords: list of y coordinates (meters, local frame)

    Returns:
        ObstacleData representing a grounding hazard
    """
    # Compute center and orientation from coordinates
    cx = sum(x_coords) / len(x_coords)
    cy = sum(y_coords) / len(y_coords)
    length = max(x_coords) - min(x_coords)
    beam = max(y_coords) - min(y_coords)
    return p.ObstacleData(
        x=cx, y=cy,
        chi=0.0,
        U=0.0,
        length=max(length, beam),
        beam=max(length, beam),
        d_safe=50.0,  # Safety margin from hazard
    )


def main():
    # ========================================================================
    # Simulation Setup
    # ========================================================================
    T_sim = 200.0          # Total simulation time (seconds)
    dt = 0.5               # Time step (seconds)
    N = int(round(T_sim / dt))  # Number of MPC steps

    print("=" * 70)
    print("PSB-MPC Taichi - Grounding Hazard Avoidance Example")
    print("=" * 70)
    print(f"Simulation time: {T_sim}s | Time step: {dt}s | Steps: {N}")

    # ========================================================================
    # Own-ship Setup (realistic coordinates from C++ example)
    # ========================================================================
    offset = 0.0
    xs_os_0 = p.ShipState4(
        x=250.0 + offset,
        y=1190.0 + offset,
        chi=0.306,          # ~17.5 degrees
        U=1.0
    )
    u_d = 1.0       # Desired speed (m/s) - slow navigation
    chi_d = 0.0     # Desired heading

    ownship_length = 150.0
    ownship_beam = 25.0

    # Waypoints (similar to C++ example)
    waypoints = [
        p.Waypoint(x=xs_os_0.x, y=xs_os_0.y),
        p.Waypoint(x=462.0, y=1257.0),
    ]

    # ========================================================================
    # Grounding Hazards (simulated from shapefile data)
    # ========================================================================
    # These represent shallow water areas, rocks, or other navigation hazards
    grounding_hazards = [
        # Hazard 1: Shoal area near the route
        p.ObstacleData(
            x=300.0, y=1100.0,
            chi=0.0, U=0.0,
            length=120.0, beam=80.0,
            d_safe=50.0,
        ),
        # Hazard 2: Rock formation
        p.ObstacleData(
            x=400.0, y=1200.0,
            chi=0.5, U=0.0,
            length=100.0, beam=60.0,
            d_safe=50.0,
        ),
        # Hazard 3: Shallow water zone
        p.ObstacleData(
            x=350.0, y=1300.0,
            chi=-0.3, U=0.0,
            length=150.0, beam=70.0,
            d_safe=50.0,
        ),
    ]

    # ========================================================================
    # Dynamic Obstacle (another ship in the area)
    # ========================================================================
    obstacles = [
        p.ObstacleData(
            x=500.0, y=1150.0,
            chi=math.pi,       # Heading toward own-ship
            U=2.0,
            length=150.0,
            beam=25.0,
            d_safe=200.0,
        ),
    ]

    # ========================================================================
    # MPC Parameters
    # ========================================================================
    params = p.PSBMPCParameters(
        T=T_sim,
        dt=dt,
        n_M=5,
        n_cbs=11,
        cpe_max_iter=50,
        cpe_tolerance=1e-4,
        cpe_n_samples=1000,
    )

    # ========================================================================
    # Initialize Solver with Grounding Hazards
    # ========================================================================
    solver = p.PSBMPC_Solver(
        params=params,
        ownship_length=ownship_length,
        ownship_beam=ownship_beam,
        grounding_hazards=grounding_hazards,  # Pass hazards here
    )

    # ========================================================================
    # Simulation Loop
    # ========================================================================
    traj_history_x = [xs_os_0.x]
    traj_history_y = [xs_os_0.y]
    traj_history_chi = [xs_os_0.chi]
    traj_history_U = [xs_os_0.U]

    # Track cost breakdowns
    grounding_cost_history = []
    collision_cost_history = []
    path_cost_history = []

    colav_active_count = 0
    start_time = time.time()

    for step in range(N):
        current_time = step * dt

        result = solver.calculate_optimal_offsets(
            xs=xs_os_0,
            obstacles=obstacles,
            waypoints=waypoints,
            u_d=u_d,
            chi_d=chi_d,
            active_mode="COLAV",
        )

        if result.total_cost > 1.0:
            colav_active_count += 1

        traj_history_x.append(result.traj_x[-1])
        traj_history_y.append(result.traj_y[-1])
        traj_history_chi.append(result.traj_chi[-1])
        traj_history_U.append(result.traj_U[-1])

        # Track cost components
        grounding_cost_history.append(result.total_cost - result.path_cost - result.collision_cost - result.colregs_cost)
        collision_cost_history.append(result.collision_cost)
        path_cost_history.append(result.path_cost)

        # Update state
        xs_os_0 = p.ShipState4(
            x=result.traj_x[-1],
            y=result.traj_y[-1],
            chi=result.traj_chi[-1],
            U=result.traj_U[-1],
        )

        # Print progress
        if step % 20 == 0:
            dist_to_wp = math.sqrt(
                (waypoints[1].x - traj_history_x[-1]) ** 2 +
                (waypoints[1].y - traj_history_y[-1]) ** 2
            )
            print(f"t={current_time:6.1f}s | "
                  f"pos=({traj_history_x[-1]:8.1f}, {traj_history_y[-1]:8.1f}) | "
                  f"chi={math.degrees(result.traj_chi[-1]):6.1f}° | "
                  f"U={result.traj_U[-1]:4.2f} | "
                  f"cost={result.total_cost:8.4f} | "
                  f"dist={dist_to_wp:8.1f}")

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

    # Cost statistics
    if grounding_cost_history:
        print(f"\nCost Breakdown (averages):")
        print(f"  Total:           {np.mean(grounding_cost_history):.4f}")
        print(f"  Collision:       {np.mean(collision_cost_history):.4f}")
        print(f"  Path:            {np.mean(path_cost_history):.4f}")

    # ========================================================================
    # Export Data
    # ========================================================================
    trajectory = np.column_stack([
        traj_history_x, traj_history_y, traj_history_chi, traj_history_U
    ])
    np.save("examples/trajectory_grounding.npy", trajectory)
    print(f"\nTrajectory saved to: examples/trajectory_grounding.npy")

    # ========================================================================
    # ASCII Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("Trajectory Overview (ASCII)")
    print("=" * 70)

    x_min = min(traj_history_x) - 50
    x_max = max(traj_history_x) + 50
    y_min = min(traj_history_y) - 50
    y_max = max(traj_history_y) + 50
    x_range = x_max - x_min
    y_range = y_max - y_min

    grid_width = 70
    grid_height = 35
    grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]

    # Plot grounding hazards
    for i, hazard in enumerate(grounding_hazards):
        gx = int((hazard.x - x_min) / x_range * grid_width)
        gy = int((hazard.y - y_min) / y_range * grid_height)
        gx = max(0, min(grid_width - 1, gx))
        gy = max(0, min(grid_height - 1, gy))
        grid[grid_height - 1 - gy][gx] = '#'

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

    # Plot start/end
    sx = int((traj_history_x[0] - x_min) / x_range * grid_width)
    sy = int((traj_history_y[0] - y_min) / y_range * grid_height)
    grid[grid_height - 1 - sy][sx] = 'S'
    ex = int((traj_history_x[-1] - x_min) / x_range * grid_width)
    ey = int((traj_history_y[-1] - y_min) / y_range * grid_height)
    grid[grid_height - 1 - ey][ex] = 'E'

    # Print
    print("  " + " " * (grid_width + 2))
    for row in grid:
        print("  " + "".join(row))
    print("  " + "-" * (grid_width + 2))
    print(f"  {x_min:.0f}" + " " * (grid_width - 10) + f"{x_max:.0f}")
    print(f"  S=Start, W=Waypoint, O=Obstacle, *=Trajectory, E=End")
    print(f"  #=Grounding Hazard (shoals, rocks, shallow water)")


if __name__ == "__main__":
    main()
