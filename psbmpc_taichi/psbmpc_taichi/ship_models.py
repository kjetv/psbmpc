"""Ship trajectory prediction models.

Implements kinematic and kinetic ship models for trajectory prediction
in the MPC framework. Ported from the C++/CUDA implementations.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .types import ShipState4, ShipState6, Waypoint
from .utils import (
    Xoshiro256pp,
    angle_diff,
    clamp,
    distance_2d,
    normalize_angle,
    ship_polygon,
)


# ============================================================================
# Kinematic Ship Model (4DOF)
# ============================================================================


class Kinematic_Ship:
    """Kinematic ship model with 4 state variables.

    State: [x, y, chi (heading), U (surge speed)]
    Uses LOS (Line-of-Sight) waypoint following for guidance.
    Supports both linear and ERK1 (Runge-Kutta) integration methods.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        chi: float = 0.0,
        U: float = 10.0,  # knots (convert as needed)
        length: float = 150.0,
        beam: float = 25.0,
        los_range: float = 500.0,  # LOS range in meters
        max_rudder: float = math.pi / 6,  # max rudder angle (30 degrees)
        time_constant: float = 20.0,  # ship response time constant
    ):
        """Initialize kinematic ship model.

        Args:
            x, y: initial position (meters, UTM or local)
            chi: initial heading (radians)
            U: initial surge speed (m/s or knots, depending on context)
            length: ship length (meters)
            beam: ship beam (meters)
            los_range: LOS range for waypoint following (meters)
            max_rudder: maximum rudder angle (radians)
            time_constant: ship response time constant (seconds)
        """
        self.x = x
        self.y = y
        self.chi = chi
        self.U = U
        self.length = length
        self.beam = beam
        self.los_range = los_range
        self.max_rudder = max_rudder
        self.time_constant = time_constant

        # Current waypoint index
        self.current_wp = 0

        # Store waypoints
        self.waypoints: List[Waypoint] = []

    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set the route waypoints.

        Args:
            waypoints: list of (x, y) tuples or Waypoint objects
        """
        processed = []
        for i, wp in enumerate(waypoints):
            if isinstance(wp, Waypoint):
                processed.append(Waypoint(x=wp.x, y=wp.y, id=i))
            else:
                processed.append(Waypoint(x=wp[0], y=wp[1], id=i))
        self.waypoints = processed
        self.current_wp = 0

    def update_guidance_references(
        self,
        u_d: float,
        chi_d: float,
        waypoints: List[Waypoint],
        xs: ShipState4,
        dt: float,
        method: str = "linear",
    ) -> ShipState4:
        """Update guidance references using LOS following.

        Args:
            u_d: desired surge speed
            chi_d: desired heading
            waypoints: route waypoints
            xs: current ship state
            dt: time step
            method: integration method ("linear" or "erk1")

        Returns:
            Updated ship state
        """
        # Store waypoints if provided
        if waypoints:
            self.waypoints = waypoints

        # Compute desired heading based on LOS
        if self.waypoints and self.current_wp < len(self.waypoints):
            wp = self.waypoints[self.current_wp]
            target_x, target_y = wp.x, wp.y

            # Compute bearing to waypoint
            bearing = math.atan2(target_y - xs.y, target_x - xs.x)
            cross_track = distance_2d(xs.x, xs.y, target_x, target_y)

            # Update current waypoint if within range
            if cross_track < self.los_range * 0.5:
                self.current_wp = min(self.current_wp + 1, len(self.waypoints) - 1)

            # Compute course offset based on cross-track error
            course_offset = normalize_angle(bearing - xs.chi)
            desired_chi = normalize_angle(xs.chi + course_offset * 0.5)
        else:
            desired_chi = chi_d

        # Integrate using selected method
        if method == "erk1":
            return self._predict_erks(xs, desired_chi, u_d, dt)
        else:
            return self._predict_linear(xs, desired_chi, u_d, dt)

    def predict(
        self,
        xs: ShipState4,
        U_d: Optional[float] = None,
        chi_d: Optional[float] = None,
        dt: float = 1.0,
        method: str = "linear",
    ) -> ShipState4:
        """Predict ship state for one time step.

        Args:
            xs: current ship state
            U_d: desired surge speed (uses current if None)
            chi_d: desired heading (uses current if None)
            dt: time step (seconds)
            method: integration method ("linear" or "erk1")

        Returns:
            Predicted ship state
        """
        if U_d is None:
            U_d = xs.U
        if chi_d is None:
            chi_d = xs.chi

        if method == "erk1":
            return self._predict_erks(xs, chi_d, U_d, dt)
        else:
            return self._predict_linear(xs, chi_d, U_d, dt)

    def _predict_linear(self, xs: ShipState4, chi_d: float, U_d: float, dt: float) -> ShipState4:
        """Linear integration (Euler method).

        State dynamics:
            x_dot = U * cos(chi)
            y_dot = U * sin(chi)
            chi_dot = (chi_d - chi) / time_constant
            U_dot = (U_d - U) / time_constant
        """
        # Heading dynamics (first-order response)
        chi_dot = (chi_d - xs.chi) / self.time_constant
        chi_new = xs.chi + chi_dot * dt

        # Surge dynamics (first-order response)
        U_dot = (U_d - xs.U) / self.time_constant
        U_new = xs.U + U_dot * dt

        # Position dynamics
        x_new = xs.x + xs.U * math.cos(xs.chi) * dt
        y_new = xs.y + xs.U * math.sin(xs.chi) * dt

        return ShipState4(x=x_new, y=y_new, chi=normalize_angle(chi_new), U=U_new)

    def _predict_erks(
        self, xs: ShipState4, chi_d: float, U_d: float, dt: float
    ) -> ShipState4:
        """Explicit Runge-Kutta 1st order (RK1/Heun) integration.

        Uses two evaluations for improved accuracy over Euler.
        """
        # State derivatives
        def derivatives(state_x, state_y, state_chi, state_U):
            chi_dot = (chi_d - state_chi) / self.time_constant
            U_dot = (U_d - state_U) / self.time_constant
            return (
                state_U * math.cos(state_chi),
                state_U * math.sin(state_chi),
                chi_dot,
                U_dot,
            )

        # Euler step (predictor)
        k1_x, k1_y, k1_chi, k1_U = derivatives(xs.x, xs.y, xs.chi, xs.U)
        pred_x = xs.x + k1_x * dt
        pred_y = xs.y + k1_y * dt
        pred_chi = xs.chi + k1_chi * dt
        pred_U = xs.U + k1_U * dt

        # RK1 correction
        k2_x, k2_y, k2_chi, k2_U = derivatives(pred_x, pred_y, pred_chi, pred_U)

        # Average
        x_new = xs.x + 0.5 * (k1_x + k2_x) * dt
        y_new = xs.y + 0.5 * (k1_y + k2_y) * dt
        chi_new = xs.chi + 0.5 * (k1_chi + k2_chi) * dt
        U_new = xs.U + 0.5 * (k1_U + k2_U) * dt

        return ShipState4(
            x=x_new, y=y_new,
            chi=normalize_angle(chi_new),
            U=U_new,
        )

    def predict_trajectory(
        self,
        xs: ShipState4,
        offsets: Optional[List[float]] = None,
        times: Optional[List[float]] = None,
        waypoints: Optional[List[Waypoint]] = None,
        T: float = 300.0,
        dt: float = 1.0,
        method: str = "linear",
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Predict full trajectory over the MPC horizon.

        Args:
            xs: initial ship state
            offsets: heading offsets to apply (list of radian offsets)
            times: time points (if None, uses range(n_steps)*dt)
            waypoints: route waypoints (uses stored if None)
            T: prediction horizon (seconds)
            dt: time step (seconds)
            method: integration method

        Returns:
            (traj_x, traj_y, traj_chi, traj_U): trajectory arrays
        """
        n_steps = int(T / dt)
        traj_x = [xs.x]
        traj_y = [xs.y]
        traj_chi = [xs.chi]
        traj_U = [xs.U]

        current_state = ShipState4(x=xs.x, y=xs.y, chi=xs.chi, U=xs.U)

        # Apply default offset if none provided
        if offsets is None:
            offsets = [1.0] * n_steps  # default 1 radian offset

        for i in range(n_steps):
            # Get offset for this step
            offset_chi = offsets[i] if i < len(offsets) else 0.0
            offset_U = 0.0  # default surge offset

            # Apply offset to desired heading
            chi_d = normalize_angle(xs.chi + offset_chi)
            U_d = xs.U + offset_U

            # Predict next state
            current_state = self.predict(
                current_state, U_d=U_d, chi_d=chi_d, dt=dt, method=method
            )

            traj_x.append(current_state.x)
            traj_y.append(current_state.y)
            traj_chi.append(current_state.chi)
            traj_U.append(current_state.U)

        return traj_x, traj_y, traj_chi, traj_U


# ============================================================================
# Kinetic Ship Model (3DOF)
# ============================================================================


class Kinetic_Ship:
    """Kinetic ship model with 6DOF state (3DOF dynamics).

    State: [x, y, psi, u, v, r]
    Includes: inertia matrix, Coriolis forces, damping.
    Lower priority than kinematic model.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
        u: float = 2.0,  # m/s
        v: float = 0.0,
        r: float = 0.0,
        mass: float = 1.0e6,  # kg
        length: float = 150.0,
        beam: float = 25.0,
        I_z: float = 5.0e10,  # moment of inertia about z
        X_u_dot: float = -0.5e6,  # hydrodynamic derivative
        Y_v_dot: float = -1.0e6,
        N_r_dot: float = -2.0e9,
        X_u: float = -5.0e4,  # hydrodynamic damping
        Y_v: float = -1.0e5,
        N_r: float = -2.0e6,
    ):
        """Initialize kinetic ship model.

        Args:
            x, y: initial position
            psi: initial heading (radians)
            u: surge velocity (m/s)
            v: sway velocity (m/s)
            r: yaw rate (rad/s)
            mass: ship mass (kg)
            length: ship length (m)
            beam: ship beam (m)
            I_z: moment of inertia about z-axis
            Hydrodynamic derivatives for X_dot, Y_dot, N_dot
            Hydrodynamic damping coefficients for X, Y, N
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.u = u
        self.v = v
        self.r = r
        self.mass = mass
        self.length = length
        self.beam = beam
        self.I_z = I_z

        # Hydrodynamic derivatives
        self.X_u_dot = X_u_dot
        self.Y_v_dot = Y_v_dot
        self.N_r_dot = N_r_dot

        # Damping coefficients
        self.X_u = X_u
        self.Y_v = Y_v
        self.N_r = N_r

        # Control inputs
        self.delta = 0.0  # rudder angle
        self.T_prop = 0.0  # propeller thrust

    def predict(
        self,
        state: ShipState6,
        dt: float = 1.0,
        delta: float = 0.0,
        T_prop: float = 0.0,
    ) -> ShipState6:
        """Predict ship state for one time step.

        Args:
            state: current ship state (ShipState6)
            dt: time step (seconds)
            delta: rudder angle (radians)
            T_prop: propeller thrust (N)

        Returns:
            Predicted ship state
        """
        # Save current state
        self.x = state.x
        self.y = state.y
        self.psi = state.psi
        self.u = state.u
        self.v = state.v
        self.r = state.r

        # Integrate for one step
        result = self.integrate(dt=dt, delta=delta, T_prop=T_prop, method="euler")
        return result

    def dynamics(
        self, u: float, v: float, r: float, psi: float,
        delta: float = 0.0, T_prop: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Compute 3DOF dynamics (u_dot, v_dot, r_dot).

        Uses standard MMG-style maneuvering model simplified for surface vessels.

        Args:
            u: surge velocity
            v: sway velocity
            r: yaw rate
            psi: heading
            delta: rudder angle
            T_prop: propeller thrust

        Returns:
            (u_dot, v_dot, r_dot)
        """
        m = self.mass
        I_z = self.I_z

        # Effective surge speed (simplified)
        U_eff = max(abs(u), 0.1)

        # Hydrodynamic forces (simplified linear model)
        # Surge
        X_hydro = self.X_u * u * abs(u) / U_eff + self.X_u_dot * 0.0  # simplified
        X_total = T_prop + X_hydro

        # Sway
        Y_hydro = self.Y_v * v * abs(v) / U_eff
        Y_rudder = -self.mass * u * delta / self.length  # simplified rudder force
        Y_total = Y_hydro + Y_rudder

        # Yaw moment
        N_hydro = self.N_r * r * abs(r) / U_eff
        N_rudder = self.mass * u * delta * self.length / 4.0  # simplified
        N_total = N_hydro + N_rudder

        # Equations of motion
        u_dot = X_total / m
        v_dot = Y_total / (m + abs(self.Y_v_dot))
        r_dot = N_total / (I_z + abs(self.N_r_dot))

        return float(u_dot), float(v_dot), float(r_dot)

    def integrate(
        self, dt: float = 1.0, delta: float = 0.0, T_prop: float = 0.0,
        method: str = "euler",
    ) -> ShipState6:
        """Integrate ship dynamics for one time step.

        Args:
            dt: time step
            delta: rudder angle (radians)
            T_prop: propeller thrust (N)
            method: integration method ("euler" or "runge_kutta")

        Returns:
            Updated ship state
        """
        if method == "runge_kutta":
            # RK4 integration
            k1_u, k1_v, k1_r = self.dynamics(self.u, self.v, self.r, self.psi, delta, T_prop)
            k1_psi = self.r

            u2 = self.u + 0.5 * dt * k1_u
            v2 = self.v + 0.5 * dt * k1_v
            r2 = self.r + 0.5 * dt * k1_r
            psi2 = self.psi + 0.5 * dt * k1_psi

            k2_u, k2_v, k2_r = self.dynamics(u2, v2, r2, psi2, delta, T_prop)
            k2_psi = r2

            u3 = self.u + 0.5 * dt * k2_u
            v3 = self.v + 0.5 * dt * k2_v
            r3 = self.r + 0.5 * dt * k2_r
            psi3 = self.psi + 0.5 * dt * k2_psi

            k3_u, k3_v, k3_r = self.dynamics(u3, v3, r3, psi3, delta, T_prop)
            k3_psi = r3

            u4 = self.u + dt * k3_u
            v4 = self.v + dt * k3_v
            r4 = self.r + dt * k3_r
            psi4 = self.psi + dt * k3_psi

            k4_u, k4_v, k4_r = self.dynamics(u4, v4, r4, psi4, delta, T_prop)
            k4_psi = r4

            self.u += dt / 6.0 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
            self.v += dt / 6.0 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            self.r += dt / 6.0 * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            self.psi = normalize_angle(self.psi + dt / 6.0 * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi))
        else:
            # Euler integration
            u_dot, v_dot, r_dot = self.dynamics(self.u, self.v, self.r, self.psi, delta, T_prop)
            self.u += u_dot * dt
            self.v += v_dot * dt
            self.r += r_dot * dt
            self.psi = normalize_angle(self.psi + self.r * dt)

        # Update position
        self.x += (self.u * math.cos(self.psi) - self.v * math.sin(self.psi)) * dt
        self.y += (self.u * math.sin(self.psi) + self.v * math.cos(self.psi)) * dt

        return ShipState6(
            x=self.x, y=self.y, psi=self.psi,
            u=self.u, v=self.v, r=self.r,
        )

    def predict_trajectory(
        self,
        T: float = 300.0,
        dt: float = 1.0,
        delta: float = 0.0,
        T_prop: float = 0.0,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """Predict full trajectory over the MPC horizon.

        Args:
            T: prediction horizon (seconds)
            dt: time step
            delta: constant rudder angle
            T_prop: constant propeller thrust

        Returns:
            (x, y, psi, u, v, r) trajectory arrays
        """
        n_steps = int(T / dt)
        traj_x, traj_y, traj_psi = [self.x], [self.y], [self.psi]
        traj_u, traj_v, traj_r = [self.u], [self.v], [self.r]

        for _ in range(n_steps):
            self.integrate(dt=dt, delta=delta, T_prop=T_prop)
            traj_x.append(self.x)
            traj_y.append(self.y)
            traj_psi.append(self.psi)
            traj_u.append(self.u)
            traj_v.append(self.v)
            traj_r.append(self.r)

        return traj_x, traj_y, traj_psi, traj_u, traj_v, traj_r


# ============================================================================
# Obstacle Ship Model
# ============================================================================


class Obstacle_Ship:
    """Simple LOS follower for obstacle ship prediction.

    Used for predicting obstacle trajectories in collision avoidance.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        chi: float = 0.0,
        U: float = 5.0,
        length: float = 100.0,
        beam: float = 20.0,
        los_range: float = 400.0,
    ):
        """Initialize obstacle ship model.

        Args:
            x, y: initial position
            chi: initial heading
            U: initial speed
            length: ship length
            beam: ship beam
            los_range: LOS range for waypoint following
        """
        self.x = x
        self.y = y
        self.chi = chi
        self.U = U
        self.length = length
        self.beam = beam
        self.los_range = los_range

        self.waypoints: List[Waypoint] = []
        self.current_wp = 0

    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set obstacle route waypoints."""
        self.waypoints = [Waypoint(x=wp[0], y=wp[1], id=i) for i, wp in enumerate(waypoints)]

    def predict(self, xs: ShipState4, T: float = 100.0, dt: float = 1.0) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Predict obstacle trajectory.

        Args:
            xs: current ship state
            T: prediction horizon (seconds)
            dt: time step (seconds)

        Returns:
            (x, y, chi, U) trajectory arrays
        """
        return self.predict_trajectory(T=T, dt=dt)

    def predict_step(self, dt: float = 1.0) -> ShipState4:
        """Predict one step of obstacle motion.

        Simple constant velocity model with LOS waypoint following.

        Args:
            dt: time step

        Returns:
            Updated ship state
        """
        if self.waypoints and self.current_wp < len(self.waypoints):
            wp = self.waypoints[self.current_wp]
            bearing = math.atan2(wp.y - self.y, wp.x - self.x)
            dist = distance_2d(self.x, self.y, wp.x, wp.y)

            # Update waypoint
            if dist < self.los_range * 0.5:
                self.current_wp = min(self.current_wp + 1, len(self.waypoints) - 1)

            # Heading control
            desired_chi = bearing
            chi_error = normalize_angle(desired_chi - self.chi)
            self.chi = normalize_angle(self.chi + chi_error * 0.1)
        else:
            # No waypoints, maintain course
            pass

        # Update position
        self.x += self.U * math.cos(self.chi) * dt
        self.y += self.U * math.sin(self.chi) * dt

        return ShipState4(x=self.x, y=self.y, chi=self.chi, U=self.U)

    def predict_trajectory(
        self,
        T: float = 300.0,
        dt: float = 1.0,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Predict full trajectory.

        Args:
            T: prediction horizon
            dt: time step

        Returns:
            (x, y, chi, U) trajectory arrays
        """
        n_steps = int(T / dt)
        traj_x, traj_y, traj_chi, traj_U = [self.x], [self.y], [self.chi], [self.U]

        for _ in range(n_steps):
            state = self.predict_step(dt)
            traj_x.append(state.x)
            traj_y.append(state.y)
            traj_chi.append(state.chi)
            traj_U.append(state.U)

        return traj_x, traj_y, traj_chi, traj_U
