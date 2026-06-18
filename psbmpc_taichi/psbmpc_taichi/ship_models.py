"""Ship trajectory prediction models.

Implements kinematic and kinetic ship models for trajectory prediction
in the MPC framework. Ported from the C++/CUDA implementations.

Supports both CPU (pure Python) and GPU (Taichi) execution modes.
"""
import math
from typing import List, Optional, Tuple

import numpy as np

from .types import ShipState4, ShipState6, Waypoint, TI_AVAILABLE, ti
from .utils import (
    Xoshiro256pp,
    angle_diff,
    clamp,
    distance_2d,
    normalize_angle,
    ship_polygon,
)


# ============================================================================
# GPU Kernels (Taichi)
# ============================================================================

if TI_AVAILABLE and ti is not None:
    # ========================================================================
    # Batched trajectory prediction kernel (all candidates in one launch)
    # ========================================================================
    @ti.kernel
    def predict_trajectory_batch_taichi(
        xs_x: ti.f32, xs_y: ti.f32, xs_chi: ti.f32, xs_U: ti.f32,
        offsets: ti.types.ndarray(),
        n_candidates: ti.i32,
        n_steps: ti.i32,
        dt: ti.f32,
        time_constant: ti.f32,
        use_rk1: ti.i32,
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        traj_chi: ti.types.ndarray(),
        traj_U: ti.types.ndarray(),
    ):
        """Batch trajectory prediction for all candidate bearings.

        Args:
            xs_x, xs_y, xs_chi, xs_U: initial ship state
            offsets: [n_candidates] heading offsets
            n_candidates: number of candidate bearings
            n_steps: number of integration steps
            dt: time step
            time_constant: ship response time constant
            use_rk1: 0=Euler, 1=RK1
            traj_x, traj_y, traj_chi, traj_U: [n_candidates x (n_steps+1)] output arrays
        """
        for c in range(n_candidates):
            offset_chi = offsets[c]
            chi_d = xs_chi + offset_chi
            # Simple normalization
            while chi_d > 3.141592653589793:
                chi_d -= 6.283185307179586
            while chi_d < -3.141592653589793:
                chi_d += 6.283185307179586

            # Store initial state
            traj_x[c, 0] = xs_x
            traj_y[c, 0] = xs_y
            traj_chi[c, 0] = xs_chi
            traj_U[c, 0] = xs_U

            # Integrate
            cx, cy, cchi, cU = xs_x, xs_y, xs_chi, xs_U
            for s in range(n_steps):
                # Heading dynamics (first-order response)
                chi_dot = (chi_d - cchi) / time_constant
                # Surge dynamics (first-order response)
                U_dot = (xs_U - cU) / time_constant
                if use_rk1 != 0:
                    # RK1 (Heun) integration - inlined to avoid tuple return issues
                    k1_x = cU * ti.cos(cchi)
                    k1_y = cU * ti.sin(cchi)
                    pred_x = cx + k1_x * dt
                    pred_y = cy + k1_y * dt
                    pred_chi = cchi + chi_dot * dt
                    pred_U = cU + U_dot * dt
                    k2_chi_dot = (chi_d - pred_chi) / time_constant
                    k2_U_dot = (xs_U - pred_U) / time_constant
                    k2_x = pred_U * ti.cos(pred_chi)
                    k2_y = pred_U * ti.sin(pred_chi)
                    cx = cx + 0.5 * (k1_x + k2_x) * dt
                    cy = cy + 0.5 * (k1_y + k2_y) * dt
                    cchi = cchi + 0.5 * (chi_dot + k2_chi_dot) * dt
                    cU = cU + 0.5 * (U_dot + k2_U_dot) * dt
                else:
                    # Euler integration
                    cx = cx + cU * ti.cos(cchi) * dt
                    cy = cy + cU * ti.sin(cchi) * dt
                    cchi = cchi + chi_dot * dt
                    cU = cU + U_dot * dt
                # Normalize heading
                while cchi > 3.141592653589793:
                    cchi -= 6.283185307179586
                while cchi < -3.141592653589793:
                    cchi += 6.283185307179586
                traj_x[c, s + 1] = cx
                traj_y[c, s + 1] = cy
                traj_chi[c, s + 1] = cchi
                traj_U[c, s + 1] = cU

    # ========================================================================
    # Batched trajectory prediction with per-step offsets
    # ========================================================================
    @ti.kernel
    def predict_trajectory_step_offsets_taichi(
        xs_x: ti.f32, xs_y: ti.f32, xs_chi: ti.f32, xs_U: ti.f32,
        offsets: ti.types.ndarray(),
        n_candidates: ti.i32,
        n_steps: ti.i32,
        dt: ti.f32,
        time_constant: ti.f32,
        use_rk1: ti.i32,
        traj_x: ti.types.ndarray(),
        traj_y: ti.types.ndarray(),
        traj_chi: ti.types.ndarray(),
        traj_U: ti.types.ndarray(),
    ):
        """Batch trajectory prediction with per-step heading offsets.

        Args:
            xs_x, xs_y, xs_chi, xs_U: initial ship state
            offsets: [n_candidates x n_steps] per-step heading offsets
            n_candidates: number of candidate bearings
            n_steps: number of integration steps
            dt: time step
            time_constant: ship response time constant
            use_rk1: 0=Euler, 1=RK1
            traj_x, traj_y, traj_chi, traj_U: [n_candidates x (n_steps+1)] output arrays
        """
        for c in range(n_candidates):
            # Store initial state
            traj_x[c, 0] = xs_x
            traj_y[c, 0] = xs_y
            traj_chi[c, 0] = xs_chi
            traj_U[c, 0] = xs_U

            # Integrate with per-step offsets
            cx, cy, cchi, cU = xs_x, xs_y, xs_chi, xs_U
            for s in range(n_steps):
                offset_idx = c * n_steps + s
                offset_chi = offsets[offset_idx]
                chi_d = cchi + offset_chi
                # Simple normalization
                while chi_d > 3.141592653589793:
                    chi_d -= 6.283185307179586
                while chi_d < -3.141592653589793:
                    chi_d += 6.283185307179586

                # Heading dynamics (first-order response)
                chi_dot = (chi_d - cchi) / time_constant
                # Surge dynamics (first-order response)
                U_dot = (cU - cU) / time_constant
                if use_rk1 != 0:
                    # RK1 (Heun) integration
                    k1_x = cU * ti.cos(cchi)
                    k1_y = cU * ti.sin(cchi)
                    pred_x = cx + k1_x * dt
                    pred_y = cy + k1_y * dt
                    pred_chi = cchi + chi_dot * dt
                    pred_U = cU + U_dot * dt
                    k2_chi_dot = (chi_d - pred_chi) / time_constant
                    k2_U_dot = (cU - pred_U) / time_constant
                    k2_x = pred_U * ti.cos(pred_chi)
                    k2_y = pred_U * ti.sin(pred_chi)
                    cx = cx + 0.5 * (k1_x + k2_x) * dt
                    cy = cy + 0.5 * (k1_y + k2_y) * dt
                    cchi = cchi + 0.5 * (chi_dot + k2_chi_dot) * dt
                    cU = cU + 0.5 * (U_dot + k2_U_dot) * dt
                else:
                    # Euler integration
                    cx = cx + cU * ti.cos(cchi) * dt
                    cy = cy + cU * ti.sin(cchi) * dt
                    cchi = cchi + chi_dot * dt
                    cU = cU + U_dot * dt
                # Normalize heading
                while cchi > 3.141592653589793:
                    cchi -= 6.283185307179586
                while cchi < -3.141592653589793:
                    cchi += 6.283185307179586
                traj_x[c, s + 1] = cx
                traj_y[c, s + 1] = cy
                traj_chi[c, s + 1] = cchi
                traj_U[c, s + 1] = cU


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

    def predict_trajectory_gpu(
        self,
        xs: ShipState4,
        offsets: List[List[float]],
        T: float = 300.0,
        dt: float = 1.0,
        method: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated batch trajectory prediction.

        Predicts trajectories for all candidate bearings in parallel.

        Args:
            xs: initial ship state
            offsets: list of candidate heading offsets (list of lists,
                     each inner list has n_steps elements)
            T: prediction horizon (seconds)
            dt: time step (seconds)
            method: integration method ("linear" or "erk1")

        Returns:
            (traj_x, traj_y, traj_chi, traj_U): trajectory arrays
                each of shape (n_candidates, n_steps+1)
        """
        n_candidates = len(offsets)
        n_steps = int(T / dt)
        use_rk1 = 1 if method == "erk1" else 0

        # Allocate output arrays on CPU
        traj_x = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_y = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_chi = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_U = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)

        # Flatten offsets for GPU
        offsets_flat = np.zeros(n_candidates * n_steps, dtype=np.float32)
        for c in range(n_candidates):
            for s in range(n_steps):
                offsets_flat[c * n_steps + s] = offsets[c][s] if s < len(offsets[c]) else 0.0

        # Launch GPU kernel
        predict_trajectory_step_offsets_taichi(
            float(xs.x), float(xs.y), float(xs.chi), float(xs.U),
            offsets_flat,
            n_candidates, n_steps,
            float(dt), float(self.time_constant),
            use_rk1,
            traj_x, traj_y, traj_chi, traj_U,
        )

        return traj_x, traj_y, traj_chi, traj_U

    def predict_trajectory_batch_gpu(
        self,
        xs: ShipState4,
        offsets: List[float],
        T: float = 300.0,
        dt: float = 1.0,
        method: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated batch trajectory prediction with constant offsets.

        Each candidate uses a constant heading offset throughout the trajectory.

        Args:
            xs: initial ship state
            offsets: list of candidate heading offsets (one per candidate)
            T: prediction horizon (seconds)
            dt: time step (seconds)
            method: integration method ("linear" or "erk1")

        Returns:
            (traj_x, traj_y, traj_chi, traj_U): trajectory arrays
                each of shape (n_candidates, n_steps+1)
        """
        n_candidates = len(offsets)
        n_steps = int(T / dt)
        use_rk1 = 1 if method == "erk1" else 0

        # Allocate output arrays on CPU
        traj_x = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_y = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_chi = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)
        traj_U = np.zeros((n_candidates, n_steps + 1), dtype=np.float32)

        # Prepare offsets array
        offsets_arr = np.array(offsets, dtype=np.float32)

        # Launch GPU kernel
        predict_trajectory_batch_taichi(
            float(xs.x), float(xs.y), float(xs.chi), float(xs.U),
            offsets_arr,
            n_candidates, n_steps,
            float(dt), float(self.time_constant),
            use_rk1,
            traj_x, traj_y, traj_chi, traj_U,
        )

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


# ============================================================================
# Obstacle Ship Model
# ============================================================================


class Obstacle_Ship(Kinematic_Ship):
    """Obstacle ship model for collision avoidance simulations.

    Extends Kinematic_Ship with obstacle-specific maneuvering behavior,
    including heading offset maneuvers and waypoint segment determination.
    """

    def __init__(
        self,
        length: float = 100.0,
        beam: float = 20.0,
        time_constant: float = 10.0,
        los_range: float = 200.0,
        current_wp: int = 0,
    ):
        """Initialize obstacle ship model.

        Args:
            length: ship length (meters)
            beam: ship beam (meters)
            time_constant: ship response time constant (seconds)
            los_range: LOS range for waypoint following (meters)
            current_wp: initial current waypoint index
        """
        super().__init__(
            x=0.0, y=0.0, chi=0.0, U=3.0,
            length=length, beam=beam,
            los_range=los_range,
            time_constant=time_constant,
        )
        self.current_wp = current_wp

    def predict(
        self,
        state: ShipState4,
        T: float = 100.0,
        dt: float = 1.0,
        U_d: Optional[float] = None,
        chi_d: Optional[float] = None,
    ) -> np.ndarray:
        """Predict obstacle ship trajectory with waypoint following.

        Args:
            state: current ship state
            T: prediction horizon (seconds)
            dt: time step (seconds)
            U_d: desired surge speed (uses current if None)
            chi_d: desired heading (uses current if None)

        Returns:
            Predicted trajectory as [4, n_steps] array
        """
        if U_d is None:
            U_d = state.U
        if chi_d is None:
            chi_d = state.chi

        n_steps = int(T / dt)
        trajectory = np.zeros((4, n_steps + 1))

        # Initialize with current state
        trajectory[0, 0] = state.x
        trajectory[1, 0] = state.y
        trajectory[2, 0] = state.chi
        trajectory[3, 0] = state.U

        # Prepare empty offset and maneuver time arrays (no maneuvers for basic predict)
        offset_sequence = np.array([])
        maneuver_times = np.array([])

        # Get waypoints as array - handle both Waypoint objects and tuples
        if self.waypoints:
            wp_coords = [(wp.x if hasattr(wp, 'x') else wp[0], 
                         wp.y if hasattr(wp, 'y') else wp[1]) for wp in self.waypoints]
            waypoints = np.array(wp_coords).T.reshape(2, -1)
        else:
            waypoints = np.zeros((2, 1))

        # Call trajectory prediction
        return self.predict_trajectory(
            trajectory=trajectory,
            offset_sequence=offset_sequence,
            maneuver_times=maneuver_times,
            u_d=U_d,
            chi_d=chi_d,
            waypoints=waypoints,
            integration_method="erk1",
            guidance_method="los",
            T=T,
            dt=dt,
        )

    def determine_active_waypoint_segment(
        self,
        waypoints: np.ndarray,
        state: ShipState4,
    ) -> int:
        """Determine which waypoint segment the ship is currently on.

        Args:
            waypoints: [2, n_waypoints] array of waypoint coordinates
            state: current ship state [x, y, chi, U]

        Returns:
            Index of the active waypoint segment
        """
        n_waypoints = waypoints.shape[1] - 1
        if n_waypoints < 1:
            return 0

        # Find the waypoint closest to the ship
        min_dist = float('inf')
        closest_wp = 0
        for i in range(n_waypoints + 1):
            dx = waypoints[0, i] - state.x
            dy = waypoints[1, i] - state.y
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                closest_wp = i

        # Ensure we're moving forward along the waypoint sequence
        self.current_wp = min(closest_wp, n_waypoints - 1) if n_waypoints > 0 else 0
        self.current_wp = max(0, self.current_wp)

        return self.current_wp

    def predict_trajectory(
        self,
        trajectory: np.ndarray,
        offset_sequence: np.ndarray,
        maneuver_times: np.ndarray,
        u_d: float,
        chi_d: float,
        waypoints: np.ndarray,
        integration_method: str = "erk1",
        guidance_method: str = "los",
        T: float = 200.0,
        dt: float = 0.5,
    ) -> np.ndarray:
        """Predict obstacle ship trajectory with heading offset maneuvers.

        Args:
            trajectory: [4, n_steps] array to store trajectory (modified in place)
            offset_sequence: heading offsets to apply [n_offsets]
            maneuver_times: time points to apply maneuvers [n_maneuvers]
            u_d: desired surge speed
            chi_d: desired heading
            waypoints: [2, n_waypoints] array of waypoint coordinates
            integration_method: "linear" or "erk1"
            guidance_method: "los" (Line-of-Sight)
            T: prediction horizon (seconds)
            dt: time step (seconds)

        Returns:
            Modified trajectory array [4, n_steps]
        """
        n_steps = int(T / dt)
        n_maneuvers = len(maneuver_times)

        # Initialize with current state
        trajectory[0, 0] = trajectory[0, 0]  # x
        trajectory[1, 0] = trajectory[1, 0]  # y
        trajectory[2, 0] = trajectory[2, 0]  # chi
        trajectory[3, 0] = trajectory[3, 0]  # U

        # Current state
        x = trajectory[0, 0]
        y = trajectory[1, 0]
        chi = trajectory[2, 0]
        U = trajectory[3, 0]

        # Determine active waypoint segment
        state = ShipState4(x=x, y=y, chi=chi, U=U)
        self.determine_active_waypoint_segment(waypoints, state)

        # Integrate
        for k in range(n_steps):
            # Determine heading offset for this time step
            offset = 0.0
            for m in range(n_maneuvers):
                if k * dt >= maneuver_times[m]:
                    # Apply offset sequence based on maneuver index
                    offset_idx = m * 2
                    if offset_idx + 1 < len(offset_sequence):
                        offset = offset_sequence[offset_idx + 1]

            # Compute desired heading with offset
            chi_d_maneuver = chi_d + offset

            # LOS guidance
            if guidance_method == "los" and self.waypoints:
                # Simplified LOS: use direct heading with offset
                chi_desired = normalize_angle(chi + offset)
            else:
                chi_desired = chi_d_maneuver

            # Integrate using selected method
            if integration_method == "erk1":
                new_state = self._predict_erks(
                    ShipState4(x=x, y=y, chi=chi, U=U),
                    chi_desired, u_d, dt
                )
                x, y, chi, U = new_state.x, new_state.y, new_state.chi, new_state.U
            else:
                new_state = self._predict_linear(
                    ShipState4(x=x, y=y, chi=chi, U=U),
                    chi_desired, u_d, dt
                )
                x, y, chi, U = new_state.x, new_state.y, new_state.chi, new_state.U

            # Store trajectory
            trajectory[0, k + 1] = x
            trajectory[1, k + 1] = y
            trajectory[2, k + 1] = chi
            trajectory[3, k + 1] = U

        return trajectory
