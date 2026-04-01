"""
=============================================================
PHASE 3 — MECHANICAL RESPONSE
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase adds on top of Phase 2:
  - Inertia effect   : Joint starts slow, builds speed gradually
  - Friction/Damping : Resistance that opposes motion
  - Realistic motion : Joint never jumps, always accelerates smoothly

Key concepts:

  1. INERTIA (Moment of Inertia — I):
     - Resistance to changes in rotational motion
     - Heavy/long links = more inertia = slower start
     - Formula: I = m × r²  (mass × radius squared)

  2. FRICTION (Damping — b):
     - Resistance proportional to speed
     - Faster motion = more friction force
     - Models real bearing/joint friction

  3. NEWTON'S SECOND LAW FOR ROTATION:
     The core equation of motion used here:

     I × α = τ_actuator - τ_friction - τ_gravity

     Where:
       I           = moment of inertia (kg·m²)
       α           = angular acceleration (deg/s²)
       τ_actuator  = torque produced by motor (N·m)
       τ_friction  = damping torque opposing motion (N·m)
       τ_gravity   = gravitational load torque (N·m)

     Solving for acceleration:
       α = (τ_actuator - τ_friction - τ_gravity) / I

     This acceleration is integrated over time to get velocity,
     and velocity is integrated to get position (angle).

=============================================================
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import build_knee_joint, RotationalJoint


# ─────────────────────────────────────────────────────────────
# MECHANICAL RESPONSE MODEL
# Adds inertia + friction on top of Phase 2's actuator model
# Uses Newton's second law for rotation at every time step
# ─────────────────────────────────────────────────────────────
class MechanicalResponseModel:
    def __init__(
        self,
        joint: RotationalJoint,
        time_constant: float = 0.5,    # actuator response speed (from Phase 2)
        damping: float = 0.05,         # friction coefficient (N·m·s/deg)
        dt: float = 0.01               # simulation time step (seconds)
    ):
        """
        Parameters:
          joint         : RotationalJoint from Phase 1
          time_constant : Actuator response lag (Phase 2 carry-over)
          damping       : Friction coefficient — higher = more resistance
                          0.0 = frictionless (unrealistic)
                          0.05 = light bearing friction (realistic)
                          0.2  = heavy friction/worn joint
          dt            : Time step for numerical integration (10ms)
        """
        self.joint        = joint
        self.time_constant = time_constant
        self.damping      = damping
        self.dt           = dt

        # Compute moment of inertia from link properties
        # I = m × r² where r = center of mass distance
        self.inertia = self._compute_inertia()

        # Simulation state
        self.time_elapsed     = 0.0
        self.angular_velocity = 0.0    # current speed (deg/s) — starts at 0
        self.time_log         = []
        self.angle_log        = []
        self.velocity_log     = []
        self.command_log      = []
        self.torque_log       = []

        print(f"\n  [PHASE 3 INIT]")
        print(f"  Moment of inertia : {self.inertia:.4f} kg·m²")
        print(f"  Damping coefficient: {self.damping} N·m·s/deg")
        print(f"  Time constant      : {self.time_constant}s")

    def _compute_inertia(self) -> float:
        """
        Computes moment of inertia of the link about the joint pivot.
        Uses point mass approximation: I = m × r²
        where r = center of mass distance from pivot.

        A higher inertia means:
          - slower acceleration at start
          - harder to stop once moving
          This is the 'slow start' effect Phase 3 requires.
        """
        m = self.joint.link.mass              # kg
        r = self.joint.link.center_of_mass    # meters
        return m * (r ** 2)                   # kg·m²

    def _compute_actuator_torque(self, current_angle: float) -> float:
        """
        Computes torque produced by actuator toward target.

        The actuator produces torque proportional to the error
        (how far from target), capped at motor's max torque.

        τ_actuator = (error / time_constant) × inertia
        """
        error        = self.joint.target_angle - current_angle
        raw_torque   = (error / self.time_constant) * self.inertia
        max_torque   = self.joint.actuator.max_torque

        # Cap to physical motor limit
        return max(-max_torque, min(max_torque, raw_torque))

    def _compute_friction_torque(self, velocity: float) -> float:
        """
        Computes friction (damping) torque opposing motion.

        τ_friction = damping × velocity

        Always opposes direction of motion (hence the sign matches velocity).
        This is viscous damping — models oil/bearing friction in real joints.
        """
        return self.damping * velocity

    def step(self) -> tuple[float, float, float]:
        """
        Advances simulation by one time step using Newton's 2nd law.

        Physics sequence every 10ms:
          1. Compute actuator torque (driving force)
          2. Compute friction torque (opposing force)
          3. Compute gravity torque (load from link weight)
          4. Net torque = actuator - friction - gravity
          5. Angular acceleration = net torque / inertia
          6. Update velocity (integrate acceleration)
          7. Cap velocity to motor limit
          8. Update angle (integrate velocity)
          9. Enforce joint limits

        Returns: (current_angle, angular_velocity, net_torque)
        """
        current_angle = self.joint.joint_output.current_angle

        # Step 1: Actuator driving torque
        tau_actuator = self._compute_actuator_torque(current_angle)

        # Step 2: Friction opposing torque
        tau_friction = self._compute_friction_torque(self.angular_velocity)

        # Step 3: Gravity load torque (from Phase 1 link physics)
        tau_gravity  = self.joint.link.gravitational_torque(current_angle)

        # Step 4: Net torque (what's left to accelerate the joint)
        tau_net = tau_actuator - tau_friction - tau_gravity

        # Step 5: Angular acceleration (α = τ / I)
        # Convert N·m to deg/s² by dividing by inertia
        alpha = tau_net / self.inertia      # deg/s² (approx)

        # Step 6: Update velocity by integrating acceleration
        self.angular_velocity += alpha * self.dt

        # Step 7: Cap velocity to motor's physical speed limit
        max_speed = self.joint.actuator.get_effective_speed()
        self.angular_velocity = max(-max_speed, min(max_speed, self.angular_velocity))

        # Step 8: Update angle by integrating velocity
        new_angle = current_angle + self.angular_velocity * self.dt

        # Step 9: Enforce joint limits — stop at boundaries
        new_angle = self.joint.joint_output.clamp_angle(new_angle)

        # If joint hit a limit, kill velocity (can't push through wall)
        if self.joint.joint_output.is_at_limit():
            self.angular_velocity = 0.0

        # Save state
        self.joint.joint_output.current_angle    = new_angle
        self.joint.joint_output.angular_velocity = self.angular_velocity
        self.time_elapsed += self.dt

        # Log everything
        self.time_log.append(self.time_elapsed)
        self.angle_log.append(new_angle)
        self.velocity_log.append(self.angular_velocity)
        self.command_log.append(self.joint.target_angle)
        self.torque_log.append(tau_net)

        return new_angle, self.angular_velocity, tau_net

    def simulate(
        self,
        target_angle: float,
        duration: float,
        label: str = ""
    ):
        """
        Full simulation run — sends command and steps through time.
        """
        # Validate and set target
        clamped = self.joint.joint_output.clamp_angle(target_angle)
        if target_angle != clamped:
            print(f"  [WARNING] Target {target_angle}° clamped to {clamped}°")
        self.joint.target_angle = clamped

        steps = int(duration / self.dt)

        print(f"\n  Simulating: {label}")
        print(f"  {'Time':>6} | {'Command':>9} | {'Actual':>9} | {'Velocity':>10} | {'Torque':>9}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")

        for i in range(steps):
            angle, vel, torque = self.step()
            t = self.time_elapsed

            if i % 50 == 0 or i == steps - 1:
                print(f"  {t:>6.2f}s | {clamped:>8.2f}° | {angle:>8.2f}° | {vel:>8.2f}°/s | {torque:>7.3f}Nm")

        print(f"\n  Final angle    : {self.joint.joint_output.current_angle:.3f}°")
        print(f"  Final velocity : {self.angular_velocity:.3f}°/s")

    def reset(self, start_angle: float = 0.0):
        """Resets simulation state for next test."""
        self.joint.joint_output.current_angle    = start_angle
        self.joint.joint_output.angular_velocity = 0.0
        self.joint.target_angle                  = start_angle
        self.angular_velocity                    = 0.0
        self.time_elapsed                        = 0.0
        self.time_log.clear()
        self.angle_log.clear()
        self.velocity_log.clear()
        self.command_log.clear()
        self.torque_log.clear()
        print(f"\n  [RESET] Joint returned to {start_angle}°")


# ─────────────────────────────────────────────────────────────
# COMPARE: WITH vs WITHOUT inertia+friction
# This is the key proof that Phase 3 is working —
# showing the difference between ideal and realistic motion
# ─────────────────────────────────────────────────────────────
def run_comparison():
    """
    Runs the same command on two models:
      Model A: No inertia, no friction (Phase 2 style — ideal)
      Model B: With inertia + friction (Phase 3 — realistic)

    The difference between them IS the Phase 3 contribution.
    """
    print("\n" + "="*60)
    print("  PHASE 3 — MECHANICAL RESPONSE")
    print("  Inertia + Friction Effects on Joint Motion")
    print("="*60)

    TARGET   = 90.0
    DURATION = 5.0

    # ── Model A: No friction (baseline from Phase 2) ─────────
    joint_a = build_knee_joint()
    model_a = MechanicalResponseModel(
        joint=joint_a,
        time_constant=0.5,
        damping=0.0,        # no friction
        dt=0.01
    )
    print("\n" + "-"*60)
    print("  MODEL A: No friction (ideal baseline)")
    print("-"*60)
    model_a.simulate(TARGET, DURATION, "0° to 90° — no friction")

    # ── Model B: With friction (realistic) ───────────────────
    joint_b = build_knee_joint()
    model_b = MechanicalResponseModel(
        joint=joint_b,
        time_constant=0.5,
        damping=0.05,       # realistic bearing friction
        dt=0.01
    )
    print("\n" + "-"*60)
    print("  MODEL B: With friction (realistic)")
    print("-"*60)
    model_b.simulate(TARGET, DURATION, "0° to 90° — with friction")

    # ── Model C: Heavy friction (worn/damaged joint) ──────────
    joint_c = build_knee_joint()
    model_c = MechanicalResponseModel(
        joint=joint_c,
        time_constant=0.5,
        damping=0.15,       # heavy friction
        dt=0.01
    )
    print("\n" + "-"*60)
    print("  MODEL C: Heavy friction (worn joint)")
    print("-"*60)
    model_c.simulate(TARGET, DURATION, "0° to 90° — heavy friction")

    # Save log and plot
    save_log(model_a, model_b, model_c)
    plot_results(model_a, model_b, model_c, TARGET)

    return model_a, model_b, model_c


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(model_a, model_b, model_c):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phase3_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 3 - MECHANICAL RESPONSE LOG\n")
        f.write("Inertia + Friction Effects\n")
        f.write("="*60 + "\n\n")

        for label, model in [
            ("MODEL A: No friction",    model_a),
            ("MODEL B: Light friction", model_b),
            ("MODEL C: Heavy friction", model_c),
        ]:
            f.write(f"{label}\n")
            f.write(f"Damping coefficient: {model.damping}\n")
            f.write(f"Moment of inertia  : {model.inertia:.4f} kg.m2\n")
            f.write(f"{'Time':>8} | {'Command':>9} | {'Angle':>9} | "
                    f"{'Velocity':>10} | {'Torque':>9}\n")
            f.write("-"*55 + "\n")
            for t, cmd, ang, vel, trq in zip(
                model.time_log, model.command_log,
                model.angle_log, model.velocity_log, model.torque_log
            ):
                f.write(
                    f"{t:>8.3f} | {cmd:>9.3f} | {ang:>9.3f} | "
                    f"{vel:>10.3f} | {trq:>9.4f}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] phase3_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(model_a, model_b, model_c, target):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "Phase 3 - Mechanical Response\n"
            "Effect of Inertia + Friction on Joint Motion",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Plot 1: Angle vs Time — all three models ──────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axhline(y=target, color='blue', linestyle='--',
                    linewidth=1.5, label=f'Command ({target}°)')
        ax1.plot(model_a.time_log, model_a.angle_log,
                 'g-', linewidth=2, label='No friction')
        ax1.plot(model_b.time_log, model_b.angle_log,
                 'r-', linewidth=2, label='Light friction')
        ax1.plot(model_c.time_log, model_c.angle_log,
                 'm-', linewidth=2, label='Heavy friction')
        ax1.set_title("Angle vs Time\n(friction comparison)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Velocity vs Time ───────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(model_a.time_log, model_a.velocity_log,
                 'g-', linewidth=2, label='No friction')
        ax2.plot(model_b.time_log, model_b.velocity_log,
                 'r-', linewidth=2, label='Light friction')
        ax2.plot(model_c.time_log, model_c.velocity_log,
                 'm-', linewidth=2, label='Heavy friction')
        ax2.set_title("Angular Velocity vs Time\n(inertia slow-start visible)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (deg/s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Torque vs Time ─────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(model_a.time_log, model_a.torque_log,
                 'g-', linewidth=2, label='No friction')
        ax3.plot(model_b.time_log, model_b.torque_log,
                 'r-', linewidth=2, label='Light friction')
        ax3.plot(model_c.time_log, model_c.torque_log,
                 'm-', linewidth=2, label='Heavy friction')
        ax3.set_title("Net Torque vs Time\n(friction reduces available torque)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Net Torque (N.m)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Tracking error comparison ─────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        err_a = [abs(c - a) for c, a in
                 zip(model_a.command_log, model_a.angle_log)]
        err_b = [abs(c - a) for c, a in
                 zip(model_b.command_log, model_b.angle_log)]
        err_c = [abs(c - a) for c, a in
                 zip(model_c.command_log, model_c.angle_log)]
        ax4.plot(model_a.time_log, err_a,
                 'g-', linewidth=2, label='No friction')
        ax4.plot(model_b.time_log, err_b,
                 'r-', linewidth=2, label='Light friction')
        ax4.plot(model_c.time_log, err_c,
                 'm-', linewidth=2, label='Heavy friction')
        ax4.set_title("Tracking Error vs Time\n(more friction = larger error)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Error (deg)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "phase3_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] phase3_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_comparison()
