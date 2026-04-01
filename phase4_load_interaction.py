"""
=============================================================
PHASE 4 — LOAD INTERACTION
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase adds on top of Phase 3:
  - External load attached to the joint (like carrying weight)
  - Increasing resistance as load increases
  - Reduced speed under heavy load
  - Heavier load = slower response = larger tracking error

Key concepts:

  1. EXTERNAL LOAD TORQUE:
     When a robot leg carries weight, that weight creates
     extra torque the actuator must fight against.

     Formula:
       tau_load = load_mass × g × load_distance × cos(angle)

     This adds directly on top of gravity torque from Phase 3.

  2. TORQUE BUDGET:
     The actuator has a fixed max torque (8.5 N·m).
     Every N·m spent fighting load/gravity/friction
     is one less N·m available for acceleration.

     tau_net = tau_actuator - tau_friction - tau_gravity - tau_load

     When tau_load is large enough:
       tau_net approaches zero → joint barely moves
       tau_net goes negative  → joint cannot move at all (stall)

  3. LOAD LEVELS SIMULATED:
     - No load     : 0.0 kg  (baseline)
     - Light load  : 0.5 kg  (small tool/sensor)
     - Medium load : 1.5 kg  (moderate payload)
     - Heavy load  : 3.0 kg  (near motor limit)

=============================================================
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import build_knee_joint, RotationalJoint


# ─────────────────────────────────────────────────────────────
# LOAD INTERACTION MODEL
# Extends Phase 3 mechanics with external payload torque
# ─────────────────────────────────────────────────────────────
class LoadInteractionModel:
    def __init__(
        self,
        joint: RotationalJoint,
        load_mass: float = 0.0,        # kg — external payload mass
        load_distance: float = 0.35,   # m  — distance from joint to load
        time_constant: float = 0.5,    # actuator response (Phase 2)
        damping: float = 0.05,         # friction coefficient (Phase 3)
        dt: float = 0.01               # simulation time step
    ):
        """
        Parameters:
          load_mass     : Mass of external payload (kg).
                          0.0 = no load, 3.0 = heavy load
          load_distance : How far from joint the load is attached (m).
                          Greater distance = greater torque arm = harder to lift
          time_constant : Actuator response lag (Phase 2 carry-over)
          damping       : Friction coefficient (Phase 3 carry-over)
          dt            : Simulation time step (10ms)
        """
        self.joint         = joint
        self.load_mass     = load_mass
        self.load_distance = load_distance
        self.time_constant = time_constant
        self.damping       = damping
        self.dt            = dt
        self.g             = 9.81      # gravity (m/s²)

        # Compute moment of inertia — link + load combined
        self.link_inertia  = (joint.link.mass *
                               joint.link.center_of_mass ** 2)
        self.load_inertia  = load_mass * load_distance ** 2
        self.total_inertia = self.link_inertia + self.load_inertia

        # Simulation state
        self.angular_velocity = 0.0
        self.time_elapsed     = 0.0

        # Logs
        self.time_log       = []
        self.angle_log      = []
        self.velocity_log   = []
        self.command_log    = []
        self.torque_log     = []
        self.load_torque_log = []

        print(f"\n  [LOAD MODEL INIT]")
        print(f"  Load mass        : {load_mass} kg")
        print(f"  Load distance    : {load_distance} m")
        print(f"  Link inertia     : {self.link_inertia:.4f} kg.m2")
        print(f"  Load inertia     : {self.load_inertia:.4f} kg.m2")
        print(f"  Total inertia    : {self.total_inertia:.4f} kg.m2")
        print(f"  Max load torque  : "
              f"{load_mass * self.g * load_distance:.3f} N.m at 0 deg")

    def compute_load_torque(self, angle_deg: float) -> float:
        """
        Computes external load torque at given angle.

        Formula: tau_load = m × g × d × cos(θ)
          - Maximum at 0° (horizontal — full weight fighting motion)
          - Zero at 90° (vertical — weight aligned with joint axis)
          - This models carrying a payload on the end of the link
        """
        angle_rad = math.radians(angle_deg)
        return self.load_mass * self.g * self.load_distance * math.cos(angle_rad)

    def compute_actuator_torque(self, current_angle: float) -> float:
        """Actuator torque toward target — same as Phase 3."""
        error      = self.joint.target_angle - current_angle
        raw_torque = (error / self.time_constant) * self.total_inertia
        max_torque = self.joint.actuator.max_torque
        return max(-max_torque, min(max_torque, raw_torque))

    def step(self) -> tuple[float, float, float, float]:
        """
        One simulation time step with full load physics.

        Torque budget every 10ms:
          tau_actuator  — motor pushing toward target
          tau_friction  — bearing resistance (Phase 3)
          tau_gravity   — link weight (Phase 1)
          tau_load      — external payload (Phase 4 NEW)
          ─────────────────────────────────────────
          tau_net = actuator - friction - gravity - load

        Returns: (angle, velocity, net_torque, load_torque)
        """
        current = self.joint.joint_output.current_angle

        # All torque components
        tau_act  = self.compute_actuator_torque(current)
        tau_fric = self.damping * self.angular_velocity
        tau_grav = self.joint.link.gravitational_torque(current)
        tau_load = self.compute_load_torque(current)

        # Net torque available for acceleration
        tau_net = tau_act - tau_fric - tau_grav - tau_load

        # Angular acceleration
        alpha = tau_net / self.total_inertia

        # Update velocity
        self.angular_velocity += alpha * self.dt

        # Cap to motor speed limit
        max_speed = self.joint.actuator.get_effective_speed()
        self.angular_velocity = max(
            -max_speed, min(max_speed, self.angular_velocity)
        )

        # Update angle
        new_angle = current + self.angular_velocity * self.dt
        new_angle = self.joint.joint_output.clamp_angle(new_angle)

        # Kill velocity at joint limits
        if self.joint.joint_output.is_at_limit():
            self.angular_velocity = 0.0

        # Save state
        self.joint.joint_output.current_angle    = new_angle
        self.joint.joint_output.angular_velocity = self.angular_velocity
        self.time_elapsed += self.dt

        # Log
        self.time_log.append(self.time_elapsed)
        self.angle_log.append(new_angle)
        self.velocity_log.append(self.angular_velocity)
        self.command_log.append(self.joint.target_angle)
        self.torque_log.append(tau_net)
        self.load_torque_log.append(tau_load)

        return new_angle, self.angular_velocity, tau_net, tau_load

    def simulate(
        self,
        target_angle: float,
        duration: float,
        label: str = ""
    ):
        """Full simulation with load interaction."""
        clamped = self.joint.joint_output.clamp_angle(target_angle)
        if target_angle != clamped:
            print(f"  [WARNING] Target {target_angle} clamped to {clamped}")
        self.joint.target_angle = clamped

        steps = int(duration / self.dt)

        print(f"\n  Simulating: {label}")
        print(f"  {'Time':>6} | {'Actual':>8} | {'Velocity':>9} | "
              f"{'Net Torque':>10} | {'Load Torque':>11}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*11}")

        for i in range(steps):
            angle, vel, tau_net, tau_load = self.step()
            t = self.time_elapsed

            if i % 50 == 0 or i == steps - 1:
                print(f"  {t:>6.2f}s | {angle:>7.2f}° | "
                      f"{vel:>8.2f}°/s | {tau_net:>9.3f}Nm | "
                      f"{tau_load:>10.3f}Nm")

        print(f"\n  Final angle    : "
              f"{self.joint.joint_output.current_angle:.3f} deg")
        print(f"  Final velocity : {self.angular_velocity:.3f} deg/s")

    def reset(self, start_angle: float = 0.0):
        """Reset for next test."""
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
        self.load_torque_log.clear()


# ─────────────────────────────────────────────────────────────
# RUN SIMULATION — 4 LOAD LEVELS
# ─────────────────────────────────────────────────────────────
def run_phase4_simulation():
    print("\n" + "="*60)
    print("  PHASE 4 — LOAD INTERACTION")
    print("  Effect of Payload Mass on Joint Response")
    print("="*60)

    TARGET   = 90.0
    DURATION = 6.0

    load_configs = [
        (0.0, "No load     (0.0 kg — baseline)"),
        (0.5, "Light load  (0.5 kg — small sensor)"),
        (1.5, "Medium load (1.5 kg — moderate payload)"),
        (3.0, "Heavy load  (3.0 kg — near motor limit)"),
    ]

    models = []

    for load_mass, label in load_configs:
        print("\n" + "-"*60)
        print(f"  {label}")
        print("-"*60)

        joint = build_knee_joint()
        model = LoadInteractionModel(
            joint=joint,
            load_mass=load_mass,
            load_distance=0.35,
            time_constant=0.5,
            damping=0.05,
            dt=0.01
        )
        model.simulate(TARGET, DURATION, label)
        models.append((load_mass, label, model))

    # Summary table
    print("\n" + "="*60)
    print("  LOAD INTERACTION SUMMARY")
    print("="*60)
    print(f"  {'Load (kg)':>10} | {'Final Angle':>12} | "
          f"{'Final Speed':>12} | {'Reached 90?':>11}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}")
    for load_mass, label, model in models:
        final_angle = model.joint.joint_output.current_angle
        final_speed = model.angular_velocity
        reached     = "YES" if final_angle >= 88.0 else "NO"
        print(f"  {load_mass:>10.1f} | {final_angle:>11.2f}° | "
              f"{final_speed:>10.2f}°/s | {reached:>11}")

    save_log(models)
    plot_results(models, TARGET)
    return models


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(models):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phase4_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 4 - LOAD INTERACTION LOG\n")
        f.write("Effect of Payload on Joint Response\n")
        f.write("="*65 + "\n\n")

        for load_mass, label, model in models:
            f.write(f"{label}\n")
            f.write(f"Load mass     : {load_mass} kg\n")
            f.write(f"Total inertia : {model.total_inertia:.4f} kg.m2\n")
            f.write(
                f"{'Time':>8} | {'Angle':>8} | {'Velocity':>10} | "
                f"{'Net Torque':>10} | {'Load Torque':>11}\n"
            )
            f.write("-"*60 + "\n")
            for t, a, v, tn, tl in zip(
                model.time_log, model.angle_log,
                model.velocity_log, model.torque_log,
                model.load_torque_log
            ):
                f.write(
                    f"{t:>8.3f} | {a:>8.3f} | {v:>10.3f} | "
                    f"{tn:>10.4f} | {tl:>11.4f}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] phase4_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(models, target):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        colors = ['green', 'blue', 'orange', 'red']
        labels = [f"{m:.1f} kg" for m, _, _ in models]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "Phase 4 - Load Interaction\n"
            "Heavier Load = Slower Response",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        # ── Plot 1: Angle vs Time ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axhline(y=target, color='black', linestyle='--',
                    linewidth=1.5, label=f'Command ({target} deg)')
        for (load_mass, _, model), color, lbl in zip(models, colors, labels):
            ax1.plot(model.time_log, model.angle_log,
                     color=color, linewidth=2, label=lbl)
        ax1.set_title("Angle vs Time\n(heavier load = slower)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Velocity vs Time ──────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        for (load_mass, _, model), color, lbl in zip(models, colors, labels):
            ax2.plot(model.time_log, model.velocity_log,
                     color=color, linewidth=2, label=lbl)
        ax2.set_title("Velocity vs Time\n(load reduces speed)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (deg/s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Load Torque vs Time ───────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        for (load_mass, _, model), color, lbl in zip(models, colors, labels):
            ax3.plot(model.time_log, model.load_torque_log,
                     color=color, linewidth=2, label=lbl)
        ax3.set_title("Load Torque vs Time\n(torque the motor must overcome)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Load Torque (N.m)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Final angle bar chart ─────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        final_angles = [
            model.joint.joint_output.current_angle
            for _, _, model in models
        ]
        load_labels = [f"{m:.1f}kg" for m, _, _ in models]
        bars = ax4.bar(load_labels, final_angles,
                       color=colors, alpha=0.8, edgecolor='black')
        ax4.axhline(y=target, color='black', linestyle='--',
                    linewidth=1.5, label=f'Target ({target} deg)')
        ax4.set_title("Final Angle vs Load\n(heavier load = further from target)",
                      fontweight='bold')
        ax4.set_xlabel("Load Mass (kg)")
        ax4.set_ylabel("Final Angle (deg)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, angle in zip(bars, final_angles):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{angle:.1f} deg",
                ha='center', va='bottom', fontsize=9
            )

        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase4_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] phase4_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase4_simulation()
