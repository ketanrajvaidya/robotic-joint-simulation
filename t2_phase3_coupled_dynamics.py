"""
=============================================================
TASK 2 — PHASE 3 : COUPLED DYNAMICS
Task: Multi-Joint Leg Simulation System
=============================================================

What this phase adds on top of Phase 1 + 2:
  - Simulates how BOTH joints move over time (not just position)
  - Uses the ActuatorResponseModel from Task 1 Phase 2
  - Shows how hip movement changes the load on the knee
  - Shows how load changes dynamically as motion happens
  - Logs torque on both joints at every time step

What is Coupled Dynamics?
  In Phase 2 we calculated WHERE the foot is (geometry).
  In Phase 3 we simulate HOW the joints actually move (physics).

  The key coupling effect:
    When the hip moves → thigh changes angle
    → gravity pulls differently on the shin
    → knee load changes even if knee didn't move

  Simple analogy:
    Hold your arm out straight (shoulder = hip, elbow = knee).
    Now tilt your whole arm up (shoulder moves).
    Your elbow didn't bend — but the weight pulling on it changed.
    That change in load is the COUPLING effect.

Key physics used:
  - First-order actuator response (from Task 1 Phase 2)
  - Gravitational torque = m × g × CoM × cos(angle)
  - Hip carries: thigh weight + shin weight (coupled load)
  - Knee carries: shin weight only (independent load)
  - Net torque = actuator torque - gravitational load
=============================================================
"""

import sys
import os
import math

# ── Import chain ─────────────────────────────────────────────
# Phase 3 imports Phase 1 (LegSystem) and Task 1 Phase 2
# (ActuatorResponseModel) — no new base classes needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t2_phase1_system_expansion import (
    build_leg_system,
    LegSystem
)
from phase2_actuator_model import ActuatorResponseModel


# ─────────────────────────────────────────────────────────────
# COUPLED DYNAMICS SIMULATOR
# Simulates both joints moving simultaneously over time,
# tracking how load on each joint changes as motion happens.
# ─────────────────────────────────────────────────────────────
class CoupledDynamicsSimulator:
    def __init__(
        self,
        leg: LegSystem,
        time_constant: float = 0.6,   # τ — how fast joints respond
        dt: float = 0.01              # time step in seconds
    ):
        """
        Parameters:
          leg           : Full LegSystem from Phase 1
          time_constant : Response speed for both actuators (seconds)
                          Higher = slower/more realistic response
          dt            : Simulation time step (10ms)
        """
        self.leg = leg
        self.dt  = dt
        self.time_constant = time_constant

        # Create one ActuatorResponseModel per joint
        # Reusing Task 1 Phase 2 class directly — no rewriting
        self.hip_model  = ActuatorResponseModel(
            joint=leg.hip,
            time_constant=time_constant,
            dt=dt
        )
        self.knee_model = ActuatorResponseModel(
            joint=leg.knee,
            time_constant=time_constant,
            dt=dt
        )

        # Time tracker
        self.time_elapsed = 0.0

    def reset(self, hip_angle: float = 0.0, knee_angle: float = 0.0):
        """
        Resets both joints to starting angles.
        Call this between simulation scenarios.
        """
        self.leg.hip.joint_output.current_angle   = hip_angle
        self.leg.hip.joint_output.angular_velocity = 0.0
        self.leg.hip.target_angle                  = hip_angle

        self.leg.knee.joint_output.current_angle   = knee_angle
        self.leg.knee.joint_output.angular_velocity = 0.0
        self.leg.knee.target_angle                  = knee_angle

        self.hip_model.time_elapsed  = 0.0
        self.knee_model.time_elapsed = 0.0
        self.time_elapsed = 0.0

        print(f"  [RESET] Hip={hip_angle}°  Knee={knee_angle}°")

    def compute_coupled_loads(self) -> dict:
        """
        Computes gravitational load on BOTH joints at current angles.
        This is the COUPLING — hip load depends on BOTH joint angles,
        knee load depends only on knee angle.

        Returns dict with:
          hip_load    : Total torque hip must overcome (N·m)
          knee_load   : Total torque knee must overcome (N·m)
          hip_net     : Net torque available at hip (actuator - gravity)
          knee_net    : Net torque available at knee (actuator - gravity)
        """
        hip_angle  = self.leg.hip.joint_output.current_angle
        knee_angle = self.leg.knee.joint_output.current_angle
        hip_rad    = math.radians(hip_angle)
        knee_rad   = math.radians(knee_angle)

        g = 9.81

        # ── Hip load: thigh weight + shin weight ──────────────
        # Thigh contribution
        thigh_torque = (self.leg.hip.link.mass *
                        g *
                        self.leg.hip.link.center_of_mass *
                        math.cos(hip_rad))

        # Shin contribution (weight at end of thigh = L1 distance)
        shin_torque_on_hip = (self.leg.knee.link.mass *
                              g *
                              self.leg.L1 *
                              math.cos(hip_rad))

        hip_load = thigh_torque + shin_torque_on_hip

        # ── Knee load: shin weight only ───────────────────────
        # Combined angle = hip + knee (shin hangs relative to thigh direction)
        combined_rad = hip_rad + knee_rad
        knee_load = (self.leg.knee.link.mass *
                     g *
                     self.leg.knee.link.center_of_mass *
                     math.cos(combined_rad))

        # ── Net torque (what's left after fighting gravity) ───
        hip_net  = self.leg.hip.actuator.max_torque  - abs(hip_load)
        knee_net = self.leg.knee.actuator.max_torque - abs(knee_load)

        return {
            "hip_load_Nm"  : round(hip_load,  4),
            "knee_load_Nm" : round(knee_load, 4),
            "hip_net_Nm"   : round(hip_net,   4),
            "knee_net_Nm"  : round(knee_net,  4),
        }

    def step(self) -> dict:
        """
        Advances both joints by one time step simultaneously.
        After each step, recomputes coupled loads.

        Returns a snapshot dict of the full system state.
        """
        # Step both actuators (using Task 1 Phase 2 model)
        hip_angle  = self.hip_model.step()
        knee_angle = self.knee_model.step()

        # Advance shared time
        self.time_elapsed += self.dt

        # Compute coupled loads at new angles
        loads = self.compute_coupled_loads()

        return {
            "time"          : round(self.time_elapsed, 4),
            "hip_angle"     : round(hip_angle,  4),
            "knee_angle"    : round(knee_angle, 4),
            "hip_target"    : self.leg.hip.target_angle,
            "knee_target"   : self.leg.knee.target_angle,
            "hip_velocity"  : round(self.leg.hip.joint_output.angular_velocity,  3),
            "knee_velocity" : round(self.leg.knee.joint_output.angular_velocity, 3),
            "hip_load_Nm"   : loads["hip_load_Nm"],
            "knee_load_Nm"  : loads["knee_load_Nm"],
            "hip_net_Nm"    : loads["hip_net_Nm"],
            "knee_net_Nm"   : loads["knee_net_Nm"],
        }

    def simulate(
        self,
        hip_target: float,
        knee_target: float,
        duration: float,
        label: str = ""
    ) -> list:
        """
        Runs a full coupled simulation — both joints move toward
        their targets simultaneously, with load tracked every step.

        Parameters:
          hip_target  : Desired hip angle (degrees)
          knee_target : Desired knee angle (degrees)
          duration    : How long to simulate (seconds)
          label       : Description for display

        Returns:
          List of step snapshots (one per time step)
        """
        # Set targets on both joints
        self.leg.hip.set_target(hip_target)
        self.leg.knee.set_target(knee_target)

        print(f"\n  {'='*60}")
        print(f"  SCENARIO: {label}")
        print(f"  Hip  target : {self.leg.hip.target_angle}°")
        print(f"  Knee target : {self.leg.knee.target_angle}°")
        print(f"  Duration    : {duration}s")
        print(f"  {'='*60}")
        print(f"  {'Time':>6} | {'Hip°':>7} | {'Knee°':>7} | "
              f"{'Hip Load':>10} | {'Knee Load':>10} | "
              f"{'Hip Net':>9} | {'Knee Net':>9}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-"
              f"{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")

        steps   = int(duration / self.dt)
        history = []

        for i in range(steps):
            snap = self.step()
            history.append(snap)

            # Print every 50 steps (every 0.5s)
            if i % 50 == 0 or i == steps - 1:
                print(
                    f"  {snap['time']:>6.2f} | "
                    f"{snap['hip_angle']:>7.2f} | "
                    f"{snap['knee_angle']:>7.2f} | "
                    f"{snap['hip_load_Nm']:>9.3f}N | "
                    f"{snap['knee_load_Nm']:>9.3f}N | "
                    f"{snap['hip_net_Nm']:>8.3f}N | "
                    f"{snap['knee_net_Nm']:>8.3f}N"
                )

        # Final state summary
        final = history[-1]
        print(f"\n  Final hip  angle : {final['hip_angle']:.3f}°  "
              f"(target: {final['hip_target']}°)")
        print(f"  Final knee angle : {final['knee_angle']:.3f}°  "
              f"(target: {final['knee_target']}°)")
        print(f"  Final hip  load  : {final['hip_load_Nm']:.3f} N·m")
        print(f"  Final knee load  : {final['knee_load_Nm']:.3f} N·m")

        return history


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_scenarios: dict):
    """Saves all coupled dynamics results to a text file."""
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "t2_phase3_log.txt"
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 2 - PHASE 3: COUPLED DYNAMICS LOG\n")
        f.write("Both joints simulated simultaneously with load tracking\n")
        f.write("=" * 75 + "\n\n")

        for name, history in all_scenarios.items():
            f.write(f"SCENARIO: {name}\n")
            f.write(
                f"{'Time':>7} | {'Hip°':>7} | {'Knee°':>7} | "
                f"{'Hip Load':>10} | {'Knee Load':>10} | "
                f"{'Hip Net':>9} | {'Knee Net':>9}\n"
            )
            f.write("-" * 70 + "\n")
            for s in history:
                f.write(
                    f"{s['time']:>7.3f} | "
                    f"{s['hip_angle']:>7.3f} | "
                    f"{s['knee_angle']:>7.3f} | "
                    f"{s['hip_load_Nm']:>9.4f}N | "
                    f"{s['knee_load_Nm']:>9.4f}N | "
                    f"{s['hip_net_Nm']:>8.4f}N | "
                    f"{s['knee_net_Nm']:>8.4f}N\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] -> t2_phase3_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_scenarios: dict):
    """
    Plots 4 graphs:
      1. Hip + Knee angles over time (both scenarios)
      2. Hip load over time (shows coupling effect)
      3. Knee load over time (changes due to hip motion)
      4. Net torque available at each joint
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 11))
        fig.suptitle(
            "Task 2 - Phase 3: Coupled Dynamics\n"
            "Joint Angles + Load Distribution Over Time",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        colors = {
            "Stand to Lift"    : ("royalblue",  "darkorange"),
            "Hip Only Motion"  : ("green",      "red"),
        }

        ax1 = fig.add_subplot(gs[0, 0])  # Angles over time
        ax2 = fig.add_subplot(gs[0, 1])  # Hip load over time
        ax3 = fig.add_subplot(gs[1, 0])  # Knee load over time
        ax4 = fig.add_subplot(gs[1, 1])  # Net torque

        for name, history in all_scenarios.items():
            times      = [s["time"]         for s in history]
            hip_angles = [s["hip_angle"]    for s in history]
            kne_angles = [s["knee_angle"]   for s in history]
            hip_loads  = [s["hip_load_Nm"]  for s in history]
            kne_loads  = [s["knee_load_Nm"] for s in history]
            hip_nets   = [s["hip_net_Nm"]   for s in history]
            kne_nets   = [s["knee_net_Nm"]  for s in history]

            c_hip, c_knee = colors.get(name, ("blue", "orange"))

            # Plot 1: Angles
            ax1.plot(times, hip_angles, color=c_hip,
                     linewidth=2, label=f"{name} - Hip")
            ax1.plot(times, kne_angles, color=c_knee,
                     linewidth=2, linestyle='--', label=f"{name} - Knee")

            # Plot 2: Hip load
            ax2.plot(times, hip_loads, color=c_hip,
                     linewidth=2, label=name)

            # Plot 3: Knee load
            ax3.plot(times, kne_loads, color=c_knee,
                     linewidth=2, label=name)

            # Plot 4: Net torque
            ax4.plot(times, hip_nets, color=c_hip,
                     linewidth=2, label=f"{name} - Hip net")
            ax4.plot(times, kne_nets, color=c_knee,
                     linewidth=2, linestyle='--', label=f"{name} - Knee net")

        # Format Plot 1
        ax1.set_title("Joint Angles Over Time", fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Format Plot 2
        ax2.set_title("Hip Load Over Time\n(Coupling: changes with hip motion)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Gravitational Load (N·m)")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linewidth=0.5)

        # Format Plot 3
        ax3.set_title("Knee Load Over Time\n(Changes even when knee is still)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Gravitational Load (N·m)")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='gray', linewidth=0.5)

        # Format Plot 4
        ax4.set_title("Net Torque Available at Each Joint",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Net Torque (N·m)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='red', linewidth=1,
                    linestyle=':', label='Zero margin')

        # Save
        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "t2_phase3_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t2_phase3_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available. Skipping graph.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase3():
    print("\n" + "=" * 60)
    print("  TASK 2 - PHASE 3: COUPLED DYNAMICS")
    print("  Both joints simulated with load tracking")
    print("=" * 60)

    # Build leg system
    leg = build_leg_system()
    sim = CoupledDynamicsSimulator(
        leg=leg,
        time_constant=0.6,
        dt=0.01
    )

    all_scenarios = {}

    # ── Scenario 1: Stand to Lift ────────────────────────────
    # Both joints move together: hip lifts, knee bends slightly
    # Shows how hip motion increases load on both joints
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    h1 = sim.simulate(
        hip_target=60.0,
        knee_target=40.0,
        duration=4.0,
        label="Stand to Lift"
    )
    all_scenarios["Stand to Lift"] = h1

    # ── Scenario 2: Hip Only Motion ──────────────────────────
    # Only hip moves — knee stays at 0°
    # KEY COUPLING DEMONSTRATION:
    # Knee load changes even though knee didn't move at all
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    h2 = sim.simulate(
        hip_target=80.0,
        knee_target=0.0,
        duration=4.0,
        label="Hip Only Motion"
    )
    all_scenarios["Hip Only Motion"] = h2

    # ── Coupling insight printout ────────────────────────────
    print("\n" + "=" * 60)
    print("  COUPLING INSIGHT")
    print("=" * 60)
    print("  Scenario 2 (Hip Only Motion):")
    print("  Knee target was 0 degrees — knee did NOT move.")
    print("  But watch the knee load column — it changed!")
    print("  This is the coupling effect:")
    print("  Hip motion changes the angle of gravity on the shin,")
    print("  which changes the torque the knee must resist.")
    print("=" * 60)

    # Save + plot
    save_log(all_scenarios)
    plot_results(all_scenarios)

    print("\n  [PHASE 3 COMPLETE] Coupled Dynamics done.")
    print("  Ready for Phase 4: Coordinated Motion\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase3()
