"""
=============================================================
TASK 2 — PHASE 6 : FAILURE PROPAGATION
Task: Multi-Joint Leg Simulation System
=============================================================

What this phase adds on top of Phase 5:
  - Simulates what happens when ONE joint fails
  - Shows how failure in one joint AFFECTS the other joint
  - Models 3 failure types: stall, overheating, noise/jitter
  - Tracks system-level collapse — not just single joint failure
  - Compares healthy vs failed behavior side by side

What is Failure Propagation?
  In Task 1 Phase 6, failure was isolated to ONE joint.
  In Task 2 Phase 6, failure SPREADS across the system.

  Example cascade:
    Knee stalls
    → Hip must compensate (tries to move harder)
    → Hip load increases beyond safe range
    → Hip overheats or stalls too
    → Entire leg stops functioning

  This is system-level failure — realistic for real robots.

3 Failure Scenarios simulated:
  1. KNEE STALL
     Knee actuator locks at current position mid-motion.
     Hip continues moving → load distribution breaks.
     Observable: foot traces wrong path, hip overloads.

  2. HIP OVERHEAT
     Hip performance degrades gradually over time.
     Output speed and torque reduce as temperature rises.
     Observable: hip slows down, knee has to compensate,
     motion becomes asymmetric and slow.

  3. NOISE + JITTER (both joints)
     Random oscillations added to both joint outputs.
     Simulates encoder failure or loose mechanical coupling.
     Observable: foot trajectory becomes erratic,
     loads spike unpredictably.
=============================================================
"""

import sys
import os
import math
import random

# ── Import chain ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t2_phase1_system_expansion import build_leg_system, LegSystem
from t2_phase3_coupled_dynamics import CoupledDynamicsSimulator
from t2_phase4_coordinated_motion import (
    CoordinatedMotionController,
    get_step_cycle_sequence
)


# ─────────────────────────────────────────────────────────────
# FAILURE PROPAGATION SIMULATOR
# Wraps CoupledDynamicsSimulator and injects failures
# at specified times during motion.
# ─────────────────────────────────────────────────────────────
class FailurePropagationSimulator:
    def __init__(
        self,
        leg: LegSystem,
        time_constant: float = 0.5,
        dt: float = 0.01
    ):
        self.leg = leg
        self.dt  = dt
        self.time_constant = time_constant

        # Base simulator from Phase 3
        self.sim = CoupledDynamicsSimulator(
            leg=leg,
            time_constant=time_constant,
            dt=dt
        )

        # Failure state flags
        self.knee_stalled      = False
        self.hip_overheating   = False
        self.noise_active      = False

        # Overheat tracking
        self.hip_temperature   = 25.0   # degrees C
        self.heat_rate         = 8.0    # degrees C per second of motion
        self.cool_rate         = 2.0    # degrees C per second at rest
        self.overheat_threshold= 80.0   # C — performance starts degrading
        self.critical_temp     = 110.0  # C — near shutdown

        # Noise parameters
        self.noise_amplitude   = 3.0    # degrees of random jitter

        # Stall locked angles
        self.knee_stall_angle  = None
        self.hip_stall_angle   = None

    def reset(self, hip_angle: float = 0.0, knee_angle: float = 0.0):
        """Resets simulation and all failure states."""
        self.sim.reset(hip_angle, knee_angle)
        self.knee_stalled    = False
        self.hip_overheating = False
        self.noise_active    = False
        self.hip_temperature = 25.0
        self.knee_stall_angle= None
        self.hip_stall_angle = None

    def inject_knee_stall(self, at_time: float, history: list):
        """
        Stalls the knee joint at a specific simulation time.
        Knee locks at its current angle — cannot move further.
        """
        current_time = history[-1]["time"] if history else 0.0
        if not self.knee_stalled and current_time >= at_time:
            self.knee_stall_angle = self.leg.knee.joint_output.current_angle
            self.knee_stalled = True
            self.leg.knee.actuator.is_stalled = True
            print(f"\n  *** KNEE STALL INJECTED at t={current_time:.2f}s ***")
            print(f"      Knee locked at {self.knee_stall_angle:.2f} degrees")

    def inject_hip_overheat(self):
        """Activates hip overheating mode."""
        if not self.hip_overheating:
            self.hip_overheating = True
            print(f"\n  *** HIP OVERHEAT MODE ACTIVATED ***")
            print(f"      Temperature rising from {self.hip_temperature:.1f}C")

    def inject_noise(self):
        """Activates noise/jitter on both joints."""
        if not self.noise_active:
            self.noise_active = True
            print(f"\n  *** NOISE/JITTER INJECTED on both joints ***")
            print(f"      Amplitude: +/- {self.noise_amplitude} degrees")

    def _apply_failures(self, snap: dict) -> dict:
        """
        Applies active failure effects to a simulation snapshot.
        Called after every step.
        """
        # ── Knee Stall ────────────────────────────────────────
        if self.knee_stalled and self.knee_stall_angle is not None:
            # Force knee back to stall angle
            self.leg.knee.joint_output.current_angle = self.knee_stall_angle
            self.leg.knee.joint_output.angular_velocity = 0.0
            snap["knee_angle"]    = self.knee_stall_angle
            snap["knee_velocity"] = 0.0
            snap["failure_knee"]  = "STALLED"

            # Hip load increases because it now carries unbalanced leg
            # Add 20% extra load to hip due to compensation attempt
            snap["hip_load_Nm"] = snap["hip_load_Nm"] * 1.20
        else:
            snap["failure_knee"] = "OK"

        # ── Hip Overheat ─────────────────────────────────────
        if self.hip_overheating:
            # Temperature rises during motion
            if abs(self.leg.hip.joint_output.angular_velocity) > 0.1:
                self.hip_temperature += self.heat_rate * self.dt
            else:
                self.hip_temperature -= self.cool_rate * self.dt
                self.hip_temperature  = max(25.0, self.hip_temperature)

            # Performance degrades as temperature rises
            if self.hip_temperature > self.overheat_threshold:
                # Reduce effective speed proportionally
                excess = self.hip_temperature - self.overheat_threshold
                max_excess = self.critical_temp - self.overheat_threshold
                degradation = min(0.85, excess / max_excess)

                # Slow down hip velocity
                self.leg.hip.joint_output.angular_velocity *= (
                    1.0 - degradation
                )
                snap["hip_velocity"] = round(
                    self.leg.hip.joint_output.angular_velocity, 3
                )
                snap["failure_hip"] = f"OVERHEAT_{self.hip_temperature:.0f}C"
            else:
                snap["failure_hip"] = f"WARM_{self.hip_temperature:.0f}C"

            snap["hip_temperature"] = round(self.hip_temperature, 1)
        else:
            snap["failure_hip"]     = "OK"
            snap["hip_temperature"] = round(self.hip_temperature, 1)

        # ── Noise / Jitter ────────────────────────────────────
        if self.noise_active:
            hip_noise  = random.uniform(
                -self.noise_amplitude, self.noise_amplitude
            )
            knee_noise = random.uniform(
                -self.noise_amplitude, self.noise_amplitude
            )
            # Apply noise to current angles
            new_hip  = self.leg.hip.joint_output.current_angle  + hip_noise
            new_knee = self.leg.knee.joint_output.current_angle + knee_noise

            # Clamp to joint limits
            new_hip  = self.leg.hip.joint_output.clamp_angle(new_hip)
            new_knee = self.leg.knee.joint_output.clamp_angle(new_knee)

            self.leg.hip.joint_output.current_angle  = new_hip
            self.leg.knee.joint_output.current_angle = new_knee

            snap["hip_angle"]    = round(new_hip,  4)
            snap["knee_angle"]   = round(new_knee, 4)
            snap["failure_noise"]= f"JITTER_{self.noise_amplitude}deg"
        else:
            snap["failure_noise"] = "OK"

        # Recompute foot position after failures applied
        hip_rad  = math.radians(snap["hip_angle"])
        knee_rad = math.radians(snap["knee_angle"])
        combined = hip_rad + knee_rad
        L1 = self.leg.L1
        L2 = self.leg.L2
        knee_x = L1 * math.cos(hip_rad)
        knee_y = L1 * math.sin(hip_rad)
        snap["foot_x"] = round(knee_x + L2 * math.cos(combined), 4)
        snap["foot_y"] = round(knee_y + L2 * math.sin(combined), 4)

        return snap

    def simulate_with_failure(
        self,
        hip_target: float,
        knee_target: float,
        duration: float,
        label: str = "",
        stall_knee_at: float = None,
        activate_overheat: bool = False,
        activate_noise: bool = False
    ) -> list:
        """
        Runs a simulation with failures injected at specified times.

        Parameters:
          hip_target        : Desired hip target angle
          knee_target       : Desired knee target angle
          duration          : How long to simulate
          label             : Scenario name
          stall_knee_at     : Time (s) to inject knee stall
          activate_overheat : Start hip overheat from beginning
          activate_noise    : Start noise/jitter from beginning
        """
        self.leg.hip.set_target(hip_target)
        self.leg.knee.set_target(knee_target)

        if activate_overheat:
            self.inject_hip_overheat()
        if activate_noise:
            self.inject_noise()

        print(f"\n  {'='*60}")
        print(f"  FAILURE SCENARIO: {label}")
        print(f"  Hip target  : {self.leg.hip.target_angle}°")
        print(f"  Knee target : {self.leg.knee.target_angle}°")
        print(f"  Duration    : {duration}s")
        print(f"  {'='*60}")
        print(f"  {'Time':>6} | {'Hip°':>7} | {'Knee°':>7} | "
              f"{'Temp':>6} | {'Failure State':>30}")
        print(f"  {'-'*70}")

        steps   = int(duration / self.dt)
        history = []

        for i in range(steps):
            # Check if knee stall should be injected
            if stall_knee_at is not None:
                self.inject_knee_stall(stall_knee_at, history)

            # Normal step
            snap = self.sim.step()

            # Apply failure effects
            snap = self._apply_failures(snap)
            history.append(snap)

            # Print every 50 steps
            if i % 50 == 0 or i == steps - 1:
                temp_str = (f"{snap.get('hip_temperature', 25.0):.0f}C")
                fail_str = (f"Hip:{snap['failure_hip'][:12]:12} | "
                           f"Knee:{snap['failure_knee'][:10]:10}")
                print(f"  {snap['time']:>6.2f} | "
                      f"{snap['hip_angle']:>7.2f} | "
                      f"{snap['knee_angle']:>7.2f} | "
                      f"{temp_str:>6} | "
                      f"{fail_str}")

        return history


# ─────────────────────────────────────────────────────────────
# HEALTHY BASELINE
# Run the same sequence without any failures for comparison
# ─────────────────────────────────────────────────────────────
def run_healthy_baseline(leg, sim) -> list:
    """Runs a clean step cycle with no failures — baseline."""
    controller = CoordinatedMotionController(sim)
    sim.reset(0.0, 0.0)
    stages   = get_step_cycle_sequence()
    history  = controller.run_sequence(stages, "Healthy Baseline")

    # Add foot positions and failure tags
    for snap in history:
        hip_rad  = math.radians(snap["hip_angle"])
        knee_rad = math.radians(snap["knee_angle"])
        combined = hip_rad + knee_rad
        knee_x   = leg.L1 * math.cos(hip_rad)
        knee_y   = leg.L1 * math.sin(hip_rad)
        snap["foot_x"]        = round(knee_x + leg.L2 * math.cos(combined), 4)
        snap["foot_y"]        = round(knee_y + leg.L2 * math.sin(combined), 4)
        snap["failure_knee"]  = "OK"
        snap["failure_hip"]   = "OK"
        snap["failure_noise"] = "OK"
        snap["hip_temperature"] = 25.0

    return history


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_scenarios: dict):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "t2_phase6_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 2 - PHASE 6: FAILURE PROPAGATION LOG\n")
        f.write("System-level failure simulation across hip + knee\n")
        f.write("=" * 85 + "\n\n")

        for name, history in all_scenarios.items():
            f.write(f"SCENARIO: {name}\n")
            f.write(
                f"{'Time':>7} | {'Hip':>7} | {'Knee':>7} | "
                f"{'FootX':>7} | {'FootY':>7} | "
                f"{'Temp':>6} | {'HipFail':>15} | {'KneeFail':>12}\n"
            )
            f.write("-" * 85 + "\n")
            for s in history:
                t = s.get("global_time", s.get("time", 0))
                f.write(
                    f"{t:>7.3f} | "
                    f"{s['hip_angle']:>7.2f} | "
                    f"{s['knee_angle']:>7.2f} | "
                    f"{s.get('foot_x', 0):>7.4f} | "
                    f"{s.get('foot_y', 0):>7.4f} | "
                    f"{s.get('hip_temperature', 25.0):>5.1f}C | "
                    f"{s.get('failure_hip',   'OK'):>15} | "
                    f"{s.get('failure_knee',  'OK'):>12}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] -> t2_phase6_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_scenarios: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(
            "Task 2 - Phase 6: Failure Propagation\n"
            "System Behavior Under Knee Stall / Overheat / Noise",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.48, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])  # Hip angles comparison
        ax2 = fig.add_subplot(gs[0, 1])  # Knee angles comparison
        ax3 = fig.add_subplot(gs[1, 0])  # Foot trajectories
        ax4 = fig.add_subplot(gs[1, 1])  # Hip temperature / load

        colors = {
            "Healthy"        : ("green",      "-"),
            "Knee Stall"     : ("red",        "--"),
            "Hip Overheat"   : ("darkorange", "-."),
            "Noise Jitter"   : ("purple",     ":"),
        }

        for name, history in all_scenarios.items():
            times      = [s.get("global_time", s.get("time", 0))
                          for s in history]
            hip_angles = [s["hip_angle"]  for s in history]
            kne_angles = [s["knee_angle"] for s in history]
            foot_xs    = [s.get("foot_x", 0) for s in history]
            foot_ys    = [s.get("foot_y", 0) for s in history]
            temps      = [s.get("hip_temperature", 25.0) for s in history]
            hip_loads  = [abs(s.get("hip_load_Nm", 0)) for s in history]

            color, ls = colors.get(name, ("blue", "-"))

            ax1.plot(times, hip_angles, color=color, linestyle=ls,
                     linewidth=2, label=name)
            ax2.plot(times, kne_angles, color=color, linestyle=ls,
                     linewidth=2, label=name)
            ax3.plot(foot_xs, foot_ys, color=color, linestyle=ls,
                     linewidth=1.5, label=name, alpha=0.8)
            ax4.plot(times, temps, color=color, linestyle=ls,
                     linewidth=2, label=f"{name} - Temp")

        # Overheat threshold line
        ax4.axhline(80,  color='orange', linewidth=1.5,
                    linestyle=':', label='Overheat threshold (80C)')
        ax4.axhline(110, color='red',    linewidth=1.5,
                    linestyle=':', label='Critical temp (110C)')

        # Stall annotation on knee plot
        ax2.axvline(2.0, color='red', linewidth=1,
                    linestyle=':', alpha=0.5, label='Stall injected')

        # Hip origin marker
        ax3.scatter(0, 0, color='black', s=100,
                    marker='x', zorder=6, label='Hip origin')

        # Format
        ax1.set_title("Hip Angle: Healthy vs Failed",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Hip Angle (deg)")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2.set_title("Knee Angle: Healthy vs Failed\n"
                      "(red dashed = stall locks knee)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Knee Angle (deg)")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3.set_title("Foot Trajectory: Healthy vs Failed\n"
                      "(failure = distorted path)",
                      fontweight='bold')
        ax3.set_xlabel("X position (m)")
        ax3.set_ylabel("Y position (m)")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.axhline(0, color='gray', linewidth=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5)

        ax4.set_title("Hip Temperature Over Time\n"
                      "(overheat scenario degradation)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Temperature (C)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "t2_phase6_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t2_phase6_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available. Skipping graph.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase6():
    print("\n" + "=" * 62)
    print("  TASK 2 - PHASE 6: FAILURE PROPAGATION")
    print("  System-level failure across hip + knee")
    print("=" * 62)

    random.seed(42)  # Reproducible noise
    all_scenarios = {}

    # ── Scenario 0: Healthy baseline ────────────────────────
    print("\n--- SCENARIO 0: HEALTHY BASELINE (no failures) ---")
    leg0  = build_leg_system()
    sim0  = CoupledDynamicsSimulator(leg0, time_constant=0.5, dt=0.01)
    h0    = run_healthy_baseline(leg0, sim0)
    all_scenarios["Healthy"] = h0

    # ── Scenario 1: Knee Stall ───────────────────────────────
    print("\n--- SCENARIO 1: KNEE STALL at t=2.0s ---")
    leg1  = build_leg_system()
    fsim1 = FailurePropagationSimulator(leg1, time_constant=0.5, dt=0.01)
    fsim1.reset(0.0, 0.0)
    h1    = fsim1.simulate_with_failure(
        hip_target=60.0, knee_target=60.0,
        duration=6.0,
        label="Knee Stall at t=2s",
        stall_knee_at=2.0
    )
    all_scenarios["Knee Stall"] = h1

    # ── Scenario 2: Hip Overheat ─────────────────────────────
    print("\n--- SCENARIO 2: HIP OVERHEAT (gradual degradation) ---")
    leg2  = build_leg_system()
    fsim2 = FailurePropagationSimulator(leg2, time_constant=0.5, dt=0.01)
    fsim2.reset(0.0, 0.0)
    h2    = fsim2.simulate_with_failure(
        hip_target=70.0, knee_target=50.0,
        duration=8.0,
        label="Hip Overheat - Gradual Degradation",
        activate_overheat=True
    )
    all_scenarios["Hip Overheat"] = h2

    # ── Scenario 3: Noise + Jitter ───────────────────────────
    print("\n--- SCENARIO 3: NOISE/JITTER on both joints ---")
    leg3  = build_leg_system()
    fsim3 = FailurePropagationSimulator(leg3, time_constant=0.5, dt=0.01)
    fsim3.reset(0.0, 0.0)
    h3    = fsim3.simulate_with_failure(
        hip_target=45.0, knee_target=45.0,
        duration=5.0,
        label="Noise and Jitter - Both Joints",
        activate_noise=True
    )
    all_scenarios["Noise Jitter"] = h3

    # ── Failure insights ─────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FAILURE PROPAGATION INSIGHTS")
    print("=" * 62)
    print("  KNEE STALL:")
    print("    - Knee locks mid-motion at stall angle")
    print("    - Hip continues moving independently")
    print("    - Foot traces wrong path — step cycle breaks")
    print("    - Hip load increases 20% due to unbalanced leg")
    print()
    print("  HIP OVERHEAT:")
    print("    - Temperature rises during continuous motion")
    print("    - Above 80C: hip speed degrades proportionally")
    print("    - Leg moves slower — coordination breaks down")
    print("    - Knee receives wrong load distribution")
    print()
    print("  NOISE/JITTER:")
    print("    - Random angle errors on both joints each step")
    print("    - Foot trajectory becomes erratic and unpredictable")
    print("    - Load spikes unpredictably at each jitter")
    print("    - Most dangerous for precision tasks")
    print("=" * 62)

    save_log(all_scenarios)
    plot_results(all_scenarios)

    print("\n  [PHASE 6 COMPLETE] Failure Propagation done.")
    print("  Ready for Phase 7: System Visualization\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase6()
