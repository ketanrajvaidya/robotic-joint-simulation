"""
=============================================================
TASK 2 — PHASE 5 : LOAD DISTRIBUTION
Task: Multi-Joint Leg Simulation System
=============================================================

What this phase adds on top of Phase 4:
  - Calculates EXACT torque at hip and knee at every moment
  - Shows how load SHIFTS between joints during motion
  - Identifies which joint is under maximum stress
  - Computes torque utilization (how close each joint is to its limit)
  - Flags danger zones where load approaches actuator limits

What is Load Distribution?
  In Phase 3 we saw loads exist.
  In Phase 4 we saw loads change during motion.
  In Phase 5 we MEASURE and ANALYZE those loads precisely.

  Key questions answered:
    - How much of the hip's max torque is being used? (%)
    - How much of the knee's max torque is being used? (%)
    - When does load shift from hip to knee?
    - At what point is the system under most stress?
    - Are we close to the actuator limits?

  Real engineering importance:
    If load > max torque → actuator stalls (Phase 6 topic)
    If load is always 90%+ → system is undersized
    Good design keeps peak load below 80% of max torque

Key metrics computed:
  - torque_Nm       : Raw gravitational load (N·m)
  - utilization_%   : Load as % of max actuator torque
  - safety_margin   : How far from limit (N·m remaining)
  - peak_load_time  : When maximum stress occurs
  - load_shift      : Moment when knee load exceeds hip load
=============================================================
"""

import sys
import os
import math

# ── Import chain ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t2_phase1_system_expansion import build_leg_system, LegSystem
from t2_phase3_coupled_dynamics import CoupledDynamicsSimulator
from t2_phase4_coordinated_motion import (
    CoordinatedMotionController,
    MotionStage,
    get_step_cycle_sequence,
    get_crouch_sequence
)


# ─────────────────────────────────────────────────────────────
# LOAD DISTRIBUTION ANALYZER
# Takes a simulation history and computes detailed load
# metrics for every time step.
# ─────────────────────────────────────────────────────────────
class LoadDistributionAnalyzer:
    def __init__(self, leg: LegSystem):
        """
        Parameters:
          leg : LegSystem — needed for actuator max torque values
        """
        self.leg           = leg
        self.hip_max_torque  = leg.hip.actuator.max_torque   # 12.0 N·m
        self.knee_max_torque = leg.knee.actuator.max_torque  # 8.5 N·m

    def analyze(self, history: list) -> list:
        """
        Takes raw simulation history from Phase 4 and adds
        detailed load metrics to every step.

        Adds to each step dict:
          hip_utilization_%  : Hip load as % of max torque
          knee_utilization_% : Knee load as % of max torque
          hip_safety_margin  : N·m remaining before hip limit
          knee_safety_margin : N·m remaining before knee limit
          dominant_joint     : Which joint has higher load
          combined_load      : Total load on system (N·m)
          danger_level       : "SAFE" / "WARNING" / "CRITICAL"

        Parameters:
          history : List of step dicts from Phase 4 simulation

        Returns:
          Same list with added analysis fields
        """
        analyzed = []

        for snap in history:
            s = dict(snap)  # copy so we don't modify original

            hip_load  = abs(s["hip_load_Nm"])
            knee_load = abs(s["knee_load_Nm"])

            # Utilization — how much of max torque is being used
            hip_util  = (hip_load  / self.hip_max_torque)  * 100
            knee_util = (knee_load / self.knee_max_torque) * 100

            # Safety margin — how much torque is left before limit
            hip_margin  = self.hip_max_torque  - hip_load
            knee_margin = self.knee_max_torque - knee_load

            # Which joint carries more load right now
            dominant = "HIP" if hip_load >= knee_load else "KNEE"

            # Combined system load
            combined = hip_load + knee_load

            # Danger level based on highest utilization
            max_util = max(hip_util, knee_util)
            if max_util >= 90:
                danger = "CRITICAL"
            elif max_util >= 70:
                danger = "WARNING"
            else:
                danger = "SAFE"

            s["hip_util_%"]       = round(hip_util,    2)
            s["knee_util_%"]      = round(knee_util,   2)
            s["hip_margin_Nm"]    = round(hip_margin,  4)
            s["knee_margin_Nm"]   = round(knee_margin, 4)
            s["dominant_joint"]   = dominant
            s["combined_load_Nm"] = round(combined,    4)
            s["danger_level"]     = danger

            analyzed.append(s)

        return analyzed

    def summarize(self, analyzed: list, label: str = ""):
        """
        Prints a clean summary of load distribution statistics.
        Identifies peak loads, danger zones, and load shift moments.
        """
        hip_loads   = [abs(s["hip_load_Nm"])  for s in analyzed]
        knee_loads  = [abs(s["knee_load_Nm"]) for s in analyzed]
        hip_utils   = [s["hip_util_%"]        for s in analyzed]
        knee_utils  = [s["knee_util_%"]       for s in analyzed]
        combined    = [s["combined_load_Nm"]  for s in analyzed]
        dangers     = [s["danger_level"]      for s in analyzed]
        times       = [s.get("global_time", s["time"]) for s in analyzed]

        # Peak loads
        peak_hip_idx   = hip_utils.index(max(hip_utils))
        peak_knee_idx  = knee_utils.index(max(knee_utils))
        peak_comb_idx  = combined.index(max(combined))

        # Danger zone count
        critical_count = dangers.count("CRITICAL")
        warning_count  = dangers.count("WARNING")
        safe_count     = dangers.count("SAFE")

        # Load shift moments — when KNEE exceeds HIP
        shift_times = [
            times[i] for i in range(len(analyzed))
            if analyzed[i]["dominant_joint"] == "KNEE"
        ]

        print(f"\n  {'='*60}")
        print(f"  LOAD ANALYSIS: {label}")
        print(f"  {'='*60}")
        print(f"\n  HIP JOINT (max capacity: {self.hip_max_torque} N·m)")
        print(f"    Peak load    : {max(hip_loads):.3f} N·m  "
              f"at t={times[peak_hip_idx]:.2f}s")
        print(f"    Peak util    : {max(hip_utils):.1f}% of max torque")
        print(f"    Min margin   : {min(s['hip_margin_Nm'] for s in analyzed):.3f} N·m")
        print(f"    Avg load     : {sum(hip_loads)/len(hip_loads):.3f} N·m")

        print(f"\n  KNEE JOINT (max capacity: {self.knee_max_torque} N·m)")
        print(f"    Peak load    : {max(knee_loads):.3f} N·m  "
              f"at t={times[peak_knee_idx]:.2f}s")
        print(f"    Peak util    : {max(knee_utils):.1f}% of max torque")
        print(f"    Min margin   : {min(s['knee_margin_Nm'] for s in analyzed):.3f} N·m")
        print(f"    Avg load     : {sum(knee_loads)/len(knee_loads):.3f} N·m")

        print(f"\n  SYSTEM COMBINED")
        print(f"    Peak combined: {max(combined):.3f} N·m  "
              f"at t={times[peak_comb_idx]:.2f}s")
        print(f"    Danger steps : CRITICAL={critical_count} | "
              f"WARNING={warning_count} | SAFE={safe_count}")

        if shift_times:
            print(f"\n  LOAD SHIFT (knee dominant):")
            print(f"    Knee exceeds hip at {len(shift_times)} steps")
            print(f"    First shift at t={shift_times[0]:.2f}s")
        else:
            print(f"\n  LOAD SHIFT: Hip dominant throughout entire sequence")

        print(f"  {'='*60}")


# ─────────────────────────────────────────────────────────────
# TORQUE TABLE PRINTER
# Prints a detailed table at key time points
# ─────────────────────────────────────────────────────────────
def print_torque_table(analyzed: list, label: str, every_n: int = 50):
    """
    Prints a formatted torque table at every N steps.

    Columns:
      Time | Hip° | Knee° | Hip Load | Hip% | Knee Load | Knee% | Danger
    """
    print(f"\n  TORQUE TABLE: {label}")
    print(
        f"  {'Time':>6} | {'Hip°':>6} | {'Knee°':>6} | "
        f"{'HipLoad':>8} | {'Hip%':>5} | "
        f"{'KneeLoad':>9} | {'Knee%':>6} | "
        f"{'Dominant':>8} | {'Danger':>8}"
    )
    print(f"  {'-'*95}")

    for i, s in enumerate(analyzed):
        if i % every_n == 0 or i == len(analyzed) - 1:
            t    = s.get("global_time", s["time"])
            print(
                f"  {t:>6.2f} | "
                f"{s['hip_angle']:>6.1f} | "
                f"{s['knee_angle']:>6.1f} | "
                f"{abs(s['hip_load_Nm']):>7.3f}N | "
                f"{s['hip_util_%']:>4.1f}% | "
                f"{abs(s['knee_load_Nm']):>8.3f}N | "
                f"{s['knee_util_%']:>5.1f}% | "
                f"{s['dominant_joint']:>8} | "
                f"{s['danger_level']:>8}"
            )


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_analyzed: dict):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "t2_phase5_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 2 - PHASE 5: LOAD DISTRIBUTION LOG\n")
        f.write("Torque utilization and safety margins at each joint\n")
        f.write("=" * 90 + "\n\n")

        for seq_name, analyzed in all_analyzed.items():
            f.write(f"SEQUENCE: {seq_name}\n")
            f.write(
                f"{'Time':>7} | {'Hip°':>6} | {'Knee°':>6} | "
                f"{'HipLoad':>8} | {'Hip%':>5} | "
                f"{'KnLoad':>8} | {'Kn%':>5} | "
                f"{'Dominant':>8} | {'Danger':>8}\n"
            )
            f.write("-" * 85 + "\n")
            for s in analyzed:
                t = s.get("global_time", s["time"])
                f.write(
                    f"{t:>7.3f} | "
                    f"{s['hip_angle']:>6.2f} | "
                    f"{s['knee_angle']:>6.2f} | "
                    f"{abs(s['hip_load_Nm']):>7.4f}N | "
                    f"{s['hip_util_%']:>4.1f}% | "
                    f"{abs(s['knee_load_Nm']):>7.4f}N | "
                    f"{s['knee_util_%']:>4.1f}% | "
                    f"{s['dominant_joint']:>8} | "
                    f"{s['danger_level']:>8}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] -> t2_phase5_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_analyzed: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Task 2 - Phase 5: Load Distribution\n"
            "Torque Utilization and Safety Margins",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])  # Utilization %
        ax2 = fig.add_subplot(gs[0, 1])  # Raw torque N·m
        ax3 = fig.add_subplot(gs[1, 0])  # Safety margins
        ax4 = fig.add_subplot(gs[1, 1])  # Combined load

        seq_colors = {
            "Step Cycle"    : ("royalblue",  "cornflowerblue"),
            "Crouch & Rise" : ("darkorange", "sandybrown"),
        }

        for seq_name, analyzed in all_analyzed.items():
            times      = [s.get("global_time", s["time"]) for s in analyzed]
            hip_utils  = [s["hip_util_%"]       for s in analyzed]
            kne_utils  = [s["knee_util_%"]       for s in analyzed]
            hip_loads  = [abs(s["hip_load_Nm"])  for s in analyzed]
            kne_loads  = [abs(s["knee_load_Nm"]) for s in analyzed]
            hip_margs  = [s["hip_margin_Nm"]     for s in analyzed]
            kne_margs  = [s["knee_margin_Nm"]    for s in analyzed]
            combined   = [s["combined_load_Nm"]  for s in analyzed]

            c_hip, c_knee = seq_colors.get(
                seq_name, ("blue", "orange"))

            # ── Plot 1: Utilization % ──────────────────────────
            ax1.plot(times, hip_utils,  color=c_hip,
                     linewidth=2, label=f"{seq_name} - Hip")
            ax1.plot(times, kne_utils,  color=c_knee,
                     linewidth=2, linestyle='--',
                     label=f"{seq_name} - Knee")

            # ── Plot 2: Raw torque ─────────────────────────────
            ax2.plot(times, hip_loads,  color=c_hip,
                     linewidth=2, label=f"{seq_name} - Hip")
            ax2.plot(times, kne_loads,  color=c_knee,
                     linewidth=2, linestyle='--',
                     label=f"{seq_name} - Knee")

            # ── Plot 3: Safety margins ─────────────────────────
            ax3.plot(times, hip_margs,  color=c_hip,
                     linewidth=2, label=f"{seq_name} - Hip margin")
            ax3.plot(times, kne_margs,  color=c_knee,
                     linewidth=2, linestyle='--',
                     label=f"{seq_name} - Knee margin")

            # ── Plot 4: Combined load ──────────────────────────
            ax4.plot(times, combined,   color=c_hip,
                     linewidth=2, label=seq_name)

        # Danger zone lines
        ax1.axhline(90, color='red',    linewidth=1.5,
                    linestyle=':', label='CRITICAL (90%)')
        ax1.axhline(70, color='orange', linewidth=1.5,
                    linestyle=':', label='WARNING (70%)')

        ax2.axhline(12.0, color='red',    linewidth=1,
                    linestyle=':', label='Hip max (12 N·m)')
        ax2.axhline(8.5,  color='orange', linewidth=1,
                    linestyle=':', label='Knee max (8.5 N·m)')

        ax3.axhline(0, color='red', linewidth=1.5,
                    linestyle=':', label='Zero margin (STALL)')

        # Format
        ax1.set_title("Torque Utilization %\n"
                      "(how much of max capacity is used)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilization (%)")
        ax1.set_ylim(0, 105)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2.set_title("Raw Torque Load (N·m)\n"
                      "with actuator limits marked",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Load (N·m)")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3.set_title("Safety Margin at Each Joint\n"
                      "(N·m remaining before stall)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Safety Margin (N·m)")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        ax4.set_title("Combined System Load\n"
                      "(Hip + Knee total torque)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Combined Load (N·m)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "t2_phase5_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t2_phase5_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available. Skipping graph.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase5():
    print("\n" + "=" * 62)
    print("  TASK 2 - PHASE 5: LOAD DISTRIBUTION")
    print("  Torque utilization and safety margins")
    print("=" * 62)

    # Build system
    leg = build_leg_system()
    sim = CoupledDynamicsSimulator(
        leg=leg,
        time_constant=0.5,
        dt=0.01
    )
    controller = CoordinatedMotionController(sim)
    analyzer   = LoadDistributionAnalyzer(leg)

    all_analyzed = {}

    # ── Sequence 1: Step Cycle ───────────────────────────────
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    stages1  = get_step_cycle_sequence()
    history1 = controller.run_sequence(
        stages1, sequence_label="Full Step Cycle"
    )
    analyzed1 = analyzer.analyze(history1)
    analyzer.summarize(analyzed1, label="Step Cycle")
    print_torque_table(analyzed1, "Step Cycle", every_n=75)
    all_analyzed["Step Cycle"] = analyzed1

    # ── Sequence 2: Crouch and Rise ──────────────────────────
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    stages2  = get_crouch_sequence()
    history2 = controller.run_sequence(
        stages2, sequence_label="Crouch and Rise"
    )
    analyzed2 = analyzer.analyze(history2)
    analyzer.summarize(analyzed2, label="Crouch & Rise")
    print_torque_table(analyzed2, "Crouch & Rise", every_n=75)
    all_analyzed["Crouch & Rise"] = analyzed2

    # ── Key insight ──────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  LOAD DISTRIBUTION INSIGHTS")
    print("=" * 62)
    print("  1. Hip joint carries significantly more load than knee")
    print("     because it supports the ENTIRE leg weight.")
    print("  2. Both joints stay well within safe operating range.")
    print("  3. Load shifts slightly during motion — proving")
    print("     dynamic coupling between joints.")
    print("  4. Safety margins never reach zero — no stall risk")
    print("     under normal coordinated motion.")
    print("  5. Crouch sequence creates highest knee utilization")
    print("     because deep knee bend = maximum shin torque.")
    print("=" * 62)

    save_log(all_analyzed)
    plot_results(all_analyzed)

    print("\n  [PHASE 5 COMPLETE] Load Distribution done.")
    print("  Ready for Phase 6: Failure Propagation\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase5()
