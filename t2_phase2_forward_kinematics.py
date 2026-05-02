"""
=============================================================
TASK 2 — PHASE 2 : KINEMATIC COUPLING (FORWARD KINEMATICS)
Task: Multi-Joint Leg Simulation System
=============================================================

What this phase adds on top of Phase 1:
  - Full Forward Kinematics engine for the 2-joint leg
  - Tracks hip position, knee position, foot position over time
  - Simulates the leg moving through a sequence of angles
  - Plots the FOOT TRAJECTORY — the path the foot draws in space
  - Saves a complete kinematics log

What is Forward Kinematics?
  Given the angles of both joints → calculate where the foot is.

  Simple analogy:
    Your arm has a shoulder (hip) and elbow (knee).
    If someone tells you "shoulder lifted 30°, elbow bent 45°"
    you can calculate exactly where your hand (foot) is.
    That calculation is called Forward Kinematics.

  Formulas used:
    knee_x = L1 × cos(θ_hip)
    knee_y = L1 × sin(θ_hip)

    foot_x = knee_x + L2 × cos(θ_hip + θ_knee)
    foot_y = knee_y + L2 × sin(θ_hip + θ_knee)

  Where:
    L1     = thigh length (0.5m)
    L2     = shin length  (0.4m)
    θ_hip  = hip joint angle (degrees → converted to radians)
    θ_knee = knee joint angle (degrees → converted to radians)

Coupling explained:
  - Changing hip angle moves the KNEE and FOOT together
  - Changing knee angle moves only the FOOT
  - They are COUPLED — you cannot move one without affecting the other
=============================================================
"""

import sys
import os
import math

# ── Import Phase 1 (which already imports Task 1) ────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t2_phase1_system_expansion import (
    build_leg_system,
    LegSystem
)


# ─────────────────────────────────────────────────────────────
# FORWARD KINEMATICS ENGINE
# The main new class in Phase 2.
# Takes a LegSystem and computes positions of all points
# as the joints move through different angles over time.
# ─────────────────────────────────────────────────────────────
class ForwardKinematics:
    def __init__(self, leg: LegSystem):
        """
        Parameters:
          leg : The LegSystem from Phase 1 (hip + knee assembled)
        """
        self.leg = leg
        self.L1  = leg.L1   # Thigh length (0.5m)
        self.L2  = leg.L2   # Shin length  (0.4m)

    # ─────────────────────────────────────────────
    # CORE CALCULATION
    # Given two angles → return all 3 positions
    # ─────────────────────────────────────────────
    def compute(self, hip_angle_deg: float, knee_angle_deg: float) -> dict:
        """
        Core forward kinematics calculation.
        Given hip and knee angles, returns positions of:
          - Hip pivot  (always at origin)
          - Knee pivot (depends on hip angle)
          - Foot tip   (depends on both angles)

        Parameters:
          hip_angle_deg  : Hip joint angle in degrees
          knee_angle_deg : Knee joint angle in degrees

        Returns:
          dict with hip, knee, foot positions and both angles
        """
        # Convert degrees to radians for math functions
        hip_rad  = math.radians(hip_angle_deg)
        knee_rad = math.radians(knee_angle_deg)

        # Hip is always fixed at origin
        hip_x, hip_y = 0.0, 0.0

        # Knee position — depends only on hip angle
        knee_x = self.L1 * math.cos(hip_rad)
        knee_y = self.L1 * math.sin(hip_rad)

        # Foot position — depends on BOTH angles combined
        # The knee angle is measured RELATIVE to the thigh direction
        # So total angle from horizontal = hip_angle + knee_angle
        combined_rad = hip_rad + knee_rad
        foot_x = knee_x + self.L2 * math.cos(combined_rad)
        foot_y = knee_y + self.L2 * math.sin(combined_rad)

        return {
            "hip_angle_deg"  : round(hip_angle_deg,  3),
            "knee_angle_deg" : round(knee_angle_deg, 3),
            "hip"            : (round(hip_x,   4), round(hip_y,   4)),
            "knee"           : (round(knee_x,  4), round(knee_y,  4)),
            "foot"           : (round(foot_x,  4), round(foot_y,  4)),
            "leg_reach_m"    : round(math.sqrt(foot_x**2 + foot_y**2), 4)
        }

    # ─────────────────────────────────────────────
    # SIMULATE MOTION SEQUENCE
    # Runs kinematics over a sequence of angle pairs
    # ─────────────────────────────────────────────
    def simulate_sequence(
        self,
        sequence: list,
        label: str = "Motion Sequence"
    ) -> list:
        """
        Runs forward kinematics over a list of (hip, knee) angle pairs.
        Returns a list of result dicts — one per step.

        Parameters:
          sequence : list of (hip_angle, knee_angle) tuples
          label    : name of this sequence for logging

        Example sequence:
          [(0, 0), (15, 10), (30, 20), (45, 30)]
          This represents the leg lifting step by step.
        """
        print(f"\n  {'─'*60}")
        print(f"  SEQUENCE: {label}")
        print(f"  {'─'*60}")
        print(f"  {'Step':>4} | {'Hip°':>6} | {'Knee°':>6} | "
              f"{'Knee (x,y)':>18} | {'Foot (x,y)':>18} | {'Reach':>7}")
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*18}-+-{'-'*18}-+-{'-'*7}")

        results = []
        for i, (hip_deg, knee_deg) in enumerate(sequence):
            r = self.compute(hip_deg, knee_deg)
            results.append(r)

            knee_str = f"({r['knee'][0]:.3f}, {r['knee'][1]:.3f})"
            foot_str = f"({r['foot'][0]:.3f}, {r['foot'][1]:.3f})"
            print(f"  {i+1:>4} | {hip_deg:>6.1f} | {knee_deg:>6.1f} | "
                  f"{knee_str:>18} | {foot_str:>18} | {r['leg_reach_m']:>7.4f}m")

        return results

    # ─────────────────────────────────────────────
    # COUPLING DEMONSTRATION
    # Shows how changing one joint affects foot
    # ─────────────────────────────────────────────
    def demonstrate_coupling(self):
        """
        Demonstrates kinematic coupling:
        1. Hip moves alone → both knee AND foot move
        2. Knee moves alone → only foot moves
        This proves the joints are COUPLED, not independent.
        """
        print(f"\n  {'─'*60}")
        print(f"  COUPLING DEMONSTRATION")
        print(f"  {'─'*60}")

        # Part 1: Fix knee at 0°, sweep hip
        print("\n  PART 1: Knee fixed at 0° — Hip sweeps from 0° to 90°")
        print("  (Shows: hip movement moves BOTH knee and foot)")
        print(f"  {'Hip°':>6} | {'Knee°':>6} | {'Knee pos':>18} | {'Foot pos':>18}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*18}-+-{'-'*18}")
        for hip_deg in [0, 15, 30, 45, 60, 75, 90]:
            r = self.compute(hip_deg, 0.0)
            knee_str = f"({r['knee'][0]:.3f}, {r['knee'][1]:.3f})"
            foot_str = f"({r['foot'][0]:.3f}, {r['foot'][1]:.3f})"
            print(f"  {hip_deg:>6} | {0:>6} | {knee_str:>18} | {foot_str:>18}")

        # Part 2: Fix hip at 45°, sweep knee
        print("\n  PART 2: Hip fixed at 45° — Knee sweeps from 0° to 120°")
        print("  (Shows: knee movement moves ONLY foot, not knee pivot)")
        print(f"  {'Hip°':>6} | {'Knee°':>6} | {'Knee pos':>18} | {'Foot pos':>18}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*18}-+-{'-'*18}")
        for knee_deg in [0, 20, 40, 60, 80, 100, 120]:
            r = self.compute(45.0, knee_deg)
            knee_str = f"({r['knee'][0]:.3f}, {r['knee'][1]:.3f})"
            foot_str = f"({r['foot'][0]:.3f}, {r['foot'][1]:.3f})"
            print(f"  {45:>6} | {knee_deg:>6} | {knee_str:>18} | {foot_str:>18}")


# ─────────────────────────────────────────────────────────────
# MOTION SEQUENCES
# 3 realistic leg motion sequences to simulate
# ─────────────────────────────────────────────────────────────

def get_standing_to_lift_sequence() -> list:
    """
    Sequence 1: Standing → Leg Lift → Return
    Simulates a robot lifting its leg and putting it back down.

    Hip goes from 0° up to 60° (leg swings forward)
    Knee goes from 0° up to 45° (slight bend during lift)
    Then returns to 0°, 0°
    """
    sequence = []
    steps = 20

    # Phase A: Lift leg (both joints move up together)
    for i in range(steps + 1):
        t = i / steps
        hip  = 0  + t * 60   # 0° → 60°
        knee = 0  + t * 45   # 0° → 45°
        sequence.append((round(hip, 2), round(knee, 2)))

    # Phase B: Lower leg back down
    for i in range(steps + 1):
        t = i / steps
        hip  = 60 - t * 60   # 60° → 0°
        knee = 45 - t * 45   # 45° → 0°
        sequence.append((round(hip, 2), round(knee, 2)))

    return sequence


def get_knee_bend_sequence() -> list:
    """
    Sequence 2: Knee Bend (hip stays fixed, knee bends fully)
    Shows the knee's contribution to foot trajectory.

    Hip fixed at 30°
    Knee goes from 0° to 120° and back
    """
    sequence = []
    steps = 20

    for i in range(steps + 1):
        t    = i / steps
        knee = t * 120   # 0° → 120°
        sequence.append((30.0, round(knee, 2)))

    for i in range(steps + 1):
        t    = i / steps
        knee = 120 - t * 120   # 120° → 0°
        sequence.append((30.0, round(knee, 2)))

    return sequence


def get_full_range_sequence() -> list:
    """
    Sequence 3: Full Range Sweep
    Both joints sweep through their full ranges simultaneously.
    Shows maximum foot workspace coverage.

    Hip: -30° → 90°
    Knee: 0° → 120°
    """
    sequence = []
    steps = 30

    for i in range(steps + 1):
        t    = i / steps
        hip  = -30 + t * 120   # -30° → 90°
        knee =   0 + t * 120   #   0° → 120°
        sequence.append((round(hip, 2), round(knee, 2)))

    return sequence


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_results: dict):
    """Saves all kinematics results to a text file."""
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "t2_phase2_log.txt"
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 2 - PHASE 2: FORWARD KINEMATICS LOG\n")
        f.write("Hip Angle + Knee Angle -> Foot Position\n")
        f.write("=" * 70 + "\n\n")

        for seq_name, results in all_results.items():
            f.write(f"SEQUENCE: {seq_name}\n")
            f.write(f"{'Step':>4} | {'Hip°':>7} | {'Knee°':>7} | "
                    f"{'Foot X':>8} | {'Foot Y':>8} | {'Reach':>7}\n")
            f.write("-" * 55 + "\n")
            for i, r in enumerate(results):
                f.write(
                    f"{i+1:>4} | {r['hip_angle_deg']:>7.2f} | "
                    f"{r['knee_angle_deg']:>7.2f} | "
                    f"{r['foot'][0]:>8.4f} | {r['foot'][1]:>8.4f} | "
                    f"{r['leg_reach_m']:>7.4f}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] → t2_phase2_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_results: dict):
    """
    Plots 4 graphs:
      1. Foot trajectory — Sequence 1 (Stand → Lift → Return)
      2. Foot trajectory — Sequence 2 (Knee bend)
      3. Foot trajectory — Sequence 3 (Full range)
      4. Combined workspace — all 3 sequences overlaid
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 11))
        fig.suptitle(
            "Task 2 — Phase 2: Forward Kinematics\nFoot Trajectory & Workspace",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        colors = {
            "Stand → Lift → Return" : ("royalblue",  "o"),
            "Knee Bend (Hip=30°)"   : ("darkorange", "s"),
            "Full Range Sweep"      : ("green",      "^"),
        }

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]

        # Individual trajectory plots
        for ax, (seq_name, results) in zip(axes[:3], all_results.items()):
            foot_x = [r["foot"][0] for r in results]
            foot_y = [r["foot"][1] for r in results]
            knee_x = [r["knee"][0] for r in results]
            knee_y = [r["knee"][1] for r in results]

            color, marker = colors[seq_name]

            # Draw leg segments at first and last position
            r0 = results[0]
            rl = results[-1]
            ax.plot([0, r0["knee"][0], r0["foot"][0]],
                    [0, r0["knee"][1], r0["foot"][1]],
                    'gray', linewidth=1.5, linestyle='--',
                    alpha=0.5, label='Start position')
            ax.plot([0, rl["knee"][0], rl["foot"][0]],
                    [0, rl["knee"][1], rl["foot"][1]],
                    'black', linewidth=1.5, linestyle='--',
                    alpha=0.5, label='End position')

            # Foot trajectory path
            ax.plot(foot_x, foot_y, color=color, linewidth=2,
                    marker=marker, markersize=3, label='Foot path')

            # Knee trajectory path
            ax.plot(knee_x, knee_y, 'purple', linewidth=1,
                    linestyle=':', marker='.', markersize=2,
                    label='Knee path', alpha=0.6)

            # Mark start and end of foot
            ax.scatter(foot_x[0],  foot_y[0],  color='green', s=80,
                       zorder=5, label='Start')
            ax.scatter(foot_x[-1], foot_y[-1], color='red',   s=80,
                       zorder=5, label='End')

            # Mark hip origin
            ax.scatter(0, 0, color='black', s=100, zorder=6,
                       marker='x', label='Hip (origin)')

            ax.set_title(seq_name, fontweight='bold', fontsize=10)
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Y position (m)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)

        # Combined workspace plot
        ax4 = axes[3]
        for seq_name, results in all_results.items():
            foot_x = [r["foot"][0] for r in results]
            foot_y = [r["foot"][1] for r in results]
            color, marker = colors[seq_name]
            ax4.plot(foot_x, foot_y, color=color, linewidth=1.5,
                     marker=marker, markersize=2, label=seq_name, alpha=0.8)

        ax4.scatter(0, 0, color='black', s=120, zorder=6,
                    marker='x', label='Hip origin')
        ax4.set_title("Combined Foot Workspace", fontweight='bold', fontsize=10)
        ax4.set_xlabel("X position (m)")
        ax4.set_ylabel("Y position (m)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        ax4.axhline(0, color='gray', linewidth=0.5)
        ax4.axvline(0, color='gray', linewidth=0.5)

        # Save
        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "t2_phase2_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] → t2_phase2_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: python -m pip install matplotlib")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase2():
    """
    Runs all Phase 2 simulations:
      1. Coupling demonstration
      2. Three motion sequences
      3. Save log
      4. Plot graphs
    """
    print("\n" + "="*60)
    print("  TASK 2 — PHASE 2: FORWARD KINEMATICS")
    print("  Hip Angle + Knee Angle → Foot Position")
    print("="*60)

    # Build leg system from Phase 1
    leg = build_leg_system()
    fk  = ForwardKinematics(leg)

    # ── Single point test ────────────────────────────────────
    print("\n--- SINGLE POINT TEST ---")
    print("  Computing foot position for hip=45°, knee=60°")
    result = fk.compute(45.0, 60.0)
    print(f"  Hip  position : {result['hip']}")
    print(f"  Knee position : {result['knee']}")
    print(f"  Foot position : {result['foot']}")
    print(f"  Leg reach     : {result['leg_reach_m']} m")
    print(f"  (Max possible : {leg.L1 + leg.L2:.2f} m — fully extended)")

    # ── Coupling demonstration ───────────────────────────────
    fk.demonstrate_coupling()

    # ── Motion sequences ─────────────────────────────────────
    all_results = {}

    seq1 = get_standing_to_lift_sequence()
    res1 = fk.simulate_sequence(seq1, label="Stand → Lift → Return")
    all_results["Stand → Lift → Return"] = res1

    seq2 = get_knee_bend_sequence()
    res2 = fk.simulate_sequence(seq2, label="Knee Bend (Hip=30°)")
    all_results["Knee Bend (Hip=30°)"] = res2

    seq3 = get_full_range_sequence()
    res3 = fk.simulate_sequence(seq3, label="Full Range Sweep")
    all_results["Full Range Sweep"] = res3

    # ── Save + Plot ──────────────────────────────────────────
    save_log(all_results)
    plot_results(all_results)

    print("\n  [PHASE 2 COMPLETE] Forward Kinematics done.")
    print("  Ready for Phase 3: Coupled Dynamics\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase2()
