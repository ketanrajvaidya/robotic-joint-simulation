"""
=============================================================
TASK 3 — PHASE 3 : FULL SYSTEM LOAD DISTRIBUTION
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase adds on top of Phase 2:
  - Computes load per leg dynamically during gait
  - Shows how load shifts between legs as robot walks
  - Identifies which leg takes maximum load
  - Identifies when system is most unstable
  - Produces load vs time graphs per leg

Key concept — Load Shifting During Gait:
  When one leg swings (lifts off ground):
    - That leg carries ZERO ground load
    - The other 3 legs must share the robot's weight
    - The leg closest to the swinging leg gets more load

  Simple analogy:
    When you lift your right foot, your left foot
    takes more of your body weight. Same physics here
    but across 4 legs.

Load model used:
  Each stance leg carries:
    - Its own link gravitational torque (hip + knee)
    - A share of the body weight distributed by position

  Swinging leg carries:
    - Only its own link weight (no ground reaction)
    - Zero body weight contribution

Metrics computed per leg per timestep:
  - gravitational_load_Nm : torque from leg links
  - body_share_Nm         : share of body weight
  - total_load_Nm         : combined load
  - utilization_%         : % of max actuator torque used
  - is_stance             : True if leg is on ground
=============================================================
"""

import sys
import os
import math

# ── Import path setup ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import Phase 1 and Phase 2
from t3_phase1_quadruped_system import (
    build_quadruped,
    QuadrupedSystem,
    CRAWL_GAIT_ORDER
)
from t3_phase2_gait_coordination import (
    CrawlGaitController,
    GaitParameters,
    LegState
)


# ─────────────────────────────────────────────────────────────
# LOAD DISTRIBUTION CALCULATOR
# Computes how load is distributed across all 4 legs
# at every timestep during the gait
# ─────────────────────────────────────────────────────────────
class SystemLoadCalculator:
    def __init__(self, quad: QuadrupedSystem):
        """
        Parameters:
          quad : QuadrupedSystem from Phase 1
        """
        self.quad        = quad
        self.body_mass   = quad.body.mass       # 5.0 kg
        self.g           = 9.81
        self.body_weight = self.body_mass * self.g  # N

        # Max torque per joint
        self.hip_max  = quad.legs["FL"].hip.actuator.max_torque   # 12.0
        self.knee_max = quad.legs["FL"].knee.actuator.max_torque  # 8.5

    def compute_leg_load(
        self,
        leg_name:  str,
        is_stance: bool
    ) -> dict:
        """
        Computes total load on a specific leg.

        Parameters:
          leg_name  : "FL", "FR", "RL", or "RR"
          is_stance : True if leg is on ground, False if swinging

        Returns:
          dict with all load metrics for this leg
        """
        leg = self.quad.legs[leg_name]

        # Gravitational torque from leg links
        hip_grav  = abs(leg.hip_gravitational_load())
        knee_grav = abs(leg.knee_gravitational_load())
        grav_load = hip_grav + knee_grav

        # Body weight share
        # Stance legs share body weight equally
        # Swinging leg gets zero body share
        if is_stance:
            body_share = self.body_weight / 3.0  # 3 legs share body weight
        else:
            body_share = 0.0

        total_load = grav_load + body_share

        # Utilization against hip max (hip carries most)
        utilization = (hip_grav / self.hip_max) * 100

        return {
            "leg"            : leg_name,
            "is_stance"      : is_stance,
            "hip_grav_Nm"    : round(hip_grav,    4),
            "knee_grav_Nm"   : round(knee_grav,   4),
            "grav_load_Nm"   : round(grav_load,   4),
            "body_share_N"   : round(body_share,  4),
            "total_load_Nm"  : round(total_load,  4),
            "utilization_%"  : round(utilization, 2),
        }

    def compute_all_loads(self, snap: dict) -> dict:
        """
        Computes load for all 4 legs from a gait snapshot.

        Parameters:
          snap : Single timestep dict from Phase 2 history

        Returns:
          dict with load data per leg + system totals
        """
        active_leg = snap["active_leg"]
        gait_phase = snap["gait_phase"]

        loads = {}
        for name in ["FL", "FR", "RL", "RR"]:
            # Leg is in swing if it is the active leg
            # AND we are in the SWING phase
            is_swing = (
                name == active_leg and
                gait_phase == LegState.SWING
            )
            loads[name] = self.compute_leg_load(
                name, is_stance=not is_swing
            )

        # System totals
        total_system = sum(
            loads[n]["total_load_Nm"]
            for n in ["FL", "FR", "RL", "RR"]
        )
        max_leg = max(
            loads, key=lambda n: loads[n]["total_load_Nm"]
        )
        min_leg = min(
            loads, key=lambda n: loads[n]["total_load_Nm"]
        )

        return {
            "time"          : snap["time"],
            "active_leg"    : active_leg,
            "gait_phase"    : gait_phase,
            "legs"          : loads,
            "total_system"  : round(total_system, 4),
            "max_load_leg"  : max_leg,
            "min_load_leg"  : min_leg,
        }

    def analyze_full_gait(self, history: list) -> list:
        """
        Runs load calculation across the full gait history.

        Parameters:
          history : Full history list from Phase 2 gait run

        Returns:
          List of load snapshots — one per timestep
        """
        print(f"\n  Computing load distribution for "
              f"{len(history)} timesteps...")

        load_history = []
        for snap in history:
            load_snap = self.compute_all_loads(snap)
            load_history.append(load_snap)

        return load_history


# ─────────────────────────────────────────────────────────────
# LOAD ANALYSIS REPORT
# Prints key findings from the load distribution
# ─────────────────────────────────────────────────────────────
def print_load_report(load_history: list):
    """
    Prints a detailed load distribution analysis report.
    Identifies peak loads, max stress leg, instability moments.
    """
    print("\n" + "=" * 65)
    print("  LOAD DISTRIBUTION ANALYSIS REPORT")
    print("=" * 65)

    # Per leg statistics
    for name in ["FL", "FR", "RL", "RR"]:
        loads = [s["legs"][name]["total_load_Nm"]
                 for s in load_history]
        utils = [s["legs"][name]["utilization_%"]
                 for s in load_history]
        swing = [s for s in load_history
                 if s["active_leg"] == name
                 and s["gait_phase"] == LegState.SWING]

        print(f"\n  {name} LEG:")
        print(f"    Peak load      : {max(loads):.3f} N·m")
        print(f"    Avg load       : {sum(loads)/len(loads):.3f} N·m")
        print(f"    Min load       : {min(loads):.3f} N·m")
        print(f"    Peak util      : {max(utils):.1f}%")
        print(f"    Swing steps    : {len(swing)}")

    # System-level findings
    total_loads = [s["total_system"] for s in load_history]
    max_load_steps = [
        s for s in load_history
        if s["total_system"] == max(total_loads)
    ]
    max_leg_counts = {}
    for s in load_history:
        ml = s["max_load_leg"]
        max_leg_counts[ml] = max_leg_counts.get(ml, 0) + 1

    print(f"\n  SYSTEM FINDINGS:")
    print(f"    Peak total load  : {max(total_loads):.3f} N·m")
    print(f"    Min total load   : {min(total_loads):.3f} N·m")
    print(f"    Avg total load   : "
          f"{sum(total_loads)/len(total_loads):.3f} N·m")
    print(f"\n    Most loaded leg (by timesteps):")
    for leg, count in sorted(
        max_leg_counts.items(),
        key=lambda x: x[1], reverse=True
    ):
        pct = count / len(load_history) * 100
        print(f"      {leg}: {count} steps ({pct:.1f}% of time)")

    print(f"\n    When system is most stressed:")
    print(f"      During STANCE phase of any leg —")
    print(f"      3 legs share body weight equally.")
    print(f"      Peak stress when heaviest leg is in stance.")

    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(load_history: list):
    log_path = os.path.join(BASE_DIR, "t3_phase3_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 3 - PHASE 3: FULL SYSTEM LOAD DISTRIBUTION LOG\n")
        f.write("Load per leg dynamically computed during crawl gait\n")
        f.write("=" * 95 + "\n\n")
        f.write(
            f"{'Time':>7} | {'Active':>6} | {'Phase':>6} | "
            f"{'FL_load':>8} | {'FR_load':>8} | "
            f"{'RL_load':>8} | {'RR_load':>8} | "
            f"{'MaxLeg':>6} | {'Total':>8}\n"
        )
        f.write("-" * 95 + "\n")
        for s in load_history:
            f.write(
                f"{s['time']:>7.3f} | "
                f"{s['active_leg']:>6} | "
                f"{s['gait_phase']:>6} | "
                f"{s['legs']['FL']['total_load_Nm']:>8.3f} | "
                f"{s['legs']['FR']['total_load_Nm']:>8.3f} | "
                f"{s['legs']['RL']['total_load_Nm']:>8.3f} | "
                f"{s['legs']['RR']['total_load_Nm']:>8.3f} | "
                f"{s['max_load_leg']:>6} | "
                f"{s['total_system']:>8.3f}\n"
            )
    print(f"\n  [LOG SAVED] -> t3_phase3_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(load_history: list):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Task 3 - Phase 3: Full System Load Distribution\n"
            "Load per Leg During Crawl Gait",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])  # Load per leg over time
        ax2 = fig.add_subplot(gs[0, 1])  # Utilization % per leg
        ax3 = fig.add_subplot(gs[1, 0])  # Total system load
        ax4 = fig.add_subplot(gs[1, 1])  # Load shift stacked

        times = [s["time"] for s in load_history]

        leg_colors = {
            "FL": "royalblue",
            "FR": "darkorange",
            "RL": "green",
            "RR": "red",
        }

        # ── Plot 1: Load per leg ──────────────────────────────
        for name, color in leg_colors.items():
            loads = [s["legs"][name]["total_load_Nm"]
                     for s in load_history]
            ax1.plot(times, loads, color=color,
                     linewidth=1.5, label=name)

        ax1.set_title("Load per Leg Over Time\n"
                      "(drops when leg swings)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Total Load (N·m)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        # ── Plot 2: Utilization % ─────────────────────────────
        for name, color in leg_colors.items():
            utils = [s["legs"][name]["utilization_%"]
                     for s in load_history]
            ax2.plot(times, utils, color=color,
                     linewidth=1.5, label=name)

        ax2.axhline(90, color='red',    linewidth=1.5,
                    linestyle=':', label='CRITICAL (90%)')
        ax2.axhline(70, color='orange', linewidth=1.5,
                    linestyle=':', label='WARNING (70%)')
        ax2.set_title("Torque Utilization % per Leg",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Utilization (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Total system load ─────────────────────────
        totals = [s["total_system"] for s in load_history]
        ax3.plot(times, totals, 'black', linewidth=2,
                 label='Total system load')
        ax3.fill_between(times, 0, totals,
                         alpha=0.2, color='gray')

        # Shade swing phases
        swing_times = [
            s["time"] for s in load_history
            if s["gait_phase"] == LegState.SWING
        ]
        if swing_times:
            ax3.axvspan(
                swing_times[0], swing_times[-1],
                alpha=0.05, color='blue'
            )

        ax3.set_title("Total System Load Over Time",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Combined Load (N·m)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Stacked load distribution ─────────────────
        fl_loads = [s["legs"]["FL"]["total_load_Nm"]
                    for s in load_history]
        fr_loads = [s["legs"]["FR"]["total_load_Nm"]
                    for s in load_history]
        rl_loads = [s["legs"]["RL"]["total_load_Nm"]
                    for s in load_history]
        rr_loads = [s["legs"]["RR"]["total_load_Nm"]
                    for s in load_history]

        ax4.stackplot(
            times,
            fl_loads, fr_loads, rl_loads, rr_loads,
            labels=["FL", "FR", "RL", "RR"],
            colors=["royalblue", "darkorange", "green", "red"],
            alpha=0.7
        )
        ax4.set_title("Stacked Load Distribution\n"
                      "(shows which leg carries most at each moment)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Load (N·m)")
        ax4.legend(fontsize=8, loc='upper right')
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(BASE_DIR, "t3_phase3_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t3_phase3_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase3():
    print("\n" + "=" * 62)
    print("  TASK 3 - PHASE 3: FULL SYSTEM LOAD DISTRIBUTION")
    print("  Computing load per leg during crawl gait")
    print("=" * 62)

    # Build quadruped
    quad   = build_quadruped()
    params = GaitParameters(
        swing_hip_angle=40.0,
        swing_knee_angle=60.0,
        stance_hip_angle=-10.0,
        stance_knee_angle=10.0,
        swing_duration=2.0,
        stance_duration=1.5,
        dt=0.01
    )
    controller = CrawlGaitController(quad, params)
    controller.reset()

    # Run 2 gait cycles
    print("\n  Running gait simulation...")
    history = controller.run(cycles=2, verbose=False)

    # Compute load distribution
    calculator   = SystemLoadCalculator(quad)
    load_history = calculator.analyze_full_gait(history)

    # Print load table (every 200 steps)
    print(f"\n  LOAD TABLE (sampled every 200 steps):")
    print(
        f"  {'Time':>6} | {'Active':>6} | "
        f"{'FL':>8} | {'FR':>8} | "
        f"{'RL':>8} | {'RR':>8} | {'MaxLeg':>6}"
    )
    print(f"  {'-'*70}")
    for i, s in enumerate(load_history):
        if i % 200 == 0 or i == len(load_history) - 1:
            print(
                f"  {s['time']:>6.2f} | "
                f"{s['active_leg']:>6} | "
                f"{s['legs']['FL']['total_load_Nm']:>7.2f}N | "
                f"{s['legs']['FR']['total_load_Nm']:>7.2f}N | "
                f"{s['legs']['RL']['total_load_Nm']:>7.2f}N | "
                f"{s['legs']['RR']['total_load_Nm']:>7.2f}N | "
                f"{s['max_load_leg']:>6}"
            )

    # Print analysis report
    print_load_report(load_history)

    # Save and plot
    save_log(load_history)
    plot_results(load_history)

    print("\n  [PHASE 3 COMPLETE] Load Distribution done.")
    print("  Ready for Phase 4: Stability + Foot Tracking\n")

    return load_history


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase3()
