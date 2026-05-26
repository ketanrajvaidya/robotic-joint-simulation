"""
=============================================================
TASK 3 — PHASE 7 : DATA + LOGGING LAYER (FOR DHRUV)
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase does:
  - Builds a structured telemetry logging system for Dhruv
  - Every timestep outputs ALL required fields
  - Format is consistent and machine-readable
  - Produces master visualization dashboard
  - Final deliverable for Task 3 submission

Required fields per timestep (from task PDF):
  - time
  - hip angles (4 legs)
  - knee angles (4 legs)
  - foot positions (x, y) for 4 legs
  - load per leg
  - system state
  - failure flags

This phase:
  1. Runs the full quadruped simulation (healthy + failure)
  2. Logs every field in structured format for Dhruv
  3. Produces all 4 required graphs:
       Graph 1: Leg angles (all 4 legs)
       Graph 2: Foot trajectory (all 4 feet)
       Graph 3: Load distribution per leg
       Graph 4: Stability state over time
  4. Produces a master dashboard combining everything
=============================================================
"""

import sys
import os
import math
import csv

# ── Import path setup ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from t3_phase1_quadruped_system import (
    build_quadruped,
    CRAWL_GAIT_ORDER
)
from t3_phase2_gait_coordination import (
    CrawlGaitController,
    GaitParameters,
    LegState
)
from t3_phase3_load_distribution import SystemLoadCalculator
from t3_phase4_stability import StabilityAnalyzer
from t3_phase5_failure_propagation import (
    get_default_params,
    run_healthy,
    FailurePropagationSimulator
)
from t3_phase6_control_interface import (
    QuadrupedControlInterface,
    GaitCommand,
    SystemStatePacket
)


# ─────────────────────────────────────────────────────────────
# TELEMETRY LOGGER
# Structured data logger for Dhruv's telemetry system
# Every field required by task PDF is logged here
# ─────────────────────────────────────────────────────────────
class TelemetryLogger:
    def __init__(self):
        self.records = []

    def log(self, packet: SystemStatePacket):
        """
        Logs one timestep of telemetry data.
        All required fields from task PDF included.
        """
        record = {
            # ── Time ──────────────────────────────────────────
            "time"              : packet.timestamp,

            # ── Hip angles (4 legs) ───────────────────────────
            "hip_FL"            : packet.hip_angles["FL"],
            "hip_FR"            : packet.hip_angles["FR"],
            "hip_RL"            : packet.hip_angles["RL"],
            "hip_RR"            : packet.hip_angles["RR"],

            # ── Knee angles (4 legs) ──────────────────────────
            "knee_FL"           : packet.knee_angles["FL"],
            "knee_FR"           : packet.knee_angles["FR"],
            "knee_RL"           : packet.knee_angles["RL"],
            "knee_RR"           : packet.knee_angles["RR"],

            # ── Foot positions (x, y) per leg ─────────────────
            "foot_FL_x"         : packet.foot_positions["FL"][0],
            "foot_FL_y"         : packet.foot_positions["FL"][1],
            "foot_FR_x"         : packet.foot_positions["FR"][0],
            "foot_FR_y"         : packet.foot_positions["FR"][1],
            "foot_RL_x"         : packet.foot_positions["RL"][0],
            "foot_RL_y"         : packet.foot_positions["RL"][1],
            "foot_RR_x"         : packet.foot_positions["RR"][0],
            "foot_RR_y"         : packet.foot_positions["RR"][1],

            # ── Load per leg ──────────────────────────────────
            "load_FL"           : packet.load_per_leg["FL"],
            "load_FR"           : packet.load_per_leg["FR"],
            "load_RL"           : packet.load_per_leg["RL"],
            "load_RR"           : packet.load_per_leg["RR"],

            # ── System state ──────────────────────────────────
            "system_state"      : packet.system_state,
            "active_leg"        : packet.active_leg,
            "gait_phase"        : packet.gait_phase,
            "gait_progress"     : packet.gait_progress,
            "polygon_area"      : packet.polygon_area,
            "com_margin"        : packet.com_margin,

            # ── Failure flags ─────────────────────────────────
            "failure_FL"        : packet.failure_flags["FL"],
            "failure_FR"        : packet.failure_flags["FR"],
            "failure_RL"        : packet.failure_flags["RL"],
            "failure_RR"        : packet.failure_flags["RR"],
            "any_failure"       : any(
                packet.failure_flags.values()
            ),
        }
        self.records.append(record)

    def save_csv(self, filename: str):
        """
        Saves telemetry to CSV format.
        CSV is the most compatible format for Dhruv's
        telemetry pipeline.
        """
        if not self.records:
            print("  [WARNING] No records to save.")
            return

        path = os.path.join(BASE_DIR, filename)
        with open(path, "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.records[0].keys()
            )
            writer.writeheader()
            writer.writerows(self.records)
        print(f"  [CSV SAVED] -> {filename} "
              f"({len(self.records)} rows)")

    def save_txt(self, filename: str):
        """Saves telemetry to structured text format."""
        if not self.records:
            return

        path = os.path.join(BASE_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("TASK 3 - PHASE 7: TELEMETRY LOG\n")
            f.write("For Dhruv Patel — Data/Telemetry Systems\n")
            f.write("=" * 110 + "\n")
            f.write(
                f"{'time':>7} | "
                f"{'hip_FL':>7} | {'hip_FR':>7} | "
                f"{'hip_RL':>7} | {'hip_RR':>7} | "
                f"{'kne_FL':>7} | {'kne_FR':>7} | "
                f"{'kne_RL':>7} | {'kne_RR':>7} | "
                f"{'ld_FL':>6} | {'ld_FR':>6} | "
                f"{'ld_RL':>6} | {'ld_RR':>6} | "
                f"{'state':>8} | {'fail':>5}\n"
            )
            f.write("-" * 110 + "\n")
            for r in self.records:
                any_fail = "YES" if r["any_failure"] else "NO"
                f.write(
                    f"{r['time']:>7.3f} | "
                    f"{r['hip_FL']:>7.2f} | "
                    f"{r['hip_FR']:>7.2f} | "
                    f"{r['hip_RL']:>7.2f} | "
                    f"{r['hip_RR']:>7.2f} | "
                    f"{r['knee_FL']:>7.2f} | "
                    f"{r['knee_FR']:>7.2f} | "
                    f"{r['knee_RL']:>7.2f} | "
                    f"{r['knee_RR']:>7.2f} | "
                    f"{r['load_FL']:>6.2f} | "
                    f"{r['load_FR']:>6.2f} | "
                    f"{r['load_RL']:>6.2f} | "
                    f"{r['load_RR']:>6.2f} | "
                    f"{r['system_state']:>8} | "
                    f"{any_fail:>5}\n"
                )
        print(f"  [TXT SAVED] -> {filename} "
              f"({len(self.records)} rows)")

    def get_field(self, field: str) -> list:
        """Returns list of values for a specific field."""
        return [r[field] for r in self.records]

    def summary(self):
        """Prints a summary of logged data."""
        if not self.records:
            return
        times  = self.get_field("time")
        states = self.get_field("system_state")
        stable = states.count("STABLE")
        marg   = states.count("MARGINAL")
        unstab = states.count("UNSTABLE")
        fail   = states.count("FAILURE")
        total  = len(self.records)

        print(f"\n  TELEMETRY SUMMARY")
        print(f"  Total records  : {total}")
        print(f"  Time range     : {times[0]:.2f}s "
              f"to {times[-1]:.2f}s")
        print(f"  STABLE         : {stable} "
              f"({stable/total*100:.1f}%)")
        print(f"  MARGINAL       : {marg} "
              f"({marg/total*100:.1f}%)")
        print(f"  UNSTABLE       : {unstab} "
              f"({unstab/total*100:.1f}%)")
        print(f"  FAILURE        : {fail} "
              f"({fail/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# REQUIRED GRAPH 1: LEG ANGLES
# ─────────────────────────────────────────────────────────────
def plot_graph1_leg_angles(
    logger: TelemetryLogger,
    save_dir: str
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Graph 1: Leg Angles — All 4 Legs",
        fontsize=14, fontweight='bold'
    )

    times = logger.get_field("time")
    leg_colors = {
        "FL": "royalblue",
        "FR": "darkorange",
        "RL": "green",
        "RR": "red",
    }

    # Hip angles
    ax1 = axes[0]
    for name, color in leg_colors.items():
        ax1.plot(times, logger.get_field(f"hip_{name}"),
                 color=color, linewidth=1.5, label=name)
    ax1.set_title("Hip Angles — All 4 Legs",
                  fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Hip Angle (deg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linewidth=0.5)

    # Knee angles
    ax2 = axes[1]
    for name, color in leg_colors.items():
        ax2.plot(times, logger.get_field(f"knee_{name}"),
                 color=color, linewidth=1.5,
                 linestyle='--', label=name)
    ax2.set_title("Knee Angles — All 4 Legs",
                  fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Knee Angle (deg)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        save_dir, "t3_phase7_graph1_leg_angles.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [SAVED] t3_phase7_graph1_leg_angles.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# REQUIRED GRAPH 2: FOOT TRAJECTORY
# ─────────────────────────────────────────────────────────────
def plot_graph2_foot_trajectory(
    logger: TelemetryLogger,
    save_dir: str
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Graph 2: Foot Trajectory — All 4 Feet",
        fontsize=14, fontweight='bold'
    )

    times = logger.get_field("time")
    leg_colors = {
        "FL": "royalblue",
        "FR": "darkorange",
        "RL": "green",
        "RR": "red",
    }

    # Y positions over time
    ax1 = axes[0]
    for name, color in leg_colors.items():
        ax1.plot(
            times,
            logger.get_field(f"foot_{name}_y"),
            color=color, linewidth=1.5, label=name
        )
    ax1.set_title("Foot Y Position Over Time\n"
                  "(peaks = swing phase)",
                  fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Y Position (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2D trajectory (x vs y)
    ax2 = axes[1]
    for name, color in leg_colors.items():
        fx = logger.get_field(f"foot_{name}_x")
        fy = logger.get_field(f"foot_{name}_y")
        ax2.plot(fx, fy, color=color,
                 linewidth=1.5, label=name, alpha=0.8)
        ax2.scatter(fx[0],  fy[0],  color=color,
                    s=60, marker='o', zorder=5)
        ax2.scatter(fx[-1], fy[-1], color=color,
                    s=60, marker='*', zorder=5)

    ax2.scatter(0, 0, color='black', s=100,
                marker='x', zorder=6, label='Body center')
    ax2.set_title("2D Foot Trajectory\n"
                  "(circle=start, star=end)",
                  fontweight='bold')
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(
        save_dir, "t3_phase7_graph2_foot_trajectory.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [SAVED] t3_phase7_graph2_foot_trajectory.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# REQUIRED GRAPH 3: LOAD DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def plot_graph3_load_distribution(
    logger: TelemetryLogger,
    save_dir: str
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Graph 3: Load Distribution Per Leg",
        fontsize=14, fontweight='bold'
    )

    times = logger.get_field("time")
    leg_colors = {
        "FL": "royalblue",
        "FR": "darkorange",
        "RL": "green",
        "RR": "red",
    }

    # Load per leg over time
    ax1 = axes[0]
    for name, color in leg_colors.items():
        ax1.plot(
            times,
            logger.get_field(f"load_{name}"),
            color=color, linewidth=1.5, label=name
        )
    ax1.set_title("Load Per Leg Over Time\n"
                  "(drops when leg swings)",
                  fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Load (N·m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Stacked load
    ax2 = axes[1]
    fl = logger.get_field("load_FL")
    fr = logger.get_field("load_FR")
    rl = logger.get_field("load_RL")
    rr = logger.get_field("load_RR")

    ax2.stackplot(
        times, fl, fr, rl, rr,
        labels=["FL", "FR", "RL", "RR"],
        colors=["royalblue", "darkorange", "green", "red"],
        alpha=0.7
    )
    ax2.set_title("Stacked Load Distribution\n"
                  "(shows which leg carries most)",
                  fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Load (N·m)")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        save_dir, "t3_phase7_graph3_load_distribution.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [SAVED] t3_phase7_graph3_load_distribution.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# REQUIRED GRAPH 4: STABILITY STATE
# ─────────────────────────────────────────────────────────────
def plot_graph4_stability_state(
    logger_healthy: TelemetryLogger,
    logger_failure: TelemetryLogger,
    save_dir: str
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Graph 4: Stability State Over Time",
        fontsize=14, fontweight='bold'
    )

    state_to_num = {
        "STABLE": 3, "MARGINAL": 2,
        "UNSTABLE": 1, "FAILURE": 0
    }
    state_colors = {
        "STABLE": "green", "MARGINAL": "orange",
        "UNSTABLE": "red", "FAILURE": "black"
    }

    for ax, logger, title in [
        (axes[0], logger_healthy, "Healthy Gait"),
        (axes[1], logger_failure, "With Leg Failure"),
    ]:
        times  = logger.get_field("time")
        states = logger.get_field("system_state")
        nums   = [state_to_num.get(s, 2) for s in states]
        colors = [state_colors.get(s, "gray") for s in states]

        ax.scatter(times, nums, c=colors, s=3, alpha=0.8)

        # Polygon area on twin axis
        ax2 = ax.twinx()
        ax2.plot(
            times,
            logger.get_field("polygon_area"),
            'royalblue', linewidth=1.5, alpha=0.5,
            label='Polygon area'
        )
        ax2.set_ylabel("Polygon Area (m²)",
                       color='royalblue', fontsize=8)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels([
            "FAILURE", "UNSTABLE", "MARGINAL", "STABLE"
        ])
        ax.set_title(f"Stability State — {title}",
                     fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("System State")
        ax.grid(True, alpha=0.3)

        patches = [
            mpatches.Patch(color='green',  label='STABLE'),
            mpatches.Patch(color='orange', label='MARGINAL'),
            mpatches.Patch(color='red',    label='UNSTABLE'),
            mpatches.Patch(color='black',  label='FAILURE'),
        ]
        ax.legend(handles=patches, fontsize=7,
                  loc='lower right')

    plt.tight_layout()
    path = os.path.join(
        save_dir, "t3_phase7_graph4_stability_state.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [SAVED] t3_phase7_graph4_stability_state.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MASTER DASHBOARD
# ─────────────────────────────────────────────────────────────
def plot_master_dashboard(
    logger_healthy: TelemetryLogger,
    logger_failure: TelemetryLogger,
    save_dir: str
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "TASK 3 — MASTER DASHBOARD\n"
        "Full Quadruped Leg Integration + "
        "Control-Ready Simulation",
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.52, wspace=0.38
    )

    logger  = logger_healthy
    times   = logger.get_field("time")
    leg_colors = {
        "FL": "royalblue", "FR": "darkorange",
        "RL": "green",     "RR": "red",
    }
    state_to_num = {
        "STABLE": 3, "MARGINAL": 2,
        "UNSTABLE": 1, "FAILURE": 0
    }

    # ── Cell 1: Hip angles ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name, color in leg_colors.items():
        ax1.plot(times, logger.get_field(f"hip_{name}"),
                 color=color, linewidth=1.5, label=name)
    ax1.set_title("Hip Angles", fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (deg)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linewidth=0.5)

    # ── Cell 2: Knee angles ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name, color in leg_colors.items():
        ax2.plot(times, logger.get_field(f"knee_{name}"),
                 color=color, linewidth=1.5,
                 linestyle='--', label=name)
    ax2.set_title("Knee Angles", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Cell 3: Foot trajectory ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for name, color in leg_colors.items():
        fx = logger.get_field(f"foot_{name}_x")
        fy = logger.get_field(f"foot_{name}_y")
        ax3.plot(fx, fy, color=color,
                 linewidth=1.5, label=name, alpha=0.8)
    ax3.scatter(0, 0, color='black', s=80,
                marker='x', zorder=5)
    ax3.set_title("Foot Trajectories", fontweight='bold')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_aspect('equal')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── Cell 4: Load per leg ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for name, color in leg_colors.items():
        ax4.plot(times, logger.get_field(f"load_{name}"),
                 color=color, linewidth=1.5, label=name)
    ax4.set_title("Load Per Leg", fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Load (N·m)")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ── Cell 5: Stacked load ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.stackplot(
        times,
        logger.get_field("load_FL"),
        logger.get_field("load_FR"),
        logger.get_field("load_RL"),
        logger.get_field("load_RR"),
        labels=["FL", "FR", "RL", "RR"],
        colors=["royalblue", "darkorange", "green", "red"],
        alpha=0.7
    )
    ax5.set_title("Load Distribution Stack",
                  fontweight='bold')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Load (N·m)")
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, alpha=0.3)

    # ── Cell 6: Stability state healthy ──────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    states  = logger.get_field("system_state")
    s_nums  = [state_to_num.get(s, 2) for s in states]
    s_colors= [
        {"STABLE": "green", "MARGINAL": "orange",
         "UNSTABLE": "red", "FAILURE": "black"}.get(s, "gray")
        for s in states
    ]
    ax6.scatter(times, s_nums, c=s_colors, s=3, alpha=0.8)
    ax6.set_yticks([0, 1, 2, 3])
    ax6.set_yticklabels([
        "FAILURE", "UNSTABLE", "MARGINAL", "STABLE"
    ])
    ax6.set_title("Stability (Healthy)", fontweight='bold')
    ax6.set_xlabel("Time (s)")
    ax6.grid(True, alpha=0.3)

    # ── Cell 7: Failure — hip angles ─────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    times_f = logger_failure.get_field("time")
    for name, color in leg_colors.items():
        ax7.plot(
            times_f,
            logger_failure.get_field(f"hip_{name}"),
            color=color, linewidth=1.5, label=name
        )
    ax7.set_title("Hip Angles (With Failure)",
                  fontweight='bold')
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Angle (deg)")
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(0, color='gray', linewidth=0.5)

    # ── Cell 8: Failure — foot trajectory ────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    for name, color in leg_colors.items():
        fx = logger_failure.get_field(f"foot_{name}_x")
        fy = logger_failure.get_field(f"foot_{name}_y")
        ax8.plot(fx, fy, color=color,
                 linewidth=1.5, label=name, alpha=0.8)
    ax8.scatter(0, 0, color='black', s=80,
                marker='x', zorder=5)
    ax8.set_title("Foot Trajectories (With Failure)",
                  fontweight='bold')
    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Y (m)")
    ax8.set_aspect('equal')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # ── Cell 9: Failure stability ─────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    states_f = logger_failure.get_field("system_state")
    s_nums_f = [state_to_num.get(s, 2) for s in states_f]
    s_cols_f = [
        {"STABLE": "green", "MARGINAL": "orange",
         "UNSTABLE": "red", "FAILURE": "black"}.get(s, "gray")
        for s in states_f
    ]
    ax9.scatter(times_f, s_nums_f,
                c=s_cols_f, s=3, alpha=0.8)
    ax9.set_yticks([0, 1, 2, 3])
    ax9.set_yticklabels([
        "FAILURE", "UNSTABLE", "MARGINAL", "STABLE"
    ])
    ax9.set_title("Stability (With Failure)",
                  fontweight='bold')
    ax9.set_xlabel("Time (s)")
    ax9.grid(True, alpha=0.3)

    path = os.path.join(
        save_dir, "t3_phase7_master_dashboard.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [SAVED] t3_phase7_master_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase7():
    print("\n" + "=" * 65)
    print("  TASK 3 - PHASE 7: DATA + LOGGING LAYER")
    print("  Structured telemetry for Dhruv + Final Graphs")
    print("=" * 65)

    params   = get_default_params()
    save_dir = BASE_DIR

    # ── Healthy simulation ───────────────────────────────────
    print("\n  [1/4] Running healthy gait simulation...")
    quad1  = build_quadruped()
    iface1 = QuadrupedControlInterface(quad1, params)
    iface1.reset()
    cmd    = GaitCommand(gait_type="crawl", velocity=0.5)

    logger_healthy = TelemetryLogger()
    total_steps = int(
        params.cycle_duration() * 2 / params.dt
    )
    for _ in range(total_steps):
        packet = iface1.step(cmd)
        logger_healthy.log(packet)

    print(f"  Healthy simulation: "
          f"{len(logger_healthy.records)} records")
    logger_healthy.summary()

    # ── Failure simulation ───────────────────────────────────
    print("\n  [2/4] Running failure simulation...")
    quad2  = build_quadruped()
    iface2 = QuadrupedControlInterface(quad2, params)
    iface2.reset()

    logger_failure = TelemetryLogger()
    failure_injected = False

    for i in range(total_steps):
        packet = iface2.step(cmd)

        # Inject RL leg failure at t=7s
        if (not failure_injected and
                packet.timestamp >= 7.0):
            quad2.legs["RL"].hip.actuator.is_stalled  = True
            quad2.legs["RL"].knee.actuator.is_stalled = True
            failure_injected = True
            print(f"  *** RL FAILURE injected at "
                  f"t={packet.timestamp:.2f}s ***")

        logger_failure.log(packet)

    print(f"  Failure simulation: "
          f"{len(logger_failure.records)} records")
    logger_failure.summary()

    # ── Save logs ────────────────────────────────────────────
    print("\n  [3/4] Saving telemetry logs...")
    logger_healthy.save_txt("t3_phase7_log_healthy.txt")
    logger_healthy.save_csv("t3_phase7_telemetry_healthy.csv")
    logger_failure.save_txt("t3_phase7_log_failure.txt")
    logger_failure.save_csv("t3_phase7_telemetry_failure.csv")

    # ── Plot required graphs ─────────────────────────────────
    print("\n  [4/4] Generating required graphs...")
    try:
        import matplotlib.pyplot as plt

        plot_graph1_leg_angles(logger_healthy, save_dir)
        plot_graph2_foot_trajectory(logger_healthy, save_dir)
        plot_graph3_load_distribution(logger_healthy, save_dir)
        plot_graph4_stability_state(
            logger_healthy, logger_failure, save_dir
        )
        plot_master_dashboard(
            logger_healthy, logger_failure, save_dir
        )

    except ImportError:
        print("  [INFO] matplotlib not available.")

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 7 COMPLETE — ALL FILES SAVED")
    print("=" * 65)
    print("  Logs (for Dhruv):")
    print("    t3_phase7_log_healthy.txt")
    print("    t3_phase7_log_failure.txt")
    print("    t3_phase7_telemetry_healthy.csv")
    print("    t3_phase7_telemetry_failure.csv")
    print("  Graphs (required by task PDF):")
    print("    t3_phase7_graph1_leg_angles.png")
    print("    t3_phase7_graph2_foot_trajectory.png")
    print("    t3_phase7_graph3_load_distribution.png")
    print("    t3_phase7_graph4_stability_state.png")
    print("  Master Dashboard:")
    print("    t3_phase7_master_dashboard.png")
    print("\n  [TASK 3 SIMULATION COMPLETE]")
    print("  Ready for REVIEW_PACKET.md + Demo Video\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase7()
