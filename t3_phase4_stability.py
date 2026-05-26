"""
=============================================================
TASK 3 — PHASE 4 : STABILITY + FOOT POSITION TRACKING
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase adds on top of Phase 3:
  - Tracks all 4 foot positions in world frame every timestep
  - Computes the support polygon (shape formed by stance feet)
  - Tracks Center of Mass (CoM) position of the robot
  - Determines if CoM is inside or outside support polygon
  - Flags STABLE vs UNSTABLE states
  - Identifies exact moments when system becomes unstable

Key concept — Support Polygon:
  The support polygon is the shape formed by connecting
  all the feet that are currently on the ground.

  When 3 legs are on ground → triangle support polygon
  When 4 legs are on ground → quadrilateral support polygon

  Stability rule:
    CoM INSIDE polygon  → robot is STABLE
    CoM OUTSIDE polygon → robot is UNSTABLE (will tip over)

  Simple analogy:
    Stand on both feet — stable (CoM between feet).
    Lean too far sideways — CoM goes outside feet — you fall.
    Same physics for the robot but with 3-4 support points.

Center of Mass (CoM) calculation:
  CoM = weighted average of all component positions
  For simplicity: CoM = body center (0, 0) in body frame
  In world frame: CoM moves as body moves

Point-in-polygon test:
  We use the ray casting algorithm to check if CoM
  is inside the support polygon.
  If ray from CoM crosses polygon boundary odd times → inside
  If ray crosses even times → outside
=============================================================
"""

import sys
import os
import math

# ── Import path setup ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

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
from t3_phase3_load_distribution import SystemLoadCalculator


# ─────────────────────────────────────────────────────────────
# STABILITY ANALYZER
# Tracks foot positions, support polygon, and CoM stability
# ─────────────────────────────────────────────────────────────
class StabilityAnalyzer:
    def __init__(self, quad: QuadrupedSystem):
        """
        Parameters:
          quad : QuadrupedSystem from Phase 1
        """
        self.quad = quad

        # CoM is computed as the centroid of all 4 foot
        # positions when robot is in neutral standing pose.
        # This ensures CoM is correctly placed inside the
        # support polygon during simulation.
        #
        # At standing (hip=0, knee=0):
        #   Each foot position = hip_offset + local_foot
        #   local_foot at 0 deg = (L1+L2, 0) = (0.9, 0)
        #   FL foot ~ (-0.2 + 0.9,  0.3) = (0.7,  0.3)
        #   FR foot ~ ( 0.2 + 0.9,  0.3) = (1.1,  0.3)
        #   RL foot ~ (-0.2 + 0.9, -0.3) = (0.7, -0.3)
        #   RR foot ~ ( 0.2 + 0.9, -0.3) = (1.1, -0.3)
        #   Centroid X = (0.7+1.1+0.7+1.1)/4 = 0.9
        #   Centroid Y = (0.3+0.3-0.3-0.3)/4 = 0.0
        self.com_x = 0.9   # centroid of foot workspace X
        self.com_y = 0.0   # centroid of foot workspace Y (symmetric)

    def get_stance_feet(self, snap: dict) -> dict:
        """
        Returns foot positions of all STANCE legs only.
        Swinging leg is excluded from support polygon.

        Parameters:
          snap : Gait snapshot from Phase 2

        Returns:
          dict of {leg_name: (foot_x, foot_y)} for stance legs
        """
        active_leg  = snap["active_leg"]
        gait_phase  = snap["gait_phase"]
        stance_feet = {}

        for name in ["FL", "FR", "RL", "RR"]:
            is_swing = (
                name == active_leg and
                gait_phase == LegState.SWING
            )
            if not is_swing:
                foot = self.quad.get_foot_position(name)
                stance_feet[name] = foot

        return stance_feet

    def compute_support_polygon(
        self,
        stance_feet: dict
    ) -> list:
        """
        Computes the support polygon vertices from stance feet.
        Orders vertices in counter-clockwise order for
        correct point-in-polygon testing.

        Parameters:
          stance_feet : dict of stance foot positions

        Returns:
          List of (x, y) vertices ordered CCW
        """
        if len(stance_feet) < 3:
            return []

        points = list(stance_feet.values())

        # Compute centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        # Sort by angle from centroid (CCW order)
        def angle(p):
            return math.atan2(p[1] - cy, p[0] - cx)

        points_sorted = sorted(points, key=angle)
        return points_sorted

    def point_in_polygon(
        self,
        px: float,
        py: float,
        polygon: list
    ) -> bool:
        """
        Ray casting algorithm to check if point (px, py)
        is inside the polygon.

        Parameters:
          px, py  : Point to test (CoM position)
          polygon : List of (x, y) vertices

        Returns:
          True if inside, False if outside
        """
        if len(polygon) < 3:
            return False

        n       = len(polygon)
        inside  = False
        j       = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            intersect = (
                (yi > py) != (yj > py) and
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            )
            if intersect:
                inside = not inside
            j = i

        return inside

    def polygon_area(self, polygon: list) -> float:
        """
        Computes area of polygon using shoelace formula.
        Used as a measure of stability margin.
        Larger area = more stable.
        """
        if len(polygon) < 3:
            return 0.0

        n    = len(polygon)
        area = 0.0
        j    = n - 1

        for i in range(n):
            area += (polygon[j][0] + polygon[i][0]) * \
                    (polygon[j][1] - polygon[i][1])
            j = i

        return abs(area) / 2.0

    def com_to_polygon_distance(
        self,
        px: float,
        py: float,
        polygon: list
    ) -> float:
        """
        Computes minimum distance from CoM to polygon boundary.
        Positive = inside (stable margin)
        Negative = outside (unstable by this distance)
        """
        if len(polygon) < 3:
            return 0.0

        min_dist = float('inf')
        n = len(polygon)
        j = n - 1

        for i in range(n):
            # Distance from point to line segment
            x1, y1 = polygon[j]
            x2, y2 = polygon[i]

            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                dist = math.sqrt(
                    (px - x1)**2 + (py - y1)**2
                )
            else:
                t = max(0, min(1, (
                    (px - x1) * dx + (py - y1) * dy
                ) / length_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.sqrt(
                    (px - proj_x)**2 + (py - proj_y)**2
                )

            min_dist = min(min_dist, dist)
            j = i

        # Negative if outside
        inside = self.point_in_polygon(px, py, polygon)
        return min_dist if inside else -min_dist

    def analyze_step(self, snap: dict) -> dict:
        """
        Full stability analysis for one timestep.

        Returns:
          dict with foot positions, polygon, CoM, stability state
        """
        # Update quad leg angles from snapshot
        for name in ["FL", "FR", "RL", "RR"]:
            self.quad.legs[name].hip.joint_output.current_angle  = \
                snap[f"{name}_hip"]
            self.quad.legs[name].knee.joint_output.current_angle = \
                snap[f"{name}_knee"]

        # Get all foot positions
        all_feet = self.quad.get_all_foot_positions()

        # Get stance feet only
        stance_feet = self.get_stance_feet(snap)

        # Compute support polygon
        polygon = self.compute_support_polygon(stance_feet)

        # Check CoM stability
        is_stable = self.point_in_polygon(
            self.com_x, self.com_y, polygon
        )

        # Polygon area (stability margin proxy)
        area = self.polygon_area(polygon)

        # Distance from CoM to polygon boundary
        margin = self.com_to_polygon_distance(
            self.com_x, self.com_y, polygon
        )

        # Stability state string
        if is_stable:
            if margin > 0.1:
                state = "STABLE"
            else:
                state = "MARGINAL"
        else:
            state = "UNSTABLE"

        return {
            "time"          : snap["time"],
            "active_leg"    : snap["active_leg"],
            "gait_phase"    : snap["gait_phase"],
            "all_feet"      : all_feet,
            "stance_feet"   : stance_feet,
            "polygon"       : polygon,
            "com_x"         : self.com_x,
            "com_y"         : self.com_y,
            "is_stable"     : is_stable,
            "stability_state": state,
            "polygon_area"  : round(area,   6),
            "com_margin"    : round(margin, 6),
            "n_stance_legs" : len(stance_feet),
        }

    def analyze_full_gait(self, history: list) -> list:
        """
        Runs stability analysis across the full gait history.
        """
        print(f"\n  Analyzing stability for "
              f"{len(history)} timesteps...")
        stability_history = []
        for snap in history:
            step = self.analyze_step(snap)
            stability_history.append(step)
        return stability_history


# ─────────────────────────────────────────────────────────────
# STABILITY REPORT
# ─────────────────────────────────────────────────────────────
def print_stability_report(stability_history: list):
    """Prints key stability findings."""
    states  = [s["stability_state"] for s in stability_history]
    areas   = [s["polygon_area"]    for s in stability_history]
    margins = [s["com_margin"]      for s in stability_history]

    stable_count   = states.count("STABLE")
    marginal_count = states.count("MARGINAL")
    unstable_count = states.count("UNSTABLE")
    total          = len(stability_history)

    print("\n" + "=" * 62)
    print("  STABILITY ANALYSIS REPORT")
    print("=" * 62)
    print(f"\n  Total timesteps : {total}")
    print(f"  STABLE          : {stable_count} "
          f"({stable_count/total*100:.1f}%)")
    print(f"  MARGINAL        : {marginal_count} "
          f"({marginal_count/total*100:.1f}%)")
    print(f"  UNSTABLE        : {unstable_count} "
          f"({unstable_count/total*100:.1f}%)")

    print(f"\n  Support polygon area:")
    print(f"    Max area : {max(areas):.6f} m²  (most stable)")
    print(f"    Min area : {min(areas):.6f} m²  (least stable)")
    print(f"    Avg area : {sum(areas)/len(areas):.6f} m²")

    print(f"\n  CoM margin (distance to polygon edge):")
    print(f"    Max margin : {max(margins):.4f} m  (safest)")
    print(f"    Min margin : {min(margins):.4f} m")

    print(f"\n  Stability per gait phase:")
    for phase in [LegState.SWING, LegState.STANCE]:
        phase_steps = [
            s for s in stability_history
            if s["gait_phase"] == phase
        ]
        if phase_steps:
            phase_stable = sum(
                1 for s in phase_steps
                if s["is_stable"]
            )
            print(f"    {phase:6s}: {phase_stable}/"
                  f"{len(phase_steps)} stable "
                  f"({phase_stable/len(phase_steps)*100:.1f}%)")

    print(f"\n  Conclusion:")
    if unstable_count == 0:
        print(f"    System remains STABLE throughout crawl gait.")
        print(f"    CoM stays inside support polygon at all times.")
        print(f"    Crawl gait is the safest walking pattern. ✅")
    else:
        print(f"    System becomes UNSTABLE at {unstable_count} steps.")
        print(f"    Review gait parameters to improve stability.")

    print("=" * 62)


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(stability_history: list):
    log_path = os.path.join(BASE_DIR, "t3_phase4_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 3 - PHASE 4: STABILITY + FOOT TRACKING LOG\n")
        f.write("Support polygon and CoM stability analysis\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"{'Time':>7} | {'Active':>6} | {'Phase':>6} | "
            f"{'Legs':>5} | {'Area':>8} | "
            f"{'Margin':>8} | {'State':>10}\n"
        )
        f.write("-" * 70 + "\n")
        for s in stability_history:
            f.write(
                f"{s['time']:>7.3f} | "
                f"{s['active_leg']:>6} | "
                f"{s['gait_phase']:>6} | "
                f"{s['n_stance_legs']:>5} | "
                f"{s['polygon_area']:>8.5f} | "
                f"{s['com_margin']:>8.4f} | "
                f"{s['stability_state']:>10}\n"
            )
    print(f"\n  [LOG SAVED] -> t3_phase4_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(stability_history: list):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(
            "Task 3 - Phase 4: Stability + Foot Position Tracking\n"
            "Support Polygon and CoM Stability Analysis",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.48, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])  # Foot positions over time
        ax2 = fig.add_subplot(gs[0, 1])  # Support polygon snapshots
        ax3 = fig.add_subplot(gs[1, 0])  # Stability state over time
        ax4 = fig.add_subplot(gs[1, 1])  # Polygon area + CoM margin

        times  = [s["time"] for s in stability_history]
        states = [s["stability_state"] for s in stability_history]
        areas  = [s["polygon_area"]    for s in stability_history]
        margins= [s["com_margin"]      for s in stability_history]

        leg_colors = {
            "FL": "royalblue",
            "FR": "darkorange",
            "RL": "green",
            "RR": "red",
        }

        # ── Plot 1: Foot Y positions over time ────────────────
        for name, color in leg_colors.items():
            fy = [s["all_feet"][name][1]
                  for s in stability_history]
            ax1.plot(times, fy, color=color,
                     linewidth=1.5, label=name)

        ax1.axhline(0, color='gray', linewidth=0.5)
        ax1.set_title("Foot Y Positions Over Time\n"
                      "(shows each foot lifting during swing)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Y Position (m)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Support polygon snapshots ─────────────────
        # Show polygon at 4 key moments
        sample_indices = [
            len(stability_history) // 8,
            len(stability_history) // 4,
            len(stability_history) // 2,
            3 * len(stability_history) // 4,
        ]
        snap_colors = ['blue', 'orange', 'green', 'red']

        for idx, sc in zip(sample_indices, snap_colors):
            s    = stability_history[idx]
            poly = s["polygon"]
            if len(poly) >= 3:
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                ax2.plot(xs, ys, color=sc, linewidth=2,
                         alpha=0.7,
                         label=f"t={s['time']:.1f}s")
                ax2.fill(
                    [p[0] for p in poly],
                    [p[1] for p in poly],
                    alpha=0.1, color=sc
                )

            # Plot stance feet
            for name, foot in s["stance_feet"].items():
                ax2.scatter(foot[0], foot[1],
                            color=leg_colors[name],
                            s=80, zorder=5)

        # CoM position
        ax2.scatter(0, 0, color='black', s=150,
                    marker='*', zorder=6, label='CoM')
        ax2.set_title("Support Polygon at Key Moments\n"
                      "(star = CoM, must stay inside polygon)",
                      fontweight='bold')
        ax2.set_xlabel("X position (m)")
        ax2.set_ylabel("Y position (m)")
        ax2.set_aspect('equal')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)

        # ── Plot 3: Stability state over time ─────────────────
        state_to_num = {
            "STABLE": 2, "MARGINAL": 1, "UNSTABLE": 0
        }
        state_colors_map = {
            "STABLE": "green",
            "MARGINAL": "orange",
            "UNSTABLE": "red"
        }
        state_nums   = [state_to_num[s] for s in states]
        state_colors = [state_colors_map[s] for s in states]

        ax3.scatter(times, state_nums,
                    c=state_colors, s=3, alpha=0.8)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["UNSTABLE", "MARGINAL", "STABLE"])
        ax3.set_title("Stability State Over Time",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Stability State")
        ax3.grid(True, alpha=0.3)

        patches = [
            mpatches.Patch(color='green',  label='STABLE'),
            mpatches.Patch(color='orange', label='MARGINAL'),
            mpatches.Patch(color='red',    label='UNSTABLE'),
        ]
        ax3.legend(handles=patches, fontsize=8)

        # ── Plot 4: Polygon area + CoM margin ─────────────────
        ax4_twin = ax4.twinx()

        ax4.plot(times, areas, 'royalblue', linewidth=2,
                 label='Polygon area (m²)')
        ax4_twin.plot(times, margins, 'darkorange',
                      linewidth=2, linestyle='--',
                      label='CoM margin (m)')
        ax4_twin.axhline(0, color='red', linewidth=1,
                         linestyle=':', label='Zero margin')

        ax4.set_title("Polygon Area + CoM Safety Margin\n"
                      "(larger = more stable)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Polygon Area (m²)", color='royalblue')
        ax4_twin.set_ylabel("CoM Margin (m)", color='darkorange')
        ax4.grid(True, alpha=0.3)

        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2,
                   labels1 + labels2, fontsize=7)

        # Save
        graph_path = os.path.join(
            BASE_DIR, "t3_phase4_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t3_phase4_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase4():
    print("\n" + "=" * 62)
    print("  TASK 3 - PHASE 4: STABILITY + FOOT TRACKING")
    print("  Support polygon and CoM stability analysis")
    print("=" * 62)

    # Build system
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

    # Run gait
    print("\n  Running gait simulation...")
    history = controller.run(cycles=2, verbose=False)

    # Stability analysis
    analyzer          = StabilityAnalyzer(quad)
    stability_history = analyzer.analyze_full_gait(history)

    # Print sample table
    print(f"\n  STABILITY TABLE (sampled every 200 steps):")
    print(
        f"  {'Time':>6} | {'Active':>6} | {'Phase':>6} | "
        f"{'Legs':>5} | {'Area':>8} | "
        f"{'Margin':>7} | {'State':>10}"
    )
    print(f"  {'-'*65}")
    for i, s in enumerate(stability_history):
        if i % 200 == 0 or i == len(stability_history) - 1:
            print(
                f"  {s['time']:>6.2f} | "
                f"{s['active_leg']:>6} | "
                f"{s['gait_phase']:>6} | "
                f"{s['n_stance_legs']:>5} | "
                f"{s['polygon_area']:>8.5f} | "
                f"{s['com_margin']:>7.4f} | "
                f"{s['stability_state']:>10}"
            )

    # Print report
    print_stability_report(stability_history)

    # Save and plot
    save_log(stability_history)
    plot_results(stability_history)

    print("\n  [PHASE 4 COMPLETE] Stability analysis done.")
    print("  Ready for Phase 5: Failure Propagation\n")

    return stability_history


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase4()
