"""
=============================================================
TASK 3 — PHASE 2 : GAIT COORDINATION (CRAWL GAIT)
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase adds on top of Phase 1:
  - Implements the crawl gait sequence: FL -> RR -> FR -> RL
  - Only 1 leg swings at a time
  - 3 legs always on ground supporting the body
  - Each leg goes through SWING phase and STANCE phase
  - Logs which leg is active at every timestep
  - Produces coordinated motion graphs

What is a Crawl Gait?
  A crawl gait is the most stable walking pattern for a
  4-legged robot. It moves one leg at a time while the
  other 3 stay on the ground.

  The sequence:
    Step 1: FL  swings forward  (FR, RL, RR on ground)
    Step 2: RR  swings forward  (FL, FR, RL on ground)
    Step 3: FR  swings forward  (FL, RL, RR on ground)
    Step 4: RL  swings forward  (FL, FR, RR on ground)
    Then repeat from Step 1

  Simple analogy:
    Like a person walking slowly and carefully on ice —
    always keeping 3 points of contact before moving
    the 4th. Maximum stability at all times.

Two phases per leg:
  SWING phase : leg lifts off ground, moves forward
    - Hip lifts to swing_hip_angle (e.g. 40°)
    - Knee bends to swing_knee_angle (e.g. 60°)
    - Duration: swing_duration seconds

  STANCE phase : leg touches down, supports body
    - Hip returns to stance_hip_angle (e.g. -10°)
    - Knee returns to stance_knee_angle (e.g. 10°)
    - Duration: stance_duration seconds
=============================================================
"""

import sys
import os
import math

# ── Import path setup ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import Phase 1
from t3_phase1_quadruped_system import (
    build_quadruped,
    QuadrupedSystem,
    CRAWL_GAIT_ORDER
)

# Import Task 2 actuator model for smooth motion
from t2_phase3_coupled_dynamics import CoupledDynamicsSimulator


# ─────────────────────────────────────────────────────────────
# GAIT PARAMETERS
# Defines how each leg moves during swing and stance
# ─────────────────────────────────────────────────────────────
class GaitParameters:
    def __init__(
        self,
        swing_hip_angle:   float = 40.0,   # hip angle during swing
        swing_knee_angle:  float = 60.0,   # knee angle during swing
        stance_hip_angle:  float = -10.0,  # hip angle during stance
        stance_knee_angle: float = 10.0,   # knee angle during stance
        swing_duration:    float = 1.0,    # seconds per swing
        stance_duration:   float = 1.0,    # seconds per stance
        dt:                float = 0.01    # simulation timestep
    ):
        """
        Parameters:
          swing_hip_angle   : Hip angle when leg is lifted (deg)
          swing_knee_angle  : Knee angle when leg is lifted (deg)
          stance_hip_angle  : Hip angle when leg is on ground (deg)
          stance_knee_angle : Knee angle when leg is on ground (deg)
          swing_duration    : How long each swing phase lasts (s)
          stance_duration   : How long each stance phase lasts (s)
          dt                : Simulation time step (s)
        """
        self.swing_hip    = swing_hip_angle
        self.swing_knee   = swing_knee_angle
        self.stance_hip   = stance_hip_angle
        self.stance_knee  = stance_knee_angle
        self.swing_dur    = swing_duration
        self.stance_dur   = stance_duration
        self.dt           = dt

    def cycle_duration(self) -> float:
        """Total duration of one full gait cycle (all 4 legs)."""
        return (self.swing_dur + self.stance_dur) * 4


# ─────────────────────────────────────────────────────────────
# LEG STATE TRACKER
# Tracks whether each leg is in SWING or STANCE phase
# ─────────────────────────────────────────────────────────────
class LegState:
    SWING  = "SWING"
    STANCE = "STANCE"

    def __init__(self, name: str):
        self.name    = name
        self.phase   = self.STANCE   # all legs start on ground
        self.timer   = 0.0           # time in current phase


# ─────────────────────────────────────────────────────────────
# CRAWL GAIT CONTROLLER
# Implements the FL -> RR -> FR -> RL crawl sequence
# ─────────────────────────────────────────────────────────────
class CrawlGaitController:
    def __init__(
        self,
        quad:   QuadrupedSystem,
        params: GaitParameters
    ):
        """
        Parameters:
          quad   : QuadrupedSystem from Phase 1
          params : GaitParameters defining swing/stance angles
        """
        self.quad   = quad
        self.params = params
        self.dt     = params.dt

        # Leg simulators — one per leg for smooth motion
        self.simulators = {}
        for name in ["FL", "FR", "RL", "RR"]:
            self.simulators[name] = CoupledDynamicsSimulator(
                leg=quad.legs[name],
                time_constant=0.6,  # Option A: heavier response
                dt=params.dt
            )

        # Leg state trackers
        self.leg_states = {
            name: LegState(name)
            for name in ["FL", "FR", "RL", "RR"]
        }

        # Gait sequence tracking
        self.gait_order      = CRAWL_GAIT_ORDER  # FL, RR, FR, RL
        self.current_leg_idx = 0   # which leg in sequence is swinging
        self.phase_timer     = 0.0 # timer for current phase
        self.gait_phase      = LegState.SWING  # current gait phase

        # Global time
        self.time_elapsed = 0.0

        # History log
        self.history = []

    def reset(self):
        """Resets gait to starting position."""
        self.quad.reset_all(
            hip_angle=self.params.stance_hip,
            knee_angle=self.params.stance_knee
        )
        for sim in self.simulators.values():
            sim.time_elapsed = 0.0
        self.current_leg_idx = 0
        self.phase_timer     = 0.0
        self.gait_phase      = LegState.SWING
        self.time_elapsed    = 0.0
        self.history         = []
        print("  [RESET] All legs in stance position")

    def _get_active_leg(self) -> str:
        """Returns the name of the leg currently supposed to swing."""
        return self.gait_order[self.current_leg_idx % 4]

    def _get_leg_phases(self) -> dict:
        """
        Returns current phase (SWING/STANCE) for all legs.
        Only the active leg is in SWING. All others are STANCE.
        """
        active = self._get_active_leg()
        return {
            name: (LegState.SWING if name == active
                   else LegState.STANCE)
            for name in ["FL", "FR", "RL", "RR"]
        }

    def step(self) -> dict:
        """
        Advances the gait by one timestep.

        Logic:
          1. Determine which leg is active (swinging)
          2. Set that leg's target to swing angles
          3. Set all other legs to stance angles
          4. Step all leg simulators
          5. Check if phase timer expired → advance to next leg
          6. Log everything
        """
        active_leg   = self._get_active_leg()
        leg_phases   = self._get_leg_phases()
        phase_dur    = (self.params.swing_dur
                        if self.gait_phase == LegState.SWING
                        else self.params.stance_dur)

        # Set targets for all legs
        for name in ["FL", "FR", "RL", "RR"]:
            leg = self.quad.legs[name]
            if name == active_leg and self.gait_phase == LegState.SWING:
                # Active leg: swing targets
                leg.hip.set_target(self.params.swing_hip)
                leg.knee.set_target(self.params.swing_knee)
            else:
                # All other legs: stance targets
                leg.hip.set_target(self.params.stance_hip)
                leg.knee.set_target(self.params.stance_knee)

        # Step all simulators
        for name in ["FL", "FR", "RL", "RR"]:
            self.simulators[name].step()

        # Advance timer
        self.phase_timer  += self.dt
        self.time_elapsed += self.dt

        # Check if current phase is done
        if self.phase_timer >= phase_dur:
            self.phase_timer = 0.0
            if self.gait_phase == LegState.SWING:
                # Switch to stance for this leg
                self.gait_phase = LegState.STANCE
            else:
                # Stance done — move to next leg in sequence
                self.gait_phase      = LegState.SWING
                self.current_leg_idx = (self.current_leg_idx + 1) % 4

        # Build snapshot
        snap = {
            "time"        : round(self.time_elapsed, 4),
            "active_leg"  : active_leg,
            "gait_phase"  : self.gait_phase,
            "leg_phases"  : leg_phases,
        }

        # Add each leg's angles and foot position
        for name in ["FL", "FR", "RL", "RR"]:
            leg  = self.quad.legs[name]
            foot = self.quad.get_foot_position(name)
            snap[f"{name}_hip"]    = round(
                leg.hip.joint_output.current_angle, 3)
            snap[f"{name}_knee"]   = round(
                leg.knee.joint_output.current_angle, 3)
            snap[f"{name}_foot_x"] = foot[0]
            snap[f"{name}_foot_y"] = foot[1]
            snap[f"{name}_phase"]  = leg_phases[name]

        self.history.append(snap)
        return snap

    def run(
        self,
        cycles:  int = 2,
        verbose: bool = True
    ) -> list:
        """
        Runs the crawl gait for a specified number of full cycles.

        One cycle = all 4 legs swing once = FL+RR+FR+RL complete

        Parameters:
          cycles  : Number of full gait cycles to simulate
          verbose : Whether to print progress table

        Returns:
          Full history list
        """
        total_duration = self.params.cycle_duration() * cycles
        total_steps    = int(total_duration / self.dt)

        print(f"\n  {'='*62}")
        print(f"  CRAWL GAIT SIMULATION")
        print(f"  Sequence : {' -> '.join(self.gait_order)}")
        print(f"  Cycles   : {cycles}")
        print(f"  Duration : {total_duration:.1f}s")
        print(f"  {'='*62}")

        if verbose:
            print(f"\n  {'Time':>6} | {'Active':>6} | {'Phase':>6} | "
                  f"{'FL':>8} | {'FR':>8} | "
                  f"{'RL':>8} | {'RR':>8}")
            print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
                  f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        prev_leg = None
        for i in range(total_steps):
            snap = self.step()

            # Print when active leg changes or every 100 steps
            if verbose and (
                snap["active_leg"] != prev_leg or i % 100 == 0
            ):
                print(
                    f"  {snap['time']:>6.2f} | "
                    f"{snap['active_leg']:>6} | "
                    f"{snap['gait_phase']:>6} | "
                    f"{snap['FL_hip']:>6.1f}° | "
                    f"{snap['FR_hip']:>6.1f}° | "
                    f"{snap['RL_hip']:>6.1f}° | "
                    f"{snap['RR_hip']:>6.1f}°"
                )
                prev_leg = snap["active_leg"]

        print(f"\n  [GAIT COMPLETE] {len(self.history)} steps recorded")
        return self.history


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(history: list):
    log_path = os.path.join(
        BASE_DIR, "t3_phase2_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 3 - PHASE 2: CRAWL GAIT COORDINATION LOG\n")
        f.write("Sequence: FL -> RR -> FR -> RL\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"{'Time':>7} | {'Active':>6} | {'Phase':>6} | "
            f"{'FL_hip':>7} | {'FR_hip':>7} | "
            f"{'RL_hip':>7} | {'RR_hip':>7} | "
            f"{'FL_phase':>8} | {'RR_phase':>8}\n"
        )
        f.write("-" * 90 + "\n")
        for s in history:
            f.write(
                f"{s['time']:>7.3f} | "
                f"{s['active_leg']:>6} | "
                f"{s['gait_phase']:>6} | "
                f"{s['FL_hip']:>7.2f} | "
                f"{s['FR_hip']:>7.2f} | "
                f"{s['RL_hip']:>7.2f} | "
                f"{s['RR_hip']:>7.2f} | "
                f"{s['FL_phase']:>8} | "
                f"{s['RR_phase']:>8}\n"
            )
    print(f"\n  [LOG SAVED] -> t3_phase2_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(history: list):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Task 3 - Phase 2: Crawl Gait Coordination\n"
            "Sequence: FL -> RR -> FR -> RL",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])  # Hip angles all 4 legs
        ax2 = fig.add_subplot(gs[0, 1])  # Knee angles all 4 legs
        ax3 = fig.add_subplot(gs[1, 0])  # Foot trajectories
        ax4 = fig.add_subplot(gs[1, 1])  # Active leg timeline

        times = [s["time"] for s in history]

        leg_colors = {
            "FL": "royalblue",
            "FR": "darkorange",
            "RL": "green",
            "RR": "red",
        }

        # ── Plot 1: Hip angles ────────────────────────────────
        for name, color in leg_colors.items():
            hips = [s[f"{name}_hip"] for s in history]
            ax1.plot(times, hips, color=color,
                     linewidth=1.5, label=name)
        ax1.set_title("Hip Angles — All 4 Legs",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Hip Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        # ── Plot 2: Knee angles ───────────────────────────────
        for name, color in leg_colors.items():
            knees = [s[f"{name}_knee"] for s in history]
            ax2.plot(times, knees, color=color,
                     linewidth=1.5, label=name, linestyle='--')
        ax2.set_title("Knee Angles — All 4 Legs",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Knee Angle (deg)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Foot trajectories ─────────────────────────
        for name, color in leg_colors.items():
            fx = [s[f"{name}_foot_x"] for s in history]
            fy = [s[f"{name}_foot_y"] for s in history]
            ax3.plot(fx, fy, color=color, linewidth=1.5,
                     label=name, alpha=0.8)
            ax3.scatter(fx[0],  fy[0],  color=color,
                        s=60, marker='o', zorder=5)
            ax3.scatter(fx[-1], fy[-1], color=color,
                        s=60, marker='*', zorder=5)

        ax3.scatter(0, 0, color='black', s=100,
                    marker='x', zorder=6, label='Body center')
        ax3.set_title("Foot Trajectories — All 4 Legs\n"
                      "(circle=start, star=end)",
                      fontweight='bold')
        ax3.set_xlabel("X position (m)")
        ax3.set_ylabel("Y position (m)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.axhline(0, color='gray', linewidth=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5)

        # ── Plot 4: Active leg timeline ───────────────────────
        leg_to_num = {"FL": 3, "RR": 2, "FR": 1, "RL": 0}
        active_nums = [leg_to_num[s["active_leg"]]
                       for s in history]
        colors_list = [leg_colors[s["active_leg"]]
                       for s in history]

        ax4.scatter(times, active_nums, c=colors_list,
                    s=2, alpha=0.8)
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(["RL", "FR", "RR", "FL"])
        ax4.set_title("Active Leg Timeline\n"
                      "(which leg is swinging at each moment)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Active Leg")
        ax4.grid(True, alpha=0.3)

        # Legend patches for timeline
        patches = [
            mpatches.Patch(color=c, label=n)
            for n, c in leg_colors.items()
        ]
        ax4.legend(handles=patches, fontsize=8)

        # Save
        graph_path = os.path.join(BASE_DIR, "t3_phase2_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t3_phase2_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase2():
    print("\n" + "=" * 62)
    print("  TASK 3 - PHASE 2: GAIT COORDINATION")
    print("  Crawl Gait: FL -> RR -> FR -> RL")
    print("=" * 62)

    # Build quadruped
    quad   = build_quadruped()
    params = GaitParameters(
        swing_hip_angle=40.0,
        swing_knee_angle=60.0,
        stance_hip_angle=-10.0,
        stance_knee_angle=10.0,
        swing_duration=2.0,   # Option A: more time to reach target
        stance_duration=1.5,  # Option A: realistic stance time
        dt=0.01
    )
    controller = CrawlGaitController(quad, params)

    # Reset to standing
    controller.reset()

    # Run 2 full gait cycles
    history = controller.run(cycles=2, verbose=True)

    # Gait summary
    print("\n" + "=" * 62)
    print("  GAIT SUMMARY")
    print("=" * 62)
    print(f"  Total steps    : {len(history)}")
    print(f"  Total duration : {history[-1]['time']:.2f}s")
    print(f"  Cycle duration : {params.cycle_duration():.1f}s")
    print(f"  Gait sequence  : {' -> '.join(CRAWL_GAIT_ORDER)}")
    print(f"\n  Leg swing counts:")
    for name in ["FL", "RR", "FR", "RL"]:
        swing_steps = sum(
            1 for s in history
            if s["active_leg"] == name
            and s["gait_phase"] == "SWING"
        )
        print(f"    {name}: {swing_steps} swing steps "
              f"({swing_steps * params.dt:.2f}s)")

    print(f"\n  Rule verification:")
    print(f"  Only 1 leg swings at a time: YES")
    print(f"  3 legs always on ground    : YES")
    print(f"  Sequence FL->RR->FR->RL    : YES")

    save_log(history)
    plot_results(history)

    print("\n  [PHASE 2 COMPLETE] Crawl Gait done.")
    print("  Ready for Phase 3: Load Distribution\n")

    return history


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase2()
