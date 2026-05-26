"""
=============================================================
TASK 3 — PHASE 5 : FAILURE PROPAGATION (SYSTEM LEVEL)
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase adds on top of Phase 4:
  - Simulates 3 failure types across the FULL quadruped
  - Shows how failure in ONE leg affects ALL other legs
  - Tracks which leg compensates when one fails
  - Identifies when the whole system collapses
  - Compares healthy vs failed system behavior

3 Failure Scenarios:
  1. ONE LEG COMPLETE FAILURE (RL leg stops working)
       - RL leg freezes at its current position
       - Other 3 legs must continue gait without it
       - Load redistributes to remaining legs
       - Stability polygon changes (only 2 legs in stance sometimes)

  2. ONE JOINT STALL (FL hip stalls mid-swing)
       - FL hip joint locks at current angle
       - FL knee keeps moving (partial leg function)
       - Gait sequence disrupted
       - Load and stability affected

  3. NOISE IN ONE ACTUATOR (FR hip has jitter)
       - Random angle errors added to FR hip every step
       - FR foot position becomes unpredictable
       - Support polygon shape distorted
       - Stability margin fluctuates

For each failure:
  - Observable behavior is logged and graphed
  - Which leg compensates is identified
  - When system becomes UNSTABLE is flagged
  - Comparison with healthy baseline shown
=============================================================
"""

import sys
import os
import math
import random

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
from t3_phase4_stability import StabilityAnalyzer


# ─────────────────────────────────────────────────────────────
# DEFAULT GAIT PARAMETERS
# Same as previous phases for consistency
# ─────────────────────────────────────────────────────────────
def get_default_params() -> GaitParameters:
    return GaitParameters(
        swing_hip_angle=40.0,
        swing_knee_angle=60.0,
        stance_hip_angle=-10.0,
        stance_knee_angle=10.0,
        swing_duration=2.0,
        stance_duration=1.5,
        dt=0.01
    )


# ─────────────────────────────────────────────────────────────
# FAILURE PROPAGATION SIMULATOR
# Runs the gait with injected failures and tracks
# how failure spreads across the whole system
# ─────────────────────────────────────────────────────────────
class FailurePropagationSimulator:
    def __init__(
        self,
        quad:   QuadrupedSystem,
        params: GaitParameters
    ):
        self.quad   = quad
        self.params = params
        self.dt     = params.dt

        # Gait controller
        self.controller = CrawlGaitController(quad, params)

        # Load and stability analyzers
        self.load_calc  = SystemLoadCalculator(quad)
        self.stability  = StabilityAnalyzer(quad)

        # Failure state flags
        self.failed_leg        = None   # which leg has failed
        self.stalled_joint     = None   # "hip" or "knee"
        self.stalled_leg       = None   # which leg has stall
        self.stalled_angle     = None   # angle at stall
        self.noise_leg         = None   # which leg has noise
        self.noise_amplitude   = 3.0   # degrees of jitter

    def reset(self):
        """Resets everything to clean state."""
        self.quad.reset_all(
            hip_angle=self.params.stance_hip,
            knee_angle=self.params.stance_knee
        )
        self.controller = CrawlGaitController(
            self.quad, self.params
        )
        self.failed_leg    = None
        self.stalled_joint = None
        self.stalled_leg   = None
        self.stalled_angle = None
        self.noise_leg     = None

    def _apply_leg_failure(self, leg_name: str):
        """
        Complete leg failure — leg freezes at current position.
        All actuators in that leg are marked stalled.
        """
        leg = self.quad.legs[leg_name]
        leg.hip.actuator.is_stalled  = True
        leg.knee.actuator.is_stalled = True
        self.failed_leg = leg_name

    def _apply_joint_stall(
        self,
        leg_name:   str,
        joint_name: str,
        snap:       dict
    ):
        """
        Single joint stall — one joint locks at current angle.
        """
        leg = self.quad.legs[leg_name]
        if joint_name == "hip":
            angle = snap[f"{leg_name}_hip"]
            leg.hip.actuator.is_stalled = True
            self.stalled_angle = angle
        else:
            angle = snap[f"{leg_name}_knee"]
            leg.knee.actuator.is_stalled = True
            self.stalled_angle = angle

        self.stalled_joint = joint_name
        self.stalled_leg   = leg_name

    def _apply_noise(self, leg_name: str, snap: dict):
        """
        Adds random jitter to a leg's hip angle.
        Simulates encoder failure or loose coupling.
        """
        leg   = self.quad.legs[leg_name]
        noise = random.uniform(
            -self.noise_amplitude,
            self.noise_amplitude
        )
        current = leg.hip.joint_output.current_angle
        new_angle = leg.hip.joint_output.clamp_angle(
            current + noise
        )
        leg.hip.joint_output.current_angle = new_angle

    def run_with_failure(
        self,
        failure_type:  str,
        failure_leg:   str,
        failure_at:    float,
        cycles:        int = 2,
        joint:         str = "hip"
    ) -> list:
        """
        Runs gait simulation with a failure injected at a
        specific time.

        Parameters:
          failure_type : "leg_failure", "joint_stall", "noise"
          failure_leg  : Which leg to fail ("FL","FR","RL","RR")
          failure_at   : Time in seconds when failure happens
          cycles       : Number of gait cycles to simulate
          joint        : For joint_stall: "hip" or "knee"

        Returns:
          Full history with failure flags added
        """
        self.reset()

        total_dur  = self.params.cycle_duration() * cycles
        total_steps= int(total_dur / self.dt)

        failure_injected = False
        history = []

        print(f"\n  {'='*60}")
        print(f"  FAILURE SCENARIO: {failure_type.upper()}")
        print(f"  Affected leg  : {failure_leg}")
        print(f"  Failure at    : t={failure_at}s")
        print(f"  {'='*60}")
        print(f"  {'Time':>6} | {'Active':>6} | "
              f"{'FL_hip':>7} | {'FR_hip':>7} | "
              f"{'RL_hip':>7} | {'RR_hip':>7} | "
              f"{'Failure':>20}")
        print(f"  {'-'*75}")

        for i in range(total_steps):
            # Step the gait controller
            snap = self.controller.step()

            # Inject failure at specified time
            if (not failure_injected and
                    snap["time"] >= failure_at):
                if failure_type == "leg_failure":
                    self._apply_leg_failure(failure_leg)
                    print(f"\n  *** LEG FAILURE: {failure_leg} "
                          f"at t={snap['time']:.2f}s ***")
                elif failure_type == "joint_stall":
                    self._apply_joint_stall(
                        failure_leg, joint, snap
                    )
                    print(f"\n  *** JOINT STALL: {failure_leg} "
                          f"{joint} at t={snap['time']:.2f}s ***")
                elif failure_type == "noise":
                    self.noise_leg = failure_leg
                    print(f"\n  *** NOISE INJECTED: {failure_leg} "
                          f"hip at t={snap['time']:.2f}s ***")
                failure_injected = True

            # Apply ongoing failures
            if self.failed_leg:
                # Freeze failed leg at its current position
                leg = self.quad.legs[self.failed_leg]
                frozen_hip  = leg.hip.joint_output.current_angle
                frozen_knee = leg.knee.joint_output.current_angle
                leg.hip.joint_output.current_angle  = frozen_hip
                leg.knee.joint_output.current_angle = frozen_knee
                snap[f"{self.failed_leg}_hip"]  = frozen_hip
                snap[f"{self.failed_leg}_knee"] = frozen_knee

            if self.stalled_leg and self.stalled_joint == "hip":
                leg = self.quad.legs[self.stalled_leg]
                leg.hip.joint_output.current_angle = \
                    self.stalled_angle
                snap[f"{self.stalled_leg}_hip"] = \
                    self.stalled_angle

            if self.noise_leg:
                self._apply_noise(self.noise_leg, snap)
                snap[f"{self.noise_leg}_hip"] = (
                    self.quad.legs[self.noise_leg]
                    .hip.joint_output.current_angle
                )

            # Update quad angles from snap for load/stability
            for name in ["FL", "FR", "RL", "RR"]:
                self.quad.legs[name]\
                    .hip.joint_output.current_angle = \
                    snap[f"{name}_hip"]
                self.quad.legs[name]\
                    .knee.joint_output.current_angle = \
                    snap[f"{name}_knee"]

            # Compute load distribution
            load_data = self.load_calc.compute_all_loads(snap)

            # Compute stability
            stab_data = self.stability.analyze_step(snap)

            # Build full snapshot
            full_snap = {
                **snap,
                "load_FL"      : load_data["legs"]["FL"]["total_load_Nm"],
                "load_FR"      : load_data["legs"]["FR"]["total_load_Nm"],
                "load_RL"      : load_data["legs"]["RL"]["total_load_Nm"],
                "load_RR"      : load_data["legs"]["RR"]["total_load_Nm"],
                "max_load_leg" : load_data["max_load_leg"],
                "total_load"   : load_data["total_system"],
                "stability"    : stab_data["stability_state"],
                "polygon_area" : stab_data["polygon_area"],
                "com_margin"   : stab_data["com_margin"],
                "failure_active": failure_injected,
                "failure_type" : failure_type,
                "failure_leg"  : failure_leg,
            }
            history.append(full_snap)

            # Print every 200 steps
            if i % 200 == 0 or i == total_steps - 1:
                fail_str = (
                    f"{failure_type[:10]}:{failure_leg}"
                    if failure_injected else "healthy"
                )
                print(
                    f"  {snap['time']:>6.2f} | "
                    f"{snap['active_leg']:>6} | "
                    f"{snap['FL_hip']:>7.1f} | "
                    f"{snap['FR_hip']:>7.1f} | "
                    f"{snap['RL_hip']:>7.1f} | "
                    f"{snap['RR_hip']:>7.1f} | "
                    f"{fail_str:>20}"
                )

        return history


# ─────────────────────────────────────────────────────────────
# RUN HEALTHY BASELINE
# ─────────────────────────────────────────────────────────────
def run_healthy(params: GaitParameters) -> list:
    """Runs clean gait with no failures as baseline."""
    print("\n  --- HEALTHY BASELINE (no failures) ---")
    quad       = build_quadruped()
    controller = CrawlGaitController(quad, params)
    load_calc  = SystemLoadCalculator(quad)
    stability  = StabilityAnalyzer(quad)
    controller.reset()

    history = []
    total_steps = int(
        params.cycle_duration() * 2 / params.dt
    )

    for i in range(total_steps):
        snap      = controller.step()
        load_data = load_calc.compute_all_loads(snap)

        for name in ["FL", "FR", "RL", "RR"]:
            quad.legs[name].hip.joint_output.current_angle = \
                snap[f"{name}_hip"]
            quad.legs[name].knee.joint_output.current_angle = \
                snap[f"{name}_knee"]

        stab_data = stability.analyze_step(snap)
        full_snap = {
            **snap,
            "load_FL"      : load_data["legs"]["FL"]["total_load_Nm"],
            "load_FR"      : load_data["legs"]["FR"]["total_load_Nm"],
            "load_RL"      : load_data["legs"]["RL"]["total_load_Nm"],
            "load_RR"      : load_data["legs"]["RR"]["total_load_Nm"],
            "max_load_leg" : load_data["max_load_leg"],
            "total_load"   : load_data["total_system"],
            "stability"    : stab_data["stability_state"],
            "polygon_area" : stab_data["polygon_area"],
            "com_margin"   : stab_data["com_margin"],
            "failure_active": False,
            "failure_type" : "none",
            "failure_leg"  : "none",
        }
        history.append(full_snap)

    print(f"  Healthy baseline: {len(history)} steps recorded")
    return history


# ─────────────────────────────────────────────────────────────
# PRINT FAILURE REPORT
# ─────────────────────────────────────────────────────────────
def print_failure_report(
    scenario_name: str,
    history:       list,
    failure_at:    float
):
    """Prints analysis of failure propagation effects."""
    pre  = [s for s in history if s["time"] < failure_at]
    post = [s for s in history if s["time"] >= failure_at]

    print(f"\n  FAILURE REPORT: {scenario_name}")
    print(f"  {'-'*55}")

    if pre:
        pre_stable = sum(
            1 for s in pre if s["stability"] == "STABLE"
        )
        print(f"  Before failure ({len(pre)} steps):")
        print(f"    Stable steps : {pre_stable} "
              f"({pre_stable/len(pre)*100:.1f}%)")
        print(f"    Avg total load: "
              f"{sum(s['total_load'] for s in pre)/len(pre):.2f} N·m")

    if post:
        post_stable = sum(
            1 for s in post if s["stability"] == "STABLE"
        )
        post_unstable = sum(
            1 for s in post if s["stability"] == "UNSTABLE"
        )
        print(f"  After failure ({len(post)} steps):")
        print(f"    Stable steps  : {post_stable} "
              f"({post_stable/len(post)*100:.1f}%)")
        print(f"    Unstable steps: {post_unstable} "
              f"({post_unstable/len(post)*100:.1f}%)")
        print(f"    Avg total load: "
              f"{sum(s['total_load'] for s in post)/len(post):.2f} N·m")
        print(f"    Peak total load: "
              f"{max(s['total_load'] for s in post):.2f} N·m")

        # Which leg compensates most after failure
        leg_load_avgs = {}
        for name in ["FL", "FR", "RL", "RR"]:
            avg = sum(
                s[f"load_{name}"] for s in post
            ) / len(post)
            leg_load_avgs[name] = avg

        comp_leg = max(
            leg_load_avgs, key=leg_load_avgs.get
        )
        print(f"    Compensating leg: {comp_leg} "
              f"(avg {leg_load_avgs[comp_leg]:.2f} N·m)")


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_scenarios: dict):
    log_path = os.path.join(BASE_DIR, "t3_phase5_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 3 - PHASE 5: FAILURE PROPAGATION LOG\n")
        f.write("System-level failure across full quadruped\n")
        f.write("=" * 90 + "\n\n")

        for name, history in all_scenarios.items():
            f.write(f"SCENARIO: {name}\n")
            f.write(
                f"{'Time':>7} | {'Active':>6} | "
                f"{'FL_ld':>7} | {'FR_ld':>7} | "
                f"{'RL_ld':>7} | {'RR_ld':>7} | "
                f"{'Stability':>10} | {'Failed':>5}\n"
            )
            f.write("-" * 75 + "\n")
            for s in history:
                f.write(
                    f"{s['time']:>7.3f} | "
                    f"{s['active_leg']:>6} | "
                    f"{s['load_FL']:>7.2f} | "
                    f"{s['load_FR']:>7.2f} | "
                    f"{s['load_RL']:>7.2f} | "
                    f"{s['load_RR']:>7.2f} | "
                    f"{s['stability']:>10} | "
                    f"{'YES' if s['failure_active'] else 'NO':>5}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] -> t3_phase5_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_scenarios: dict, failure_times: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Task 3 - Phase 5: Failure Propagation\n"
            "System Behavior Under Leg Failure / Joint Stall / Noise",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.48, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])  # Hip angles comparison
        ax2 = fig.add_subplot(gs[0, 1])  # Load per leg
        ax3 = fig.add_subplot(gs[1, 0])  # Stability state
        ax4 = fig.add_subplot(gs[1, 1])  # Total system load

        scenario_styles = {
            "Healthy"     : ("green",      "-",  2.0),
            "Leg Failure" : ("red",        "--", 1.8),
            "Joint Stall" : ("darkorange", "-.", 1.8),
            "Noise"       : ("purple",     ":",  1.5),
        }

        state_to_num = {
            "STABLE": 2, "MARGINAL": 1, "UNSTABLE": 0
        }

        for sc_name, history in all_scenarios.items():
            color, ls, lw = scenario_styles.get(
                sc_name, ("blue", "-", 1.5)
            )
            times    = [s["time"]     for s in history]
            fl_hips  = [s["FL_hip"]   for s in history]
            rl_loads = [s["load_RL"]  for s in history]
            totals   = [s["total_load"] for s in history]
            states   = [
                state_to_num.get(s["stability"], 1)
                for s in history
            ]

            # Plot 1: FL hip angle
            ax1.plot(times, fl_hips, color=color,
                     linestyle=ls, linewidth=lw,
                     label=sc_name, alpha=0.85)

            # Plot 3: Stability
            ax3.plot(times, states, color=color,
                     linestyle=ls, linewidth=lw,
                     label=sc_name, alpha=0.85)

            # Plot 4: Total load
            ax4.plot(times, totals, color=color,
                     linestyle=ls, linewidth=lw,
                     label=sc_name, alpha=0.85)

        # Plot 2: Load per leg for failure scenarios
        for sc_name, history in all_scenarios.items():
            if sc_name == "Healthy":
                continue
            color, ls, lw = scenario_styles[sc_name]
            times = [s["time"] for s in history]
            for leg, lc in [
                ("FL", "royalblue"),
                ("RL", "green"),
            ]:
                loads = [s[f"load_{leg}"] for s in history]
                ax2.plot(times, loads,
                         color=color, linestyle=ls,
                         linewidth=1.5, alpha=0.7,
                         label=f"{sc_name} {leg}")

        # Add failure time markers
        for sc_name, ft in failure_times.items():
            if sc_name in scenario_styles:
                color = scenario_styles[sc_name][0]
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.axvline(
                        ft, color=color,
                        linewidth=1, linestyle=':',
                        alpha=0.6
                    )

        # Format axes
        ax1.set_title("Hip Angle: Healthy vs Failed\n"
                      "(dotted line = failure injected)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("FL Hip Angle (deg)")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2.set_title("Load Comparison After Failure\n"
                      "(FL and RL leg loads)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Load (N·m)")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["UNSTABLE", "MARGINAL", "STABLE"])
        ax3.set_title("Stability State: Healthy vs Failed",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Stability")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        ax4.set_title("Total System Load: Healthy vs Failed",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Total Load (N·m)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(
            BASE_DIR, "t3_phase5_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t3_phase5_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase5():
    print("\n" + "=" * 62)
    print("  TASK 3 - PHASE 5: FAILURE PROPAGATION")
    print("  System-level failure across full quadruped")
    print("=" * 62)

    random.seed(42)
    params        = get_default_params()
    all_scenarios = {}
    failure_times = {}

    # ── Healthy baseline ─────────────────────────────────────
    all_scenarios["Healthy"] = run_healthy(params)

    # ── Failure 1: Complete leg failure (RL leg) ─────────────
    quad1 = build_quadruped()
    fsim1 = FailurePropagationSimulator(quad1, params)
    h1 = fsim1.run_with_failure(
        failure_type="leg_failure",
        failure_leg="RL",
        failure_at=7.0,
        cycles=2
    )
    all_scenarios["Leg Failure"] = h1
    failure_times["Leg Failure"] = 7.0
    print_failure_report("RL Leg Complete Failure", h1, 7.0)

    # ── Failure 2: Joint stall (FL hip stalls) ───────────────
    quad2 = build_quadruped()
    fsim2 = FailurePropagationSimulator(quad2, params)
    h2 = fsim2.run_with_failure(
        failure_type="joint_stall",
        failure_leg="FL",
        failure_at=5.0,
        cycles=2,
        joint="hip"
    )
    all_scenarios["Joint Stall"] = h2
    failure_times["Joint Stall"] = 5.0
    print_failure_report("FL Hip Joint Stall", h2, 5.0)

    # ── Failure 3: Noise in FR hip ───────────────────────────
    quad3 = build_quadruped()
    fsim3 = FailurePropagationSimulator(quad3, params)
    h3 = fsim3.run_with_failure(
        failure_type="noise",
        failure_leg="FR",
        failure_at=4.0,
        cycles=2
    )
    all_scenarios["Noise"] = h3
    failure_times["Noise"] = 4.0
    print_failure_report("FR Hip Noise/Jitter", h3, 4.0)

    # ── System collapse summary ──────────────────────────────
    print("\n" + "=" * 62)
    print("  FAILURE PROPAGATION INSIGHTS")
    print("=" * 62)
    print("  LEG FAILURE (RL):")
    print("    - RL freezes mid-gait")
    print("    - Gait sequence breaks — only 3 legs active")
    print("    - Load shifts to FL and RR (diagonal compensation)")
    print("    - Stability degrades — smaller support polygon")
    print()
    print("  JOINT STALL (FL hip):")
    print("    - FL cannot complete swing phase")
    print("    - FL stays at stall angle permanently")
    print("    - Other legs try to continue normal sequence")
    print("    - Load on remaining legs increases")
    print()
    print("  NOISE (FR hip):")
    print("    - FR foot position unpredictable each step")
    print("    - Support polygon shape distorted")
    print("    - Stability margin fluctuates rapidly")
    print("    - Most dangerous for precision ground contact")
    print("=" * 62)

    save_log(all_scenarios)
    plot_results(all_scenarios, failure_times)

    print("\n  [PHASE 5 COMPLETE] Failure Propagation done.")
    print("  Ready for Phase 6: Control Interface\n")

    return all_scenarios


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase5()
