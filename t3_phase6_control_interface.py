"""
=============================================================
TASK 3 — PHASE 6 : CONTROL INTERFACE LAYER (CRITICAL)
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this phase adds:
  - Defines a clean control-ready interface for Rajaryan
  - Input  : desired gait phase + desired velocity
  - Output : joint angles, foot positions, system state
  - Every timestep outputs a structured telemetry packet
  - System state = STABLE / UNSTABLE / FAILURE

This is the CRITICAL phase — the task PDF marks it as such.
Without this interface, the simulation cannot be handed
to the control team.

What "Control-Ready" means:
  Rajaryan (control team) should be able to:
    1. Send a gait command (which gait phase, how fast)
    2. Receive back all joint angles and foot positions
    3. Know if the robot is stable or not
    4. Detect failures automatically
  WITHOUT knowing anything about the internal simulation.

Interface design:
  INPUT (what Rajaryan sends):
    - gait_phase  : "STANCE" or "SWING"
    - velocity    : 0.0 to 1.0 (0=slow, 1=fast)
    - active_leg  : which leg should swing ("FL","FR","RL","RR")

  OUTPUT (what system returns every timestep):
    - timestamp
    - hip_angles   : {FL, FR, RL, RR}
    - knee_angles  : {FL, FR, RL, RR}
    - foot_positions : {FL, FR, RL, RR} as (x, y)
    - load_per_leg : {FL, FR, RL, RR} in N·m
    - system_state : "STABLE" / "UNSTABLE" / "FAILURE"
    - failure_flags: {FL, FR, RL, RR} True/False
    - gait_progress: 0.0 to 1.0 (how far through cycle)
=============================================================
"""

import sys
import os
import math
import json

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
from t3_phase5_failure_propagation import get_default_params


# ─────────────────────────────────────────────────────────────
# GAIT COMMAND
# What Rajaryan sends to the control interface
# ─────────────────────────────────────────────────────────────
class GaitCommand:
    def __init__(
        self,
        gait_type:  str   = "crawl",   # "crawl" only for now
        velocity:   float = 0.5,        # 0.0=slow to 1.0=fast
        active_leg: str   = "FL"        # which leg to swing
    ):
        """
        Parameters:
          gait_type  : Type of gait ("crawl")
          velocity   : Speed factor 0.0 to 1.0
          active_leg : Which leg should be swinging
        """
        self.gait_type  = gait_type
        self.velocity   = max(0.0, min(1.0, velocity))
        self.active_leg = active_leg

    def to_dict(self) -> dict:
        return {
            "gait_type"  : self.gait_type,
            "velocity"   : self.velocity,
            "active_leg" : self.active_leg,
        }


# ─────────────────────────────────────────────────────────────
# SYSTEM STATE PACKET
# What the control interface returns every timestep
# This is what Rajaryan reads
# ─────────────────────────────────────────────────────────────
class SystemStatePacket:
    def __init__(
        self,
        timestamp:      float,
        hip_angles:     dict,
        knee_angles:    dict,
        foot_positions: dict,
        load_per_leg:   dict,
        system_state:   str,
        failure_flags:  dict,
        gait_progress:  float,
        active_leg:     str,
        gait_phase:     str,
        polygon_area:   float,
        com_margin:     float,
    ):
        self.timestamp      = timestamp
        self.hip_angles     = hip_angles      # {FL:deg, FR:deg, ...}
        self.knee_angles    = knee_angles     # {FL:deg, FR:deg, ...}
        self.foot_positions = foot_positions  # {FL:(x,y), ...}
        self.load_per_leg   = load_per_leg    # {FL:Nm, ...}
        self.system_state   = system_state    # STABLE/UNSTABLE/FAILURE
        self.failure_flags  = failure_flags   # {FL:bool, ...}
        self.gait_progress  = gait_progress   # 0.0 to 1.0
        self.active_leg     = active_leg
        self.gait_phase     = gait_phase
        self.polygon_area   = polygon_area
        self.com_margin     = com_margin

    def to_dict(self) -> dict:
        """Converts packet to dict for logging/telemetry."""
        return {
            "timestamp"      : round(self.timestamp, 4),
            "hip_angles"     : {
                k: round(v, 3)
                for k, v in self.hip_angles.items()
            },
            "knee_angles"    : {
                k: round(v, 3)
                for k, v in self.knee_angles.items()
            },
            "foot_positions" : {
                k: (round(v[0], 4), round(v[1], 4))
                for k, v in self.foot_positions.items()
            },
            "load_per_leg"   : {
                k: round(v, 3)
                for k, v in self.load_per_leg.items()
            },
            "system_state"   : self.system_state,
            "failure_flags"  : self.failure_flags,
            "gait_progress"  : round(self.gait_progress, 3),
            "active_leg"     : self.active_leg,
            "gait_phase"     : self.gait_phase,
            "polygon_area"   : round(self.polygon_area, 5),
            "com_margin"     : round(self.com_margin, 4),
        }

    def print_summary(self):
        """Prints a readable one-line summary of the packet."""
        flags = [
            k for k, v in self.failure_flags.items() if v
        ]
        flag_str = ",".join(flags) if flags else "none"
        print(
            f"  t={self.timestamp:>6.2f}s | "
            f"Active:{self.active_leg} | "
            f"Phase:{self.gait_phase:>6} | "
            f"State:{self.system_state:>8} | "
            f"Progress:{self.gait_progress:.2f} | "
            f"Failures:{flag_str}"
        )


# ─────────────────────────────────────────────────────────────
# CONTROL INTERFACE
# The main class Rajaryan will use directly
# ─────────────────────────────────────────────────────────────
class QuadrupedControlInterface:
    def __init__(
        self,
        quad:   QuadrupedSystem,
        params: GaitParameters
    ):
        """
        Parameters:
          quad   : QuadrupedSystem from Phase 1
          params : GaitParameters for gait timing
        """
        self.quad   = quad
        self.params = params

        # Internal subsystems
        self.controller = CrawlGaitController(quad, params)
        self.load_calc  = SystemLoadCalculator(quad)
        self.stability  = StabilityAnalyzer(quad)

        # Failure tracking
        self.failure_flags = {
            name: False
            for name in ["FL", "FR", "RL", "RR"]
        }

        # Gait cycle tracking
        self.cycle_duration  = params.cycle_duration()
        self.step_duration   = (
            params.swing_dur + params.stance_dur
        )
        self.gait_progress   = 0.0
        self.total_steps     = 0

        # Telemetry log
        self.telemetry_log   = []

    def reset(self):
        """Resets the interface to initial state."""
        self.quad.reset_all(
            hip_angle=self.params.stance_hip,
            knee_angle=self.params.stance_knee
        )
        self.controller = CrawlGaitController(
            self.quad, self.params
        )
        self.failure_flags = {
            n: False for n in ["FL", "FR", "RL", "RR"]
        }
        self.gait_progress = 0.0
        self.total_steps   = 0
        self.telemetry_log = []

    def detect_failures(self) -> dict:
        """
        Automatically detects failures by checking
        actuator stall flags and temperature limits.

        Returns:
          dict {leg_name: True/False}
        """
        flags = {}
        for name in ["FL", "FR", "RL", "RR"]:
            leg = self.quad.legs[name]
            hip_failed  = leg.hip.actuator.is_stalled
            knee_failed = leg.knee.actuator.is_stalled
            temp_failed = leg.hip.actuator.temperature > 100.0
            flags[name] = hip_failed or knee_failed or temp_failed
        return flags

    def get_system_state(
        self,
        stability_state: str,
        failure_flags:   dict
    ) -> str:
        """
        Determines overall system state string.

        Priority:
          FAILURE  > UNSTABLE > MARGINAL > STABLE

        Returns:
          "STABLE" / "MARGINAL" / "UNSTABLE" / "FAILURE"
        """
        any_failure = any(failure_flags.values())
        if any_failure:
            return "FAILURE"
        return stability_state

    def step(self, command: GaitCommand) -> SystemStatePacket:
        """
        Main interface method — Rajaryan calls this every timestep.

        Takes a GaitCommand, advances simulation one step,
        returns a complete SystemStatePacket.

        Parameters:
          command : GaitCommand from control team

        Returns:
          SystemStatePacket with full system state
        """
        # Apply velocity scaling to time constant
        # Higher velocity = lower time constant = faster response
        base_tc = 0.6
        tc = base_tc * (1.0 - command.velocity * 0.5)
        for sim in self.controller.simulators.values():
            sim.time_constant = max(0.2, tc)

        # Step gait controller
        snap = self.controller.step()

        # Update quad angles from snap
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

        # Detect failures
        self.failure_flags = self.detect_failures()

        # Overall system state
        sys_state = self.get_system_state(
            stab_data["stability_state"],
            self.failure_flags
        )

        # Gait progress (0.0 to 1.0 within current cycle)
        self.total_steps += 1
        cycle_steps = int(
            self.cycle_duration / self.params.dt
        )
        self.gait_progress = (
            self.total_steps % cycle_steps
        ) / cycle_steps

        # Build packet
        packet = SystemStatePacket(
            timestamp      = snap["time"],
            hip_angles     = {
                n: round(snap[f"{n}_hip"], 3)
                for n in ["FL", "FR", "RL", "RR"]
            },
            knee_angles    = {
                n: round(snap[f"{n}_knee"], 3)
                for n in ["FL", "FR", "RL", "RR"]
            },
            foot_positions = {
                n: self.quad.get_foot_position(n)
                for n in ["FL", "FR", "RL", "RR"]
            },
            load_per_leg   = {
                n: load_data["legs"][n]["total_load_Nm"]
                for n in ["FL", "FR", "RL", "RR"]
            },
            system_state   = sys_state,
            failure_flags  = dict(self.failure_flags),
            gait_progress  = self.gait_progress,
            active_leg     = snap["active_leg"],
            gait_phase     = snap["gait_phase"],
            polygon_area   = stab_data["polygon_area"],
            com_margin     = stab_data["com_margin"],
        )

        # Add to telemetry log
        self.telemetry_log.append(packet.to_dict())

        return packet

    def run(
        self,
        command: GaitCommand,
        cycles:  int = 2,
        verbose: bool = True
    ) -> list:
        """
        Runs the control interface for a number of cycles.

        Parameters:
          command : GaitCommand to execute
          cycles  : Number of gait cycles
          verbose : Print state packets

        Returns:
          List of SystemStatePackets
        """
        total_dur   = self.params.cycle_duration() * cycles
        total_steps = int(total_dur / self.params.dt)
        packets     = []

        print(f"\n  {'='*65}")
        print(f"  CONTROL INTERFACE RUNNING")
        print(f"  Gait type  : {command.gait_type}")
        print(f"  Velocity   : {command.velocity} "
              f"({'slow' if command.velocity < 0.4 else 'medium' if command.velocity < 0.7 else 'fast'})")
        print(f"  Cycles     : {cycles}")
        print(f"  Duration   : {total_dur:.1f}s")
        print(f"  {'='*65}")

        if verbose:
            print(f"\n  {'Time':>6} | {'Active':>6} | "
                  f"{'Phase':>6} | {'State':>8} | "
                  f"{'Progress':>8} | {'Failures':>10}")
            print(f"  {'-'*60}")

        prev_state = None
        for i in range(total_steps):
            packet = self.step(command)
            packets.append(packet)

            # Print when state changes or every 200 steps
            if verbose and (
                packet.system_state != prev_state
                or i % 200 == 0
                or i == total_steps - 1
            ):
                flags = [
                    k for k, v in packet.failure_flags.items()
                    if v
                ]
                flag_str = ",".join(flags) if flags else "none"
                print(
                    f"  {packet.timestamp:>6.2f} | "
                    f"{packet.active_leg:>6} | "
                    f"{packet.gait_phase:>6} | "
                    f"{packet.system_state:>8} | "
                    f"{packet.gait_progress:>8.3f} | "
                    f"{flag_str:>10}"
                )
                prev_state = packet.system_state

        return packets


# ─────────────────────────────────────────────────────────────
# SAVE TELEMETRY LOG
# ─────────────────────────────────────────────────────────────
def save_telemetry(interface: QuadrupedControlInterface):
    """Saves full telemetry log to structured text file."""
    log_path = os.path.join(BASE_DIR, "t3_phase6_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 3 - PHASE 6: CONTROL INTERFACE TELEMETRY\n")
        f.write("Structured output for Rajaryan (Control) + "
                "Dhruv (Data)\n")
        f.write("=" * 95 + "\n\n")
        f.write(
            f"{'Time':>7} | {'Active':>6} | {'Phase':>6} | "
            f"{'State':>8} | {'Progress':>8} | "
            f"{'FL_hip':>7} | {'FR_hip':>7} | "
            f"{'RL_hip':>7} | {'RR_hip':>7} | "
            f"{'Failures':>10}\n"
        )
        f.write("-" * 95 + "\n")
        for pkt in interface.telemetry_log:
            flags = [
                k for k, v in pkt["failure_flags"].items()
                if v
            ]
            flag_str = ",".join(flags) if flags else "none"
            f.write(
                f"{pkt['timestamp']:>7.3f} | "
                f"{pkt['active_leg']:>6} | "
                f"{pkt['gait_phase']:>6} | "
                f"{pkt['system_state']:>8} | "
                f"{pkt['gait_progress']:>8.3f} | "
                f"{pkt['hip_angles']['FL']:>7.2f} | "
                f"{pkt['hip_angles']['FR']:>7.2f} | "
                f"{pkt['hip_angles']['RL']:>7.2f} | "
                f"{pkt['hip_angles']['RR']:>7.2f} | "
                f"{flag_str:>10}\n"
            )
    print(f"\n  [LOG SAVED] -> t3_phase6_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(
    packets_slow:   list,
    packets_medium: list,
    packets_fast:   list
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Task 3 - Phase 6: Control Interface Layer\n"
            "System Response at Different Velocity Commands",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])  # Hip angles vs velocity
        ax2 = fig.add_subplot(gs[0, 1])  # Foot positions
        ax3 = fig.add_subplot(gs[1, 0])  # Gait progress
        ax4 = fig.add_subplot(gs[1, 1])  # System state

        configs = [
            (packets_slow,   "Slow (v=0.2)",   "royalblue",  "-"),
            (packets_medium, "Medium (v=0.5)", "darkorange", "--"),
            (packets_fast,   "Fast (v=0.9)",   "green",      "-."),
        ]

        state_to_num = {
            "STABLE": 2, "MARGINAL": 1,
            "UNSTABLE": 0, "FAILURE": -1
        }

        for packets, label, color, ls in configs:
            times    = [p.timestamp      for p in packets]
            fl_hips  = [p.hip_angles["FL"] for p in packets]
            fl_feet_y= [p.foot_positions["FL"][1]
                        for p in packets]
            progress = [p.gait_progress   for p in packets]
            states   = [
                state_to_num.get(p.system_state, 1)
                for p in packets
            ]

            ax1.plot(times, fl_hips,   color=color,
                     linestyle=ls, linewidth=2, label=label)
            ax2.plot(times, fl_feet_y, color=color,
                     linestyle=ls, linewidth=2, label=label)
            ax3.plot(times, progress,  color=color,
                     linestyle=ls, linewidth=2, label=label)
            ax4.plot(times, states,    color=color,
                     linestyle=ls, linewidth=2, label=label)

        # Format
        ax1.set_title("FL Hip Angle at Different Velocities\n"
                      "(faster velocity = snappier response)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Hip Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2.set_title("FL Foot Y Position at Different Velocities",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Foot Y Position (m)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3.set_title("Gait Progress (0=cycle start, 1=cycle end)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Gait Progress")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4.set_yticks([-1, 0, 1, 2])
        ax4.set_yticklabels([
            "FAILURE", "UNSTABLE", "MARGINAL", "STABLE"
        ])
        ax4.set_title("System State at Different Velocities",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("System State")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(
            BASE_DIR, "t3_phase6_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t3_phase6_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase6():
    print("\n" + "=" * 65)
    print("  TASK 3 - PHASE 6: CONTROL INTERFACE LAYER")
    print("  Building control-ready interface for Rajaryan")
    print("=" * 65)

    params = get_default_params()

    # ── Test 1: Slow velocity ────────────────────────────────
    print("\n  [TEST 1] Slow velocity command (v=0.2)")
    quad1  = build_quadruped()
    iface1 = QuadrupedControlInterface(quad1, params)
    iface1.reset()
    cmd_slow   = GaitCommand(
        gait_type="crawl", velocity=0.2, active_leg="FL"
    )
    packets_slow = iface1.run(cmd_slow, cycles=2, verbose=True)

    # ── Test 2: Medium velocity ──────────────────────────────
    print("\n  [TEST 2] Medium velocity command (v=0.5)")
    quad2  = build_quadruped()
    iface2 = QuadrupedControlInterface(quad2, params)
    iface2.reset()
    cmd_med    = GaitCommand(
        gait_type="crawl", velocity=0.5, active_leg="FL"
    )
    packets_med = iface2.run(cmd_med, cycles=2, verbose=False)

    # ── Test 3: Fast velocity ────────────────────────────────
    print("\n  [TEST 3] Fast velocity command (v=0.9)")
    quad3  = build_quadruped()
    iface3 = QuadrupedControlInterface(quad3, params)
    iface3.reset()
    cmd_fast   = GaitCommand(
        gait_type="crawl", velocity=0.9, active_leg="FL"
    )
    packets_fast = iface3.run(cmd_fast, cycles=2, verbose=False)

    # ── Interface summary ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CONTROL INTERFACE SUMMARY")
    print("=" * 65)

    for label, packets in [
        ("Slow   (v=0.2)", packets_slow),
        ("Medium (v=0.5)", packets_med),
        ("Fast   (v=0.9)", packets_fast),
    ]:
        stable = sum(
            1 for p in packets
            if p.system_state == "STABLE"
        )
        total  = len(packets)
        print(f"\n  {label}:")
        print(f"    Total packets : {total}")
        print(f"    Stable        : {stable} "
              f"({stable/total*100:.1f}%)")
        print(f"    Final state   : {packets[-1].system_state}")
        print(f"    Sample packet :")
        packets[100].print_summary()

    print(f"\n  Interface is ready for Rajaryan.")
    print(f"  Usage:")
    print(f"    interface = QuadrupedControlInterface(quad, params)")
    print(f"    command   = GaitCommand(velocity=0.5)")
    print(f"    packet    = interface.step(command)")
    print(f"    state     = packet.system_state")
    print(f"    angles    = packet.hip_angles")

    # Save telemetry from medium velocity test
    save_telemetry(iface2)

    # Plot
    plot_results(packets_slow, packets_med, packets_fast)

    print("\n  [PHASE 6 COMPLETE] Control Interface done.")
    print("  Ready for Phase 7: Data + Logging Layer\n")

    return iface2, packets_med


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase6()
