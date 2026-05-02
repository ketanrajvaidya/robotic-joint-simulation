"""
=============================================================
TASK 2 — PHASE 4 : COORDINATED MOTION
Task: Multi-Joint Leg Simulation System
=============================================================

What this phase adds on top of Phase 3:
  - Simulates a FULL motion sequence: Standing -> Lifting -> Lowering
  - Both joints move in coordination — scripted timing
  - Motion is broken into stages, each with specific targets
  - Tracks foot trajectory throughout the full sequence
  - Shows how a real robotic leg step cycle works

What is Coordinated Motion?
  In Phase 3, we just moved joints to single targets.
  In Phase 4, we chain MULTIPLE targets together in sequence.

  Real walking needs coordination:
    Stage 1 (STAND)  : Both joints neutral — leg straight down
    Stage 2 (LIFT)   : Hip swings forward, knee bends — leg rises
    Stage 3 (EXTEND) : Knee extends forward — foot reaches out
    Stage 4 (LOWER)  : Hip and knee lower — foot touches ground
    Stage 5 (RETURN) : Joints return to neutral — ready for next step

  Simple analogy:
    Walking is not just "bend knee" or "lift hip" — it is a
    choreographed sequence where both joints move at specific
    times and rates to produce smooth, efficient motion.
    Phase 4 builds that choreography.

Key new concept — Motion Stages:
  Each stage has:
    - A target hip angle
    - A target knee angle
    - A duration (how long to hold/reach that target)
    - A label (what this stage represents physically)
=============================================================
"""

import sys
import os
import math

# ── Import chain ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t2_phase1_system_expansion import build_leg_system, LegSystem
from t2_phase3_coupled_dynamics import CoupledDynamicsSimulator


# ─────────────────────────────────────────────────────────────
# MOTION STAGE
# Defines a single stage of coordinated movement.
# A full motion sequence is a list of stages played in order.
# ─────────────────────────────────────────────────────────────
class MotionStage:
    def __init__(
        self,
        label: str,
        hip_target: float,
        knee_target: float,
        duration: float
    ):
        """
        Parameters:
          label       : What this stage is (e.g. "Lift leg")
          hip_target  : Target hip angle for this stage (degrees)
          knee_target : Target knee angle for this stage (degrees)
          duration    : How long to run this stage (seconds)
        """
        self.label       = label
        self.hip_target  = hip_target
        self.knee_target = knee_target
        self.duration    = duration

    def __repr__(self):
        return (f"Stage[{self.label}] "
                f"hip={self.hip_target}° knee={self.knee_target}° "
                f"dur={self.duration}s")


# ─────────────────────────────────────────────────────────────
# COORDINATED MOTION CONTROLLER
# Plays a sequence of MotionStages one after another,
# producing a full leg movement sequence.
# ─────────────────────────────────────────────────────────────
class CoordinatedMotionController:
    def __init__(
        self,
        sim: CoupledDynamicsSimulator
    ):
        """
        Parameters:
          sim : CoupledDynamicsSimulator from Phase 3
                (already has leg + both actuator models inside)
        """
        self.sim = sim

    def run_sequence(
        self,
        stages: list,
        sequence_label: str = "Motion Sequence"
    ) -> list:
        """
        Runs a full sequence of motion stages one after another.
        Returns complete history with stage labels attached.

        Parameters:
          stages         : List of MotionStage objects
          sequence_label : Name of the full sequence

        Returns:
          Full history list — every time step from all stages combined
        """
        print(f"\n{'='*62}")
        print(f"  COORDINATED MOTION: {sequence_label}")
        print(f"{'='*62}")

        # Print stage plan first
        print(f"\n  Motion Plan:")
        total_dur = 0
        for i, stage in enumerate(stages):
            print(f"    Stage {i+1}: [{stage.label:20s}] "
                  f"Hip={stage.hip_target:>6}°  "
                  f"Knee={stage.knee_target:>6}°  "
                  f"Duration={stage.duration}s")
            total_dur += stage.duration
        print(f"    Total duration: {total_dur}s\n")

        full_history = []
        stage_start_time = 0.0

        for i, stage in enumerate(stages):
            print(f"\n  --- Stage {i+1}: {stage.label} ---")
            print(f"      Hip target  : {stage.hip_target}°")
            print(f"      Knee target : {stage.knee_target}°")
            print(f"      Duration    : {stage.duration}s")

            # Run this stage using Phase 3 simulator
            stage_history = self.sim.simulate(
                hip_target  = stage.hip_target,
                knee_target = stage.knee_target,
                duration    = stage.duration,
                label       = stage.label
            )

            # Tag each step with stage info and global time
            for snap in stage_history:
                snap["stage"]            = stage.label
                snap["stage_index"]      = i + 1
                snap["global_time"]      = round(
                    stage_start_time + snap["time"], 4
                )
                # Compute foot position at this step
                hip_rad  = math.radians(snap["hip_angle"])
                knee_rad = math.radians(snap["knee_angle"])
                combined = hip_rad + knee_rad
                L1 = self.sim.leg.L1
                L2 = self.sim.leg.L2
                knee_x = L1 * math.cos(hip_rad)
                knee_y = L1 * math.sin(hip_rad)
                foot_x = knee_x + L2 * math.cos(combined)
                foot_y = knee_y + L2 * math.sin(combined)
                snap["foot_x"] = round(foot_x, 4)
                snap["foot_y"] = round(foot_y, 4)

                full_history.append(snap)

            stage_start_time += stage.duration

            # Print stage end state
            last = stage_history[-1]
            print(f"      End hip  angle : {last['hip_angle']:.2f}°")
            print(f"      End knee angle : {last['knee_angle']:.2f}°")
            print(f"      End foot pos   : "
                  f"({full_history[-1]['foot_x']:.3f}, "
                  f"{full_history[-1]['foot_y']:.3f}) m")

        print(f"\n  [SEQUENCE COMPLETE] {sequence_label}")
        print(f"  Total steps recorded: {len(full_history)}")

        return full_history


# ─────────────────────────────────────────────────────────────
# MOTION SEQUENCES
# Two pre-defined coordinated motion sequences
# ─────────────────────────────────────────────────────────────

def get_step_cycle_sequence() -> list:
    """
    A full walking step cycle:
      Stage 1 — STAND    : Neutral position, leg down
      Stage 2 — LIFT     : Hip swings forward, knee bends
      Stage 3 — EXTEND   : Knee straightens, foot reaches forward
      Stage 4 — LOWER    : Leg comes down to touch ground
      Stage 5 — RETURN   : Return to neutral standing position

    This is the fundamental motion of a walking robot leg.
    """
    return [
        MotionStage("STAND",   hip=0.0,  knee=0.0,  duration=1.0),
        MotionStage("LIFT",    hip=30.0, knee=60.0,  duration=1.5),
        MotionStage("EXTEND",  hip=50.0, knee=20.0,  duration=1.5),
        MotionStage("LOWER",   hip=20.0, knee=5.0,   duration=1.0),
        MotionStage("RETURN",  hip=0.0,  knee=0.0,   duration=1.0),
    ]


def get_crouch_sequence() -> list:
    """
    A crouch-and-rise sequence:
      Stage 1 — NEUTRAL  : Standing straight
      Stage 2 — CROUCH   : Hip drops back, knee bends deep
      Stage 3 — HOLD     : Hold crouch position
      Stage 4 — RISE     : Return to standing

    Simulates a robot lowering its body (like before jumping
    or picking something up from the ground).
    """
    return [
        MotionStage("NEUTRAL",  hip=0.0,   knee=0.0,   duration=0.5),
        MotionStage("CROUCH",   hip=-20.0, knee=90.0,  duration=2.0),
        MotionStage("HOLD",     hip=-20.0, knee=90.0,  duration=1.0),
        MotionStage("RISE",     hip=0.0,   knee=0.0,   duration=2.0),
    ]


# Monkey-patch MotionStage to accept keyword args cleanly
def MotionStage(label, hip, knee, duration):
    """Factory for MotionStage — cleaner call syntax."""
    ms = object.__new__(_MotionStage)
    ms.label       = label
    ms.hip_target  = hip
    ms.knee_target = knee
    ms.duration    = duration
    return ms

class _MotionStage:
    def __repr__(self):
        return (f"Stage[{self.label}] "
                f"hip={self.hip_target}° knee={self.knee_target}°"
                f" dur={self.duration}s")


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(all_sequences: dict):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "t2_phase4_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("TASK 2 - PHASE 4: COORDINATED MOTION LOG\n")
        f.write("Full leg motion sequences with foot trajectory\n")
        f.write("=" * 80 + "\n\n")

        for seq_name, history in all_sequences.items():
            f.write(f"SEQUENCE: {seq_name}\n")
            f.write(
                f"{'GTime':>7} | {'Stage':>12} | "
                f"{'Hip':>7} | {'Knee':>7} | "
                f"{'Foot X':>8} | {'Foot Y':>8} | "
                f"{'HipLoad':>9} | {'KneeLoad':>9}\n"
            )
            f.write("-" * 80 + "\n")
            for s in history:
                f.write(
                    f"{s['global_time']:>7.3f} | "
                    f"{s['stage']:>12} | "
                    f"{s['hip_angle']:>7.2f} | "
                    f"{s['knee_angle']:>7.2f} | "
                    f"{s['foot_x']:>8.4f} | "
                    f"{s['foot_y']:>8.4f} | "
                    f"{s['hip_load_Nm']:>8.4f}N | "
                    f"{s['knee_load_Nm']:>8.4f}N\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] -> t2_phase4_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(all_sequences: dict):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            "Task 2 - Phase 4: Coordinated Motion\n"
            "Standing -> Lifting -> Lowering Full Sequence",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])  # Hip angle over time
        ax2 = fig.add_subplot(gs[0, 1])  # Knee angle over time
        ax3 = fig.add_subplot(gs[1, 0])  # Foot trajectory
        ax4 = fig.add_subplot(gs[1, 1])  # Load distribution

        # Stage colors for shading
        stage_colors = {
            "STAND"  : "#d4e6f1",
            "LIFT"   : "#d5f5e3",
            "EXTEND" : "#fdebd0",
            "LOWER"  : "#f9ebea",
            "RETURN" : "#e8daef",
            "NEUTRAL": "#d4e6f1",
            "CROUCH" : "#fdebd0",
            "HOLD"   : "#f9ebea",
            "RISE"   : "#d5f5e3",
        }

        seq_colors = ["royalblue", "darkorange"]

        for seq_idx, (seq_name, history) in enumerate(
                all_sequences.items()):

            gtimes     = [s["global_time"]   for s in history]
            hip_angles = [s["hip_angle"]      for s in history]
            kne_angles = [s["knee_angle"]     for s in history]
            foot_xs    = [s["foot_x"]         for s in history]
            foot_ys    = [s["foot_y"]         for s in history]
            hip_loads  = [s["hip_load_Nm"]    for s in history]
            kne_loads  = [s["knee_load_Nm"]   for s in history]
            stages     = [s["stage"]          for s in history]
            c = seq_colors[seq_idx]

            # ── Plot 1: Hip angle ──────────────────────────────
            ax1.plot(gtimes, hip_angles, color=c,
                     linewidth=2, label=seq_name)

            # Shade stage regions (only for first sequence)
            if seq_idx == 0:
                prev_stage = stages[0]
                start_t    = gtimes[0]
                for i, (t, st) in enumerate(zip(gtimes, stages)):
                    if st != prev_stage or i == len(gtimes) - 1:
                        ax1.axvspan(
                            start_t, t,
                            alpha=0.15,
                            color=stage_colors.get(prev_stage, "gray"),
                            label=f"_{prev_stage}"
                        )
                        ax1.text(
                            (start_t + t) / 2,
                            ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0
                            else -5,
                            prev_stage, fontsize=6,
                            ha='center', va='bottom', alpha=0.7
                        )
                        prev_stage = st
                        start_t    = t

            # ── Plot 2: Knee angle ─────────────────────────────
            ax2.plot(gtimes, kne_angles, color=c,
                     linewidth=2, label=seq_name, linestyle='--')

            # ── Plot 3: Foot trajectory ────────────────────────
            ax3.plot(foot_xs, foot_ys, color=c,
                     linewidth=2, label=seq_name, marker='.',
                     markersize=1)
            ax3.scatter(foot_xs[0],  foot_ys[0],
                        color='green', s=80, zorder=5,
                        label=f"Start ({seq_name})")
            ax3.scatter(foot_xs[-1], foot_ys[-1],
                        color='red',   s=80, zorder=5,
                        label=f"End ({seq_name})")

            # ── Plot 4: Load distribution ──────────────────────
            ax4.plot(gtimes, hip_loads, color=c,
                     linewidth=2, label=f"{seq_name} - Hip")
            ax4.plot(gtimes, kne_loads, color=c,
                     linewidth=1.5, linestyle=':',
                     label=f"{seq_name} - Knee")

        # Format axes
        ax1.set_title("Hip Angle Over Time\n(with stage labels)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Hip Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2.set_title("Knee Angle Over Time",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Knee Angle (deg)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linewidth=0.5)

        ax3.set_title("Foot Trajectory\n(path foot traces in space)",
                      fontweight='bold')
        ax3.set_xlabel("X position (m)")
        ax3.set_ylabel("Y position (m)")
        ax3.scatter(0, 0, color='black', s=100,
                    marker='x', zorder=6, label='Hip origin')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        ax3.axhline(0, color='gray', linewidth=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5)

        ax4.set_title("Load on Each Joint Over Time\n"
                      "(Hip load vs Knee load)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Gravitational Load (N·m)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='gray', linewidth=0.5)

        # Save
        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "t2_phase4_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] -> t2_phase4_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not available. Skipping graph.")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_phase4():
    print("\n" + "=" * 62)
    print("  TASK 2 - PHASE 4: COORDINATED MOTION")
    print("  Standing -> Lifting -> Lowering Full Sequence")
    print("=" * 62)

    leg = build_leg_system()
    sim = CoupledDynamicsSimulator(
        leg=leg,
        time_constant=0.5,
        dt=0.01
    )
    controller = CoordinatedMotionController(sim)

    all_sequences = {}

    # ── Sequence 1: Full Step Cycle ──────────────────────────
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    stages1 = get_step_cycle_sequence()
    history1 = controller.run_sequence(
        stages1,
        sequence_label="Full Step Cycle"
    )
    all_sequences["Step Cycle"] = history1

    # ── Sequence 2: Crouch and Rise ──────────────────────────
    sim.reset(hip_angle=0.0, knee_angle=0.0)
    stages2 = get_crouch_sequence()
    history2 = controller.run_sequence(
        stages2,
        sequence_label="Crouch and Rise"
    )
    all_sequences["Crouch and Rise"] = history2

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  PHASE 4 SUMMARY")
    print("=" * 62)
    for seq_name, history in all_sequences.items():
        foot_xs = [s["foot_x"] for s in history]
        foot_ys = [s["foot_y"] for s in history]
        x_range = max(foot_xs) - min(foot_xs)
        y_range = max(foot_ys) - min(foot_ys)
        print(f"\n  Sequence: {seq_name}")
        print(f"    Total steps    : {len(history)}")
        print(f"    Foot X range   : {min(foot_xs):.3f} to "
              f"{max(foot_xs):.3f} m  (span: {x_range:.3f} m)")
        print(f"    Foot Y range   : {min(foot_ys):.3f} to "
              f"{max(foot_ys):.3f} m  (span: {y_range:.3f} m)")

    save_log(all_sequences)
    plot_results(all_sequences)

    print("\n  [PHASE 4 COMPLETE] Coordinated Motion done.")
    print("  Ready for Phase 5: Load Distribution\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase4()
