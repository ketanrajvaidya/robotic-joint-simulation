"""
=============================================================
PHASE 2 — ACTUATOR MODELING
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase adds on top of Phase 1:
  - Input command (desired angle)
  - Output response (actual angle over time)
  - Response delay using a first-order system (time constant)
  - Gradual movement — joint never jumps instantly to target

Key concept — First Order System:
  A first-order system means the output "chases" the input
  gradually, like a car accelerating toward a speed limit.

  Formula used:
    error          = target_angle - current_angle
    velocity       = error × (1 / time_constant)
    new_angle      = current_angle + velocity × dt

  time_constant (τ):
    - Small τ = fast response (snappy but unrealistic)
    - Large τ = slow response (sluggish, realistic for heavy joints)
    - We use τ = 0.5 seconds for a realistic servo motor

=============================================================
"""

import sys
import os
import math
import time

# ── import Phase 1 ──────────────────────────────────────────
# We import the joint we built in Phase 1 so Phase 2 builds
# directly on top of it — no duplication.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import (
    build_knee_joint,
    RotationalJoint,
    Actuator,
    JointOutput,
    Link
)


# ─────────────────────────────────────────────────────────────
# ACTUATOR RESPONSE MODEL
# Models how an actuator responds to a command over time.
# Uses a first-order lag system — the most common model
# for real servo motors and hydraulic actuators.
# ─────────────────────────────────────────────────────────────
class ActuatorResponseModel:
    def __init__(
        self,
        joint: RotationalJoint,
        time_constant: float = 0.5,   # τ (tau) in seconds — response speed
        dt: float = 0.01              # simulation time step in seconds (10ms)
    ):
        """
        Parameters:
          joint         : The RotationalJoint from Phase 1
          time_constant : How quickly actuator responds to commands (seconds).
                          Lower = faster response. Real servos: 0.1 to 1.0s
          dt            : Time step for numerical simulation (seconds).
                          Smaller = more accurate but slower to compute.
        """
        self.joint = joint
        self.time_constant = time_constant  # τ
        self.dt = dt                        # simulation time step

        # Simulation state
        self.time_elapsed = 0.0             # total time simulated (seconds)
        self.command_log = []               # log of all commands sent
        self.response_log = []              # log of actual responses

    def send_command(self, target_angle: float):
        """
        Sends a new angle command to the joint.
        Validates and stores it — actual motion happens in step().
        """
        self.joint.set_target(target_angle)
        print(f"\n  [COMMAND] Target angle set to {self.joint.target_angle}°")

    def step(self):
        """
        Advances the simulation by one time step (dt seconds).

        Physics:
          1. Calculate error = how far we are from target
          2. Calculate velocity = error scaled by responsiveness
          3. Cap velocity to actuator's max speed
          4. Update current angle
          5. Enforce joint limits

        This is called repeatedly in a loop to simulate motion over time.
        """
        current = self.joint.joint_output.current_angle
        target  = self.joint.target_angle

        # Step 1: How far are we from target?
        error = target - current

        # Step 2: Velocity from first-order system
        # The closer we are to target, the slower we move (smooth approach)
        raw_velocity = error / self.time_constant

        # Step 3: Cap to actuator's physical speed limit
        max_speed = self.joint.actuator.get_effective_speed()  # deg/s after gearing
        velocity = max(-max_speed, min(max_speed, raw_velocity))

        # Step 4: Update angle
        new_angle = current + velocity * self.dt

        # Step 5: Clamp to joint limits (Phase 5 enforcement)
        new_angle = self.joint.joint_output.clamp_angle(new_angle)

        # Store velocity for analysis
        self.joint.joint_output.angular_velocity = velocity
        self.joint.joint_output.current_angle = new_angle

        # Advance time
        self.time_elapsed += self.dt

        # Log this step
        self.command_log.append(target)
        self.response_log.append(new_angle)

        return new_angle

    def simulate(
        self,
        target_angle: float,
        duration: float,
        label: str = ""
    ) -> tuple[list, list, list]:
        """
        Runs a full simulation of the joint moving to a target angle.

        Parameters:
          target_angle : Desired angle to reach (degrees)
          duration     : How long to simulate (seconds)
          label        : Description for logging

        Returns:
          times     : list of time values
          commands  : list of target angles (flat line = constant command)
          responses : list of actual angles over time
        """
        self.send_command(target_angle)

        times     = []
        commands  = []
        responses = []

        steps = int(duration / self.dt)

        print(f"\n  Simulating: {label}")
        print(f"  {'Time':>6} | {'Command':>10} | {'Actual':>10} | {'Error':>8} | {'Velocity':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

        for i in range(steps):
            actual = self.step()
            t      = self.time_elapsed
            cmd    = self.joint.target_angle
            err    = cmd - actual
            vel    = self.joint.joint_output.angular_velocity

            times.append(t)
            commands.append(cmd)
            responses.append(actual)

            # Print every 50 steps (every 0.5s) to avoid flooding terminal
            if i % 50 == 0 or i == steps - 1:
                print(f"  {t:>6.2f}s | {cmd:>9.2f}° | {actual:>9.2f}° | {err:>7.2f}° | {vel:>8.2f}°/s")

        final_error = abs(self.joint.target_angle - self.joint.joint_output.current_angle)
        print(f"\n  Final angle : {self.joint.joint_output.current_angle:.3f}°")
        print(f"  Final error : {final_error:.3f}°")

        return times, commands, responses

    def reset(self, start_angle: float = 0.0):
        """Resets joint back to a starting angle for a new test."""
        self.joint.joint_output.current_angle = start_angle
        self.joint.joint_output.angular_velocity = 0.0
        self.joint.target_angle = start_angle
        self.time_elapsed = 0.0
        self.command_log.clear()
        self.response_log.clear()
        print(f"\n  [RESET] Joint returned to {start_angle}°")


# ─────────────────────────────────────────────────────────────
# SIMULATION SCENARIOS
# Three tests that demonstrate Phase 2 behavior clearly
# ─────────────────────────────────────────────────────────────
def run_phase2_simulation():
    """
    Runs three simulation scenarios showing actuator response behavior.
    """
    print("\n" + "="*60)
    print("  PHASE 2 — ACTUATOR MODELING")
    print("="*60)

    # Build the joint from Phase 1
    joint = build_knee_joint()
    model = ActuatorResponseModel(
        joint=joint,
        time_constant=0.5,   # τ = 0.5s — realistic servo response
        dt=0.01              # 10ms time step
    )

    # ── TEST 1: Simple move from 0° to 90° ──────────────────
    print("\n" + "─"*60)
    print("  TEST 1: Move from 0° → 90° (normal command)")
    print("─"*60)

    t1, cmd1, resp1 = model.simulate(
        target_angle=90.0,
        duration=3.0,
        label="0° to 90° in 3 seconds"
    )

    # ── TEST 2: Quick reversal — 90° back to 30° ────────────
    model.reset(start_angle=90.0)
    print("\n" + "─"*60)
    print("  TEST 2: Reversal from 90° → 30° (direction change)")
    print("─"*60)

    t2, cmd2, resp2 = model.simulate(
        target_angle=30.0,
        duration=3.0,
        label="90° to 30° reversal"
    )

    # ── TEST 3: Command beyond limit (150° → clamped to 120°) 
    model.reset(start_angle=0.0)
    print("\n" + "─"*60)
    print("  TEST 3: Over-limit command (0° → 150°, limit is 120°)")
    print("─"*60)

    t3, cmd3, resp3 = model.simulate(
        target_angle=150.0,    # will be clamped to 120°
        duration=5.0,
        label="Over-limit command clamped to 120°"
    )

    # ── SAVE LOG ─────────────────────────────────────────────
    save_log(t1, cmd1, resp1, t2, cmd2, resp2, t3, cmd3, resp3)

    # ── PLOT ─────────────────────────────────────────────────
    plot_results(t1, cmd1, resp1, t2, cmd2, resp2, t3, cmd3, resp3)

    return (t1, cmd1, resp1), (t2, cmd2, resp2), (t3, cmd3, resp3)


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(t1, cmd1, resp1, t2, cmd2, resp2, t3, cmd3, resp3):
    """Saves input vs output log to a text file."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_log.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 2 — ACTUATOR MODELING LOG\n")
        f.write("Input (Command) vs Output (Actual Angle)\n")
        f.write("="*55 + "\n\n")

        for label, times, cmds, resps in [
            ("TEST 1: 0° → 90°",   t1, cmd1, resp1),
            ("TEST 2: 90° → 30°",  t2, cmd2, resp2),
            ("TEST 3: 0° → 150° (clamped to 120°)", t3, cmd3, resp3),
        ]:
            f.write(f"{label}\n")
            f.write(f"{'Time':>8} | {'Command':>10} | {'Actual':>10} | {'Error':>8}\n")
            f.write("-"*45 + "\n")
            for t, c, r in zip(times, cmds, resps):
                f.write(f"{t:>8.3f} | {c:>10.3f} | {r:>10.3f} | {abs(c-r):>8.3f}\n")
            f.write("\n")

    print(f"\n  [LOG SAVED] → phase2_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(t1, cmd1, resp1, t2, cmd2, resp2, t3, cmd3, resp3):
    """
    Plots input command vs actual output for all three tests.
    This is the Phase 7 graph for Phase 2 data.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "Phase 2 — Actuator Response Model\nInput Command vs Actual Output",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Plot 1: Test 1 ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t1, cmd1,  'b--', linewidth=1.5, label='Command (input)')
        ax1.plot(t1, resp1, 'r-',  linewidth=2.0, label='Actual (output)')
        ax1.set_title("Test 1: 0° → 90°", fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 100)

        # ── Plot 2: Test 2 ───────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t2, cmd2,  'b--', linewidth=1.5, label='Command (input)')
        ax2.plot(t2, resp2, 'r-',  linewidth=2.0, label='Actual (output)')
        ax2.set_title("Test 2: 90° → 30° (reversal)", fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angle (°)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Test 3 ───────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t3, cmd3,  'b--', linewidth=1.5, label='Command (input)')
        ax3.plot(t3, resp3, 'r-',  linewidth=2.0, label='Actual (output)')
        ax3.axhline(y=120, color='orange', linestyle=':', linewidth=1.5, label='Joint limit (120°)')
        ax3.set_title("Test 3: Over-limit → clamped to 120°", fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle (°)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Error over time (Test 1) ─────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        error = [abs(c - r) for c, r in zip(cmd1, resp1)]
        ax4.plot(t1, error, 'g-', linewidth=2.0, label='Tracking error')
        ax4.set_title("Tracking Error over Time (Test 1)", fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Error (°)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Save
        graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] → phase2_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")
        print("  Then re-run this file to see the graphs.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase2_simulation()
