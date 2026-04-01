"""
=============================================================
PHASE 5 — LIMITS + CONSTRAINTS
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase adds on top of Phase 4:
  - Hard joint limits    : Physical stops at 0° and 120°
  - Soft joint limits    : Warning zones before hard stops
  - Actuator torque limit: Motor cannot exceed max torque
  - Actuator speed limit : Motor cannot exceed max speed
  - Limit detection      : System detects and reports when limits are hit
  - Constraint behavior  : Joint stops dead at limits, cannot push further

Key concepts:

  1. HARD LIMITS (Physical Stops):
     - The joint physically cannot move beyond 0° or 120°
     - Like a door that cannot open past 180° — the hinge blocks it
     - When hit: velocity = 0, position locked at boundary
     - Any command beyond this is ignored

  2. SOFT LIMITS (Warning Zones):
     - Buffer zone near hard limits (e.g. 5° before each stop)
     - System slows down when entering soft limit zone
     - Gives warning before hitting hard stop
     - Models real robot safety systems

  3. ACTUATOR LIMITS:
     - Max torque: 8.5 N·m — cannot produce more force than this
     - Max speed : 18°/s  — cannot move faster than this
     - When torque limit hit: acceleration stops, speed is maintained
     - When speed limit hit : no further acceleration allowed

  4. CONSTRAINT SCENARIOS SIMULATED:
     - Commanding beyond max limit (>120°)
     - Commanding below min limit (<0°)
     - High speed approach to limit (momentum test)
     - Actuator torque saturation under load

=============================================================
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import build_knee_joint, RotationalJoint


# ─────────────────────────────────────────────────────────────
# LIMITS + CONSTRAINTS MODEL
# Full constraint system built on top of all previous phases
# ─────────────────────────────────────────────────────────────
class LimitsConstraintsModel:
    def __init__(
        self,
        joint: RotationalJoint,
        soft_limit_margin: float = 5.0,   # degrees before hard stop
        time_constant: float = 0.5,
        damping: float = 0.05,
        load_mass: float = 0.0,
        load_distance: float = 0.35,
        dt: float = 0.01
    ):
        """
        Parameters:
          soft_limit_margin : Degrees before hard stop where
                              system starts slowing down (safety zone)
          time_constant     : Actuator response (Phase 2)
          damping           : Friction (Phase 3)
          load_mass         : External payload (Phase 4)
          load_distance     : Load moment arm (Phase 4)
          dt                : Simulation time step
        """
        self.joint             = joint
        self.soft_margin       = soft_limit_margin
        self.time_constant     = time_constant
        self.damping           = damping
        self.load_mass         = load_mass
        self.load_distance     = load_distance
        self.dt                = dt
        self.g                 = 9.81

        # Hard limits from Phase 1
        self.hard_min = joint.joint_output.min_angle   # 0°
        self.hard_max = joint.joint_output.max_angle   # 120°

        # Soft limits — warning zones
        self.soft_min = self.hard_min + soft_limit_margin   # 5°
        self.soft_max = self.hard_max - soft_limit_margin   # 115°

        # Inertia
        self.link_inertia  = (joint.link.mass *
                               joint.link.center_of_mass ** 2)
        self.load_inertia  = load_mass * load_distance ** 2
        self.total_inertia = self.link_inertia + self.load_inertia

        # State
        self.angular_velocity  = 0.0
        self.time_elapsed      = 0.0
        self.limit_hit_log     = []   # records when/which limit was hit

        # Logs
        self.time_log          = []
        self.angle_log         = []
        self.velocity_log      = []
        self.command_log       = []
        self.torque_log        = []
        self.limit_zone_log    = []   # 'none', 'soft_min', 'soft_max',
                                      # 'hard_min', 'hard_max'

        print(f"\n  [PHASE 5 INIT]")
        print(f"  Hard limits  : {self.hard_min}° to {self.hard_max}°")
        print(f"  Soft limits  : {self.soft_min}° to {self.soft_max}°")
        print(f"  Max torque   : {joint.actuator.max_torque} N.m")
        print(f"  Max speed    : {joint.actuator.get_effective_speed()} deg/s")

    # ── Limit zone detection ─────────────────────────────────
    def get_limit_zone(self, angle: float) -> str:
        """
        Returns which limit zone the joint is currently in.

        Zones:
          'normal'   : Free movement zone
          'soft_min' : Approaching minimum — slow down warning
          'soft_max' : Approaching maximum — slow down warning
          'hard_min' : At minimum hard stop — cannot go further
          'hard_max' : At maximum hard stop — cannot go further
        """
        if angle <= self.hard_min:
            return 'hard_min'
        elif angle >= self.hard_max:
            return 'hard_max'
        elif angle <= self.soft_min:
            return 'soft_min'
        elif angle >= self.soft_max:
            return 'soft_max'
        else:
            return 'normal'

    # ── Speed scaling in soft limit zone ────────────────────
    def get_speed_scale(self, angle: float, zone: str) -> float:
        """
        Scales speed down when approaching soft limits.
        In the soft limit zone, max speed is reduced to 30%
        to prevent slamming into the hard stop.

        Returns a scale factor between 0.3 and 1.0.
        """
        if zone == 'soft_min':
            # How close to hard min? 0=at soft, 1=at hard
            proximity = (self.soft_min - angle) / self.soft_margin
            return max(0.3, 1.0 - 0.7 * proximity)
        elif zone == 'soft_max':
            proximity = (angle - self.soft_max) / self.soft_margin
            return max(0.3, 1.0 - 0.7 * proximity)
        return 1.0   # full speed in normal zone

    # ── Torque computations ──────────────────────────────────
    def compute_actuator_torque(self, current: float) -> float:
        """Actuator torque — capped at motor max (torque limit)."""
        error      = self.joint.target_angle - current
        raw_torque = (error / self.time_constant) * self.total_inertia
        max_torque = self.joint.actuator.max_torque
        # ACTUATOR TORQUE LIMIT enforced here
        return max(-max_torque, min(max_torque, raw_torque))

    def compute_load_torque(self, angle: float) -> float:
        """External payload torque (Phase 4)."""
        rad = math.radians(angle)
        return self.load_mass * self.g * self.load_distance * math.cos(rad)

    # ── Main simulation step ─────────────────────────────────
    def step(self) -> tuple[float, float, float, str]:
        """
        One time step with full limit + constraint enforcement.

        Constraint sequence:
          1. Detect current limit zone
          2. Compute all torques
          3. Apply speed scaling if in soft zone
          4. Update velocity and angle
          5. Enforce hard limits — stop dead if hit
          6. Log limit events

        Returns: (angle, velocity, net_torque, limit_zone)
        """
        current = self.joint.joint_output.current_angle

        # Step 1: Detect limit zone
        zone = self.get_limit_zone(current)

        # Step 2: All torques
        tau_act  = self.compute_actuator_torque(current)
        tau_fric = self.damping * self.angular_velocity
        tau_grav = self.joint.link.gravitational_torque(current)
        tau_load = self.compute_load_torque(current)
        tau_net  = tau_act - tau_fric - tau_grav - tau_load

        # Step 3: Angular acceleration
        alpha = tau_net / max(self.total_inertia, 0.001)

        # Step 4: Update velocity
        self.angular_velocity += alpha * self.dt

        # Apply speed scaling in soft limit zones
        max_speed  = self.joint.actuator.get_effective_speed()
        speed_scale = self.get_speed_scale(current, zone)
        scaled_max  = max_speed * speed_scale

        # SPEED LIMIT enforced here
        self.angular_velocity = max(
            -scaled_max, min(scaled_max, self.angular_velocity)
        )

        # Step 5: Update angle
        new_angle = current + self.angular_velocity * self.dt

        # HARD LIMIT enforcement — clamp and kill velocity
        if new_angle <= self.hard_min:
            new_angle = self.hard_min
            if self.angular_velocity < 0:
                # Log limit hit event
                self.limit_hit_log.append({
                    'time': self.time_elapsed,
                    'limit': 'hard_min',
                    'angle': new_angle,
                    'velocity_before': self.angular_velocity
                })
            self.angular_velocity = 0.0

        elif new_angle >= self.hard_max:
            new_angle = self.hard_max
            if self.angular_velocity > 0:
                self.limit_hit_log.append({
                    'time': self.time_elapsed,
                    'limit': 'hard_max',
                    'angle': new_angle,
                    'velocity_before': self.angular_velocity
                })
            self.angular_velocity = 0.0

        # Save state
        self.joint.joint_output.current_angle    = new_angle
        self.joint.joint_output.angular_velocity = self.angular_velocity
        self.time_elapsed += self.dt

        # Log
        self.time_log.append(self.time_elapsed)
        self.angle_log.append(new_angle)
        self.velocity_log.append(self.angular_velocity)
        self.command_log.append(self.joint.target_angle)
        self.torque_log.append(tau_net)
        self.limit_zone_log.append(zone)

        return new_angle, self.angular_velocity, tau_net, zone

    def simulate(
        self,
        target_angle: float,
        duration: float,
        label: str = ""
    ):
        """Full simulation with limit reporting."""
        # Command is clamped to hard limits
        original  = target_angle
        clamped   = self.joint.joint_output.clamp_angle(target_angle)
        if original != clamped:
            print(f"  [LIMIT] Command {original}° exceeds hard limit."
                  f" Clamped to {clamped}°")
        self.joint.target_angle = clamped

        steps = int(duration / self.dt)

        print(f"\n  Simulating: {label}")
        print(f"  {'Time':>6} | {'Angle':>8} | {'Velocity':>9} | "
              f"{'Torque':>8} | {'Zone':>10}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")

        for i in range(steps):
            angle, vel, torque, zone = self.step()
            t = self.time_elapsed

            if i % 50 == 0 or i == steps - 1:
                print(f"  {t:>6.2f}s | {angle:>7.2f}° | "
                      f"{vel:>8.2f}°/s | {torque:>7.3f}Nm | {zone:>10}")

        # Report limit hits
        if self.limit_hit_log:
            print(f"\n  [LIMIT EVENTS DETECTED]")
            for event in self.limit_hit_log:
                print(f"    t={event['time']:.2f}s | "
                      f"Limit: {event['limit']:>8} | "
                      f"Angle: {event['angle']:.1f}° | "
                      f"Velocity before stop: "
                      f"{event['velocity_before']:.2f} deg/s")
        else:
            print(f"\n  [NO LIMIT EVENTS] Joint stayed within bounds")

        print(f"\n  Final angle    : "
              f"{self.joint.joint_output.current_angle:.3f}°")

    def reset(self, start_angle: float = 0.0):
        """Reset for next test."""
        self.joint.joint_output.current_angle    = start_angle
        self.joint.joint_output.angular_velocity = 0.0
        self.joint.target_angle                  = start_angle
        self.angular_velocity                    = 0.0
        self.time_elapsed                        = 0.0
        self.limit_hit_log.clear()
        self.time_log.clear()
        self.angle_log.clear()
        self.velocity_log.clear()
        self.command_log.clear()
        self.torque_log.clear()
        self.limit_zone_log.clear()
        print(f"\n  [RESET] Joint at {start_angle}°")


# ─────────────────────────────────────────────────────────────
# RUN SIMULATION — 4 CONSTRAINT SCENARIOS
# ─────────────────────────────────────────────────────────────
def run_phase5_simulation():
    print("\n" + "="*60)
    print("  PHASE 5 — LIMITS + CONSTRAINTS")
    print("  Joint and Actuator Limit Enforcement")
    print("="*60)

    models = []

    # ── TEST 1: Command beyond max limit ─────────────────────
    print("\n" + "-"*60)
    print("  TEST 1: Command beyond max limit (0° → 150°)")
    print("  Expected: Clamped to 120°, stops at hard max")
    print("-"*60)
    joint1 = build_knee_joint()
    m1 = LimitsConstraintsModel(joint1, soft_limit_margin=5.0, dt=0.01)
    m1.simulate(150.0, 8.0, "0° to 150° (hard max = 120°)")
    models.append(("Beyond max limit", m1))

    # ── TEST 2: Command below min limit ──────────────────────
    print("\n" + "-"*60)
    print("  TEST 2: Command below min limit (60° → -30°)")
    print("  Expected: Clamped to 0°, stops at hard min")
    print("-"*60)
    joint2 = build_knee_joint()
    joint2.joint_output.current_angle = 60.0
    m2 = LimitsConstraintsModel(joint2, soft_limit_margin=5.0, dt=0.01)
    m2.angular_velocity = 0.0
    m2.simulate(-30.0, 6.0, "60° to -30° (hard min = 0°)")
    models.append(("Below min limit", m2))

    # ── TEST 3: Normal move — no limits hit ──────────────────
    print("\n" + "-"*60)
    print("  TEST 3: Normal move within limits (0° → 90°)")
    print("  Expected: Smooth motion, no limit events")
    print("-"*60)
    joint3 = build_knee_joint()
    m3 = LimitsConstraintsModel(joint3, soft_limit_margin=5.0, dt=0.01)
    m3.simulate(90.0, 6.0, "0° to 90° (within limits)")
    models.append(("Normal within limits", m3))

    # ── TEST 4: Torque limit — heavy load ────────────────────
    print("\n" + "-"*60)
    print("  TEST 4: Actuator torque limit (2.0 kg load)")
    print("  Expected: Reduced speed, torque saturates at 8.5 N.m")
    print("-"*60)
    joint4 = build_knee_joint()
    m4 = LimitsConstraintsModel(
        joint4,
        soft_limit_margin=5.0,
        load_mass=2.0,
        load_distance=0.35,
        dt=0.01
    )
    m4.simulate(90.0, 8.0, "0° to 90° with 2.0kg load (torque limit)")
    models.append(("Torque limit with load", m4))

    save_log(models)
    plot_results(models)
    return models


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(models):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phase5_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 5 - LIMITS + CONSTRAINTS LOG\n")
        f.write("Joint and Actuator Constraint Enforcement\n")
        f.write("="*65 + "\n\n")

        for label, model in models:
            f.write(f"TEST: {label}\n")
            f.write(f"Hard limits : {model.hard_min} to {model.hard_max} deg\n")
            f.write(f"Soft limits : {model.soft_min} to {model.soft_max} deg\n")

            if model.limit_hit_log:
                f.write("LIMIT EVENTS:\n")
                for ev in model.limit_hit_log:
                    f.write(
                        f"  t={ev['time']:.3f}s | "
                        f"{ev['limit']} | "
                        f"angle={ev['angle']:.1f} deg | "
                        f"vel={ev['velocity_before']:.2f} deg/s\n"
                    )
            else:
                f.write("LIMIT EVENTS: None\n")

            f.write(
                f"{'Time':>8} | {'Angle':>8} | {'Velocity':>10} | "
                f"{'Torque':>8} | {'Zone':>10}\n"
            )
            f.write("-"*55 + "\n")
            for t, a, v, tr, z in zip(
                model.time_log, model.angle_log,
                model.velocity_log, model.torque_log,
                model.limit_zone_log
            ):
                f.write(
                    f"{t:>8.3f} | {a:>8.3f} | {v:>10.3f} | "
                    f"{tr:>8.4f} | {z:>10}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] phase5_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(models):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "Phase 5 - Limits + Constraints\n"
            "Joint and Actuator Limit Enforcement",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.45, wspace=0.35)

        colors = ['red', 'blue', 'green', 'orange']

        # ── Plot 1: All tests angle vs time ──────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axhline(y=120, color='red', linestyle=':',
                    linewidth=1.5, label='Hard max (120°)')
        ax1.axhline(y=115, color='orange', linestyle=':',
                    linewidth=1.0, label='Soft max (115°)')
        ax1.axhline(y=5,   color='orange', linestyle=':',
                    linewidth=1.0, label='Soft min (5°)')
        ax1.axhline(y=0,   color='blue', linestyle=':',
                    linewidth=1.5, label='Hard min (0°)')
        for (label, model), color in zip(models, colors):
            ax1.plot(model.time_log, model.angle_log,
                     color=color, linewidth=2,
                     label=label[:20])
        ax1.set_title("All Tests — Angle vs Time\n"
                      "(limit zones shown)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Test 1 detail — hitting max limit ─────────
        ax2 = fig.add_subplot(gs[0, 1])
        _, m1 = models[0]
        ax2.plot(m1.time_log, m1.angle_log,
                 'r-', linewidth=2, label='Actual angle')
        ax2.plot(m1.time_log, m1.command_log,
                 'b--', linewidth=1.5, label='Command (150°→120°)')
        ax2.axhline(y=120, color='red', linestyle=':',
                    linewidth=2, label='Hard max limit')
        ax2.axhline(y=115, color='orange', linestyle=':',
                    linewidth=1.5, label='Soft max limit')
        ax2.set_title("Test 1: Hitting Max Limit\n"
                      "(joint stops at 120°)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angle (deg)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Test 2 detail — hitting min limit ─────────
        ax3 = fig.add_subplot(gs[1, 0])
        _, m2 = models[1]
        ax3.plot(m2.time_log, m2.angle_log,
                 'b-', linewidth=2, label='Actual angle')
        ax3.plot(m2.time_log, m2.command_log,
                 'b--', linewidth=1.5, label='Command (-30°→0°)')
        ax3.axhline(y=0, color='blue', linestyle=':',
                    linewidth=2, label='Hard min limit')
        ax3.axhline(y=5, color='orange', linestyle=':',
                    linewidth=1.5, label='Soft min limit')
        ax3.set_title("Test 2: Hitting Min Limit\n"
                      "(joint stops at 0°)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle (deg)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── Plot 4: Velocity showing soft limit slowdown ──────
        ax4 = fig.add_subplot(gs[1, 1])
        for (label, model), color in zip(models, colors):
            ax4.plot(model.time_log, model.velocity_log,
                     color=color, linewidth=2,
                     label=label[:20])
        ax4.axhline(y=18, color='black', linestyle='--',
                    linewidth=1.5, label='Max speed (18°/s)')
        ax4.set_title("Velocity vs Time\n"
                      "(speed drops near limits)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity (deg/s)")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase5_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] phase5_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase5_simulation()
