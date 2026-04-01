"""
=============================================================
PHASE 6 — FAILURE CONDITIONS
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase adds on top of Phase 5:
  - Actuator stall     : Motor stops completely under overload
  - Overheating        : Performance degrades as temperature rises
  - Inconsistent motion: Random jitter/noise in output

Key concepts:

  1. ACTUATOR STALL:
     - Happens when load torque exceeds motor's max torque
     - Motor draws maximum current but produces no motion
     - Observable: velocity = 0, current = max, temperature rising
     - In real robots: triggers overcurrent protection

  2. OVERHEATING:
     - Motor generates heat when running (I²R losses)
     - Heat builds up faster under load
     - As temperature rises: max torque reduces (thermal derating)
     - At critical temp: motor shuts down or severely limited
     - Formula: temp_rise = current² × resistance × dt
                torque_limit = max_torque × (1 - heat_factor)

  3. INCONSISTENT MOTION:
     - Models worn bearings, loose connections, encoder noise
     - Random noise added to velocity output
     - Results in jitter, stuttering, unpredictable position
     - Observable: jagged position curve instead of smooth

  4. FAILURE PROGRESSION:
     Normal → Degraded → Stall/Shutdown
     This is the realistic failure arc of any actuator

=============================================================
"""

import sys
import os
import math
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import build_knee_joint, RotationalJoint


# ─────────────────────────────────────────────────────────────
# FAILURE CONDITIONS MODEL
# Extends all previous phases with three failure modes
# ─────────────────────────────────────────────────────────────
class FailureConditionsModel:
    def __init__(
        self,
        joint: RotationalJoint,
        # Failure mode flags
        enable_stall: bool = False,
        enable_overheating: bool = False,
        enable_noise: bool = False,
        # Stall parameters
        stall_load_mass: float = 0.0,
        stall_load_distance: float = 0.35,
        # Overheating parameters
        thermal_resistance: float = 2.0,    # °C per watt
        critical_temp: float = 80.0,        # °C — shutdown temperature
        ambient_temp: float = 25.0,         # °C — starting temperature
        cooling_rate: float = 0.5,          # °C/s natural cooling
        # Noise parameters
        noise_amplitude: float = 0.5,       # degrees of random jitter
        # Standard parameters
        time_constant: float = 0.5,
        damping: float = 0.05,
        dt: float = 0.01
    ):
        self.joint              = joint
        self.enable_stall       = enable_stall
        self.enable_overheating = enable_overheating
        self.enable_noise       = enable_noise
        self.stall_load_mass    = stall_load_mass
        self.stall_load_distance = stall_load_distance
        self.thermal_resistance = thermal_resistance
        self.critical_temp      = critical_temp
        self.ambient_temp       = ambient_temp
        self.cooling_rate       = cooling_rate
        self.noise_amplitude    = noise_amplitude
        self.time_constant      = time_constant
        self.damping            = damping
        self.dt                 = dt
        self.g                  = 9.81

        # Inertia
        link  = joint.link
        self.link_inertia  = link.mass * link.center_of_mass ** 2
        self.load_inertia  = stall_load_mass * stall_load_distance ** 2
        self.total_inertia = self.link_inertia + self.load_inertia

        # Thermal state
        self.temperature       = ambient_temp   # current motor temp (°C)
        self.is_stalled        = False
        self.is_overheated     = False
        self.angular_velocity  = 0.0
        self.time_elapsed      = 0.0

        # Performance degradation factor (1.0 = full, 0.0 = dead)
        self.performance_factor = 1.0

        # Logs
        self.time_log         = []
        self.angle_log        = []
        self.velocity_log     = []
        self.temp_log         = []
        self.torque_log       = []
        self.performance_log  = []
        self.failure_events   = []

        print(f"\n  [PHASE 6 INIT]")
        print(f"  Stall mode      : {enable_stall}")
        print(f"  Overheat mode   : {enable_overheating}")
        print(f"  Noise mode      : {enable_noise}")
        print(f"  Stall load      : {stall_load_mass} kg")
        print(f"  Critical temp   : {critical_temp} °C")
        print(f"  Noise amplitude : {noise_amplitude} deg")

    # ── Thermal model ────────────────────────────────────────
    def update_temperature(self, tau_actuator: float):
        """
        Updates motor temperature based on current draw.

        Physics:
          Power dissipated = torque × angular_velocity (approx)
          Simplified: heat ∝ torque² (I²R losses)
          Temperature rise = heat × thermal_resistance × dt
          Natural cooling reduces temperature toward ambient

        As temperature rises:
          performance_factor decreases linearly
          At critical_temp: performance_factor = 0 (shutdown)
        """
        # Heat generated proportional to torque squared
        heat_generated = (abs(tau_actuator) ** 2) * self.thermal_resistance * self.dt * 0.1

        # Natural cooling toward ambient
        cooling = self.cooling_rate * (self.temperature - self.ambient_temp) * self.dt

        # Update temperature
        self.temperature += heat_generated - cooling
        self.temperature  = max(self.ambient_temp, self.temperature)

        # Performance degradation — linear from ambient to critical
        temp_range = self.critical_temp - self.ambient_temp
        temp_above = self.temperature - self.ambient_temp
        self.performance_factor = max(0.0, 1.0 - (temp_above / temp_range))

        # Check for overheat shutdown
        if self.temperature >= self.critical_temp and not self.is_overheated:
            self.is_overheated = True
            self.failure_events.append({
                'time': self.time_elapsed,
                'type': 'OVERHEAT_SHUTDOWN',
                'temp': self.temperature,
                'performance': self.performance_factor
            })
            print(f"\n  [FAILURE] OVERHEAT at t={self.time_elapsed:.2f}s"
                  f" | Temp: {self.temperature:.1f}°C"
                  f" | Performance: {self.performance_factor:.2f}")

    # ── Noise model ──────────────────────────────────────────
    def apply_noise(self, velocity: float) -> float:
        """
        Adds random noise to velocity output.
        Models worn bearings, encoder errors, loose connections.

        The noise amplitude scales with velocity —
        faster motion = more vibration = more noise.
        At zero velocity: small constant noise (static friction jitter)
        """
        if not self.enable_noise:
            return velocity

        # Scale noise with speed — faster = noisier
        speed_factor = max(0.3, abs(velocity) / 18.0)
        noise = random.gauss(0, self.noise_amplitude * speed_factor)
        return velocity + noise

    # ── Stall detection ──────────────────────────────────────
    def check_stall(self, tau_net: float, velocity: float):
        """
        Detects actuator stall condition.
        Stall occurs when:
          - Net torque is negative (load > motor)
          - AND velocity is effectively zero
          - AND motor is trying to move (has a target)
        """
        if not self.enable_stall:
            return

        target_error = abs(self.joint.target_angle -
                           self.joint.joint_output.current_angle)

        if (tau_net < 0 and
                abs(velocity) < 0.5 and
                target_error > 1.0 and
                not self.is_stalled):
            self.is_stalled = True
            self.failure_events.append({
                'time': self.time_elapsed,
                'type': 'STALL',
                'angle': self.joint.joint_output.current_angle,
                'net_torque': tau_net
            })
            print(f"\n  [FAILURE] STALL DETECTED at t={self.time_elapsed:.2f}s"
                  f" | Angle: {self.joint.joint_output.current_angle:.1f}°"
                  f" | Net torque: {tau_net:.3f} N.m")

    # ── Main simulation step ─────────────────────────────────
    def step(self) -> tuple[float, float, float, float, float]:
        """
        One time step with all failure modes active.

        Failure sequence per step:
          1. Compute all torques
          2. Apply performance degradation (overheating)
          3. Update temperature
          4. Check for stall
          5. Apply noise to velocity
          6. Update position
          7. Enforce limits

        Returns: (angle, velocity, net_torque, temperature, performance)
        """
        current = self.joint.joint_output.current_angle

        # ── Torques ──────────────────────────────────────────
        error      = self.joint.target_angle - current
        raw_torque = (error / self.time_constant) * self.total_inertia

        # Apply performance degradation from overheating
        # Degraded motor produces less torque
        effective_max = (self.joint.actuator.max_torque *
                         self.performance_factor)
        tau_act  = max(-effective_max, min(effective_max, raw_torque))
        tau_fric = self.damping * self.angular_velocity
        tau_grav = self.joint.link.gravitational_torque(current)

        # Load torque (for stall testing)
        tau_load = 0.0
        if self.stall_load_mass > 0:
            rad = math.radians(current)
            tau_load = (self.stall_load_mass * self.g *
                        self.stall_load_distance * math.cos(rad))

        tau_net = tau_act - tau_fric - tau_grav - tau_load

        # ── Update temperature ────────────────────────────────
        if self.enable_overheating:
            self.update_temperature(tau_act)

        # ── Check stall ───────────────────────────────────────
        self.check_stall(tau_net, self.angular_velocity)

        # ── If stalled — no movement ──────────────────────────
        if self.is_stalled and self.enable_stall:
            # Motor is drawing current but not moving
            # Temperature still rises during stall
            if self.enable_overheating:
                self.temperature += (self.thermal_resistance *
                                     self.dt * 2.0)  # stall heats faster
            self.angular_velocity = 0.0
            new_angle = current

        else:
            # ── Normal physics ────────────────────────────────
            alpha = tau_net / max(self.total_inertia, 0.001)
            self.angular_velocity += alpha * self.dt

            # Speed limit (reduced by performance factor)
            max_speed = (self.joint.actuator.get_effective_speed() *
                         self.performance_factor)
            max_speed = max(1.0, max_speed)  # minimum 1 deg/s
            self.angular_velocity = max(
                -max_speed, min(max_speed, self.angular_velocity)
            )

            # Apply noise
            noisy_velocity = self.apply_noise(self.angular_velocity)

            # Update angle
            new_angle = current + noisy_velocity * self.dt
            new_angle = self.joint.joint_output.clamp_angle(new_angle)

            if self.joint.joint_output.is_at_limit():
                self.angular_velocity = 0.0

        # Save state
        self.joint.joint_output.current_angle    = new_angle
        self.joint.joint_output.angular_velocity = self.angular_velocity
        self.time_elapsed += self.dt

        # Log
        self.time_log.append(self.time_elapsed)
        self.angle_log.append(new_angle)
        self.velocity_log.append(self.angular_velocity)
        self.temp_log.append(self.temperature)
        self.torque_log.append(tau_net)
        self.performance_log.append(self.performance_factor)

        return (new_angle, self.angular_velocity,
                tau_net, self.temperature, self.performance_factor)

    def simulate(self, target_angle: float, duration: float, label: str = ""):
        """Full simulation run."""
        clamped = self.joint.joint_output.clamp_angle(target_angle)
        self.joint.target_angle = clamped
        steps = int(duration / self.dt)

        print(f"\n  Simulating: {label}")
        print(f"  {'Time':>6} | {'Angle':>8} | {'Velocity':>9} | "
              f"{'Temp':>7} | {'Perf':>6} | {'Status':>12}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-"
              f"{'-'*7}-+-{'-'*6}-+-{'-'*12}")

        for i in range(steps):
            angle, vel, torque, temp, perf = self.step()
            t = self.time_elapsed

            if i % 50 == 0 or i == steps - 1:
                if self.is_stalled:
                    status = "STALLED"
                elif self.is_overheated:
                    status = "OVERHEATED"
                elif perf < 0.5:
                    status = "DEGRADED"
                else:
                    status = "normal"

                print(f"  {t:>6.2f}s | {angle:>7.2f}° | "
                      f"{vel:>8.2f}°/s | {temp:>6.1f}C | "
                      f"{perf:>5.2f}x | {status:>12}")

        print(f"\n  Final angle       : "
              f"{self.joint.joint_output.current_angle:.2f}°")
        print(f"  Final temperature : {self.temperature:.1f}°C")
        print(f"  Performance left  : {self.performance_factor:.2f}x")
        print(f"  Stalled           : {self.is_stalled}")
        print(f"  Overheated        : {self.is_overheated}")
        print(f"  Failure events    : {len(self.failure_events)}")

    def reset(self, start_angle: float = 0.0):
        self.joint.joint_output.current_angle    = start_angle
        self.joint.joint_output.angular_velocity = 0.0
        self.joint.target_angle                  = start_angle
        self.angular_velocity  = 0.0
        self.time_elapsed      = 0.0
        self.temperature       = self.ambient_temp
        self.performance_factor = 1.0
        self.is_stalled        = False
        self.is_overheated     = False
        self.time_log.clear()
        self.angle_log.clear()
        self.velocity_log.clear()
        self.temp_log.clear()
        self.torque_log.clear()
        self.performance_log.clear()
        self.failure_events.clear()


# ─────────────────────────────────────────────────────────────
# RUN ALL THREE FAILURE SCENARIOS
# ─────────────────────────────────────────────────────────────
def run_phase6_simulation():
    print("\n" + "="*60)
    print("  PHASE 6 — FAILURE CONDITIONS")
    print("  Stall | Overheating | Inconsistent Motion")
    print("="*60)

    random.seed(42)  # reproducible noise

    # ── FAILURE 1: ACTUATOR STALL ────────────────────────────
    print("\n" + "-"*60)
    print("  FAILURE 1: ACTUATOR STALL")
    print("  Load exceeds motor torque → joint cannot move")
    print("-"*60)
    joint1 = build_knee_joint()
    m1 = FailureConditionsModel(
        joint1,
        enable_stall=True,
        enable_overheating=False,
        enable_noise=False,
        stall_load_mass=3.5,      # heavy enough to stall motor
        stall_load_distance=0.35,
        dt=0.01
    )
    m1.simulate(90.0, 6.0, "Stall test — 3.5kg load")

    # ── FAILURE 2: OVERHEATING ───────────────────────────────
    print("\n" + "-"*60)
    print("  FAILURE 2: OVERHEATING")
    print("  Motor runs hot → performance degrades → shutdown")
    print("-"*60)
    joint2 = build_knee_joint()
    m2 = FailureConditionsModel(
        joint2,
        enable_stall=False,
        enable_overheating=True,
        enable_noise=False,
        stall_load_mass=1.0,
        thermal_resistance=3.5,   # high thermal resistance = heats fast
        critical_temp=80.0,
        ambient_temp=25.0,
        cooling_rate=0.3,
        dt=0.01
    )
    m2.simulate(90.0, 10.0, "Overheating test — high thermal load")

    # ── FAILURE 3: INCONSISTENT MOTION ──────────────────────
    print("\n" + "-"*60)
    print("  FAILURE 3: INCONSISTENT MOTION")
    print("  Noise/jitter in output — worn bearings/encoder error")
    print("-"*60)
    joint3 = build_knee_joint()
    m3 = FailureConditionsModel(
        joint3,
        enable_stall=False,
        enable_overheating=False,
        enable_noise=True,
        noise_amplitude=2.5,      # significant jitter
        dt=0.01
    )
    m3.simulate(90.0, 6.0, "Inconsistent motion — high noise")

    # ── FAILURE 4: ALL FAILURES COMBINED ────────────────────
    print("\n" + "-"*60)
    print("  FAILURE 4: ALL FAILURES COMBINED")
    print("  Worst case — stall + overheat + noise together")
    print("-"*60)
    joint4 = build_knee_joint()
    m4 = FailureConditionsModel(
        joint4,
        enable_stall=True,
        enable_overheating=True,
        enable_noise=True,
        stall_load_mass=2.0,
        thermal_resistance=3.0,
        critical_temp=80.0,
        noise_amplitude=1.5,
        dt=0.01
    )
    m4.simulate(90.0, 10.0, "Combined failure — all modes active")

    save_log(m1, m2, m3, m4)
    plot_results(m1, m2, m3, m4)
    return m1, m2, m3, m4


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_log(m1, m2, m3, m4):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phase6_log.txt"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 6 - FAILURE CONDITIONS LOG\n")
        f.write("Stall | Overheating | Inconsistent Motion\n")
        f.write("="*65 + "\n\n")

        for label, model in [
            ("FAILURE 1: STALL",       m1),
            ("FAILURE 2: OVERHEAT",    m2),
            ("FAILURE 3: NOISE",       m3),
            ("FAILURE 4: COMBINED",    m4),
        ]:
            f.write(f"{label}\n")
            f.write(f"Final angle      : "
                    f"{model.joint.joint_output.current_angle:.2f} deg\n")
            f.write(f"Final temperature: {model.temperature:.1f} C\n")
            f.write(f"Performance left : {model.performance_factor:.2f}\n")
            f.write(f"Stalled          : {model.is_stalled}\n")
            f.write(f"Overheated       : {model.is_overheated}\n")

            if model.failure_events:
                f.write("FAILURE EVENTS:\n")
                for ev in model.failure_events:
                    f.write(f"  {ev}\n")

            f.write(f"{'Time':>8} | {'Angle':>8} | {'Velocity':>10} | "
                    f"{'Temp':>8} | {'Perf':>6}\n")
            f.write("-"*55 + "\n")
            for t, a, v, tmp, p in zip(
                model.time_log, model.angle_log,
                model.velocity_log, model.temp_log,
                model.performance_log
            ):
                f.write(f"{t:>8.3f} | {a:>8.3f} | {v:>10.3f} | "
                        f"{tmp:>8.2f} | {p:>6.3f}\n")
            f.write("\n")

    print(f"\n  [LOG SAVED] phase6_log.txt")


# ─────────────────────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────────────────────
def plot_results(m1, m2, m3, m4):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(
            "Phase 6 - Failure Conditions\n"
            "Stall | Overheating | Inconsistent Motion | Combined",
            fontsize=14, fontweight='bold'
        )
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               hspace=0.55, wspace=0.35)

        # ── Plot 1: Angle vs Time — all failures ─────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axhline(y=90, color='black', linestyle='--',
                    linewidth=1.5, label='Target (90 deg)')
        ax1.plot(m1.time_log, m1.angle_log,
                 'r-', linewidth=2, label='Stall')
        ax1.plot(m2.time_log, m2.angle_log,
                 'orange', linewidth=2, label='Overheat')
        ax1.plot(m3.time_log, m3.angle_log,
                 'b-', linewidth=1.5, label='Noise/Jitter', alpha=0.8)
        ax1.plot(m4.time_log, m4.angle_log,
                 'm-', linewidth=2, label='Combined')
        ax1.set_title("Angle vs Time\n(all failure modes)",
                      fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Plot 2: Temperature vs Time ───────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axhline(y=80, color='red', linestyle='--',
                    linewidth=1.5, label='Critical temp (80C)')
        ax2.plot(m2.time_log, m2.temp_log,
                 'orange', linewidth=2, label='Overheat model')
        ax2.plot(m4.time_log, m4.temp_log,
                 'm-', linewidth=2, label='Combined model')
        ax2.set_title("Temperature vs Time\n(overheating progression)",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Temperature (C)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Performance factor vs Time ────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(m2.time_log, m2.performance_log,
                 'orange', linewidth=2, label='Overheat')
        ax3.plot(m4.time_log, m4.performance_log,
                 'm-', linewidth=2, label='Combined')
        ax3.axhline(y=1.0, color='green', linestyle='--',
                    linewidth=1.5, label='Full performance')
        ax3.axhline(y=0.0, color='red', linestyle='--',
                    linewidth=1.5, label='Shutdown')
        ax3.set_title("Performance Factor vs Time\n"
                      "(1.0=full, 0.0=shutdown)",
                      fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Performance (x)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.1, 1.2)

        # ── Plot 4: Noise comparison ──────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        # Show first 3 seconds for clarity
        steps_3s = int(3.0 / 0.01)
        t3  = m3.time_log[:steps_3s]
        a3  = m3.angle_log[:steps_3s]
        t_clean = [t for t in m1.time_log if t <= 3.0]
        a_clean = m1.angle_log[:len(t_clean)]
        ax4.plot(t_clean, a_clean,
                 'g-', linewidth=2, label='Clean (no noise)')
        ax4.plot(t3, a3,
                 'b-', linewidth=1, label='Noisy output',
                 alpha=0.8)
        ax4.set_title("Inconsistent Motion — Noise Effect\n"
                      "(first 3 seconds)",
                      fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle (deg)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ── Plot 5: Torque vs Time ────────────────────────────
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(m1.time_log, m1.torque_log,
                 'r-', linewidth=2, label='Stall')
        ax5.plot(m2.time_log, m2.torque_log,
                 'orange', linewidth=2, label='Overheat')
        ax5.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.8)
        ax5.set_title("Net Torque vs Time\n"
                      "(negative = stall condition)",
                      fontweight='bold')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Net Torque (N.m)")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # ── Plot 6: Velocity comparison ───────────────────────
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(m1.time_log, m1.velocity_log,
                 'r-', linewidth=2, label='Stall (zero)')
        ax6.plot(m2.time_log, m2.velocity_log,
                 'orange', linewidth=2, label='Overheat (drops)')
        ax6.plot(m3.time_log, m3.velocity_log,
                 'b-', linewidth=1, label='Noise (jitter)',
                 alpha=0.7)
        ax6.set_title("Velocity vs Time\n"
                      "(each failure has unique signature)",
                      fontweight='bold')
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Velocity (deg/s)")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        graph_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase6_graph.png"
        )
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"  [GRAPH SAVED] phase6_graph.png")
        plt.show()

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_phase6_simulation()
