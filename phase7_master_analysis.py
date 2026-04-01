"""
=============================================================
PHASE 7 — ANALYSIS + VISUALIZATION
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

What this phase does:
  - Runs the COMPLETE simulation combining all 6 phases
  - Generates all required plots:
      1. Angle vs Time
      2. Input vs Output (command vs actual)
      3. Load vs Response
      4. Performance degradation
      5. Failure behavior
  - Explains where performance degrades
  - Produces the final deliverable graphs for GitHub

This is the MASTER file — it imports and runs everything
built across Phases 1-6 and produces the complete analysis.

=============================================================
"""

import sys
import os
import math
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import build_knee_joint


# ─────────────────────────────────────────────────────────────
# MASTER SIMULATION ENGINE
# Combines all physics from Phases 1-6 in one clean class
# ─────────────────────────────────────────────────────────────
class MasterSimulation:
    def __init__(
        self,
        load_mass: float = 0.0,
        damping: float = 0.05,
        time_constant: float = 0.5,
        enable_noise: bool = False,
        noise_amplitude: float = 0.5,
        enable_overheat: bool = False,
        thermal_resistance: float = 2.0,
        critical_temp: float = 80.0,
        dt: float = 0.01
    ):
        self.joint          = build_knee_joint()
        self.load_mass      = load_mass
        self.load_distance  = 0.35
        self.damping        = damping
        self.time_constant  = time_constant
        self.enable_noise   = enable_noise
        self.noise_amplitude = noise_amplitude
        self.enable_overheat = enable_overheat
        self.thermal_resistance = thermal_resistance
        self.critical_temp  = critical_temp
        self.ambient_temp   = 25.0
        self.dt             = dt
        self.g              = 9.81

        # Inertia
        link = self.joint.link
        self.inertia = (link.mass * link.center_of_mass ** 2 +
                        load_mass * self.load_distance ** 2)

        # State
        self.velocity       = 0.0
        self.time_elapsed   = 0.0
        self.temperature    = 25.0
        self.performance    = 1.0

        # Logs
        self.times      = []
        self.angles     = []
        self.commands   = []
        self.velocities = []
        self.temps      = []
        self.perfs      = []
        self.torques    = []

    def step(self):
        current = self.joint.joint_output.current_angle
        target  = self.joint.target_angle

        # Actuator torque — degraded by temperature
        error      = target - current
        raw_torque = (error / self.time_constant) * self.inertia
        eff_max    = self.joint.actuator.max_torque * self.performance
        tau_act    = max(-eff_max, min(eff_max, raw_torque))

        # Friction, gravity, load
        tau_fric = self.damping * self.velocity
        tau_grav = self.joint.link.gravitational_torque(current)
        rad      = math.radians(current)
        tau_load = (self.load_mass * self.g *
                    self.load_distance * math.cos(rad))
        tau_net  = tau_act - tau_fric - tau_grav - tau_load

        # Overheating
        if self.enable_overheat:
            heat = (abs(tau_act) ** 2) * self.thermal_resistance * self.dt * 0.1
            cool = 0.3 * (self.temperature - self.ambient_temp) * self.dt
            self.temperature = max(self.ambient_temp,
                                   self.temperature + heat - cool)
            temp_range = self.critical_temp - self.ambient_temp
            self.performance = max(0.0, 1.0 - (
                (self.temperature - self.ambient_temp) / temp_range))

        # Physics
        alpha     = tau_net / max(self.inertia, 0.001)
        self.velocity += alpha * self.dt
        max_speed = self.joint.actuator.get_effective_speed() * self.performance
        max_speed = max(1.0, max_speed)
        self.velocity = max(-max_speed, min(max_speed, self.velocity))

        # Noise
        if self.enable_noise:
            self.velocity += random.gauss(0, self.noise_amplitude * 0.1)

        # Update angle
        new_angle = current + self.velocity * self.dt
        new_angle = self.joint.joint_output.clamp_angle(new_angle)
        if self.joint.joint_output.is_at_limit():
            self.velocity = 0.0

        self.joint.joint_output.current_angle = new_angle
        self.time_elapsed += self.dt

        # Log
        self.times.append(self.time_elapsed)
        self.angles.append(new_angle)
        self.commands.append(target)
        self.velocities.append(self.velocity)
        self.temps.append(self.temperature)
        self.perfs.append(self.performance)
        self.torques.append(tau_net)

        return new_angle

    def run(self, target: float, duration: float):
        clamped = self.joint.joint_output.clamp_angle(target)
        self.joint.target_angle = clamped
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
        return self


# ─────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS FOR PHASE 7 ANALYSIS
# ─────────────────────────────────────────────────────────────
def run_all_scenarios():
    print("\n" + "="*60)
    print("  PHASE 7 — MASTER ANALYSIS + VISUALIZATION")
    print("  Complete System Behavior Summary")
    print("="*60)

    random.seed(42)
    TARGET   = 90.0
    DURATION = 8.0

    # ── Scenario 1: Ideal (no load, no failures) ────────────
    print("\n  Running Scenario 1: Ideal baseline...")
    s1 = MasterSimulation(load_mass=0.0, damping=0.05)
    s1.run(TARGET, DURATION)

    # ── Scenario 2: With friction ────────────────────────────
    print("  Running Scenario 2: With friction...")
    s2 = MasterSimulation(load_mass=0.0, damping=0.15)
    s2.run(TARGET, DURATION)

    # ── Scenario 3: Medium load ──────────────────────────────
    print("  Running Scenario 3: Medium load (1.5kg)...")
    s3 = MasterSimulation(load_mass=1.5, damping=0.05)
    s3.run(TARGET, DURATION)

    # ── Scenario 4: Heavy load ───────────────────────────────
    print("  Running Scenario 4: Heavy load (3.0kg)...")
    s4 = MasterSimulation(load_mass=3.0, damping=0.05)
    s4.run(TARGET, DURATION)

    # ── Scenario 5: Overheating ──────────────────────────────
    print("  Running Scenario 5: Overheating...")
    s5 = MasterSimulation(
        load_mass=1.0,
        enable_overheat=True,
        thermal_resistance=3.5,
        critical_temp=80.0
    )
    s5.run(TARGET, DURATION)

    # ── Scenario 6: Noise/Jitter ────────────────────────────
    print("  Running Scenario 6: Noise/Jitter...")
    s6 = MasterSimulation(
        load_mass=0.0,
        enable_noise=True,
        noise_amplitude=2.0
    )
    s6.run(TARGET, DURATION)

    print("\n  All scenarios complete. Generating analysis...")

    # Print summary table
    print("\n" + "="*70)
    print("  COMPLETE SYSTEM ANALYSIS SUMMARY")
    print("="*70)
    print(f"  {'Scenario':<28} | {'Final Angle':>11} | "
          f"{'Final Temp':>10} | {'Performance':>11}")
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*10}-+-{'-'*11}")

    for label, sim in [
        ("Ideal (no load)",          s1),
        ("With friction",             s2),
        ("Medium load (1.5kg)",       s3),
        ("Heavy load (3.0kg)",        s4),
        ("Overheating (1.0kg load)",  s5),
        ("Noise/Jitter",              s6),
    ]:
        fa   = sim.joint.joint_output.current_angle
        temp = sim.temperature
        perf = sim.performance
        print(f"  {label:<28} | {fa:>10.2f}° | "
              f"{temp:>9.1f}°C | {perf:>10.2f}x")

    return s1, s2, s3, s4, s5, s6


# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────
def save_master_log(scenarios):
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "phase7_master_log.txt"
    )
    labels = [
        "Ideal baseline",
        "With friction",
        "Medium load 1.5kg",
        "Heavy load 3.0kg",
        "Overheating",
        "Noise/Jitter"
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("PHASE 7 - MASTER ANALYSIS LOG\n")
        f.write("Complete System Behavior — All Scenarios\n")
        f.write("="*65 + "\n\n")

        for label, sim in zip(labels, scenarios):
            f.write(f"SCENARIO: {label}\n")
            f.write(f"Final angle      : "
                    f"{sim.joint.joint_output.current_angle:.3f} deg\n")
            f.write(f"Final temperature: {sim.temperature:.1f} C\n")
            f.write(f"Final performance: {sim.performance:.3f}x\n")
            f.write(f"{'Time':>8} | {'Command':>8} | "
                    f"{'Actual':>8} | {'Error':>8} | "
                    f"{'Velocity':>10}\n")
            f.write("-"*55 + "\n")
            for t, cmd, ang, vel in zip(
                sim.times, sim.commands,
                sim.angles, sim.velocities
            ):
                f.write(
                    f"{t:>8.3f} | {cmd:>8.3f} | "
                    f"{ang:>8.3f} | {abs(cmd-ang):>8.3f} | "
                    f"{vel:>10.3f}\n"
                )
            f.write("\n")

    print(f"\n  [LOG SAVED] phase7_master_log.txt")
    return log_path


# ─────────────────────────────────────────────────────────────
# MASTER PLOTS — ALL 5 REQUIRED BY PDF
# ─────────────────────────────────────────────────────────────
def generate_master_plots(s1, s2, s3, s4, s5, s6):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        colors = {
            'ideal':    '#2ecc71',
            'friction': '#3498db',
            'med_load': '#f39c12',
            'hvy_load': '#e74c3c',
            'overheat': '#9b59b6',
            'noise':    '#1abc9c',
        }

        # ════════════════════════════════════════════════════
        # FIGURE 1: ANGLE VS TIME (PDF requirement 1)
        # ════════════════════════════════════════════════════
        fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10))
        fig1.suptitle(
            "Phase 7 — Graph 1: Angle vs Time\n"
            "Complete System Response Under All Conditions",
            fontsize=13, fontweight='bold'
        )

        ax = axes1[0]
        ax.axhline(y=90, color='black', linestyle='--',
                   linewidth=1.5, label='Target (90 deg)', alpha=0.7)
        ax.plot(s1.times, s1.angles, color=colors['ideal'],
                linewidth=2.5, label='Ideal (no load)')
        ax.plot(s2.times, s2.angles, color=colors['friction'],
                linewidth=2, label='With friction')
        ax.plot(s3.times, s3.angles, color=colors['med_load'],
                linewidth=2, label='Medium load 1.5kg')
        ax.plot(s4.times, s4.angles, color=colors['hvy_load'],
                linewidth=2, label='Heavy load 3.0kg')
        ax.set_title("Angle vs Time — Normal Conditions",
                     fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 110)

        ax2 = axes1[1]
        ax2.axhline(y=90, color='black', linestyle='--',
                    linewidth=1.5, label='Target (90 deg)', alpha=0.7)
        ax2.plot(s1.times, s1.angles, color=colors['ideal'],
                 linewidth=2.5, label='Ideal baseline')
        ax2.plot(s5.times, s5.angles, color=colors['overheat'],
                 linewidth=2, label='Overheating')
        ax2.plot(s6.times, s6.angles, color=colors['noise'],
                 linewidth=1.5, label='Noise/Jitter', alpha=0.85)
        ax2.set_title("Angle vs Time — Failure Conditions",
                      fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angle (deg)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path1 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase7_graph1_angle_vs_time.png"
        )
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] phase7_graph1_angle_vs_time.png")

        # ════════════════════════════════════════════════════
        # FIGURE 2: INPUT VS OUTPUT (PDF requirement 2)
        # ════════════════════════════════════════════════════
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle(
            "Phase 7 — Graph 2: Input Command vs Actual Output\n"
            "Tracking Error Analysis",
            fontsize=13, fontweight='bold'
        )

        for ax, sim, label, color in [
            (axes2[0], s1, "Ideal — Input vs Output", colors['ideal']),
            (axes2[1], s3, "Medium Load — Input vs Output",
             colors['med_load']),
        ]:
            ax.plot(sim.times, sim.commands, 'b--',
                    linewidth=1.5, label='Input command')
            ax.plot(sim.times, sim.angles, color=color,
                    linewidth=2, label='Actual output')
            error = [abs(c - a) for c, a in
                     zip(sim.commands, sim.angles)]
            ax2b = ax.twinx()
            ax2b.fill_between(sim.times, error, alpha=0.15,
                               color='red', label='Tracking error')
            ax2b.set_ylabel("Error (deg)", color='red')
            ax2b.tick_params(axis='y', labelcolor='red')
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle (deg)")
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase7_graph2_input_vs_output.png"
        )
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] phase7_graph2_input_vs_output.png")

        # ════════════════════════════════════════════════════
        # FIGURE 3: LOAD VS RESPONSE (PDF requirement 3)
        # ════════════════════════════════════════════════════
        fig3, axes3 = plt.subplots(1, 3, figsize=(16, 6))
        fig3.suptitle(
            "Phase 7 — Graph 3: Load vs Response\n"
            "Effect of Payload on Joint Performance",
            fontsize=13, fontweight='bold'
        )

        load_sims   = [s1, s2, s3, s4]
        load_labels = ['0.0kg', '0.0kg+friction',
                       '1.5kg', '3.0kg']
        load_colors = [colors['ideal'], colors['friction'],
                       colors['med_load'], colors['hvy_load']]

        # Plot A: Angle vs Time for all loads
        ax_a = axes3[0]
        ax_a.axhline(y=90, color='black', linestyle='--',
                     linewidth=1.5, label='Target')
        for sim, lbl, col in zip(load_sims, load_labels, load_colors):
            ax_a.plot(sim.times, sim.angles, color=col,
                      linewidth=2, label=lbl)
        ax_a.set_title("Angle vs Time\nby load level",
                       fontweight='bold')
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Angle (deg)")
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

        # Plot B: Final angle vs load (bar chart)
        ax_b = axes3[1]
        load_vals   = [0.0, 0.0, 1.5, 3.0]
        final_angles = [s.joint.joint_output.current_angle
                        for s in load_sims]
        bars = ax_b.bar(load_labels, final_angles,
                        color=load_colors, alpha=0.85,
                        edgecolor='black')
        ax_b.axhline(y=90, color='black', linestyle='--',
                     linewidth=1.5, label='Target (90 deg)')
        for bar, angle in zip(bars, final_angles):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{angle:.1f}",
                ha='center', va='bottom', fontsize=9
            )
        ax_b.set_title("Final Angle vs Load\n(performance summary)",
                       fontweight='bold')
        ax_b.set_xlabel("Load configuration")
        ax_b.set_ylabel("Final angle (deg)")
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.3, axis='y')

        # Plot C: Velocity vs Time for all loads
        ax_c = axes3[2]
        for sim, lbl, col in zip(load_sims, load_labels, load_colors):
            ax_c.plot(sim.times, sim.velocities, color=col,
                      linewidth=2, label=lbl)
        ax_c.set_title("Velocity vs Time\nload reduces speed",
                       fontweight='bold')
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("Velocity (deg/s)")
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)

        plt.tight_layout()
        path3 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase7_graph3_load_vs_response.png"
        )
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] phase7_graph3_load_vs_response.png")

        # ════════════════════════════════════════════════════
        # FIGURE 4: PERFORMANCE DEGRADATION ANALYSIS
        # ════════════════════════════════════════════════════
        fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
        fig4.suptitle(
            "Phase 7 — Graph 4: Performance Degradation Analysis\n"
            "Where and Why Performance Drops",
            fontsize=13, fontweight='bold'
        )

        # Tracking error over time
        ax41 = axes4[0, 0]
        for sim, lbl, col in zip(load_sims, load_labels, load_colors):
            error = [abs(c - a) for c, a in
                     zip(sim.commands, sim.angles)]
            ax41.plot(sim.times, error, color=col,
                      linewidth=2, label=lbl)
        ax41.set_title("Tracking Error vs Time",
                       fontweight='bold')
        ax41.set_xlabel("Time (s)")
        ax41.set_ylabel("Error (deg)")
        ax41.legend(fontsize=8)
        ax41.grid(True, alpha=0.3)

        # Temperature rise
        ax42 = axes4[0, 1]
        ax42.axhline(y=80, color='red', linestyle='--',
                     linewidth=1.5, label='Critical (80C)')
        ax42.plot(s5.times, s5.temps,
                  color=colors['overheat'], linewidth=2,
                  label='Overheat scenario')
        ax42.fill_between(s5.times, s5.temps,
                           alpha=0.2, color=colors['overheat'])
        ax42.set_title("Temperature Rise Over Time",
                       fontweight='bold')
        ax42.set_xlabel("Time (s)")
        ax42.set_ylabel("Temperature (C)")
        ax42.legend(fontsize=8)
        ax42.grid(True, alpha=0.3)

        # Performance factor
        ax43 = axes4[1, 0]
        ax43.plot(s5.times, s5.perfs,
                  color=colors['overheat'], linewidth=2,
                  label='Performance (overheat)')
        ax43.axhline(y=1.0, color='green', linestyle='--',
                     linewidth=1.5, label='Full (1.0x)')
        ax43.fill_between(s5.times, s5.perfs, 1.0,
                           alpha=0.15, color='red',
                           label='Lost performance')
        ax43.set_title("Motor Performance Factor\n"
                       "(thermal derating)",
                       fontweight='bold')
        ax43.set_xlabel("Time (s)")
        ax43.set_ylabel("Performance (x)")
        ax43.legend(fontsize=8)
        ax43.grid(True, alpha=0.3)
        ax43.set_ylim(0, 1.2)

        # Net torque comparison
        ax44 = axes4[1, 1]
        ax44.axhline(y=0, color='black', linewidth=0.8)
        for sim, lbl, col in zip(load_sims, load_labels, load_colors):
            ax44.plot(sim.times, sim.torques, color=col,
                      linewidth=2, label=lbl, alpha=0.85)
        ax44.set_title("Net Torque vs Time\n"
                       "(positive=accelerating, negative=decelerating)",
                       fontweight='bold')
        ax44.set_xlabel("Time (s)")
        ax44.set_ylabel("Net Torque (N.m)")
        ax44.legend(fontsize=8)
        ax44.grid(True, alpha=0.3)

        plt.tight_layout()
        path4 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase7_graph4_performance_degradation.png"
        )
        plt.savefig(path4, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] phase7_graph4_performance_degradation.png")

        # ════════════════════════════════════════════════════
        # FIGURE 5: MASTER SUMMARY DASHBOARD
        # ════════════════════════════════════════════════════
        fig5 = plt.figure(figsize=(16, 12))
        fig5.suptitle(
            "Phase 7 — Master Summary Dashboard\n"
            "Actuated Joint System — Complete Behavior Analysis",
            fontsize=14, fontweight='bold'
        )
        gs5 = gridspec.GridSpec(3, 3, figure=fig5,
                                hspace=0.55, wspace=0.38)

        # 1. All scenarios angle
        ax51 = fig5.add_subplot(gs5[0, :2])
        ax51.axhline(y=90, color='black', linestyle='--',
                     linewidth=1.5, alpha=0.6)
        for sim, lbl, col in [
            (s1, 'Ideal',       colors['ideal']),
            (s2, 'Friction',    colors['friction']),
            (s3, 'Load 1.5kg',  colors['med_load']),
            (s4, 'Load 3.0kg',  colors['hvy_load']),
            (s5, 'Overheat',    colors['overheat']),
            (s6, 'Noise',       colors['noise']),
        ]:
            ax51.plot(sim.times, sim.angles, color=col,
                      linewidth=2, label=lbl)
        ax51.set_title("All Scenarios — Angle vs Time",
                       fontweight='bold')
        ax51.set_xlabel("Time (s)")
        ax51.set_ylabel("Angle (deg)")
        ax51.legend(fontsize=8, ncol=3)
        ax51.grid(True, alpha=0.3)

        # 2. Final angle summary bar
        ax52 = fig5.add_subplot(gs5[0, 2])
        sim_list = [s1, s2, s3, s4, s5, s6]
        lbl_list = ['Ideal', 'Friction',
                    '1.5kg', '3.0kg', 'Overheat', 'Noise']
        col_list = list(colors.values())
        fa_list  = [s.joint.joint_output.current_angle
                    for s in sim_list]
        bars = ax52.barh(lbl_list, fa_list, color=col_list,
                         alpha=0.85, edgecolor='black')
        ax52.axvline(x=90, color='black', linestyle='--',
                     linewidth=1.5)
        ax52.set_title("Final Angle\nSummary",
                       fontweight='bold')
        ax52.set_xlabel("Final angle (deg)")
        ax52.grid(True, alpha=0.3, axis='x')

        # 3. Input vs output (ideal)
        ax53 = fig5.add_subplot(gs5[1, 0])
        ax53.plot(s1.times, s1.commands, 'b--',
                  linewidth=1.5, label='Command')
        ax53.plot(s1.times, s1.angles,
                  color=colors['ideal'], linewidth=2,
                  label='Actual')
        ax53.set_title("Input vs Output\n(ideal)",
                       fontweight='bold')
        ax53.set_xlabel("Time (s)")
        ax53.set_ylabel("Angle (deg)")
        ax53.legend(fontsize=8)
        ax53.grid(True, alpha=0.3)

        # 4. Input vs output (loaded)
        ax54 = fig5.add_subplot(gs5[1, 1])
        ax54.plot(s3.times, s3.commands, 'b--',
                  linewidth=1.5, label='Command')
        ax54.plot(s3.times, s3.angles,
                  color=colors['med_load'], linewidth=2,
                  label='Actual (1.5kg)')
        ax54.set_title("Input vs Output\n(1.5kg load)",
                       fontweight='bold')
        ax54.set_xlabel("Time (s)")
        ax54.set_ylabel("Angle (deg)")
        ax54.legend(fontsize=8)
        ax54.grid(True, alpha=0.3)

        # 5. Temperature
        ax55 = fig5.add_subplot(gs5[1, 2])
        ax55.axhline(y=80, color='red', linestyle='--',
                     linewidth=1.5, label='Critical')
        ax55.plot(s5.times, s5.temps,
                  color=colors['overheat'], linewidth=2,
                  label='Temperature')
        ax55.set_title("Temperature Rise",
                       fontweight='bold')
        ax55.set_xlabel("Time (s)")
        ax55.set_ylabel("Temp (C)")
        ax55.legend(fontsize=8)
        ax55.grid(True, alpha=0.3)

        # 6. Load vs response
        ax56 = fig5.add_subplot(gs5[2, 0])
        for sim, lbl, col in zip(
            [s1, s3, s4],
            ['0kg', '1.5kg', '3.0kg'],
            [colors['ideal'], colors['med_load'], colors['hvy_load']]
        ):
            ax56.plot(sim.times, sim.velocities,
                      color=col, linewidth=2, label=lbl)
        ax56.set_title("Load vs Response\n(velocity)",
                       fontweight='bold')
        ax56.set_xlabel("Time (s)")
        ax56.set_ylabel("Velocity (deg/s)")
        ax56.legend(fontsize=8)
        ax56.grid(True, alpha=0.3)

        # 7. Noise comparison
        ax57 = fig5.add_subplot(gs5[2, 1])
        steps_4s = int(4.0 / 0.01)
        ax57.plot(s1.times[:steps_4s], s1.angles[:steps_4s],
                  color=colors['ideal'], linewidth=2,
                  label='Clean')
        ax57.plot(s6.times[:steps_4s], s6.angles[:steps_4s],
                  color=colors['noise'], linewidth=1.5,
                  label='Noisy', alpha=0.85)
        ax57.set_title("Clean vs Noisy\nmotion",
                       fontweight='bold')
        ax57.set_xlabel("Time (s)")
        ax57.set_ylabel("Angle (deg)")
        ax57.legend(fontsize=8)
        ax57.grid(True, alpha=0.3)

        # 8. Performance factor
        ax58 = fig5.add_subplot(gs5[2, 2])
        ax58.plot(s5.times, s5.perfs,
                  color=colors['overheat'], linewidth=2,
                  label='Performance')
        ax58.axhline(y=1.0, color='green', linestyle='--',
                     linewidth=1.5, label='Full (1.0x)')
        ax58.set_title("Performance\nDegradation",
                       fontweight='bold')
        ax58.set_xlabel("Time (s)")
        ax58.set_ylabel("Performance (x)")
        ax58.legend(fontsize=8)
        ax58.set_ylim(0, 1.2)
        ax58.grid(True, alpha=0.3)

        path5 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "phase7_graph5_master_dashboard.png"
        )
        plt.savefig(path5, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] phase7_graph5_master_dashboard.png")
        plt.show()

        return [path1, path2, path3, path4, path5]

    except ImportError:
        print("\n  [INFO] matplotlib not installed.")
        print("  Run: pip install matplotlib")
        return []


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run all scenarios
    s1, s2, s3, s4, s5, s6 = run_all_scenarios()

    # Save master log
    save_master_log([s1, s2, s3, s4, s5, s6])

    # Generate all graphs
    print("\n  Generating all Phase 7 graphs...")
    generate_master_plots(s1, s2, s3, s4, s5, s6)

    print("\n" + "="*60)
    print("  PHASE 7 COMPLETE")
    print("  All graphs saved. Simulation complete.")
    print("="*60)
    print("\n  FILES GENERATED:")
    print("  phase7_master_log.txt")
    print("  phase7_graph1_angle_vs_time.png")
    print("  phase7_graph2_input_vs_output.png")
    print("  phase7_graph3_load_vs_response.png")
    print("  phase7_graph4_performance_degradation.png")
    print("  phase7_graph5_master_dashboard.png")
