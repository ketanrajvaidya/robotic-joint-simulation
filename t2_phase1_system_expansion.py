"""
=============================================================
TASK 2 — PHASE 1 : SYSTEM EXPANSION
Task: Multi-Joint Leg Simulation System
=============================================================

What this file does:
  - Takes the existing Knee Joint from Task 1 (unchanged)
  - Adds a NEW Hip Joint using the exact same classes
  - Assembles both joints into a LegSystem

Leg structure being modeled:
  
  [HIP JOINT]
       |
    (thigh)          ← L1 = 0.5m, 1.5kg
       |
  [KNEE JOINT]
       |
    (shin)           ← L2 = 0.4m, 1.2kg  (same as Task 1)
       |
    (foot)

Design principle:
  - Nothing from Task 1 is rewritten
  - We import Task 1 classes directly
  - Only NEW things are added here: HipJoint + LegSystem
=============================================================
"""

import sys
import os
import math

# ── Import Task 1 foundation ─────────────────────────────────
# All classes (Actuator, JointOutput, Link, RotationalJoint)
# and the build_knee_joint() function come directly from Task 1.
# We do NOT rewrite them — we build ON TOP of them.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phase1_joint_definition import (
    Actuator,
    JointOutput,
    Link,
    RotationalJoint,
    build_knee_joint       # Task 1 knee joint — reused directly
)


# ─────────────────────────────────────────────────────────────
# NEW: BUILD HIP JOINT
# The hip joint is LARGER and STRONGER than the knee joint
# because it must support the weight of the ENTIRE leg
# (thigh + shin + foot) not just the shin.
#
# Hip joint specs (different from knee):
#   Range     : -30° to +90°  (backward tilt to forward lift)
#   Torque    : 12.0 N·m      (stronger — carries more load)
#   Speed     : 150 deg/s     (slightly slower — heavier load)
#   Gear ratio: 12:1          (more reduction = more torque)
#   Link      : Thigh, 0.5m, 1.5kg
# ─────────────────────────────────────────────────────────────
def build_hip_joint() -> RotationalJoint:
    """
    Builds and returns a pre-configured hip joint
    for the robotic leg simulation.

    Hip is the TOP joint — connects torso to thigh.
    It carries the full weight of the leg below it.
    """

    # Hip motor: stronger than knee motor (more torque needed)
    actuator = Actuator(
        name="DC Servo 24V Hip",
        max_torque=12.0,    # 12.0 N·m — must lift entire leg
        max_speed=150.0,    # 150 deg/s before gearing
        gear_ratio=12.0     # 12:1 reduction → 12.5 deg/s at output
    )

    # Hip output:
    #   -30° = leg tilted backward (behind body)
    #     0° = leg straight down (neutral/standing)
    #   +90° = leg fully lifted forward
    joint_output = JointOutput(
        min_angle=-30.0,
        max_angle=90.0,
        initial_angle=0.0   # Starts in neutral/standing position
    )

    # Thigh link: longer and heavier than shin (Task 1 lower leg)
    link = Link(
        name="Thigh",
        length=0.5,         # 50 cm (thigh is longer than shin)
        mass=1.5,           # 1.5 kg (heavier than shin's 1.2 kg)
        center_of_mass=0.25 # CoM at midpoint of thigh
    )

    # Assemble hip joint (same RotationalJoint class as Task 1)
    joint = RotationalJoint(
        name="Hip Joint",
        joint_type="rotational",
        axis="Z",           # Same Z-axis as knee (sagittal plane motion)
        actuator=actuator,
        joint_output=joint_output,
        link=link
    )

    return joint


# ─────────────────────────────────────────────────────────────
# NEW: LEG SYSTEM
# This is the main new class in Task 2.
# It holds BOTH joints together and treats them as ONE system.
#
# Think of it like this:
#   Task 1: You had one engine (knee)
#   Task 2: You now have a car (hip + knee working together)
#
# The LegSystem is what all future Task 2 phases will use.
# ─────────────────────────────────────────────────────────────
class LegSystem:
    def __init__(
        self,
        hip_joint: RotationalJoint,
        knee_joint: RotationalJoint
    ):
        """
        LegSystem parameters:
          hip_joint  : The upper joint (built by build_hip_joint())
          knee_joint : The lower joint (built by build_knee_joint() from Task 1)

        Physical layout:
          - Hip pivot is the origin point (0, 0)
          - Knee pivot position depends on hip angle
          - Foot position depends on BOTH hip and knee angles
        """
        self.hip   = hip_joint
        self.knee  = knee_joint

        # Leg identity
        self.name = "Robotic Leg (Hip + Knee)"

        # Physical constants
        self.L1 = hip_joint.link.length    # Thigh length (m)
        self.L2 = knee_joint.link.length   # Shin length (m)

    # ─────────────────────────────────────────────
    # GEOMETRY: Where are the joints and foot?
    # ─────────────────────────────────────────────
    def hip_position(self) -> tuple:
        """
        Hip pivot is always fixed at the origin.
        Returns (0.0, 0.0) — this is the reference point.
        """
        return (0.0, 0.0)

    def knee_position(self) -> tuple:
        """
        Calculates where the KNEE pivot is in 2D space.
        This depends entirely on the hip angle.

        Formula (basic trigonometry):
          knee_x = L1 × cos(hip_angle)
          knee_y = L1 × sin(hip_angle)

        Example:
          Hip at 0°  → knee is directly below hip (0, -0.5)
          Hip at 90° → knee is in front of hip (0.5, 0)
        """
        hip_angle_rad = math.radians(self.hip.joint_output.current_angle)
        knee_x = self.L1 * math.cos(hip_angle_rad)
        knee_y = self.L1 * math.sin(hip_angle_rad)
        return (round(knee_x, 4), round(knee_y, 4))

    def foot_position(self) -> tuple:
        """
        Calculates where the FOOT is in 2D space.
        This depends on BOTH hip AND knee angles.

        This is basic Forward Kinematics (Phase 2 will expand this).
        For now it gives us the foot position for the describe() output.

        Formula:
          foot_x = knee_x + L2 × cos(hip_angle + knee_angle)
          foot_y = knee_y + L2 × sin(hip_angle + knee_angle)
        """
        hip_angle_rad  = math.radians(self.hip.joint_output.current_angle)
        knee_angle_rad = math.radians(self.knee.joint_output.current_angle)
        combined_angle = hip_angle_rad + knee_angle_rad

        knee_x, knee_y = self.knee_position()
        foot_x = knee_x + self.L2 * math.cos(combined_angle)
        foot_y = knee_y + self.L2 * math.sin(combined_angle)
        return (round(foot_x, 4), round(foot_y, 4))

    def total_leg_length(self) -> float:
        """
        Returns the total leg reach (L1 + L2) in meters.
        This is the maximum distance foot can be from hip.
        """
        return self.L1 + self.L2

    # ─────────────────────────────────────────────
    # LOAD: How much torque does each joint carry?
    # ─────────────────────────────────────────────
    def hip_gravitational_load(self) -> float:
        """
        Calculates total gravitational torque on the HIP.
        Hip must support BOTH thigh and shin weights.

        Two components:
          1. Thigh's own weight (same as Task 1 gravitational_torque)
          2. Shin's weight acting through the thigh as a moment arm
        """
        hip_angle = self.hip.joint_output.current_angle
        hip_angle_rad = math.radians(hip_angle)

        # Component 1: Thigh's own gravitational torque
        thigh_torque = self.hip.link.gravitational_torque(hip_angle)

        # Component 2: Shin (knee link) weight acting at the end of thigh
        # The shin hangs from the knee, which is at distance L1 from hip
        shin_mass    = self.knee.link.mass
        g            = 9.81
        shin_torque  = shin_mass * g * self.L1 * math.cos(hip_angle_rad)

        return round(thigh_torque + shin_torque, 4)

    def knee_gravitational_load(self) -> float:
        """
        Calculates gravitational torque on the KNEE.
        Knee only supports the shin (same as Task 1).

        Uses the exact same gravitational_torque() method from Task 1 Link class.
        """
        knee_angle = self.knee.joint_output.current_angle
        return round(self.knee.link.gravitational_torque(knee_angle), 4)

    # ─────────────────────────────────────────────
    # SET TARGETS: Command both joints
    # ─────────────────────────────────────────────
    def set_targets(self, hip_angle: float, knee_angle: float):
        """
        Sets target angles for BOTH joints simultaneously.
        Each joint handles its own clamping (from Task 1).

        Parameters:
          hip_angle  : Desired hip angle (degrees) — range: -30° to 90°
          knee_angle : Desired knee angle (degrees) — range: 0° to 120°
        """
        print(f"\n  [COMMAND] Hip  target → {hip_angle}°")
        print(f"  [COMMAND] Knee target → {knee_angle}°")
        self.hip.set_target(hip_angle)
        self.knee.set_target(knee_angle)

    # ─────────────────────────────────────────────
    # DESCRIBE: Full system summary
    # ─────────────────────────────────────────────
    def describe(self):
        """
        Prints a complete human-readable summary of the leg system.
        Shows both joints, geometry, and load at current angles.
        """
        print("\n" + "=" * 60)
        print(f"  LEG SYSTEM: {self.name}")
        print("=" * 60)
        print(f"  Total reach (L1 + L2) : {self.total_leg_length():.2f} m")
        print(f"  Thigh length (L1)     : {self.L1:.2f} m")
        print(f"  Shin length  (L2)     : {self.L2:.2f} m")

        # ── Hip Joint ──
        print("\n  ┌─ HIP JOINT ──────────────────────────────────────┐")
        print(f"  │  Type          : {self.hip.joint_type.capitalize()}")
        print(f"  │  Axis          : {self.hip.axis}-axis")
        print(f"  │  Range         : {self.hip.joint_output.min_angle}° to {self.hip.joint_output.max_angle}°")
        print(f"  │  Current angle : {self.hip.joint_output.current_angle}°")
        print(f"  │  Target angle  : {self.hip.target_angle}°")
        print(f"  │  Actuator      : {self.hip.actuator.name}")
        print(f"  │  Max torque    : {self.hip.actuator.max_torque} N·m")
        print(f"  │  Gear ratio    : {self.hip.actuator.gear_ratio}:1")
        print(f"  │  Output speed  : {self.hip.actuator.get_effective_speed():.2f} deg/s")
        print(f"  │  Grav. load    : {self.hip_gravitational_load()} N·m  (thigh + shin)")
        print(f"  └──────────────────────────────────────────────────┘")

        # ── Knee Joint ──
        print("\n  ┌─ KNEE JOINT ─────────────────────────────────────┐")
        print(f"  │  Type          : {self.knee.joint_type.capitalize()}")
        print(f"  │  Axis          : {self.knee.axis}-axis")
        print(f"  │  Range         : {self.knee.joint_output.min_angle}° to {self.knee.joint_output.max_angle}°")
        print(f"  │  Current angle : {self.knee.joint_output.current_angle}°")
        print(f"  │  Target angle  : {self.knee.target_angle}°")
        print(f"  │  Actuator      : {self.knee.actuator.name}")
        print(f"  │  Max torque    : {self.knee.actuator.max_torque} N·m")
        print(f"  │  Gear ratio    : {self.knee.actuator.gear_ratio}:1")
        print(f"  │  Output speed  : {self.knee.actuator.get_effective_speed():.2f} deg/s")
        print(f"  │  Grav. load    : {self.knee_gravitational_load()} N·m  (shin only)")
        print(f"  └──────────────────────────────────────────────────┘")

        # ── Geometry ──
        hip_pos  = self.hip_position()
        knee_pos = self.knee_position()
        foot_pos = self.foot_position()

        print("\n  ┌─ GEOMETRY (current angles) ───────────────────────┐")
        print(f"  │  Hip  pivot  (x, y) : {hip_pos}")
        print(f"  │  Knee pivot  (x, y) : {knee_pos}")
        print(f"  │  Foot tip    (x, y) : {foot_pos}")
        print(f"  └──────────────────────────────────────────────────┘")

        print("=" * 60)


# ─────────────────────────────────────────────────────────────
# BUILDER FUNCTION
# Returns a ready-to-use LegSystem object.
# All Task 2 phases will call this function — just like
# Task 1 phases called build_knee_joint().
# ─────────────────────────────────────────────────────────────
def build_leg_system() -> LegSystem:
    """
    Builds and returns a complete two-joint leg system.
    Combines the Task 1 knee joint with the new hip joint.

    Returns:
      LegSystem object with hip + knee ready for simulation.
    """
    hip_joint  = build_hip_joint()   # New — defined in this file
    knee_joint = build_knee_joint()  # From Task 1 — imported directly

    leg = LegSystem(
        hip_joint=hip_joint,
        knee_joint=knee_joint
    )

    return leg


# ─────────────────────────────────────────────────────────────
# DEMO — Run when file is executed directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  TASK 2 — PHASE 1: SYSTEM EXPANSION")
    print("  Building Hip + Knee Leg System")
    print("="*60)

    # Build the full leg
    leg = build_leg_system()

    # Show full system at default position (both joints at 0°)
    print("\n--- DEFAULT POSITION (both joints at 0°) ---")
    leg.describe()

    # ── Test 1: Standing position ────────────────────────────
    print("\n--- TEST: STANDING POSITION ---")
    print("  Setting hip = 0°, knee = 0° (leg straight down)")
    leg.hip.joint_output.current_angle  = 0.0
    leg.knee.joint_output.current_angle = 0.0
    knee_pos = leg.knee_position()
    foot_pos = leg.foot_position()
    print(f"  Knee position : {knee_pos}")
    print(f"  Foot position : {foot_pos}")
    print(f"  Hip load      : {leg.hip_gravitational_load()} N·m")
    print(f"  Knee load     : {leg.knee_gravitational_load()} N·m")

    # ── Test 2: Leg lifted forward ───────────────────────────
    print("\n--- TEST: LEG LIFTED FORWARD ---")
    print("  Setting hip = 45°, knee = 30°")
    leg.hip.joint_output.current_angle  = 45.0
    leg.knee.joint_output.current_angle = 30.0
    knee_pos = leg.knee_position()
    foot_pos = leg.foot_position()
    print(f"  Knee position : {knee_pos}")
    print(f"  Foot position : {foot_pos}")
    print(f"  Hip load      : {leg.hip_gravitational_load()} N·m")
    print(f"  Knee load     : {leg.knee_gravitational_load()} N·m")

    # ── Test 3: Fully bent knee ──────────────────────────────
    print("\n--- TEST: FULLY BENT KNEE ---")
    print("  Setting hip = 0°, knee = 120°")
    leg.hip.joint_output.current_angle  = 0.0
    leg.knee.joint_output.current_angle = 120.0
    knee_pos = leg.knee_position()
    foot_pos = leg.foot_position()
    print(f"  Knee position : {knee_pos}")
    print(f"  Foot position : {foot_pos}")
    print(f"  Hip load      : {leg.hip_gravitational_load()} N·m")
    print(f"  Knee load     : {leg.knee_gravitational_load()} N·m")

    # ── Test 4: set_targets() with clamping ─────────────────
    print("\n--- TEST: set_targets() WITH OUT-OF-RANGE VALUE ---")
    leg.set_targets(hip_angle=100.0, knee_angle=150.0)
    # Hip max is 90° → should clamp to 90°
    # Knee max is 120° → should clamp to 120°
    print(f"  Hip  target after clamp : {leg.hip.target_angle}°")
    print(f"  Knee target after clamp : {leg.knee.target_angle}°")

    # ── Gravitational load table ─────────────────────────────
    print("\n--- GRAVITATIONAL LOAD TABLE (hip angle sweep) ---")
    print(f"  {'Hip°':>6} | {'Knee°':>6} | {'Hip Load (N·m)':>16} | {'Knee Load (N·m)':>16}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*16}-+-{'-'*16}")
    for hip_deg in [-30, -15, 0, 15, 30, 45, 60, 75, 90]:
        leg.hip.joint_output.current_angle  = hip_deg
        leg.knee.joint_output.current_angle = 60.0   # fixed knee at 60°
        h_load = leg.hip_gravitational_load()
        k_load = leg.knee_gravitational_load()
        print(f"  {hip_deg:>6} | {60:>6} | {h_load:>16.4f} | {k_load:>16.4f}")

    print("\n  [PHASE 1 COMPLETE] LegSystem ready for Phase 2 (Kinematics)\n")
