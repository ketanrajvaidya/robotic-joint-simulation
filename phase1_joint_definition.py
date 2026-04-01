"""
=============================================================
PHASE 1 — JOINT SYSTEM DEFINITION
Task: Actuated Joint System Modeling + Mechatronic Behavior Simulation
=============================================================

This file defines the core building blocks of the robotic joint system:
  - Actuator     : The motor/driver that produces motion
  - JointOutput  : The rotating shaft/output of the joint
  - Link         : The rigid arm attached to the joint output
  - RotationalJoint : The complete joint combining all three components

Design principle: Each component is defined separately, then assembled
into the RotationalJoint — mimicking how a real mechatronic system is built.
"""

import math


# ─────────────────────────────────────────────
# COMPONENT 1: ACTUATOR
# The actuator is the "engine" of the joint.
# It receives a command (desired angle or speed)
# and tries to produce the corresponding motion.
# ─────────────────────────────────────────────
class Actuator:
    def __init__(
        self,
        name: str,
        max_torque: float,      # Maximum torque the actuator can produce (N·m)
        max_speed: float,       # Maximum angular speed (degrees/second)
        gear_ratio: float = 1.0 # Gear ratio (output speed = motor speed / gear_ratio)
    ):
        """
        Actuator parameters:
          name        : Identifier (e.g. "DC Motor 12V")
          max_torque  : Upper torque limit — actuator cannot exceed this (N·m)
          max_speed   : Upper speed limit — actuator cannot move faster (deg/s)
          gear_ratio  : Gearing between motor shaft and joint output.
                        A ratio of 10 means motor turns 10x for every 1x at output.
                        Higher ratio = more torque, less speed.
        """
        self.name = name
        self.max_torque = max_torque        # N·m
        self.max_speed = max_speed          # degrees/second
        self.gear_ratio = gear_ratio

        # Runtime state
        self.current_torque = 0.0           # Torque currently being produced
        self.is_stalled = False             # True if actuator has stalled (Phase 6)
        self.temperature = 25.0            # Starting temperature in °C (Phase 6)

    def get_effective_speed(self) -> float:
        """
        Returns the speed at the output shaft (after gearing).
        Output speed is reduced by gear ratio.
        """
        return self.max_speed / self.gear_ratio

    def status(self) -> dict:
        return {
            "name": self.name,
            "max_torque_Nm": self.max_torque,
            "max_speed_deg_s": self.max_speed,
            "gear_ratio": self.gear_ratio,
            "output_speed_deg_s": self.get_effective_speed(),
            "stalled": self.is_stalled,
            "temperature_C": self.temperature,
        }


# ─────────────────────────────────────────────
# COMPONENT 2: JOINT OUTPUT
# The joint output is the rotating shaft —
# the physical part that changes angle over time.
# It is what the link is attached to.
# ─────────────────────────────────────────────
class JointOutput:
    def __init__(
        self,
        min_angle: float,   # Minimum allowable angle (degrees)
        max_angle: float,   # Maximum allowable angle (degrees)
        initial_angle: float = 0.0  # Starting angle (degrees)
    ):
        """
        JointOutput parameters:
          min_angle     : Hard lower limit — joint physically cannot go below this.
          max_angle     : Hard upper limit — joint physically cannot go above this.
          initial_angle : Where the joint starts (usually 0° = neutral position).

        Range of motion (ROM) = max_angle - min_angle
        For this task: ROM = 0° to 120° = 120° total range.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = initial_angle  # Live angle — updated during simulation
        self.angular_velocity = 0.0         # Current speed (degrees/second)

    @property
    def range_of_motion(self) -> float:
        """Returns the total range of motion in degrees."""
        return self.max_angle - self.min_angle

    def is_at_limit(self) -> str | None:
        """
        Checks if the joint is at either hard stop.
        Returns 'min', 'max', or None.
        """
        if self.current_angle <= self.min_angle:
            return "min"
        if self.current_angle >= self.max_angle:
            return "max"
        return None

    def clamp_angle(self, angle: float) -> float:
        """
        Clamps an angle to the allowed range.
        Used to enforce joint limits (Phase 5).
        """
        return max(self.min_angle, min(self.max_angle, angle))

    def status(self) -> dict:
        return {
            "min_angle_deg": self.min_angle,
            "max_angle_deg": self.max_angle,
            "current_angle_deg": round(self.current_angle, 3),
            "angular_velocity_deg_s": round(self.angular_velocity, 3),
            "range_of_motion_deg": self.range_of_motion,
            "at_limit": self.is_at_limit(),
        }


# ─────────────────────────────────────────────
# COMPONENT 3: LINK
# The link is the rigid arm/rod connected to the
# joint output. It's the "limb" that moves in space.
# Its position depends entirely on the joint angle.
# ─────────────────────────────────────────────
class Link:
    def __init__(
        self,
        name: str,
        length: float,          # Length of the link in meters
        mass: float,            # Mass of the link in kg
        center_of_mass: float   # Distance from joint to link's center of mass (m)
    ):
        """
        Link parameters:
          name            : Identifier (e.g. "Lower leg link")
          length          : How long the arm is (meters). Affects moment arm.
          mass            : How heavy the link is (kg).
                            More mass = more inertia = harder to accelerate.
          center_of_mass  : Where the link's weight is concentrated.
                            Used to calculate gravitational load torque.
        """
        self.name = name
        self.length = length                    # meters
        self.mass = mass                        # kg
        self.center_of_mass = center_of_mass    # meters from joint pivot

    def gravitational_torque(self, joint_angle_deg: float, g: float = 9.81) -> float:
        """
        Calculates the torque due to gravity acting on the link (N·m).
        This is the load the actuator must overcome just to hold position.

        Formula: τ = m × g × CoM × cos(θ)
          - At θ=0° (horizontal), gravity torque is maximum
          - At θ=90° (vertical), gravity torque is zero
        """
        angle_rad = math.radians(joint_angle_deg)
        return self.mass * g * self.center_of_mass * math.cos(angle_rad)

    def tip_position(self, joint_angle_deg: float) -> tuple[float, float]:
        """
        Returns the (x, y) position of the link tip in 2D space.
        Assumes joint pivot is at origin (0, 0).
        Useful for visualization (Phase 7).
        """
        angle_rad = math.radians(joint_angle_deg)
        x = self.length * math.cos(angle_rad)
        y = self.length * math.sin(angle_rad)
        return round(x, 4), round(y, 4)

    def status(self) -> dict:
        return {
            "name": self.name,
            "length_m": self.length,
            "mass_kg": self.mass,
            "center_of_mass_m": self.center_of_mass,
        }


# ─────────────────────────────────────────────
# MAIN ASSEMBLY: ROTATIONAL JOINT
# Combines Actuator + JointOutput + Link into
# one complete mechatronic joint system.
# This is the object that will be simulated in
# all subsequent phases.
# ─────────────────────────────────────────────
class RotationalJoint:
    def __init__(
        self,
        name: str,
        joint_type: str,        # Always "rotational" for this task
        axis: str,              # Axis of rotation: "X", "Y", or "Z"
        actuator: Actuator,
        joint_output: JointOutput,
        link: Link
    ):
        """
        RotationalJoint parameters:
          name         : Joint identifier (e.g. "Knee Joint")
          joint_type   : Type of joint — "rotational" means it spins around an axis
          axis         : Which axis it rotates around (e.g. "Z" for 2D planar motion)
          actuator     : The motor assembly driving this joint
          joint_output : The rotating shaft with angle limits
          link         : The arm/limb attached to the output
        """
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        self.actuator = actuator
        self.joint_output = joint_output
        self.link = link

        # Command target — what angle we're trying to reach
        self.target_angle = joint_output.current_angle

    def set_target(self, angle: float):
        """
        Sets a new target angle for the joint.
        The angle is clamped to the allowed range automatically.
        """
        clamped = self.joint_output.clamp_angle(angle)
        if angle != clamped:
            print(f"  [WARNING] Target {angle}° out of range. Clamped to {clamped}°.")
        self.target_angle = clamped

    def describe(self):
        """Prints a full human-readable system summary."""
        print("=" * 55)
        print(f"  JOINT SYSTEM: {self.name}")
        print("=" * 55)
        print(f"  Type      : {self.joint_type.capitalize()}")
        print(f"  Axis      : {self.axis}-axis")
        print(f"  Target    : {self.target_angle}°")
        print()

        print("  [ ACTUATOR ]")
        for k, v in self.actuator.status().items():
            print(f"    {k:<28}: {v}")
        print()

        print("  [ JOINT OUTPUT ]")
        for k, v in self.joint_output.status().items():
            print(f"    {k:<28}: {v}")
        print()

        print("  [ LINK ]")
        for k, v in self.link.status().items():
            print(f"    {k:<28}: {v}")

        angle = self.joint_output.current_angle
        grav_torque = self.link.gravitational_torque(angle)
        tip_x, tip_y = self.link.tip_position(angle)
        print()
        print(f"  [ COMPUTED AT CURRENT ANGLE ({angle}°) ]")
        print(f"    Gravitational torque      : {grav_torque:.3f} N·m")
        print(f"    Link tip position (x, y)  : ({tip_x} m, {tip_y} m)")
        print("=" * 55)


# ─────────────────────────────────────────────
# SYSTEM INSTANTIATION
# This creates the actual joint object that
# all future phases will import and simulate.
# ─────────────────────────────────────────────
def build_knee_joint() -> RotationalJoint:
    """
    Builds and returns a pre-configured knee joint
    for a robotic leg simulation.
    """
    # Motor: 12V DC servo with gear reduction
    actuator = Actuator(
        name="DC Servo 12V",
        max_torque=8.5,     # 8.5 N·m — enough to support leg weight
        max_speed=180.0,    # 180 deg/s before gearing
        gear_ratio=10.0     # 10:1 gear reduction → 18 deg/s at output
    )

    # Knee output: 0° to 120° range (straight to fully bent)
    joint_output = JointOutput(
        min_angle=0.0,
        max_angle=120.0,
        initial_angle=0.0   # Starts in straight/extended position
    )

    # Lower leg link: 0.4m long, 1.2kg, CoM at 0.2m from knee
    link = Link(
        name="Lower Leg",
        length=0.4,         # 40 cm
        mass=1.2,           # 1.2 kg
        center_of_mass=0.2  # CoM at midpoint of link
    )

    # Assemble
    joint = RotationalJoint(
        name="Knee Joint",
        joint_type="rotational",
        axis="Z",           # Z-axis = motion in the sagittal (forward/back) plane
        actuator=actuator,
        joint_output=joint_output,
        link=link
    )

    return joint


# ─────────────────────────────────────────────
# RUN DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPhase 1 — Joint System Definition\n")

    joint = build_knee_joint()
    joint.describe()

    print("\n--- Testing set_target() ---")
    joint.set_target(90.0)
    print(f"  New target angle: {joint.target_angle}°")

    joint.set_target(150.0)   # Should warn and clamp to 120°
    print(f"  Clamped target:   {joint.target_angle}°")

    print("\n--- Gravitational torque across range ---")
    for deg in [0, 30, 60, 90, 120]:
        torque = joint.link.gravitational_torque(deg)
        tip = joint.link.tip_position(deg)
        print(f"  At {deg:>3}°: torque = {torque:>6.3f} N·m | tip = {tip}")
