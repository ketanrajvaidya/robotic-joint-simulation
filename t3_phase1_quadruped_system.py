"""
=============================================================
TASK 3 — PHASE 1 : 4-LEG SYSTEM EXPANSION
Task: Full Quadruped Leg Integration + Control-Ready Simulation
=============================================================

What this file does:
  - Takes the Task 2 LegSystem (hip + knee) unchanged
  - Creates 4 identical copies: FL, FR, RL, RR
  - Attaches all 4 legs to a central body reference
  - Defines body offsets for each leg position
  - Builds the QuadrupedSystem — foundation for all Task 3 phases

Quadruped layout (top view):
  
        FL ──────── FR
         |          |
         |   BODY   |
         |          |
        RL ──────── RR

  FL = Front Left
  FR = Front Right
  RL = Rear Left
  RR = Rear Right

Body dimensions:
  - Body length (front to rear) : 0.6m
  - Body width  (left to right) : 0.4m
  - Body origin : center of body at (0, 0)

Leg attachment points (hip pivot positions):
  FL : (-0.2,  0.3)   left side, front
  FR : ( 0.2,  0.3)   right side, front
  RL : (-0.2, -0.3)   left side, rear
  RR : ( 0.2, -0.3)   right side, rear

Design principle:
  - LegSystem from Task 2 is reused EXACTLY — not rewritten
  - Each leg is an independent LegSystem instance
  - QuadrupedSystem assembles them with body offsets
  - All Task 3 phases will import from this file
=============================================================
"""

import sys
import os
import math

# ── Import path setup ────────────────────────────────────────
# Task 3 is in Task3/ folder
# Task 2 files are in Task2/ folder (one level up, then Task2)
# Task 1 files are in Task1/ folder (one level up, then Task1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(BASE_DIR, '..', 'Task1')
TASK2_DIR = os.path.join(BASE_DIR, '..', 'Task2')

sys.path.append(TASK1_DIR)
sys.path.append(TASK2_DIR)
sys.path.append(BASE_DIR)

# Import LegSystem and builder from Task 2 Phase 1
from t2_phase1_system_expansion import (
    LegSystem,
    build_leg_system,
    build_hip_joint
)

# Import Task 1 foundation
from phase1_joint_definition import build_knee_joint


# ─────────────────────────────────────────────────────────────
# LEG POSITION CONSTANTS
# Defines where each leg attaches to the body
# ─────────────────────────────────────────────────────────────
LEG_POSITIONS = {
    "FL": (-0.2,  0.3),   # Front Left
    "FR": ( 0.2,  0.3),   # Front Right
    "RL": (-0.2, -0.3),   # Rear Left
    "RR": ( 0.2, -0.3),   # Rear Right
}

# Crawl gait order (Phase 2 will use this)
CRAWL_GAIT_ORDER = ["FL", "RR", "FR", "RL"]


# ─────────────────────────────────────────────────────────────
# QUADRUPED BODY
# Represents the central body connecting all 4 legs
# ─────────────────────────────────────────────────────────────
class QuadrupedBody:
    def __init__(
        self,
        length: float = 0.6,   # front to rear (m)
        width:  float = 0.4,   # left to right (m)
        mass:   float = 5.0    # body mass (kg)
    ):
        """
        Parameters:
          length : Body length front to rear (m)
          width  : Body width left to right (m)
          mass   : Body mass (kg) — used for CoM calculation
        """
        self.length = length
        self.width  = width
        self.mass   = mass

        # Body center of mass — always at center of body
        self.com_x  = 0.0
        self.com_y  = 0.0

    def status(self) -> dict:
        return {
            "length_m" : self.length,
            "width_m"  : self.width,
            "mass_kg"  : self.mass,
            "com"      : (self.com_x, self.com_y),
        }


# ─────────────────────────────────────────────────────────────
# QUADRUPED SYSTEM
# The main class for Task 3.
# Holds the body + all 4 legs + their positions.
# ─────────────────────────────────────────────────────────────
class QuadrupedSystem:
    def __init__(
        self,
        body:   QuadrupedBody,
        legs:   dict,           # {"FL": LegSystem, "FR": ..., ...}
        positions: dict         # {"FL": (x, y), "FR": ..., ...}
    ):
        """
        Parameters:
          body      : QuadrupedBody — central body reference
          legs      : dict of 4 LegSystem objects keyed by leg name
          positions : dict of (x, y) body attachment points per leg

        All leg names must be: FL, FR, RL, RR
        """
        self.body      = body
        self.legs      = legs        # {"FL": LegSystem, ...}
        self.positions = positions   # {"FL": (x, y), ...}

        # Validate all 4 legs present
        for name in ["FL", "FR", "RL", "RR"]:
            assert name in legs,     f"Missing leg: {name}"
            assert name in positions,f"Missing position: {name}"

        self.name = "Quadruped Robot (4-Leg System)"

    # ─────────────────────────────────────────────
    # GEOMETRY: World-frame foot positions
    # ─────────────────────────────────────────────
    def get_foot_position(self, leg_name: str) -> tuple:
        """
        Returns foot position in WORLD frame (relative to body center).

        Each leg computes its foot position relative to its own
        hip pivot. We add the body offset to get world position.

        Parameters:
          leg_name : "FL", "FR", "RL", or "RR"

        Returns:
          (world_x, world_y) — foot position in world frame
        """
        leg    = self.legs[leg_name]
        offset = self.positions[leg_name]  # (ox, oy) body offset

        # Local foot position from leg's own hip pivot
        local_foot = leg.foot_position()

        # World position = body offset + local foot position
        world_x = offset[0] + local_foot[0]
        world_y = offset[1] + local_foot[1]

        return (round(world_x, 4), round(world_y, 4))

    def get_all_foot_positions(self) -> dict:
        """Returns world-frame foot positions for all 4 legs."""
        return {
            name: self.get_foot_position(name)
            for name in ["FL", "FR", "RL", "RR"]
        }

    def get_hip_world_position(self, leg_name: str) -> tuple:
        """
        Returns the hip pivot position in world frame.
        This is just the body attachment offset for that leg.
        """
        return self.positions[leg_name]

    # ─────────────────────────────────────────────
    # LOAD: Total load per leg
    # ─────────────────────────────────────────────
    def get_leg_load(self, leg_name: str) -> dict:
        """
        Returns the gravitational load at hip and knee
        for a specific leg.
        """
        leg = self.legs[leg_name]
        return {
            "hip_load_Nm"  : leg.hip_gravitational_load(),
            "knee_load_Nm" : leg.knee_gravitational_load(),
            "total_Nm"     : round(
                abs(leg.hip_gravitational_load()) +
                abs(leg.knee_gravitational_load()), 4
            )
        }

    def get_all_loads(self) -> dict:
        """Returns load dict for all 4 legs."""
        return {
            name: self.get_leg_load(name)
            for name in ["FL", "FR", "RL", "RR"]
        }

    # ─────────────────────────────────────────────
    # STATE: Full system snapshot
    # ─────────────────────────────────────────────
    def get_system_state(self) -> dict:
        """
        Returns a complete snapshot of the quadruped system.
        This is what Phase 6 (Control Interface) will output
        every timestep.
        """
        state = {
            "legs"  : {},
            "feet"  : self.get_all_foot_positions(),
            "loads" : self.get_all_loads(),
            "body"  : self.body.status(),
        }

        for name in ["FL", "FR", "RL", "RR"]:
            leg = self.legs[name]
            state["legs"][name] = {
                "hip_angle"  : leg.hip.joint_output.current_angle,
                "knee_angle" : leg.knee.joint_output.current_angle,
                "hip_target" : leg.hip.target_angle,
                "knee_target": leg.knee.target_angle,
                "is_stalled" : leg.hip.actuator.is_stalled or
                               leg.knee.actuator.is_stalled,
            }

        return state

    # ─────────────────────────────────────────────
    # COMMANDS: Set targets on specific legs
    # ─────────────────────────────────────────────
    def set_leg_targets(
        self,
        leg_name:   str,
        hip_angle:  float,
        knee_angle: float
    ):
        """
        Sets target angles for a specific leg.

        Parameters:
          leg_name   : "FL", "FR", "RL", or "RR"
          hip_angle  : Desired hip angle (degrees)
          knee_angle : Desired knee angle (degrees)
        """
        self.legs[leg_name].set_targets(hip_angle, knee_angle)

    def set_all_targets(
        self,
        hip_angle:  float,
        knee_angle: float
    ):
        """
        Sets the same target angles on ALL 4 legs simultaneously.
        Used for standing, crouching, or uniform movements.
        """
        for name in ["FL", "FR", "RL", "RR"]:
            leg = self.legs[name]
            leg.hip.set_target(hip_angle)
            leg.knee.set_target(knee_angle)

    def reset_all(
        self,
        hip_angle:  float = 0.0,
        knee_angle: float = 0.0
    ):
        """
        Resets all 4 legs to specified angles.
        Default is standing position (0°, 0°).
        """
        for name in ["FL", "FR", "RL", "RR"]:
            leg = self.legs[name]
            leg.hip.joint_output.current_angle   = hip_angle
            leg.hip.joint_output.angular_velocity = 0.0
            leg.hip.target_angle                  = hip_angle
            leg.knee.joint_output.current_angle   = knee_angle
            leg.knee.joint_output.angular_velocity = 0.0
            leg.knee.target_angle                  = knee_angle
            leg.hip.actuator.is_stalled            = False
            leg.knee.actuator.is_stalled           = False

    # ─────────────────────────────────────────────
    # DESCRIBE: Full system printout
    # ─────────────────────────────────────────────
    def describe(self):
        """Prints a complete readable summary of the quadruped."""
        print("\n" + "=" * 65)
        print(f"  QUADRUPED SYSTEM: {self.name}")
        print("=" * 65)

        # Body
        print(f"\n  BODY")
        print(f"    Length : {self.body.length}m")
        print(f"    Width  : {self.body.width}m")
        print(f"    Mass   : {self.body.mass}kg")
        print(f"    CoM    : {self.body.com_x, self.body.com_y}")

        # Each leg
        print(f"\n  LEGS")
        print(f"  {'Leg':>4} | {'Attach(x,y)':>14} | "
              f"{'Hip°':>6} | {'Knee°':>6} | "
              f"{'Foot(x,y)':>16} | {'HipLoad':>9} | {'KnLoad':>8}")
        print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*6}-+-{'-'*6}-+"
              f"-{'-'*16}-+-{'-'*9}-+-{'-'*8}")

        for name in ["FL", "FR", "RL", "RR"]:
            leg    = self.legs[name]
            attach = self.positions[name]
            foot   = self.get_foot_position(name)
            load   = self.get_leg_load(name)
            hip_a  = leg.hip.joint_output.current_angle
            kne_a  = leg.knee.joint_output.current_angle

            print(
                f"  {name:>4} | "
                f"({attach[0]:>5.2f},{attach[1]:>5.2f}) | "
                f"{hip_a:>6.1f} | "
                f"{kne_a:>6.1f} | "
                f"({foot[0]:>6.3f},{foot[1]:>6.3f}) | "
                f"{load['hip_load_Nm']:>8.3f}N | "
                f"{load['knee_load_Nm']:>7.3f}N"
            )

        print("=" * 65)


# ─────────────────────────────────────────────────────────────
# BUILDER FUNCTION
# All Task 3 phases call this one function to get a
# ready-to-use QuadrupedSystem — just like Task 2 phases
# called build_leg_system()
# ─────────────────────────────────────────────────────────────
def build_quadruped() -> QuadrupedSystem:
    """
    Builds and returns a complete 4-leg quadruped system.

    Creates:
      - 1 QuadrupedBody (central body)
      - 4 LegSystem instances (FL, FR, RL, RR)
      - Attaches legs to body at correct positions

    Returns:
      QuadrupedSystem ready for simulation
    """
    # Central body
    body = QuadrupedBody(
        length=0.6,
        width=0.4,
        mass=5.0
    )

    # Build 4 independent leg instances
    # Each is a fresh LegSystem — same specs, independent state
    legs = {
        "FL": build_leg_system(),
        "FR": build_leg_system(),
        "RL": build_leg_system(),
        "RR": build_leg_system(),
    }

    # Name each leg for clarity
    for name, leg in legs.items():
        leg.name = f"{name} Leg"

    # Assemble quadruped
    quad = QuadrupedSystem(
        body=body,
        legs=legs,
        positions=LEG_POSITIONS
    )

    return quad


# ─────────────────────────────────────────────────────────────
# DEMO — Run when executed directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  TASK 3 — PHASE 1: 4-LEG SYSTEM EXPANSION")
    print("  Building Full Quadruped System")
    print("=" * 65)

    # Build quadruped
    quad = build_quadruped()

    # ── Test 1: Default standing position ────────────────────
    print("\n--- TEST 1: STANDING POSITION (all legs at 0°, 0°) ---")
    quad.describe()

    # ── Test 2: All legs lifted slightly ─────────────────────
    print("\n--- TEST 2: SLIGHT LIFT (hip=20°, knee=30°) ---")
    quad.set_all_targets(hip_angle=20.0, knee_angle=30.0)
    # Manually set current angles to targets for display
    for name in ["FL", "FR", "RL", "RR"]:
        quad.legs[name].hip.joint_output.current_angle  = 20.0
        quad.legs[name].knee.joint_output.current_angle = 30.0
    quad.describe()

    # ── Test 3: Single leg target ────────────────────────────
    print("\n--- TEST 3: FL LEG LIFTED (hip=45°, knee=60°) ---")
    quad.reset_all(0.0, 0.0)
    quad.legs["FL"].hip.joint_output.current_angle  = 45.0
    quad.legs["FL"].knee.joint_output.current_angle = 60.0
    all_feet = quad.get_all_foot_positions()
    print(f"  All foot positions:")
    for name, pos in all_feet.items():
        print(f"    {name}: {pos}")

    all_loads = quad.get_all_loads()
    print(f"\n  All leg loads:")
    for name, load in all_loads.items():
        print(f"    {name}: hip={load['hip_load_Nm']:.3f}N·m  "
              f"knee={load['knee_load_Nm']:.3f}N·m  "
              f"total={load['total_Nm']:.3f}N·m")

    # ── Test 4: Full system state snapshot ───────────────────
    print("\n--- TEST 4: SYSTEM STATE SNAPSHOT ---")
    quad.reset_all(0.0, 0.0)
    state = quad.get_system_state()
    print(f"  Body CoM   : {state['body']['com']}")
    print(f"  Leg states :")
    for name, ls in state["legs"].items():
        print(f"    {name}: hip={ls['hip_angle']}°  "
              f"knee={ls['knee_angle']}°  "
              f"stalled={ls['is_stalled']}")
    print(f"  Foot positions:")
    for name, pos in state["feet"].items():
        print(f"    {name}: {pos}")

    # ── Test 5: Load table at different angles ───────────────
    print("\n--- TEST 5: LOAD TABLE (hip angle sweep, all legs) ---")
    print(f"  {'Hip°':>5} | {'Knee°':>6} | "
          f"{'FL Total':>10} | {'FR Total':>10} | "
          f"{'RL Total':>10} | {'RR Total':>10}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}"
          f"-+-{'-'*10}-+-{'-'*10}")

    for hip_deg in [0, 15, 30, 45, 60]:
        for name in ["FL", "FR", "RL", "RR"]:
            quad.legs[name].hip.joint_output.current_angle  = hip_deg
            quad.legs[name].knee.joint_output.current_angle = 30.0
        loads = quad.get_all_loads()
        print(
            f"  {hip_deg:>5} | {30:>6} | "
            f"{loads['FL']['total_Nm']:>10.4f} | "
            f"{loads['FR']['total_Nm']:>10.4f} | "
            f"{loads['RL']['total_Nm']:>10.4f} | "
            f"{loads['RR']['total_Nm']:>10.4f}"
        )

    print("\n  [PHASE 1 COMPLETE]")
    print("  QuadrupedSystem ready for Phase 2 (Gait Coordination)\n")
