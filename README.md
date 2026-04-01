# Actuated Joint System Modeling + Mechatronic Behavior Simulation
**Robotics Systems — Test Task 1**
**Author: Ketan Rajvaidya**

---

## Project Overview

This project builds a complete mechatronic joint simulation for a single rotational joint of a robotic leg. The simulation models realistic actuator behavior — including response delay, mechanical inertia, friction, load interaction, physical limits, and failure conditions. It does NOT assume instant motion or ignore resistance. Every phase adds a new layer of physical realism on top of the previous one.

The system is designed to eventually become part of a multi-joint robotic leg simulation.

---

## Repository Structure

```
robotic-joint-simulation/
│
├── phase1_joint_definition.py        # Joint, actuator, link classes
├── phase2_actuator_model.py          # Response delay simulation
├── phase3_mechanical_response.py     # Inertia + friction
├── phase4_load_interaction.py        # External payload effects
├── phase5_limits_constraints.py      # Hard + soft limit enforcement
├── phase6_failure_conditions.py      # Stall, overheat, noise
├── phase7_master_analysis.py         # Master analysis + all graphs
│
├── logs/
│   ├── phase2_log.txt
│   ├── phase3_log.txt
│   ├── phase4_log.txt
│   ├── phase5_log.txt
│   ├── phase6_log.txt
│   └── phase7_master_log.txt
│
└── graphs/
    ├── phase2_graph.png
    ├── phase3_graph.png
    ├── phase4_graph.png
    ├── phase5_graph.png
    ├── phase6_graph.png
    ├── phase7_graph1_angle_vs_time.png
    ├── phase7_graph2_input_vs_output.png
    ├── phase7_graph3_load_vs_response.png
    ├── phase7_graph4_performance_degradation.png
    └── phase7_graph5_master_dashboard.png
```

---

## How to Run

### Requirements
```bash
python -m pip install numpy matplotlib
```

### Run individual phases
```bash
python phase1_joint_definition.py
python phase2_actuator_model.py
python phase3_mechanical_response.py
python phase4_load_interaction.py
python phase5_limits_constraints.py
python phase6_failure_conditions.py
python phase7_master_analysis.py
```

> All files must be in the same folder. Each phase imports from phase1.

---

## System Definition

### Joint Type
Rotational joint — models the knee joint of a robotic leg. Rotates around the Z-axis (sagittal plane — forward and backward motion).

### Range of Motion
0 degrees (fully extended) to 120 degrees (fully bent).

### Components

| Component | Description | Values |
|---|---|---|
| Actuator | DC Servo 12V motor | Max torque: 8.5 Nm, Max speed: 180 deg/s |
| Gear Reduction | 10:1 gear ratio | Output speed: 18 deg/s |
| Joint Output | Rotating shaft | Range: 0 to 120 degrees |
| Link | Lower leg arm | Length: 0.4m, Mass: 1.2kg, CoM: 0.2m from pivot |

---

## Phase-by-Phase Documentation

### Phase 1 — Joint System Definition
Defines the three core components as Python classes: Actuator, JointOutput, and Link. These are assembled into a RotationalJoint class via a build_knee_joint() factory function that all subsequent phases import. The Actuator class already contains is_stalled and temperature fields, scaffolded for Phase 6. The JointOutput class contains clamp_angle() which enforces limits at the data layer.

### Phase 2 — Actuator Modeling
Implements a first-order lag system to simulate realistic actuator response:
- velocity = (target - current) / time_constant
- new_angle = current_angle + velocity x dt

Time constant of 0.5 seconds models a realistic servo motor response. The joint never jumps to target instantly. Three tests: normal move, direction reversal, and over-limit command clamping.

### Phase 3 — Mechanical Response
Adds Newton's second law for rotation:
- I x alpha = tau_actuator - tau_friction - tau_gravity

Moment of inertia I = m x r^2 causes slow start. Damping coefficient models bearing friction. Three models compared: no friction, light friction (0.05), and heavy friction (0.15).

### Phase 4 — Load Interaction
Adds external payload torque:
- tau_load = mass x g x distance x cos(angle)

Four load levels tested: 0.0 kg, 0.5 kg, 1.5 kg, and 3.0 kg. The 3.0 kg load produces 10.3 Nm of resistance — exceeding the motor's 8.5 Nm maximum. The joint never moved under this load, demonstrating a physics-based stall.

### Phase 5 — Limits and Constraints
Implements a two-layer limit system:

Hard limits — absolute physical stops at 0 and 120 degrees. When hit: velocity zeroed, position locked.

Soft limits — warning zones 5 degrees inside each hard stop. When entered: maximum speed reduced to 30% to prevent slamming.

Actuator limits — torque capped at 8.5 Nm, speed capped at 18 deg/s at all times. All limit events logged with exact timestamp.

### Phase 6 — Failure Conditions
Three failure modes simulated:

Actuator Stall — detected when net torque is negative and velocity is near zero. Motor draws current but produces no motion.

Overheating — temperature rises based on heat = torque^2 x thermal_resistance x dt. Performance degrades linearly from 1.0 toward 0.0 as temperature rises toward 80C.

Inconsistent Motion — Gaussian noise added to velocity output. Models worn bearings or encoder errors. Results in jagged irregular motion.

---

## Actuator Behavior

The actuator is modeled as a first-order system:
- Response delay: 0.5 second time constant
- Gradual approach: velocity proportional to error
- Torque saturation: hard cap at 8.5 Nm
- Speed saturation: hard cap at 18 deg/s after gear reduction
- Thermal derating: torque and speed reduce as temperature rises
- Stall detection: automatic when load exceeds motor capability

---

## Load Interaction

Load torque formula: tau = m x g x d x cos(theta)

- Maximum at 0 degrees (horizontal) — gravity fully opposes motion
- Zero at 90 degrees (vertical) — gravity aligns with joint axis
- Total inertia increases with load: I_total = I_link + I_load

| Load | Max Resistance | Outcome |
|---|---|---|
| 0.0 kg | 0.0 Nm | Reaches target easily |
| 0.5 kg | 1.7 Nm | Reaches target |
| 1.5 kg | 5.1 Nm | Degraded performance |
| 3.0 kg | 10.3 Nm | Complete stall |

---

## Limits and Failure Behavior

### Joint Limits
- Hard min: 0 degrees — absolute stop
- Hard max: 120 degrees — absolute stop
- Soft min: 5 degrees — speed reduced to 30%
- Soft max: 115 degrees — speed reduced to 30%

### Actuator Limits
- Max torque: 8.5 Nm
- Max speed: 18 deg/s

### Failure Signatures

| Failure | Observable Behavior | Graph Signature |
|---|---|---|
| Stall | Angle = 0, velocity = 0 | Flat line at 0 degrees |
| Overheating | Angle rises then falls back | Inverted U shape |
| Noise/Jitter | Reaches target but rough path | Jagged line |

---

## Assumptions

- 2D planar motion only (Z-axis rotation)
- Rigid link — no flex or vibration in the arm
- Point mass approximation for inertia calculation
- Viscous damping only — no static friction modeled
- Simplified thermal model — heat proportional to torque squared
- Single joint — no coupling effects from other joints
- Gear reduction is lossless
- Load attached at fixed 0.35m from pivot

---

## Key Results Summary

| Scenario | Final Angle | Notes |
|---|---|---|
| Ideal (no load) | 91.4 deg | Slight overshoot |
| With friction | 85.6 deg | Friction reduces performance |
| Medium load 1.5kg | 97.9 deg | Slower but reaches target |
| Heavy load 3.0kg | 0.0 deg | Complete stall |
| Overheating | 0.0 deg | Motor degraded to 0.55x |
| Noise/Jitter | 89.7 deg | Rough path but reaches target |

---

## Where Performance Degrades

1. Load above 60% of motor torque — significant slowdown
2. Load above 100% of motor torque — complete stall
3. Temperature above 50C — noticeable torque reduction
4. Temperature at 80C — full shutdown
5. Heavy friction above 0.10 — cannot reach target
6. Soft limit zone — speed reduced 70% intentionally

---

## Dependencies

| Library | Purpose |
|---|---|
| Python 3.10+ | Core language |
| matplotlib | All graph generation |
| numpy | Numerical operations |
| math | Trigonometry (built-in) |
| random | Noise generation (built-in) |

Install: python -m pip install numpy matplotlib

---

## Future Work

This simulation extends into multi-joint coordinated simulation. The next step connects this knee joint to a hip and ankle joint, coordinating motion across all three for a complete robotic leg gait simulation.
