# REVIEW_PACKET.md
# Task 2 — Multi-Joint Leg Simulation System
# Candidate: Ketan Rajvaidya
# Company: Blackhole Infiverse

---

## DEMO VIDEO

**Link:** https://www.youtube.com/watch?v=I4rN7o3yBPE

**Contents covered in video:**
- Phase 1: System expansion — hip + knee leg assembly
- Phase 2: Forward kinematics and foot trajectory graphs
- Phase 3: Coupled dynamics demonstration
- Phase 4: Coordinated motion sequences (step cycle + crouch)
- Phase 5: Load distribution and torque utilization analysis
- Phase 6: Failure propagation — knee stall, overheat, noise jitter
- Phase 7: Master dashboard walkthrough and system summary

---

## 1. SYSTEM DEFINITION

### What Was Built
A physics-based multi-joint robotic leg simulation consisting of:
- **Hip Joint** — upper joint connecting torso to thigh
- **Knee Joint** — lower joint connecting thigh to shin
- **Full Leg System** — both joints coupled and simulated together

### Leg Structure
```
[HIP JOINT]   — Rotational, Z-axis, range: -30° to +90°
      |
   (thigh)     — L1 = 0.50m, mass = 1.5kg, CoM at 0.25m
      |
[KNEE JOINT]  — Rotational, Z-axis, range: 0° to +120°
      |
   (shin)      — L2 = 0.40m, mass = 1.2kg, CoM at 0.20m
      |
   (foot)      — endpoint tracked in 2D space
```

### Total Leg Reach
- Maximum reach (fully extended): **0.90m**
- Reachable workspace: swept arc from -30° to +90° hip, 0° to 120° knee

### Evolution from Task 1
Task 1 built a single knee joint with actuator modeling, mechanical
response, load interaction, limits, and failure conditions.

Task 2 expands that to a full two-joint leg system with:
- Hip joint added using identical Task 1 classes
- Forward kinematics coupling both joints to foot position
- Coupled dynamics — hip motion affects knee load
- Coordinated motion sequences (step cycle, crouch)
- System-level load distribution analysis
- Failure propagation across both joints

---

## 2. ACTUATOR BEHAVIOR

### Hip Actuator
| Property | Value |
|---|---|
| Model | DC Servo 24V Hip |
| Max Torque | 12.0 N·m |
| Max Speed | 150 deg/s (before gearing) |
| Gear Ratio | 12:1 |
| Output Speed | 12.5 deg/s at joint |
| Response Model | First-order lag (τ = 0.5s) |

### Knee Actuator (from Task 1)
| Property | Value |
|---|---|
| Model | DC Servo 12V |
| Max Torque | 8.5 N·m |
| Max Speed | 180 deg/s (before gearing) |
| Gear Ratio | 10:1 |
| Output Speed | 18.0 deg/s at joint |
| Response Model | First-order lag (τ = 0.5s) |

### Actuator Response Model
Both actuators use the first-order lag system from Task 1 Phase 2:
```
error    = target_angle - current_angle
velocity = error / time_constant
new_angle = current_angle + velocity × dt
```
This ensures:
- No instant jumps to target angle
- Realistic gradual approach
- Speed capped at physical actuator limit

---

## 3. HIP + KNEE INTERACTION (Kinematic Coupling)

### Forward Kinematics
Given hip angle (θ1) and knee angle (θ2), foot position is:
```
knee_x = L1 × cos(θ1)
knee_y = L1 × sin(θ1)

foot_x = knee_x + L2 × cos(θ1 + θ2)
foot_y = knee_y + L2 × sin(θ1 + θ2)
```

### Coupling Effect
The joints are NOT independent:
- Moving the **hip** moves BOTH the knee pivot AND the foot
- Moving the **knee** moves ONLY the foot
- Hip motion changes the gravity angle on the shin →
  knee load changes even when knee does not move

This was demonstrated in Phase 3 (Coupled Dynamics) where
knee target was set to 0° but knee load changed throughout
the simulation as hip moved.

### Load Coupling
```
Hip load  = thigh_torque + shin_weight × L1 × cos(θ_hip)
Knee load = shin_mass × g × CoM × cos(θ_hip + θ_knee)
```
Hip carries both thigh AND shin weight.
Knee carries only shin weight.
Hip load is consistently 3-4x higher than knee load.

---

## 4. LOAD DISTRIBUTION

### Normal Operating Loads (Step Cycle)
| Joint | Peak Load | Peak Utilization | Safety Margin |
|---|---|---|---|
| Hip | ~9.5 N·m | ~79% | ~2.5 N·m |
| Knee | ~2.5 N·m | ~29% | ~6.0 N·m |

### Key Findings
- Hip operates at 79% utilization — working hard but safe
- Knee operates at 29% utilization — significant reserve capacity
- Crouch position creates highest combined load (~12 N·m)
- Safety margins never reach zero in healthy coordinated motion
- Load shifts dynamically during motion — not constant

### Load Shift Points
- During leg lift: hip load decreases as angle approaches 90°
- During extension: knee load drops as combined angle reduces
- During crouch: both joints near peak load simultaneously

---

## 5. LIMITS AND CONSTRAINTS

### Joint Limits
| Joint | Min Angle | Max Angle | Range |
|---|---|---|---|
| Hip | -30° | +90° | 120° |
| Knee | 0° | +120° | 120° |

### Enforcement
- All targets automatically clamped to joint range
- Out-of-range commands log a WARNING and clamp silently
- Limits enforced every simulation step — not just at command time

### Actuator Limits
- Speed capped at output shaft speed (after gear reduction)
- Torque tracked against max_torque — used in utilization calculation
- Temperature tracked (Phase 6) — degrades performance above 80°C

---

## 6. FAILURE BEHAVIOR AND PROPAGATION

### Failure 1 — Knee Stall
**Trigger:** Knee actuator torque cannot overcome load  
**Behavior:** Knee locks at current angle mid-motion  
**Propagation:**
- Hip continues moving with locked knee
- Foot traces wrong path — step cycle breaks
- Hip load increases ~20% (compensating for unbalanced leg)
- System cannot complete intended motion

**Observable in graphs:**
- Knee angle flatlines at stall point
- Foot trajectory deviates from healthy arc
- Hip load spikes after stall

### Failure 2 — Hip Overheat
**Trigger:** Continuous motion raises hip temperature  
**Behavior:** Gradual performance degradation above 80°C  
**Propagation:**
- Hip velocity reduces proportionally to temperature excess
- Motion slows — knee receives wrong load timing
- Coordination between joints breaks down
- Near 110°C — near complete shutdown

**Observable in graphs:**
- Hip angle rises slower than healthy
- Temperature curve crosses 80°C threshold
- Foot trajectory correct shape but compressed/slower

### Failure 3 — Noise and Jitter
**Trigger:** Encoder failure or loose mechanical coupling  
**Behavior:** Random ±3° angle error on both joints every step  
**Propagation:**
- Both joints oscillate randomly around target
- Foot position becomes completely unpredictable
- Load spikes at every jitter event
- Most dangerous failure for ground contact tasks

**Observable in graphs:**
- Hip and knee angle graphs show erratic zigzag
- Foot trajectory is scattered dots instead of smooth arc

---

## 7. PHASE SUMMARY

| Phase | What Was Built | Output Files |
|---|---|---|
| Phase 1 | Hip joint + LegSystem assembly | t2_phase1_system_expansion.py |
| Phase 2 | Forward kinematics engine | t2_phase2_forward_kinematics.py, t2_phase2_log.txt, t2_phase2_graph.png |
| Phase 3 | Coupled dynamics simulator | t2_phase3_coupled_dynamics.py, t2_phase3_log.txt, t2_phase3_graph.png |
| Phase 4 | Coordinated motion sequences | t2_phase4_coordinated_motion.py, t2_phase4_log.txt, t2_phase4_graph.png |
| Phase 5 | Load distribution analysis | t2_phase5_load_distribution.py, t2_phase5_log.txt, t2_phase5_graph.png |
| Phase 6 | Failure propagation simulation | t2_phase6_failure_propagation.py, t2_phase6_log.txt, t2_phase6_graph.png |
| Phase 7 | Master visualization dashboard | t2_phase7_visualization.py, 5 graph PNG files |

---

## 8. ASSUMPTIONS

1. **2D planar motion only** — all motion in the sagittal plane (Z-axis rotation). No lateral or torsional forces modeled.

2. **Rigid links** — thigh and shin are perfectly rigid. No flex, bend, or vibration in the links.

3. **Point foot** — foot is modeled as a single point. No foot geometry, contact surface, or ground reaction force.

4. **Simplified thermal model** — hip temperature rises linearly during motion, cools linearly at rest. Real thermal behavior is more complex.

5. **Open-loop control** — no feedback from actual position to correct errors. Commands are sent and the first-order model approximates response.

6. **Gravity only** — no external forces, no contact forces, no inertial coupling beyond gravitational torque.

7. **Identical time constants** — both joints use τ = 0.5s. Real robots may have different response characteristics per joint.

8. **No joint friction** — friction and backlash not modeled. Real joints have both.

9. **Noise is random uniform** — jitter modeled as uniform random distribution. Real encoder noise is typically Gaussian.

---

## 9. LIMITATIONS

1. **Single leg only** — no quadruped coordination. No ground reaction, no stability analysis.

2. **No inverse kinematics** — cannot specify foot position and compute required joint angles. Only forward direction.

3. **No inertial dynamics** — mass moment of inertia not modeled for acceleration/deceleration. Only gravitational load.

4. **No collision detection** — leg can pass through ground in simulation. No physical constraint on foot position.

5. **No control system interface** — no PID, no trajectory planning, no gait controller. Designed for Rajaryan (Control team) to integrate.

---

## 10. INTEGRATION NOTES

### For Suraj (Mechanical)
- All physical parameters (link lengths, masses, CoM positions) are defined in `t2_phase1_system_expansion.py` in `build_hip_joint()` and `build_knee_joint()`
- Gravitational torque calculations use standard Newton mechanics
- Safety margins and peak loads are logged in `t2_phase5_log.txt`

### For Rajaryan (Control)
- `LegSystem.set_targets(hip, knee)` is the command interface
- `ActuatorResponseModel` in `phase2_actuator_model.py` defines response behavior
- `CoupledDynamicsSimulator.step()` advances simulation one dt step
- Replace open-loop targets with closed-loop PID commands here

### For Dhruv (Data Systems)
- All simulation logs saved as `.txt` files with consistent column format
- Every step dict contains: time, hip_angle, knee_angle, foot_x, foot_y, hip_load_Nm, knee_load_Nm
- Log format uses `|` delimiter — compatible with CSV parsing after replace
- Failure states logged as string fields: `failure_hip`, `failure_knee`, `failure_noise`

---

## 11. HOW TO RUN

```bash
# Run phases in order
python t2_phase1_system_expansion.py
python t2_phase2_forward_kinematics.py
python t2_phase3_coupled_dynamics.py
python t2_phase4_coordinated_motion.py
python t2_phase5_load_distribution.py
python t2_phase6_failure_propagation.py
python t2_phase7_visualization.py
```

### Requirements
```
python >= 3.10
numpy
matplotlib
```

Install:
```bash
python -m pip install numpy matplotlib
```

---

## 12. FILE STRUCTURE

```
robotic joint simulation/
│
├── Task 1 Files (unchanged foundation)
│   ├── phase1_joint_definition.py
│   ├── phase2_actuator_model.py
│   ├── phase3_mechanical_response.py
│   ├── phase4_load_interaction.py
│   ├── phase5_limits_constraints.py
│   └── phase6_failure_conditions.py
│
├── Task 2 Files
│   ├── t2_phase1_system_expansion.py
│   ├── t2_phase2_forward_kinematics.py
│   ├── t2_phase3_coupled_dynamics.py
│   ├── t2_phase4_coordinated_motion.py
│   ├── t2_phase5_load_distribution.py
│   ├── t2_phase6_failure_propagation.py
│   └── t2_phase7_visualization.py
│
├── Logs
│   ├── t2_phase2_log.txt
│   ├── t2_phase3_log.txt
│   ├── t2_phase4_log.txt
│   ├── t2_phase5_log.txt
│   └── t2_phase6_log.txt
│
├── Graphs
│   ├── t2_phase2_graph.png
│   ├── t2_phase3_graph.png
│   ├── t2_phase4_graph.png
│   ├── t2_phase5_graph.png
│   ├── t2_phase6_graph.png
│   ├── t2_phase7_graph1_hip_angle_vs_time.png
│   ├── t2_phase7_graph2_knee_angle_vs_time.png
│   ├── t2_phase7_graph3_foot_trajectory.png
│   ├── t2_phase7_graph4_torque_distribution.png
│   └── t2_phase7_master_dashboard.png
│
└── REVIEW_PACKET.md   ← this file
```

---

*Task 2 — Multi-Joint Leg Simulation System*
*Candidate: Ketan Rajvaidya*
*Blackhole Infiverse Robotics Systems Test*
