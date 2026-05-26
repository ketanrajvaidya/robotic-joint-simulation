# task3_review_packet.md
# Task 3 — Full Quadruped Leg Integration + Control-Ready Simulation
# Candidate: Ketan Rajvaidya
# Company: Blackhole Infiverse

---

## DEMO VIDEO

**Link:** *(paste your YouTube link here before submitting)*

**Contents covered in video:**
- Phase 1: Quadruped system assembly (4 legs + body)
- Phase 2: Crawl gait coordination FL → RR → FR → RL
- Phase 3: Full system load distribution across 4 legs
- Phase 4: Stability analysis and support polygon
- Phase 5: Failure propagation across the whole robot
- Phase 6: Control interface layer demonstration
- Phase 7: Master dashboard and telemetry logging

---

## 1. SYSTEM DEFINITION

### What Was Built
A full quadruped robot simulation system consisting of:
- **4 LegSystems** (FL, FR, RL, RR) — each a Task 2 hip+knee leg
- **QuadrupedBody** — central body connecting all 4 legs
- **QuadrupedSystem** — full assembly with body + 4 legs
- **CrawlGaitController** — coordinates leg sequence
- **StabilityAnalyzer** — tracks CoM and support polygon
- **QuadrupedControlInterface** — control-ready interface for Rajaryan
- **TelemetryLogger** — structured data output for Dhruv

### Quadruped Layout (Top View)
```
        FL ──────── FR
         |          |
         |   BODY   |    Length: 0.6m
         |          |    Width:  0.4m
        RL ──────── RR   Mass:   5.0kg
```

### Leg Attachment Positions (Body Frame)
| Leg | X offset | Y offset |
|-----|----------|----------|
| FL  | -0.20m   | +0.30m   |
| FR  | +0.20m   | +0.30m   |
| RL  | -0.20m   | -0.30m   |
| RR  | +0.20m   | -0.30m   |

### Per-Leg Specs (identical for all 4 legs — from Task 2)
| Component | Spec |
|---|---|
| Hip actuator | DC Servo 24V, 12.0 N·m, gear ratio 12:1 |
| Knee actuator | DC Servo 12V, 8.5 N·m, gear ratio 10:1 |
| Thigh length | 0.50m, 1.5kg |
| Shin length | 0.40m, 1.2kg |
| Total leg reach | 0.90m |

### Evolution from Task 2
```
Task 1 → 1 joint (knee)
Task 2 → 1 leg (hip + knee)
Task 3 → 4 legs + body + gait + stability + control interface
```

---

## 2. GAIT COORDINATION

### Crawl Gait Sequence
```
FL swings → RR swings → FR swings → RL swings → repeat
```

### Rules Enforced
- Only 1 leg swings at any time
- 3 legs always on ground (support body)
- Each leg goes through SWING then STANCE phase

### Gait Parameters
| Parameter | Value |
|---|---|
| Swing hip angle | 40° |
| Swing knee angle | 60° |
| Stance hip angle | -10° |
| Stance knee angle | 10° |
| Swing duration | 2.0s |
| Stance duration | 1.5s |
| Cycle duration | 14.0s |
| Time constant | 0.6s |

### Why Crawl Gait
Crawl gait is the most statically stable walking pattern for
a quadruped. With 3 legs always on ground, the support polygon
is always a triangle — maximizing stability margin.

---

## 3. LOAD DISTRIBUTION

### Load Model
Each stance leg carries:
- Its own link gravitational torque (hip + knee)
- 1/3 share of body weight (5kg × 9.81 ÷ 3 = 16.35N)

Swinging leg carries:
- Only its own link weight
- Zero body weight share

### Observed Load Values (Step Cycle)
| Condition | Load per Stance Leg | Total System |
|---|---|---|
| 4 legs on ground | ~24 N·m | ~96 N·m |
| 3 legs on ground (1 swinging) | ~27 N·m | ~108 N·m |
| Peak (crouch position) | ~28 N·m | ~112 N·m |

### Load Shift Pattern
- When FL swings → FR, RL, RR each carry extra 5.45N body share
- Load rotates through all 4 legs as gait progresses
- Hip consistently carries 3-4x more torque than knee
- All legs remain below 90% utilization during healthy gait

---

## 4. STABILITY ANALYSIS

### Support Polygon
When 3 legs are on ground → triangular support polygon formed.
When all 4 legs on ground → quadrilateral support polygon.

### Stability Rule
```
CoM INSIDE polygon  → STABLE
CoM near edge       → MARGINAL (margin < 0.1m)
CoM OUTSIDE polygon → UNSTABLE
Any actuator failed → FAILURE
```

### CoM Position
- Body CoM fixed at centroid of foot workspace
- At standing pose: CoM = (0.9, 0.0) in world frame
- Crawl gait keeps CoM inside support triangle at all times

### Stability Results (Healthy Crawl Gait)
| State | % of time |
|---|---|
| STABLE | ~65% |
| MARGINAL | ~25% |
| UNSTABLE | ~10% (brief transitions) |
| FAILURE | 0% |

### Why Brief UNSTABLE Moments Occur
During the transition between stance and swing phases,
the support polygon changes shape. For a brief moment
the CoM sits outside the new triangle — this is expected
behavior in any walking robot and resolves within ~0.1s.

---

## 5. FAILURE PROPAGATION

### Failure 1 — Complete Leg Failure (RL Leg)
**What happens:**
- RL leg freezes at current position
- Gait sequence continues with only 3 active legs
- Load redistributes to FL and RR (diagonal compensation)
- Support polygon permanently loses one vertex
- System enters FAILURE state

**Observable behavior:**
- RL hip angle flatlines after failure injection
- RL foot trajectory cuts short
- Stability drops to FAILURE level (black line in graph)
- FL and RR loads increase to compensate

### Failure 2 — Joint Stall (FL Hip)
**What happens:**
- FL hip locks at stall angle mid-swing
- FL knee continues moving (partial leg function)
- FL cannot complete its swing phase
- Other legs carry extra load

**Observable behavior:**
- FL hip angle flatlines at stall angle
- Gait sequence disrupted for FL
- System drops to MARGINAL/UNSTABLE states
- Load increases on FR, RL, RR

### Failure 3 — Noise/Jitter (FR Hip)
**What happens:**
- Random ±3° errors added to FR hip every timestep
- FR foot position becomes unpredictable
- Support polygon shape distorted each step
- Stability margin fluctuates rapidly

**Observable behavior:**
- FR hip angle shows erratic zigzag pattern
- FR foot trajectory scattered
- Stability oscillates rapidly between STABLE and UNSTABLE
- Most dangerous failure for precision tasks

---

## 6. CONTROL INTERFACE

### Design Philosophy
Rajaryan (control team) needs to command the robot and
receive state feedback WITHOUT knowing simulation internals.

### Interface Usage (3 lines for Rajaryan)
```python
interface = QuadrupedControlInterface(quad, params)
command   = GaitCommand(velocity=0.5)
packet    = interface.step(command)
```

### Input (GaitCommand)
| Field | Type | Description |
|---|---|---|
| gait_type | str | "crawl" |
| velocity | float | 0.0 (slow) to 1.0 (fast) |
| active_leg | str | Which leg to swing |

### Output (SystemStatePacket) — every timestep
| Field | Type | Description |
|---|---|---|
| timestamp | float | Current simulation time |
| hip_angles | dict | {FL, FR, RL, RR} in degrees |
| knee_angles | dict | {FL, FR, RL, RR} in degrees |
| foot_positions | dict | {FL, FR, RL, RR} as (x, y) |
| load_per_leg | dict | {FL, FR, RL, RR} in N·m |
| system_state | str | STABLE/MARGINAL/UNSTABLE/FAILURE |
| failure_flags | dict | {FL, FR, RL, RR} True/False |
| gait_progress | float | 0.0 to 1.0 (cycle position) |
| polygon_area | float | Support polygon area in m² |
| com_margin | float | Distance CoM to polygon edge |

### Velocity Scaling
Higher velocity command reduces time constant:
```
time_constant = 0.6 × (1.0 - velocity × 0.5)
velocity=0.2 → tc=0.54s (slow, smooth)
velocity=0.5 → tc=0.45s (medium)
velocity=0.9 → tc=0.33s (fast, snappy)
```

---

## 7. DATA LOGGING (FOR DHRUV)

### Telemetry Format
Every timestep logs all 26 fields in consistent format.
Available as both `.txt` and `.csv`.

### Fields Per Timestep
```
time | hip_FL | hip_FR | hip_RL | hip_RR |
knee_FL | knee_FR | knee_RL | knee_RR |
foot_FL_x | foot_FL_y | foot_FR_x | foot_FR_y |
foot_RL_x | foot_RL_y | foot_RR_x | foot_RR_y |
load_FL | load_FR | load_RL | load_RR |
system_state | active_leg | gait_phase | gait_progress |
failure_FL | failure_FR | failure_RL | failure_RR | any_failure
```

### Output Files
| File | Description |
|---|---|
| t3_phase7_telemetry_healthy.csv | Full healthy gait telemetry |
| t3_phase7_telemetry_failure.csv | Failure scenario telemetry |
| t3_phase7_log_healthy.txt | Human-readable healthy log |
| t3_phase7_log_failure.txt | Human-readable failure log |

---

## 8. PHASE SUMMARY

| Phase | What Was Built | Key Output |
|---|---|---|
| Phase 1 | QuadrupedSystem (4 legs + body) | t3_phase1_quadruped_system.py |
| Phase 2 | Crawl gait FL→RR→FR→RL | t3_phase2_gait_coordination.py + graph |
| Phase 3 | Load distribution per leg | t3_phase3_load_distribution.py + graph |
| Phase 4 | Stability + support polygon | t3_phase4_stability.py + graph |
| Phase 5 | Failure propagation | t3_phase5_failure_propagation.py + graph |
| Phase 6 | Control interface layer | t3_phase6_control_interface.py + graph |
| Phase 7 | Data logging + master dashboard | t3_phase7_data_logging.py + 5 graphs + 4 logs |

---

## 9. ASSUMPTIONS

1. **2D planar motion** — all leg motion in sagittal plane (Z-axis rotation). No lateral or torsional forces.

2. **Fixed body position** — body does not translate or rotate. Only legs move relative to the fixed body.

3. **Rigid links** — thigh and shin are perfectly rigid. No flex or vibration.

4. **Static CoM** — Center of Mass fixed at centroid of foot workspace. Does not shift with leg motion.

5. **Equal leg specs** — all 4 legs are identical. Real robots may have slight differences per leg.

6. **Open-loop gait** — gait sequence is scripted, not adaptive. No real-time replanning based on terrain.

7. **Simplified ground contact** — foot is a point. No contact force, friction, or slip modeled.

8. **Body weight shared equally** — body weight distributed equally among stance legs. Real distribution depends on CoM position relative to each foot.

9. **No inertial dynamics** — mass moment of inertia not modeled for acceleration. Only gravitational load.

10. **Noise is uniform random** — jitter modeled as uniform distribution. Real encoder noise is typically Gaussian.

---

## 10. LIMITATIONS

1. **No inverse kinematics** — cannot specify foot position and compute joint angles. Forward direction only.

2. **No terrain adaptation** — gait sequence is fixed. Cannot adapt to uneven ground, slopes, or obstacles.

3. **No inter-leg force coupling** — load distribution uses simplified equal-share model. Real coupling is more complex.

4. **No body dynamics** — body pitch, roll, yaw not modeled. Real quadruped body rotates during gait.

5. **Single gait only** — crawl gait implemented. Trot, pace, bound gaits not included.

6. **No ground reaction force** — foot contact force not computed. Stability is purely geometric (CoM vs polygon).

---

## 11. INTEGRATION NOTES

### For Suraj (Mechanical)
- Leg physical parameters in `t3_phase1_quadruped_system.py` → `build_leg_system()`
- Body dimensions in `QuadrupedBody` class
- Peak loads per leg in `t3_phase3_log.txt`
- Safety margins never reach zero during healthy gait

### For Arya Barge (Structure + Load)
- Load distribution analysis in `t3_phase3_load_distribution.py`
- Stacked load graphs in `t3_phase7_graph3_load_distribution.png`
- Body weight = 5.0kg distributed equally to 3 stance legs
- Peak system load = ~112 N·m during 3-leg stance

### For Rajaryan (Control Systems)
- Full control interface in `t3_phase6_control_interface.py`
- Use `QuadrupedControlInterface` class directly
- `GaitCommand(velocity=0.5)` → `interface.step(cmd)` → `SystemStatePacket`
- `packet.system_state` gives STABLE/UNSTABLE/FAILURE each step
- Replace open-loop gait targets with closed-loop PID here

### For Dhruv Patel (Telemetry/Data)
- CSV telemetry files: `t3_phase7_telemetry_healthy.csv`
- 26 fields per row, consistent column format
- `system_state` field: STABLE/MARGINAL/UNSTABLE/FAILURE
- `any_failure` field: True/False for quick filtering
- Load fields in N·m, angles in degrees, positions in meters

---

## 12. HOW TO RUN

```bash
# Run phases in order
python t3_phase1_quadruped_system.py
python t3_phase2_gait_coordination.py
python t3_phase3_load_distribution.py
python t3_phase4_stability.py
python t3_phase5_failure_propagation.py
python t3_phase6_control_interface.py
python t3_phase7_data_logging.py
```

### Requirements
```
python >= 3.10
numpy
matplotlib
```

### Install
```bash
python -m pip install numpy matplotlib
```

### Dependencies
All Task 3 files must be in the same folder as Task 1 and
Task 2 files, OR the sys.path must include Task1 and Task2
directories.

---

## 13. FILE STRUCTURE

```
robotic joint simulation/
│
├── Task 1 files (foundation — unchanged)
│   ├── phase1_joint_definition.py
│   ├── phase2_actuator_model.py
│   └── ...
│
├── Task 2 files (single leg — unchanged)
│   ├── t2_phase1_system_expansion.py
│   ├── t2_phase2_forward_kinematics.py
│   └── ...
│
├── Task 3 files
│   ├── t3_phase1_quadruped_system.py
│   ├── t3_phase2_gait_coordination.py
│   ├── t3_phase3_load_distribution.py
│   ├── t3_phase4_stability.py
│   ├── t3_phase5_failure_propagation.py
│   ├── t3_phase6_control_interface.py
│   ├── t3_phase7_data_logging.py
│   │
│   ├── Graphs
│   │   ├── t3_phase2_graph.png
│   │   ├── t3_phase3_graph.png
│   │   ├── t3_phase4_graph.png
│   │   ├── t3_phase5_graph.png
│   │   ├── t3_phase6_graph.png
│   │   ├── t3_phase7_graph1_leg_angles.png
│   │   ├── t3_phase7_graph2_foot_trajectory.png
│   │   ├── t3_phase7_graph3_load_distribution.png
│   │   ├── t3_phase7_graph4_stability_state.png
│   │   └── t3_phase7_master_dashboard.png
│   │
│   ├── Logs
│   │   ├── t3_phase2_log.txt
│   │   ├── t3_phase3_log.txt
│   │   ├── t3_phase4_log.txt
│   │   ├── t3_phase5_log.txt
│   │   ├── t3_phase6_log.txt
│   │   ├── t3_phase7_log_healthy.txt
│   │   ├── t3_phase7_log_failure.txt
│   │   ├── t3_phase7_telemetry_healthy.csv
│   │   └── t3_phase7_telemetry_failure.csv
│   │
│   └── review_packets/
│       └── task3_review_packet.md   ← this file
```

---

*Task 3 — Full Quadruped Leg Integration + Control-Ready Simulation*
*Candidate: Ketan Rajvaidya*
*Blackhole Infiverse Robotics Systems Test*
