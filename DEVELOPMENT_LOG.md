# Development Log — 2D Self-Driving Car Simulation

Chronological record of implementation progress, issues encountered, fixes applied, parameter changes, and conclusions.  
**Last updated:** June 2026 · **Current config:** `config.py`

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Implementation Phases](#implementation-phases)
3. [Issues, Fixes & Parameter Changes](#issues-fixes--parameter-changes)
4. [Current Parameters (Reference)](#current-parameters-reference)
5. [Conclusions & Recommendations](#conclusions--recommendations)

---

## Project Summary

| Item | Detail |
|------|--------|
| **Stack** | Python, Pygame, OpenCV, NumPy, PyTorch |
| **Architecture** | Modular OOP: `config`, `track_manager`, `car`, `environment`, `agent`, `main` |
| **Track sources** | PNG image parser + procedural Catmull-Rom splines |
| **Perception** | 4-frame stacked 84×84 vision + 7-ray LiDAR |
| **RL** | Hybrid CNN + LiDAR Double DQN, 6 combined discrete actions |
| **Phases** | 1 Sandbox → 2 Sensors → 3 RL loop (**all complete**) |

---

## Implementation Phases

### Phase 1 — Sandbox Mechanics ✅

**Delivered:**
- `config.py`, `track_manager.py`, `car.py`, `main.py` (manual mode)
- PNG track parsing (white road, black walls, green start/finish)
- Procedural spline track generation
- Kinematic bicycle model + WASD keyboard driving
- Collision detection via wall pixel masks

**Exit criteria met:** Car drives on track, collides with walls, start/finish line visible.

---

### Phase 2 — Sensors ✅

**Delivered:**
- `cast_rays()` — 7 normalized LiDAR distances from car nose
- `capture_vision_frame()` — oriented crop → OpenCV 84×84 binary mask
- 4-frame stack `(4, 84, 84)` for CNN input
- Debug overlays: ray lines, ray bars, OpenCV `Vision Crop` / `Vision Stack` windows

**Exit criteria met:** Rays terminate at walls; vision stack updates with car motion.

---

### Phase 3 — RL Loop ✅

**Delivered:**
- `environment.py` — `reset()` / `step(action)` with rewards and lap integration
- `agent.py` — Hybrid Double DQN (vision + rays)
- `main.py` — `--mode train` and `--mode play`
- Checkpoint save/load to `checkpoints/best_agent.pth`

**Exit criteria met:** Training loop runs; agent improves track progress (52%+ lap progress in ~20 episodes after major RL overhaul; full laps still require longer training).

---

## Issues, Fixes & Parameter Changes

### Issue 1 — Procedural track not rendering

| | |
|---|---|
| **When** | Phase 1 |
| **Symptom** | Only green start/finish line visible; no white road loop |
| **Cause** | Road drawn as two overlapping polygons (outer fill + inner punch) instead of a single ring |
| **Fix** | Single ring polygon: outer boundary forward, inner boundary reversed |
| **Files** | `track_manager.py` |

---

### Issue 2 — PNG start/finish line incorrect

| | |
|---|---|
| **When** | Phase 1 |
| **Symptom** | Green line along one side of road, not across track width |
| **Cause** | Line drawn vertically along road edge instead of inner→outer wall |
| **Fix** | Horizontal line from inner ellipse to outer ellipse at track right side |
| **Files** | `scripts/generate_default_track.py`, `assets/track.png` |

---

### Issue 3 — Lap counter too easy to exploit (back-and-forth)

| | |
|---|---|
| **When** | Phase 1 (lap logic v1) |
| **Symptom** | Rocking over green line incremented laps without full circuit |
| **Parameters (broken)** | Any crossing onto green pixels counted a lap |
| **Fix (v2)** | Directed line crossing + re-arm distance (`MIN_LAP_REARM_DISTANCE` 450 px) |
| **Fix (v3)** | Zone-entry lap tracker with heading match |
| **Fix (v4 — current)** | Anti-exploit: distance + angle + far-point proof; forward-only progress (`velocity >= 0`) |

**Lap parameters evolution:**

| Parameter | v2 | v3 | v4 (current) |
|-----------|----|----|--------------|
| Detection | Pixel on/off | Directed side crossing | Finish zone + multi-proof |
| `MIN_LAP_DISTANCE` | — | 180 px | `LAP_DISTANCE_FRACTION` 0.42 × circuit (~829 px) |
| `MIN_LAP_ANGLE_PROGRESS` | — | — | 3.5 rad (~200°) |
| `LAP_FAR_DISTANCE_FRACTION` | — | — | 0.72 |
| Reverse progress | Counted | Counted | **Erases** distance progress |
| `SPAWN_OFFSET_BEFORE_LINE` | — | 25 px | **55 px** (spawn before line) |

**Files:** `track_manager.py` (`LapTracker`), `config.py`

---

### Issue 4 — Lap counter never incremented (over-strict)

| | |
|---|---|
| **When** | After lap logic v3 |
| **Symptom** | `lap:%` stuck at 0% during normal driving |
| **Cause** | `in_finish_zone(nose)` bug — tuple passed as single arg (`missing y`) |
| **Fix** | `in_finish_zone(*nose)` |
| **Files** | `track_manager.py` |

---

### Issue 5 — RL reward hacking: wiggle for track progress

| | |
|---|---|
| **When** | Phase 3, ~60–500 episodes (first DQN) |
| **Symptom** | Agent wiggles back/forth near start; no full laps; avoids walls carefully |
| **Cause** | Flat `+0.1` per step; race-axis forward reward; reverse gear from brake; progress counted while reversing |

**Parameters when issue occurred:**

| Parameter | Value (broken) |
|-----------|----------------|
| `REWARD_STEP` | 0.1 every step |
| `REWARD_FORWARD_SCALE` | 0.12 (race-axis only) |
| `REWARD_REVERSE_SCALE` | 0.15 |
| Brake behavior | Could go negative (`velocity -= braking`) |
| Lap progress | Total path distance including reverse |
| Actions | 4 mutually exclusive (left / right / accel / brake) |
| Agent | Vision-only CNN, 4 actions |
| Best checkpoint | Highest `episode_reward` |

**Fixes applied (incremental):**

1. Forward-only displacement rewards (race direction)
2. Brake clamped: `velocity = max(0, velocity - braking)` — no reverse
3. `REWARD_REVERSE_SCALE` → **0.5**
4. `REWARD_NEGATIVE_VELOCITY` = 0.05 (safety net)
5. Lap progress only when `velocity >= 0`; reverse erodes distance

**Files:** `environment.py`, `car.py`, `config.py`, `track_manager.py`

---

### Issue 6 — RL stuck at first right wall (~500 episodes)

| | |
|---|---|
| **When** | Phase 3, after Issue 5 fixes |
| **Symptom** | Best agent drives to first right wall and stops; 0 laps; low exploration value |
| **Cause** | Multiple compounding problems (see below) |

**Root causes:**

| # | Problem |
|---|---------|
| 1 | **4 exclusive actions** — cannot accelerate and steer same frame |
| 2 | **Vision-only CNN** — wall distance hard to learn from pixels alone |
| 3 | **Race-direction reward** — wrong signal on curved track sections |
| 4 | **Best checkpoint by raw reward** — rewarded “safe short drive” over progress |
| 5 | **Weak training** — batch 64, target update every 10 steps, 1 grad step/env step |

**Parameters when issue occurred:**

| Category | Value (broken) |
|----------|----------------|
| `NUM_ACTIONS` | 4 |
| `LEARNING_RATE` | 1e-4 |
| `BATCH_SIZE` | 64 |
| `TARGET_UPDATE_EVERY` | 10 |
| `TRAIN_UPDATES_PER_STEP` | 1 |
| `REPLAY_BUFFER_SIZE` | 50,000 |
| Rewards | Race-axis forward + step bonus |
| Network | `CNNQNetwork` (vision only) |
| Checkpoint criterion | `max(episode_reward)` |

**Fixes applied (major RL overhaul):**

| Change | Before | After (current) |
|--------|--------|-----------------|
| **Actions** | 4 exclusive | **6 combined** (accel±steer, coast, brake) |
| **Network** | CNN only | **Hybrid CNN + 7 rays** |
| **Algorithm** | DQN | **Double DQN** + Huber loss |
| `LEARNING_RATE` | 1e-4 | **3e-4** |
| `BATCH_SIZE` | 64 | **128** |
| `TARGET_UPDATE_EVERY` | 10 | **500** |
| `TRAIN_UPDATES_PER_STEP` | 1 | **2** |
| `REPLAY_BUFFER_SIZE` | 50,000 | **80,000** |
| `MIN_REPLAY_TO_TRAIN` | 64 | **500** |
| `EPSILON_END` | 0.05 | **0.08** |
| Checkpoint criterion | episode reward | **`progress% × 1000 + reward`** |
| Rewards | Race-axis | Path distance, lap % delta, wall proximity, speed/idle |

**Early validation (20 episodes, fresh train):** `best_prog` **52.2%** vs ~0% with old agent.

**Files:** `config.py`, `car.py`, `environment.py`, `agent.py`, `main.py`

---

## Current Parameters (Reference)

### Rewards (current)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `REWARD_PATH_SCALE` | 0.2 | Per pixel on road with speed > 0 |
| `REWARD_WALL_PROXIMITY` | 2.0 | Penalty when `min_ray < 0.45` |
| `REWARD_PROGRESS_DELTA` | 0.15 | Per % lap progress gained |
| `REWARD_SPEED_CLEAR` | 0.08 | Speed bonus when center ray clear |
| `REWARD_IDLE` | 0.08 | Penalty when stopped with clear path |
| `REWARD_OFF_ROAD` | 3.0 | Per step off drivable surface |
| `REWARD_CRASH` | -80.0 | Episode-ending collision |
| `REWARD_LAP` | 500.0 | Valid lap completion |

### RL / Agent (current)

| Parameter | Value |
|-----------|-------|
| `NUM_ACTIONS` | 6 |
| `LEARNING_RATE` | 3e-4 |
| `GAMMA` | 0.99 |
| `EPSILON_START` / `END` / `DECAY` | 1.0 / 0.08 / 0.995 (per episode) |
| `BATCH_SIZE` | 128 |
| `REPLAY_BUFFER_SIZE` | 80,000 |
| `TARGET_UPDATE_EVERY` | 500 |
| `TRAIN_UPDATES_PER_STEP` | 2 |
| `MAX_EPISODE_STEPS` | 4,000 |

### Lap detection (current)

| Parameter | Value |
|-----------|-------|
| `SPAWN_OFFSET_BEFORE_LINE` | 55 px |
| `LAP_DISTANCE_FRACTION` | 0.42 (~829 px on default track) |
| `MIN_LAP_ANGLE_PROGRESS` | 3.5 rad |
| `LAP_FAR_DISTANCE_FRACTION` | 0.72 |
| `FINISH_ZONE_RADIUS` | 38 px |
| Forward-only progress | `velocity >= 0` required |

### Vision / Sensors (current)

| Parameter | Value |
|-----------|-------|
| `VISION_SIZE` | 84 |
| `FRAME_STACK` | 4 |
| `VISION_CROP_SIZE` | 200 |
| `RAY_COUNT` | 7 |
| `RAY_MAX_DISTANCE` | 300 px |

---

## Conclusions & Recommendations

### What worked

1. **Modular phases** — manual driving and sensors before RL caught physics/rendering bugs early.
2. **LiDAR + vision hybrid** — largest single improvement for wall avoidance and training stability.
3. **Combined steer/throttle actions** — essential for cornering; exclusive actions created impossible control.
4. **Lap progress as checkpoint metric** — prevents “park safely” from beating exploratory policies.
5. **Anti-exploit lap rules** — forward-only progress + multi-proof lap validation stops figure-8 cheats.

### What did not work

1. **Flat per-step survival reward** — enables wiggle/local optimum without crashing.
2. **Race-direction-only forward reward** — misleading on oval curves.
3. **Vision-only DQN** — too slow to learn spatial wall distances in 500 episodes.
4. **Saving best agent by raw episode reward** — optimizes short safe episodes over track coverage.
5. **Allowing reverse from brake** — enabled backward progress farming.

### Recommended training workflow

```powershell
# Delete incompatible old checkpoints after architecture changes
Remove-Item checkpoints\best_agent.pth -ErrorAction SilentlyContinue

# Train headless (faster)
python main.py --mode train --episodes 1000

# Monitor log columns: progress, best_prog, min_ray
python main.py --mode train --episodes 200 --render-train

# Evaluate
python main.py --mode play
```

### Success metrics to watch

| Metric | Poor | Improving | Good |
|--------|------|-----------|------|
| `best_prog` | < 15% | 30–60% | > 80% |
| `laps` (per 1000 ep) | 0 | occasional | consistent |
| `min_ray` at crash | < 0.1 | 0.1–0.3 | agent steers before < 0.2 |
| Episode steps | < 50 (instant crash) | 100–500 | 1000+ |

### Possible next steps (not yet implemented)

- Track **waypoint / checkpoint** rewards along centerline for denser curvature signal
- **Curriculum**: train on simpler procedural tracks first, transfer to PNG oval
- **Prioritized experience replay** for crash/near-miss transitions
- Longer training (2000–5000+ episodes) for full lap completion with vision stack

---

## File Map (current)

```text
car-driving-sims/
├── config.py              # All hyperparameters
├── main.py                # manual | train | play
├── car.py                 # Physics, sensors, actions
├── track_manager.py       # Tracks + LapTracker
├── environment.py         # RL env + reward shaping
├── agent.py               # Hybrid Double DQN
├── assets/track.png       # Default oval track
├── checkpoints/           # Saved .pth weights (gitignored)
├── scripts/               # test_lap_*, test_sensors, test_rl_loop
├── STEPS.md               # Original design spec
├── PROJECT_OVERVIEW.md    # Architecture overview
└── DEVELOPMENT_LOG.md     # This file
```

---

## Revision History

| Date | Milestone |
|------|-----------|
| Phase 1 | Sandbox, tracks, manual drive |
| Phase 1 fixes | Procedural ring polygon, PNG finish line, lap logic iterations |
| Phase 2 | Raycasting + OpenCV frame stack |
| Phase 3 | Environment, DQN, train/play modes |
| Post-Phase 3 | Reward anti-wiggle, brake clamp, forward-only lap progress |
| RL overhaul | Hybrid DQN, 6 actions, progress-based checkpoints |
