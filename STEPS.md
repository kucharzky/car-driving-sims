## Track Design & Map Loading Mechanics

The simulation supports two distinct methods for handling tracks. Both methods track structural loops and detect the crossing of a designated Start/Finish line.

### Method A: Procedural Generation (Bézier Splines)

* **How it works:** The script generates a set of random control points in a radial layout, applies a Smoothing Spline or a Closed Bézier Curve equation to loop them, and extrudes parallel inner/outer boundaries to create a track width.
* **Pros:** Infinite training environments; prevents the AI from overfitting to one specific shape.

### Method B: PNG Image Parser

* **How it works:** The `TrackManager` loads a custom static image asset (`track.png`).
* **Color Coding Rules:**
* **Road/Driveable Area:** White pixels (`255, 255, 255`).
* **Walls/Obstacles:** Black pixels (`0, 0, 0`). Collision is triggered if the car hits a black pixel.
* **Start/Finish Line:** Pure green pixels (`0, 255, 0`). Used to track lap completions and reset the timer.



| Feature | Procedural Spline Track | PNG Image Track |
| --- | --- | --- |
| **Generation Speed** | Instant (Dynamic) | Static (Requires Pre-drawing) |
| **Flexibility** | High (Infinite Variations) | Low (Fixed Map Geometry) |
| **Complexity** | Complex Mathematical Splines | Simple Pixel Array Lookups |

---

## Physics Engine & Car Kinematics

To ensure realistic vehicle movement without overcomplicating calculations, the simulation utilizes a **2D Kinematic Bicycle Model**.

* **State Variables:** Position ($x, y$), velocity ($v$), heading angle ($\theta$), and steering angle ($\phi$).
* **Motion Equations:**

$$x_{t+1} = x_t + v \cdot \cos(\theta) \cdot \Delta t$$


$$y_{t+1} = y_t + v \cdot \sin(\theta) \cdot \Delta t$$


$$\theta_{t+1} = \theta_t + \frac{v}{L} \cdot \tan(\phi) \cdot \Delta t$$



*(Where $L$ is the wheelbase length of the car).*
* **Constraints:** Hard thresholds are applied to max acceleration, braking deceleration, and steering limits to emulate physical boundaries.

---

## Computer Vision & Sensor Integration

The agent perceives its world through two customizable vision layers:

### Vector Space Layer (Raycasting/LiDAR)

The car projects multi-directional lines (e.g., 5, 7, or 12 rays) radiating from its bumper at fixed angles (e.g., -90°, -45°, 0°, 45°, 90°). It calculates the intersection point between these vectors and the track walls, outputting distance floats.

### Computer Vision Layer (Pixel Matrix Input)

1. **Frame Capture:** The area around the car is cropped from the Pygame window or rendered from a car-mounted top-down perspective.
2. **Preprocessing (OpenCV):** The image is downscaled (e.g., to $84 \times 84$ pixels), converted to grayscale, and normalized into binary states (0 for off-track, 1 for road).
3. **Frame Stacking:** To allow the neural network to understand **velocity and direction**, a stack of 4 consecutive frames is grouped together as a single input state.

---

## Reinforcement Learning Brain

The agent learns via a **Deep Q-Network (DQN)** for discrete controls, or **Proximal Policy Optimization (PPO)** for continuous operations.

```
+-----------------------------------------------------------+
|                       ENVIRONMENT                         |
|  [Track Canvas] -> Render Screen State -> Track Collisions |
+-----------------------------------------------------------+
       ^                                             |
       | Action                                      | State / Reward
       | (Steer, Gas)                                v
+-----------------------------------------------------------+
|                          AGENT                            |
|  [CNN Encoder] -> Deep Q-Network -> Action Selection      |
+-----------------------------------------------------------+

```

### The Neural Network Architecture (CNN)

* **Input:** Tensor of shape `(4, 84, 84)` (4 stacked grayscale screens).
* **Convolutional Layers:** 3 layers to extract spatial road features (edges, curves, distance to walls).
* **Fully Connected Layers:** Converts spatial features into action values.
* **Output:** Q-Values for discrete commands (Turn Left, Turn Right, Accelerate, Brake).

### Reward Function Formula

The reward design balances speed against safety:

* **Step Reward:** Each step surviving without a crash yields a small positive reward ($+0.1$).
* **Velocity Bonus:** Higher reward for moving fast along the track vector ($+v \cdot \cos(\Delta\theta)$).
* **Crash Penalty:** Hitting a track boundary immediately drops a major penalty ($-100$) and terminates the episode.
* **Lap Completion:** Passing the green Start/Finish pixel loop yields a major bonus ($+500$).

---

## Step-by-Step Implementation Strategy

### Phase 1: Sandbox Mechanics (Days 1–2)

* Write `config.py` and `track_manager.py` to parse custom PNG track graphics.
* Build `car.py` using Pygame to enable human keyboard driving (`W`, `A`, `S`, `D`) over the image boundaries.

### Phase 2: Sensor Arrays (Days 3–4)

* Code raycasting or sub-surface window captures around the vehicle coordinates.
* Feed captured pixel visuals directly into an isolated window to ensure OpenCV transforms match spatial positions accurately.

### Phase 3: The RL Loop (Days 5–7)

* Construct the PyTorch CNN network inside `agent.py`.
* Establish the step environment loop: `State -> Action -> Reward -> New State`.
* Let the agent train overnight, saving model weights (`.pth` files) iteratively when high scores are achieved.

```

```