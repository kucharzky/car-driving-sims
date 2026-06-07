# Autonomous Driving Simulation: RL & Computer Vision from Scratch

This project aims to build a lightweight, 2D self-driving car simulation environment designed specifically to train an autonomous agent using Reinforcement Learning (RL) and Computer Vision (CV). 

---

## 1. Project Preparation & Environment

### Directory Structure
```text
self_driving_car/
│
├── config.py             # Hyperparameters, physical constants, screen sizing
├── main.py               # Main execution loop (Manual vs. AI Training Mode)
├── car.py                # Vehicle kinematics, physics, and sensor simulation
├── environment.py        # Pygame wrapper, collision rules, reward logic
├── track_manager.py      # Map generation (Procedural Spline & PNG Parser)
├── agent.py              # PyTorch RL model architecture and training loops
└── requirements.txt      # Python dependencies