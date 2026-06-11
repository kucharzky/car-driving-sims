"""PyTorch DQN agent with CNN + LiDAR hybrid encoder."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config

Transition = Tuple[
    np.ndarray,
    np.ndarray,
    int,
    float,
    np.ndarray,
    np.ndarray,
    bool,
]


def _conv2d_size(size: int, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


class HybridDQN(nn.Module):
    """CNN vision encoder fused with LiDAR ray distances."""

    def __init__(self, num_actions: int = config.NUM_ACTIONS) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(config.FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out = _conv2d_size(
            _conv2d_size(_conv2d_size(config.VISION_SIZE, 8, 4), 4, 2),
            3,
            1,
        )
        self.conv_flat = 64 * conv_out * conv_out
        self.ray_encoder = nn.Sequential(
            nn.Linear(config.RAY_COUNT, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.conv_flat + 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, vision: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        visual = self.conv(vision).view(vision.size(0), -1)
        ray_feat = self.ray_encoder(rays)
        fused = torch.cat([visual, ray_feat], dim=1)
        return self.head(fused)


class ReplayBuffer:
    """Fixed-size experience replay for DQN."""

    def __init__(self, capacity: int = config.REPLAY_BUFFER_SIZE) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self._buffer, batch_size)
        states, rays, actions, rewards, next_states, next_rays, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(rays).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.stack(next_rays).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )


class Agent:
    """Double DQN with hybrid vision + LiDAR observations."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy_net = HybridDQN().to(self.device)
        self.target_net = HybridDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayBuffer()
        self.train_steps = 0
        self.epsilon = config.EPSILON_START

    def select_action(
        self,
        state: np.ndarray,
        rays: np.ndarray,
        epsilon: float | None = None,
    ) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.randint(0, config.NUM_ACTIONS - 1)

        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            rays_t = torch.from_numpy(rays).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t, rays_t)
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        rays: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_rays: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push((state, rays, action, reward, next_state, next_rays, done))

    def train_step(self) -> float | None:
        if len(self.memory) < config.MIN_REPLAY_TO_TRAIN:
            return None

        states, rays, actions, rewards, next_states, next_rays, dones = self.memory.sample(
            config.BATCH_SIZE
        )

        states_t = torch.from_numpy(states).to(self.device)
        rays_t = torch.from_numpy(rays).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        next_rays_t = torch.from_numpy(next_rays).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        self.policy_net.train()
        q_values = self.policy_net(states_t, rays_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t, next_rays_t).argmax(dim=1)
            next_q = (
                self.target_net(next_states_t, next_rays_t)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target = rewards_t + config.GAMMA * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % config.TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def train_updates(self, count: int = config.TRAIN_UPDATES_PER_STEP) -> float | None:
        """Run multiple gradient steps per environment step."""
        last_loss: float | None = None
        for _ in range(count):
            loss = self.train_step()
            if loss is not None:
                last_loss = loss
        return last_loss

    def decay_epsilon(self) -> None:
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", config.EPSILON_END))
        self.train_steps = int(checkpoint.get("train_steps", 0))
