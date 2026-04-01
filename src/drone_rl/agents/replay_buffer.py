"""Circular replay buffer for off-policy RL."""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ReplayBuffer:
    """Standard circular replay buffer."""

    def __init__(self, capacity: int = 200_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self._buf.append((
            np.asarray(state,      dtype=np.float32),
            np.asarray(action,     dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Batch:
        idxs = np.random.choice(len(self._buf), size=batch_size,
                                replace=len(self._buf) < batch_size)
        batch = [self._buf[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s),  dtype=torch.float32),
            torch.as_tensor(np.array(a),  dtype=torch.float32),
            torch.as_tensor(np.array(r, dtype=np.float32)).unsqueeze(1),
            torch.as_tensor(np.array(ns), dtype=torch.float32),
            torch.as_tensor(np.array(d, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buf) >= batch_size
