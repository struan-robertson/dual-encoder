"""Methods used to update variables on the fly using streamed data."""

import torch


class AdaptiveNormalisation:
    """Use Exponential Moving Average (EMA) for normalising a stream of data."""

    def __init__(
        self,
        initial_mean: torch.Tensor,
        initial_std: torch.Tensor,
        device,
        momentum: float = 0.99,
    ):
        self.mean = initial_mean.clone().to(device)
        self.std = initial_std.clone().to(device)
        self.momentum = momentum
        self.n_samples = 0
        self.device = device

    def __call__(self, batch: torch.Tensor, *, update: bool = True):
        # Normalise with current statistics
        normalised = (batch - self.mean[None, :, None, None]) / self.std[
            None, :, None, None
        ]

        if update:
            batch_mean = batch.mean(dim=[0, 2, 3])
            batch_std = batch.std(dim=[0, 2, 3])

            # Start with a high momentum, decrease over time
            # Improves early training stability
            effective_momentum = min(
                self.momentum, self.n_samples / (self.n_samples + 1)
            )

            self.mean = (
                effective_momentum * self.mean + (1 - effective_momentum) * batch_mean
            )
            self.std = (
                effective_momentum * self.std + (1 - effective_momentum) * batch_std
            )
            self.n_samples += batch.size(0)

        return normalised

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "momentum": self.momentum,
            "n_samples": self.n_samples,
        }

    def load_state_dict(self, state_dict: dict):
        self.mean = state_dict["mean"].to(self.device)
        self.std = state_dict["std"].to(self.device)
        self.momentum = state_dict["momentum"]
        self.n_samples = state_dict["n_samples"]


class DifficultyScheduler:
    """Curriculum difficulty scheduling."""

    def __init__(
        self,
        initial_difficulty: float = 0.1,
        max_difficulty=1.0,
        peak_steps: int = 1000,
    ):
        self.difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.step_amount = max_difficulty / peak_steps

    def step(self):
        # Calculate how well sperated the pairs are
        # Higher violation_score means network is struggling
        difficulty = self.difficulty + self.step_amount
        self.difficulty = min(self.max_difficulty, difficulty)

    def get_difficulty(self):
        return self.difficulty

    def state_dict(self):
        return {
            "difficulty": self.difficulty,
            "max_difficulty": self.max_difficulty,
            "step_amount": self.step_amount,
        }

    def load_state_dict(self, state_dict: dict):
        self.difficulty = state_dict["difficulty"]
        self.max_difficulty = state_dict["max_difficulty"]
        self.step_amount = state_dict["step_amount"]
