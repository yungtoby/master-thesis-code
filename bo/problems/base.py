from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch as t

Params = Dict[str, t.Tensor]

class BaseProblemFamily(ABC):
    def __init__(self, device: str | t.device, dtype: t.dtype):
        self.device = t.device(device)
        self.dtype = dtype

    @abstractmethod
    def sample_params(self, B: int, seed: Optional[int] = None) -> Params:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: t.Tensor, params: Params) -> t.Tensor:
        raise NotImplementedError

    @abstractmethod
    def costs(self, X: t.Tensor, params: Params) -> t.Tensor:
        raise NotImplementedError

    def optimal_value_on_grid(self, X_grid: t.Tensor, params: Params) -> t.Tensor:
        y = self.evaluate(X_grid, params).squeeze(-1)  # [B, N]
        return y.max(dim=1).values