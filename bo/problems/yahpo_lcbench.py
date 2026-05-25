from __future__ import annotations

from typing import Optional
import random
import numpy as np
import torch as t

from yahpo_gym import BenchmarkSet
from bo.problems.base import BaseProblemFamily, Params


class YAHPOLCBenchProblemFamily(BaseProblemFamily):
    '''
    YAHPO LCBench problem family.

    This class builds a finite candidate cache:
        X_encoded: [B, N, d]
        y_grid:    [B, N]
        costs:     [B, N]

    The environment then optimizes over the finite candidate set by action index.
    '''

    provides_candidate_cache = True

    feature_keys = [
        'batch_size',
        'learning_rate',
        'max_dropout',
        'max_units',
        'momentum',
        'num_layers',
        'weight_decay',
    ]

    log_keys = {
        'learning_rate',
        'weight_decay',
    }


    def __init__(self, device, dtype, instances, objective_key='val_accuracy', cost_key='time', epoch=51, objective_scale=0.01):
        super().__init__(device=device, dtype=dtype)

        # Initialize list of instance strings
        self.instances = [str(i) for i in instances]
        if len(self.instances) == 0:
            raise ValueError('YAHPOLCBenchProblemFamily requires at least one instance.')

        # Initialize objective, cost key, epoch and scale
        self.objective_key = objective_key
        self.cost_key = cost_key
        self.epoch = int(epoch)
        self.objective_scale = float(objective_scale)

        self._bench_cache = {}
        self._bounds = None

        # Initialize bounds from the first instance
        self._init_bounds(self.instances[0])



    def build_candidate_cache(self, B, n_candidates, seed = None):
        rng = random.Random(seed)

        # Initialize empty tensors
        X = t.empty((B, n_candidates, len(self.feature_keys)), device=self.device, dtype=self.dtype)
        y_grid = t.empty((B, n_candidates), device=self.device, dtype=self.dtype)
        costs = t.empty((B, n_candidates), device=self.device, dtype=self.dtype)

        instances = []
        raw_configs = []

        # For each lane:
        for b_idx in range(B):
            
            # Choose an instance out of the scenario
            instance = rng.choice(self.instances)
            instances.append(instance)

            lane_configs = []

            # For each candidate in the lane
            for n_idx in range(n_candidates):

                # Sample a config
                cfg = self._sample_config(instance)

                # Append lane config to list of lane configs
                lane_configs.append(cfg)

                # Insert tensor into X tensor
                X[b_idx, n_idx] = t.tensor(self._encode_config(cfg), device=self.device, dtype=self.dtype)
                
                # Insert validation and cost into tensors
                y, c = self._evaluate_config(instance, cfg)
                y_grid[b_idx, n_idx] = y
                costs[b_idx, n_idx] = c

            raw_configs.append(lane_configs)

        params = {
            'instances': instances,
            'raw_configs': raw_configs,
        }

        return X, y_grid, costs, params



    def update_lane_params(self, old_params, lanes, new_params):
        if old_params is None:
            return new_params

        for local_idx, lane in enumerate(lanes.tolist()):
            old_params['instances'][lane] = new_params['instances'][local_idx]
            old_params['raw_configs'][lane] = new_params['raw_configs'][local_idx]

        return old_params



    def _get_benchmark(self, instance: str):
        if instance not in self._bench_cache:
            self._bench_cache[instance] = BenchmarkSet('lcbench', instance=instance)
        return self._bench_cache[instance]



    def _init_bounds(self, instance: str) -> None:
        b = self._get_benchmark(instance)
        cs = b.get_opt_space(drop_fidelity_params=True)

        bounds = {}
        for key in self.feature_keys:
            hp = cs.get_hyperparameter(key)

            if not hasattr(hp, 'lower') or not hasattr(hp, 'upper'):
                raise TypeError(
                    f'Hyperparameter {key} does not expose lower/upper bounds. '
                    'The first LCBench implementation assumes numeric hyperparameters.'
                )

            lo = float(hp.lower)
            hi = float(hp.upper)
            is_log = key in self.log_keys or bool(getattr(hp, 'log', False))

            if is_log:
                lo = np.log(max(lo, 1e-12))
                hi = np.log(max(hi, 1e-12))

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValueError(f'Invalid bounds for {key}: lower={lo}, upper={hi}')

            bounds[key] = (lo, hi, is_log)

        self._bounds = bounds



    def _sample_config(self, instance: str) -> dict:
        b = self._get_benchmark(instance)
        cs = b.get_opt_space(drop_fidelity_params=True)

        cfg = cs.sample_configuration(1).get_dictionary()
        cfg['epoch'] = self.epoch

        # Make sure instance/task id is consistent.
        cfg['OpenML_task_id'] = str(instance)

        return cfg



    def _encode_config(self, cfg: dict) -> list[float]:
        encoded = []

        for key in self.feature_keys:
            value = float(cfg[key])
            lo, hi, is_log = self._bounds[key]

            if is_log:
                value = np.log(max(value, 1e-12))

            z = (value - lo) / (hi - lo)
            z = float(np.clip(z, 0.0, 1.0))
            encoded.append(z)

        return encoded



    def _evaluate_config(self, instance: str, cfg: dict) -> tuple[float, float]:
        b = self._get_benchmark(instance)
        out = b.objective_function(cfg)

        if isinstance(out, list):
            out = out[0]

        y = float(out[self.objective_key]) * self.objective_scale
        c = float(out[self.cost_key])

        if not np.isfinite(y):
            raise RuntimeError(f'Non-finite objective from YAHPO: {y}')
        if not np.isfinite(c) or c < 0:
            raise RuntimeError(f'Invalid cost from YAHPO: {c}')

        return y, c
    



















    # These are only here to satisfy the BaseProblemFamily API.
    # The YAHPO family should be used through build_candidate_cache().
    def sample_params(self, B: int, seed: Optional[int] = None) -> Params:
        return {}

    def evaluate(self, X: t.Tensor, params: Params) -> t.Tensor:
        raise RuntimeError(
            'YAHPOLCBenchProblemFamily should be used through build_candidate_cache(), '
            'not evaluate(X, params).'
        )

    def costs(self, X: t.Tensor, params: Params) -> t.Tensor:
        raise RuntimeError(
            'YAHPOLCBenchProblemFamily should be used through build_candidate_cache(), '
            'not costs(X, params).'
        )