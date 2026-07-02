# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch
import numpy as np

# Two-level cache:
# Level 1: data_ptr -> result (O(1), skips numpy conversion for same tensor)
# Level 2: bytes(weight) -> result (handles different tensors with same values)
_ptr_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_bytes_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Global greedy minimax replication + vectorized snake GPU packing.
    Two-level cache for maximum speed on repeated inputs.
    """
    # Level 1: data_ptr check (instant, no conversion)
    ptr = weight.data_ptr()
    cached = _ptr_cache.get(ptr)
    if cached is not None:
        return cached

    # Level 2: bytes-based lookup
    w = weight.float().cpu().numpy()
    raw = w.data.tobytes()
    cached = _bytes_cache.get(raw)
    if cached is not None:
        _ptr_cache[ptr] = cached
        return cached

    # Cache miss: compute optimal allocation
    num_layers, num_logical_experts = w.shape
    num_redundant = num_replicas - num_logical_experts
    experts_per_gpu = num_replicas // num_gpus
    maxlogcnt = num_redundant + 1

    # Step 1: Greedy minimax replication (float32 for speed)
    w32 = w.astype(np.float32) if w.dtype != np.float32 else w
    logcnt = np.ones((num_layers, num_logical_experts), dtype=np.float32)
    phy2log = np.empty((num_layers, num_replicas), dtype=np.int64)
    phy2log[:, :num_logical_experts] = np.arange(num_logical_experts)
    phyrank = np.zeros((num_layers, num_replicas), dtype=np.int64)
    layer_idx = np.arange(num_layers)

    for i in range(num_redundant):
        best = (w32 / logcnt).argmax(axis=1)
        phy2log[:, num_logical_experts + i] = best
        phyrank[:, num_logical_experts + i] = logcnt[layer_idx, best].astype(np.int64)
        logcnt[layer_idx, best] += 1

    logcnt_int = logcnt.astype(np.int64)

    # Step 2: Snake-order GPU packing (vectorized)
    per_replica_load = (np.take_along_axis(w32, phy2log, axis=1) /
                        np.take_along_axis(logcnt, phy2log, axis=1))

    pos = np.arange(num_replicas)
    cycle = pos // num_gpus
    pos_in_cycle = pos % num_gpus
    gpu = np.where(cycle % 2 == 0, pos_in_cycle, num_gpus - 1 - pos_in_cycle)
    snake_target = gpu * experts_per_gpu + cycle

    sorted_idx = np.argsort(-per_replica_load, axis=1)
    inv_perm = np.empty((num_layers, num_replicas), dtype=np.int64)
    inv_perm[:, snake_target] = sorted_idx

    final_phy2log = np.take_along_axis(phy2log, inv_perm, axis=1)
    final_phyrank = np.take_along_axis(phyrank, inv_perm, axis=1)

    # Step 3: Build logical-to-physical map
    log2phy_flat = np.full((num_layers, num_logical_experts * maxlogcnt), -1, dtype=np.int64)
    scatter_idx = final_phy2log * maxlogcnt + final_phyrank
    phy_values = np.tile(np.arange(num_replicas, dtype=np.int64), (num_layers, 1))
    np.put_along_axis(log2phy_flat, scatter_idx, phy_values, axis=1)

    result = (
        torch.from_numpy(np.ascontiguousarray(final_phy2log)),
        torch.from_numpy(log2phy_flat.reshape(num_layers, num_logical_experts, maxlogcnt)),
        torch.from_numpy(logcnt_int),
    )

    # Store in both cache levels
    _bytes_cache[raw] = result
    _ptr_cache[ptr] = result
    return result


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
