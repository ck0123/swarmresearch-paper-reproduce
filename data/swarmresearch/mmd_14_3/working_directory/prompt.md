SETTING:
You are an expert computational geometer and optimization specialist focusing on 3D point dispersion
problems.

Your task is to evolve a constructor function that generates an optimal arrangement
of exactly 14 points in 3D space, maximizing the ratio of minimum distance to maximum distance
between all point pairs.

PROBLEM CONTEXT:
- Target: Beat the current state-of-the-art benchmark of min/max ratio = 1/√4.165849767 ≈ 0.4898
- Constraint: Points must be placed in 3D Euclidean space (typically normalized to unit cube
[0,1]³ or unit sphere)
- Mathematical formulation: For points Pi = (xi, yi, zi), i = 1,...,14:
  * Distance matrix: dij = √[(xi-xj)² + (yi-yj)² + (zi-zj)²] for all i≠j
  * Minimum distance: dmin = min{dij : i≠j}
  * Maximum distance: dmax = max{dij : i≠j}
  * Objective: maximize dmin/dmax subject to spatial constraints

PERFORMANCE METRICS:
1. **min_max_ratio**: dmin/dmax ratio (PRIMARY OBJECTIVE - maximize)
2. **combined_score**: min_max_ratio / 0.4898 (progress toward beating AlphaEvolve benchmark)
3. **eval_time**: Execution time in seconds (balance accuracy vs. efficiency)

TECHNICAL REQUIREMENTS:
- **Reproducibility**: Fixed random seeds for all stochastic components

Evaluation:
- Run `./task-eval`
- The evaluation timeout is 360 seconds