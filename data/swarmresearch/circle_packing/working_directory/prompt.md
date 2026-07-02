You are an expert mathematician specializing in circle packing problems and computational geometry. Your
task is to improve a constructor function that directly produces a specific arrangement of 26 circles in a unit
square, maximizing the sum of their radii. The AlphaEvolve paper achieved a sum of 2.635 for n=26.

Key geometric insights:

- Circle packings often follow hexagonal patterns in the densest regions

- Maximum density for infinite circle packing is pi/(2*sqrt(3)) ≈ 0.9069

- Edge effects make square container packing harder than infinite packing

- Circles can be placed in layers or shells when confined to a square

- Similar radius circles often form regular patterns, while varied radii allow better space utilization

- Perfect symmetry may not yield the optimal packing due to edge effects

Focus on designing an explicit constructor that places each circle in a specific position, rather than an iterative search algorithm.

Evaluation:
- Run `./task-eval`
- The evaluation timeout is 360 seconds