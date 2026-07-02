You are an expert signal processing engineer specializing in real-time adaptive filtering algorithms.

Your task is to improve a signal processing algorithm that filters volatile, non-stationary time series data using a sliding window approach. The algorithm must minimize noise while preserving signal dynamics with minimal computational latency and phase delay. Focus on the multi-objective optimization of: (1) Slope change minimization - reducing spurious directional reversals, (2) Lag error minimization - maintaining responsiveness, (3) Tracking accuracy - preserving genuine signal trends, and (4) False reversal penalty - avoiding noise-induced trend changes. Consider advanced techniques like adaptive filtering (Kalman filters, particle filters), multi-scale processing(wavelets, EMD), predictive enhancement (polynomial fitting, neural networks), and trend detection methods.

Evaluation:
- Run `./task-eval`
- The evaluation timeout is 360 seconds