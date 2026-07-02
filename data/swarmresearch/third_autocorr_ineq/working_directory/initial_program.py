# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    num_intervals: int = 800
    learning_rate: float = 0.005


class C3OptimizerAtN:
    """Optimizer that works at a specific resolution N."""
    def __init__(self, N, lr):
        self.N = N
        self.domain_width = 0.5
        self.dx = self.domain_width / N
        self.lr = lr

    def _pnorm_objective(self, f_values: jnp.ndarray, p: float) -> jnp.ndarray:
        integral_f = jnp.sum(f_values) * self.dx
        integral_f_sq = integral_f**2

        padded_f = jnp.pad(f_values, (0, self.N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real
        scaled_conv_f_f = conv_f_f * self.dx

        abs_conv = jnp.abs(scaled_conv_f_f)
        max_val = jnp.max(abs_conv)
        normalized = abs_conv / (max_val + 1e-20)
        lp_approx = max_val * jnp.power(jnp.mean(jnp.power(normalized, p)), 1.0/p)

        c3_ratio = lp_approx / jnp.maximum(integral_f_sq, 1e-10)
        integral_penalty = 5.0 * jnp.exp(-20.0 * integral_f_sq)
        return c3_ratio + integral_penalty

    def _hard_objective(self, f_values: jnp.ndarray) -> jnp.ndarray:
        integral_f = jnp.sum(f_values) * self.dx
        integral_f_sq = integral_f**2

        padded_f = jnp.pad(f_values, (0, self.N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real
        scaled_conv_f_f = conv_f_f * self.dx

        max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))
        c3_ratio = max_abs_conv / jnp.maximum(integral_f_sq, 1e-10)
        integral_penalty = 5.0 * jnp.exp(-20.0 * integral_f_sq)
        return c3_ratio + integral_penalty

    def hard_c3(self, f_values):
        integral_f = float(jnp.sum(f_values) * self.dx)
        integral_f_sq = integral_f**2
        if integral_f_sq < 1e-10:
            return 1e10
        padded_f = jnp.pad(f_values, (0, self.N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real
        scaled_conv_f_f = conv_f_f * self.dx
        max_abs_conv = float(jnp.max(jnp.abs(scaled_conv_f_f)))
        return max_abs_conv / integral_f_sq

    def run_pnorm_phase(self, f_values, p, num_steps, lr):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.01,
            peak_value=lr,
            warmup_steps=min(2000, num_steps // 20),
            decay_steps=num_steps,
            end_value=lr * 1e-4,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )
        opt_state = optimizer.init(f_values)

        @jax.jit
        def train_step(f_values, opt_state):
            loss, grads = jax.value_and_grad(self._pnorm_objective)(f_values, p)
            updates, opt_state_new = optimizer.update(grads, opt_state, f_values)
            f_values_new = optax.apply_updates(f_values, updates)
            return f_values_new, opt_state_new, loss

        best_f = f_values
        best_c3 = float('inf')

        for step in range(num_steps):
            f_values, opt_state, loss = train_step(f_values, opt_state)
            if step % 10000 == 0 or step == num_steps - 1:
                c3_val = self.hard_c3(f_values)
                if c3_val < best_c3:
                    best_c3 = c3_val
                    best_f = f_values

        return best_f, best_c3

    def run_hard_phase(self, f_values, num_steps, lr):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.01,
            peak_value=lr,
            warmup_steps=min(1000, num_steps // 20),
            decay_steps=num_steps,
            end_value=lr * 1e-5,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=schedule),
        )
        opt_state = optimizer.init(f_values)

        @jax.jit
        def train_step(f_values, opt_state):
            loss, grads = jax.value_and_grad(self._hard_objective)(f_values)
            updates, opt_state_new = optimizer.update(grads, opt_state, f_values)
            f_values_new = optax.apply_updates(f_values, updates)
            return f_values_new, opt_state_new, loss

        best_f = f_values
        best_c3 = float('inf')

        for step in range(num_steps):
            f_values, opt_state, loss = train_step(f_values, opt_state)
            if step % 5000 == 0 or step == num_steps - 1:
                c3_val = self.hard_c3(f_values)
                if c3_val < best_c3:
                    best_c3 = c3_val
                    best_f = f_values

        return best_f, best_c3

    def run_full_sweep(self, f_values, step_mult=1.0):
        lr = self.lr
        best_f = f_values
        best_c3 = float('inf')

        phases = [
            (4.0,    int(12000 * step_mult), lr),
            (8.0,    int(12000 * step_mult), lr * 0.8),
            (16.0,   int(15000 * step_mult), lr * 0.6),
            (32.0,   int(25000 * step_mult), lr * 0.5),
            (64.0,   int(40000 * step_mult), lr * 0.4),
            (128.0,  int(55000 * step_mult), lr * 0.3),
            (256.0,  int(65000 * step_mult), lr * 0.2),
            (512.0,  int(65000 * step_mult), lr * 0.14),
            (1024.0, int(55000 * step_mult), lr * 0.09),
            (2048.0, int(45000 * step_mult), lr * 0.06),
        ]

        for p, num_steps, phase_lr in phases:
            f_values, c3 = self.run_pnorm_phase(f_values, p, num_steps, phase_lr)
            if c3 < best_c3:
                best_c3 = c3
                best_f = f_values
            f_values = best_f

        return best_f, best_c3


def upsample(f_values, old_N, new_N):
    """Upsample function values using linear interpolation."""
    old_x = np.linspace(0, 1, old_N, endpoint=False)
    new_x = np.linspace(0, 1, new_N, endpoint=False)
    return jnp.array(np.interp(new_x, old_x, np.array(f_values)))


def run():
    hypers = Hyperparameters()
    N_final = hypers.num_intervals
    lr = hypers.learning_rate
    print(f"N_final={N_final}")

    overall_best_f = None
    overall_best_c3 = float('inf')

    # Three-tier resolution: N=100 -> N=300 -> N=800
    # Idea: N=300 gives better basin identification than N=100
    # while still being 7x faster per step than N=800

    # Tier 1: Quick screen at N=100 with many seeds
    N_low = 100
    opt_low = C3OptimizerAtN(N_low, lr * 2)
    candidates_100 = []

    print(f"  Tier 1: N={N_low} with 40 seeds...")
    for seed in range(40):
        key = jax.random.PRNGKey(seed * 7 + 11)
        f_init = jax.random.normal(key, (N_low,)) * 0.3 + 2.0
        f_opt, c3 = opt_low.run_full_sweep(f_init, step_mult=0.2)
        candidates_100.append((c3, f_opt))

    candidates_100.sort(key=lambda x: x[0])
    print(f"  N={N_low} best 5: {[f'{c:.6f}' for c, _ in candidates_100[:5]]}")

    # Tier 2: Refine top 5 at N=300 (more accurate basin identification)
    N_mid = 300
    opt_mid = C3OptimizerAtN(N_mid, lr * 1.5)
    candidates_300 = []

    print(f"  Tier 2: Refining top 5 at N={N_mid}...")
    for rank in range(5):
        c3_100, f_100 = candidates_100[rank]
        f_up = upsample(f_100, N_low, N_mid)
        f_opt, c3 = opt_mid.run_full_sweep(f_up, step_mult=0.4)
        candidates_300.append((c3, f_opt))
        print(f"    Candidate {rank}: {c3_100:.6f} -> {c3:.8f}")

    candidates_300.sort(key=lambda x: x[0])

    # Tier 3: Full refinement at N=800
    opt_hi = C3OptimizerAtN(N_final, lr)

    print(f"  Tier 3: Refining best at N={N_final}...")
    c3_300, f_300 = candidates_300[0]
    f_up = upsample(f_300, N_mid, N_final)
    f_opt, c3 = opt_hi.run_full_sweep(f_up, step_mult=0.7)
    print(f"    Best N=300: {c3_300:.8f} -> {c3:.8f}")
    overall_best_c3 = c3
    overall_best_f = f_opt

    # Hard max refinement
    print("  Hard max refinement...")
    for trial in range(20):
        perturb_key = jax.random.PRNGKey(trial * 997 + 500)
        perturb_scale = 0.002 + trial * 0.0015
        f_perturbed = overall_best_f + jax.random.normal(perturb_key, (N_final,)) * perturb_scale
        f_result, c3_val = opt_hi.run_hard_phase(f_perturbed, 40000, lr * 0.012)
        if c3_val < overall_best_c3:
            overall_best_c3 = c3_val
            overall_best_f = f_result
            print(f"    Trial {trial}: C3 = {c3_val:.8f} ***")

    print(f"Final best C3: {overall_best_c3:.8f}")

    loss_val = overall_best_c3
    f_values_np = np.array(overall_best_f)

    return f_values_np, float(overall_best_c3), float(loss_val), N_final


# EVOLVE-BLOCK-END
