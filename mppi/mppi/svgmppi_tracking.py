from __future__ import annotations

# 新增 SVGM-PPI 版本，基于现有 MPPI 控制器实现。
# 注意：代码中所有注释都用英文，以遵守指令。

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .mppi_tracking import MPPI  # type: ignore


class SVGMPPI(MPPI):
    """An MPPI variant that leverages Stein Variational Gradient Descent (SVGD)
    to obtain guide samples before performing the standard MPPI updates.

    Compared with the base MPPI implementation, this controller maintains a
    small set of guide control sequences ("particles"). These particles are
    evolved using SVGD to approximate the posterior of optimal control
    sequences. The best particle is then selected as the nominal sequence that
    guides subsequent MPPI sampling.
    """

    def __init__(self, config, env, jrng,
                 temperature: float = 0.01,
                 damping: float = 0.001,
                 track=None):
        super().__init__(config, env, jrng, temperature, damping, track)

        # Retrieve additional hyper-parameters from the config; fall back to
        # reasonable defaults if they are missing.
        self.guide_sample_num: int = getattr(config, "guide_sample_num", 1)
        self.num_svgd_iteration: int = getattr(config, "num_svgd_iteration", 10)
        self.svgd_step_size: float = getattr(config, "svgd_step_size", 0.1)
        self.is_use_nominal_solution: bool = getattr(config, "is_use_nominal_solution", True)

        # Initialize guide samples as a stack of the zero control sequence.
        self.guide_samples = jnp.tile(jnp.expand_dims(self.a_opt, 0),
                                      (self.guide_sample_num, 1, 1))  # [N, n_steps, a_shape]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, env_state, reference_traj):  # type: ignore
        """Perform one control update.

        This method first refines the guide particles via SVGD, then selects the
        best particle to act as the nominal sequence, and finally executes the
        standard MPPI iterations defined in the base class.
        """
        # 1. Perform SVGD on guide particles.
        self.guide_samples = self._run_svgd(self.guide_samples, env_state, reference_traj)

        # 2. Pick the best particle according to the cost.
        guide_costs = self._batch_cost(self.guide_samples, env_state, reference_traj)
        best_idx = int(jnp.argmin(guide_costs))
        best_particle = self.guide_samples[best_idx]

        # 3. Optionally use the best particle as the nominal sequence.
        if self.is_use_nominal_solution:
            self.a_opt = best_particle

        # 4. Proceed with the standard MPPI update (will overwrite self.a_opt).
        super().update(env_state, reference_traj)

        # 5. Cache the selected nominal sequence for external access.
        self.nominal_control_seq = best_particle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_svgd(self, particles: jnp.ndarray, env_state, reference_traj) -> jnp.ndarray:  # type: ignore
        """Run a fixed number of SVGD iterations to update the particles."""
        rng_key = self.jrng.new_key()
        for _ in range(self.num_svgd_iteration):
            rng_key, sub_key = jax.random.split(rng_key)
            particles = self._svgd_step(particles, env_state, reference_traj, sub_key)
        return particles

    @partial(jax.jit, static_argnums=0)
    def _svgd_step(self, particles: jnp.ndarray, env_state, reference_traj, rng_key) -> jnp.ndarray:  # type: ignore
        """One SVGD update over the full particle set."""
        num_particles = particles.shape[0]

        # Compute gradients of log-posterior w.r.t. each particle.
        def single_cost_fn(ctrl_seq):
            return self._trajectory_cost(ctrl_seq, env_state, reference_traj, rng_key)

        grad_log_post = -jax.vmap(jax.grad(single_cost_fn))(particles) / self.temperature  # [N, n_steps, a_shape]

        # Compute RBF kernel matrix among particles.
        flat_particles = particles.reshape(num_particles, -1)  # [N, D]
        pairwise_d2 = jnp.sum((jnp.expand_dims(flat_particles, 1) - jnp.expand_dims(flat_particles, 0)) ** 2, axis=-1)  # [N, N]
        # Median heuristic for bandwidth.
        median_d2 = jnp.median(pairwise_d2)
        h = median_d2 / jnp.log(num_particles + 1.0) + 1e-6
        kappa = jnp.exp(-pairwise_d2 / h)  # [N, N]

        # Gradient of the kernel.
        diff = jnp.expand_dims(flat_particles, 1) - jnp.expand_dims(flat_particles, 0)  # [N, N, D]
        grad_kappa = -2.0 / h * diff * jnp.expand_dims(kappa, -1)  # [N, N, D]

        # SVGD update direction.
        phi = (
            jnp.matmul(kappa, grad_log_post.reshape(num_particles, -1)).reshape(num_particles, *particles.shape[1:]) +
            jnp.sum(grad_kappa, axis=1).reshape(num_particles, *particles.shape[1:])
        ) / num_particles

        # Euler update.
        particles = particles + self.svgd_step_size * phi
        particles = jnp.clip(particles, -1.0, 1.0)
        return particles

    # ------------------------------------------------------------------
    # Cost computation utilities
    # ------------------------------------------------------------------
    def _batch_cost(self, particles: jnp.ndarray, env_state, reference_traj) -> jnp.ndarray:  # type: ignore
        """Vectorized cost for a batch of control sequences."""
        cost_fn = partial(self._trajectory_cost, env_state=env_state, reference_traj=reference_traj, rng_key=self.jrng.new_key())
        return jax.vmap(cost_fn)(particles)

    def _trajectory_cost(self, ctrl_seq: jnp.ndarray, env_state, reference_traj, rng_key) -> jnp.ndarray:  # type: ignore
        """Return scalar cost (negative reward) for one control sequence."""
        traj = self.rollout(ctrl_seq, env_state, rng_key)
        if self.config.state_predictor in self.config.cartesian_models:
            reward = self.env.reward_fn_xy(traj, reference_traj,
                                           self.env.costmap_jax,
                                           self.env.costmap_origin[0],
                                           self.env.costmap_origin[1],
                                           self.env.costmap_resolution)
        else:
            reward = self.env.reward_fn_sey(traj, reference_traj)
        return -jnp.sum(reward) 