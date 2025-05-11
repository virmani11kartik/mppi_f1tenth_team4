"""An MPPI based planner."""
import jax
import jax.numpy as jnp
import os, sys
sys.path.append("../")
from functools import partial
import numpy as np


class MPPI():
    """An MPPI based planner."""
    def __init__(self, config, env, jrng, 
                 temperature=0.01, damping=0.001, track=None):
        self.config = config
        self.n_iterations = config.n_iterations
        self.n_steps = config.n_steps
        self.n_samples = config.n_samples
        self.temperature = temperature
        self.damping = damping
        self.a_std = jnp.array(config.control_sample_std)
        self.a_cov_shift = config.a_cov_shift
        self.adaptive_covariance = (config.adaptive_covariance and self.n_iterations > 1) or self.a_cov_shift
        self.a_shape = config.control_dim
        self.env = env
        self.jrng = jrng
        self.init_state(self.env, self.a_shape)
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))
        self.track = track
        # --------------- SVG‑MPPI state & hyper‑parameters -----------------
        self.svg_Kg  = config.svg_Kg
        self.svg_N   = config.svg_N
        self.svg_L   = config.svg_L
        self.svg_eps = config.svg_eps
        sigma_init   = config.svg_sigma_init
        self.sigma_floor = jnp.asarray(float(config.svg_sigma_floor), dtype=jnp.float32)

        self.Vg = jnp.zeros((self.svg_Kg, self.n_steps, self.a_shape))

        # Guide particle(s)   Vg.shape = [Kg, n_steps, a_shape]
        self.Vg = jnp.zeros((self.svg_Kg, self.n_steps, self.a_shape))
        # Shared diagonal covariance  Σ_g  (we store its inverse once)

        # initialise Σ_g, Σ_g⁻¹ and a_cov at one place
        self.reset_svg_covariance(sigma_init)

        self.cur_state = None 
        # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _transport_guides_core(self, Vg, Sigma_g, env_state,
                            reference_traj, rng_key):
        """
        JIT‑compiled, side‑effect‑free core.
        Args:
            Vg         : [Kg, T, m]  current guide set
            Sigma_g    : [T, m]      diagonal covariance
        Returns:
            Vg_new     : [Kg, T, m]  updated guide
            Sigma_hat  : [T, m]      adaptive diagonal covariance
            key_out    : PRNG key after use
        """
        Sigma_diag = Sigma_g                        # [T,m]
        Kg, N      = self.svg_Kg, self.svg_N

        # ---------- dummy tensors so scan has known shapes -----------------
        dummy_samples = jnp.zeros((Kg, N, self.n_steps, self.a_shape))
        dummy_w       = jnp.ones((Kg*N,)) / (Kg*N)

        def one_svgd_iter(carry, _):
            """
            carry = (Vg, key, V_samples, w_flat)
            Returns same‑shape carry.
            """
            Vg, key, _, _ = carry
            key, sub = jax.random.split(key)

            # 1) sample N trajectories per guide
            noise = jax.random.normal(
                sub, shape=(Kg, N, self.n_steps, self.a_shape)
            ) * jnp.sqrt(Sigma_diag)                           # broadcast diag
            V_samples = Vg[:, None] + noise          # [Kg,N,T,m]

            # 2) rollout and cost
            flat_V  = V_samples.reshape(-1, self.n_steps, self.a_shape)
            flat_st = jax.vmap(self.rollout, in_axes=(0, None, None))(
                flat_V, env_state, key)

            costs = jax.vmap(
                self.env.reward_fn_xy,
                (0, None, None, None, None, None)
            )(flat_st, reference_traj,
            self.env.costmap_jax,
            self.env.costmap_origin[0],
            self.env.costmap_origin[1],
            self.env.costmap_resolution)

            seq_cost = jnp.mean(costs, axis=1)                 # [Kg*N]
            w_flat   = jax.nn.softmax(
                -(seq_cost - jnp.min(seq_cost)) / self.temperature)

            # 3) surrogate gradient
            score = (V_samples - Vg[:, None]) / Sigma_diag      # [Kg,N,T,m]
            score = score.reshape(-1, self.n_steps, self.a_shape)
            grad  = -jnp.tensordot(w_flat, score, axes=(0, 0))   # [T,m]

            Vg_new = Vg - self.svg_eps * grad[None]             # apply ε step
            return (Vg_new, key, V_samples, w_flat), None

        # ---------- run L SVGD iterations via lax.scan ----------------------
        (Vg_new, key_out, V_samples_last, w_last), _ = jax.lax.scan(
            one_svgd_iter,
            (Vg, rng_key, dummy_samples, dummy_w),
            None,
            length=self.svg_L)

        # ---------- adaptive covariance  Σ̂  from last batch ----------------
        diff2 = (V_samples_last - Vg_new[0]) ** 2                # [Kg,N,T,m]
        Sigma_hat = jnp.maximum(
            jnp.tensordot(w_last, diff2.reshape(-1, self.n_steps, self.a_shape),
                        axes=(0, 0)),                          # weighted var
            self.sigma_floor                                     # floor
        )                                                        # [T,m]
        return Vg_new, Sigma_hat, key_out
    

    def transport_guides(self, env_state, reference_traj, rng_key):
        """
        Stateful wrapper: calls the pure JIT core, then stores the results
        back into self.Vg, self.Sigma_g, self.Sigma_g_inv.
        """
        Vg_new, Sigma_hat, _ = self._transport_guides_core(
            self.Vg, self.Sigma_g, env_state, reference_traj, rng_key)

        # bring concrete device arrays to Python side
        Vg_new    = jax.device_get(Vg_new)
        Sigma_hat = jax.device_get(Sigma_hat)

        # update class state
        self.Vg          = Vg_new
        self.Sigma_g     = Sigma_hat
        self.Sigma_g_inv = 1.0 / Sigma_hat

        # first guide becomes nominal sequence for MPPI
        return Vg_new[0], Sigma_hat
# -------------------------------------------------------------------


    # ─── Modified helper (replaces previous 1‑D version) ────────────────────
    def reset_svg_covariance(self, sigma_init=0.05):
        # diagonal variance per time‑step
        diag = (sigma_init ** 2) * jnp.ones((self.n_steps, self.a_shape))
        self.Sigma_g     = diag
        self.Sigma_g_inv = 1.0 / diag
        self.a_cov       = diag            # keep MPPI sampler consistent
    # -----------------------------------------------------------------------

    def init_state(self, env, a_shape):
        # uses random as a hack to support vmap
        # we should find a non-hack approach to initializing the state
        dim_a = jnp.prod(a_shape)  # np.int32
        self.env = env
        self.a_opt = 0.0*jax.random.uniform(self.jrng.new_key(), shape=(self.n_steps,
                                                dim_a))  # [n_steps, dim_a]
        # a_cov: [n_steps, dim_a, dim_a]
        if self.a_cov_shift:
            # note: should probably store factorized cov,
            # e.g. cholesky, for faster sampling
            self.a_cov = (self.a_std**2)*jnp.tile(jnp.eye(dim_a), (self.n_steps, 1, 1))
            self.a_cov_init = self.a_cov.copy()
        else:
            self.a_cov = None
            self.a_cov_init = self.a_cov
            
            
    def update(self, env_state, reference_traj):
        self.cur_state = env_state
        # (0) shift previous optimal sequence as usual
        self.a_opt, self.a_cov = self.shift_prev_opt(self.a_opt, self.a_cov)

        # -----------------------------------------------------------------

        # (1) SVG‑MPPI: move guide & adopt it as new nominal sequence Ū
        (U_nom, Sigma_hat) = self.transport_guides(env_state, reference_traj, self.jrng.new_key())
        self.a_opt = U_nom                      # start MPPI from the guide
        # you may also copy Σ_g into self.a_cov if adaptive_covariance = False
        self.a_cov = Sigma_hat 
        # -----------------------------------------------------------------

        # (2) vanilla MPPI refinement iterations
        for _ in range(self.n_iterations):
            self.a_opt, self.a_cov, self.states, self.traj_opt = self.iteration_step(self.a_opt, self.a_cov, self.jrng.new_key(), env_state, reference_traj)
        
        if self.track is not None and self.config.state_predictor in self.config.cartesian_models:
            self.states = self.convert_cartesian_to_frenet_jax(self.states)
            self.traj_opt = self.convert_cartesian_to_frenet_jax(self.traj_opt)
        self.sampled_states = self.states

    
    @partial(jax.jit, static_argnums=(0))
    def shift_prev_opt(self, a_opt, a_cov):
        a_opt = jnp.concatenate([a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((self.a_shape,)),
                                                axis=0)])  # [n_steps, a_shape]
        if self.a_cov_shift:
            a_cov = jnp.concatenate([a_cov[1:, :],
                                    jnp.expand_dims((self.a_std**2)*jnp.eye(self.a_shape),
                                                    axis=0)])
        else:
            a_cov = self.a_cov_init
        return a_opt, a_cov
    
    
    # 注意：取消 JIT 装饰以便每次调用时都可感知最新的环境信息（例如动态更新的 costmap）
    def iteration_step(self, a_opt, a_cov, rng_da, env_state, reference_traj):
        rng_da, rng_da_split1, rng_da_split2 = jax.random.split(rng_da, 3)
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * self.a_std - a_opt,
            jnp.ones_like(a_opt) * self.a_std - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape)
        )  # [n_samples, n_steps, dim_a]

        actions = jnp.clip(jnp.expand_dims(a_opt, axis=0) + da, -1.0, 1.0)
        states = jax.vmap(self.rollout, in_axes=(0, None, None))(
            actions, env_state, rng_da_split1
        )
        
        if self.config.state_predictor in self.config.cartesian_models:
            reward = jax.vmap(self.env.reward_fn_xy, in_axes=(0, None, None, None, None, None))( # type: ignore
                states, reference_traj,
                self.env.costmap_jax,
                self.env.costmap_origin[0],
                self.env.costmap_origin[1],
                self.env.costmap_resolution,
            )
        else:
            reward = jax.vmap(self.env.reward_fn_sey, in_axes=(0, None))(
                states, reference_traj
            ) # [n_samples, n_steps]          
        
        R = jax.vmap(self.returns)(reward) # [n_samples, n_steps], pylint: disable=invalid-name

        # ----------  Quadratic term  Δuᵀ Σ⁻¹ Δu  (Eq.8) --------------------------
        # self.Sigma_g_inv : [a_shape]  diagonal inverse variances
        guide_cost = 0.5 * jnp.sum((da ** 2) * self.Sigma_g_inv, axis=(1, 2))  # [n_samples]
        R_aug = R + guide_cost[:, None]     # add same cost to every step pos

        # ------------ MPPI softmax weight with the augmented return ---------------
        w = jax.vmap(self.weights, 1, 1)(R_aug)       # [n_samples, n_steps]
        # --------------------------------------------------------------------------


        # Original MPPI weight
        # w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)  # [n_steps, dim_a]
        if self.adaptive_covariance:
            a_cov = jax.vmap(jax.vmap(jnp.outer))(
                da, da
            )  # [n_samples, n_steps, a_shape, a_shape]
            a_cov = jax.vmap(jnp.average, (1, None, 1))(
                a_cov, 0, w
            )  # a_cov: [n_steps, a_shape, a_shape]
            a_cov = a_cov + jnp.eye(self.a_shape)*0.00001 # prevent loss of rank when one sample is heavily weighted
            
        if self.config.render:
            traj_opt = self.rollout(a_opt, env_state, rng_da_split2)
        else:
            traj_opt = states[0]
            
        return a_opt, a_cov, states, traj_opt

   
    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]


    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w
    
    
    @partial(jax.jit, static_argnums=0)
    def rollout(self, actions, env_state, rng_key):
        """
        # actions: [n_steps, a_shape]
        # env: {.step(states, actions), .reward(states)}
        # env_state: np.float32
        # actions: # a_0, ..., a_{n_steps}. [n_steps, a_shape]
        # states: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
        """
    
        def rollout_step(env_state, actions, rng_key):
            actions = jnp.reshape(actions, self.env.a_shape)
            (env_state, env_var, mb_dyna) = self.env.step(env_state, actions, rng_key)
            return env_state
        
        states = []
        for t in range(self.n_steps):
            env_state = rollout_step(env_state, actions[t, :], rng_key)
            states.append(env_state)
            
        return jnp.asarray(states)
    
    
    # @partial(jax.jit, static_argnums=(0))
    def convert_cartesian_to_frenet_jax(self, states):
        states_shape = (*states.shape[:-1], 7)
        states = states.reshape(-1, states.shape[-1])
        converted_states = self.track.vmap_cartesian_to_frenet_jax(states[:, (0, 1, 4)])
        states_frenet = jnp.concatenate([converted_states[:, :2], 
                                         states[:, 2:4] * jnp.cos(states[:, 6:7]),
                                         converted_states[:, 2:3],
                                         states[:, 2:4] * jnp.sin(states[:, 6:7])], axis=-1)
        return states_frenet.reshape(states_shape)