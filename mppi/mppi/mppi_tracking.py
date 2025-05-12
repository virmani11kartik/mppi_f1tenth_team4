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

        self.svg_Kg       = config.svg_Kg     # e.g. 1
        self.svg_sigma_init = float(config.svg_sigma_init)
        self.Vg = jnp.zeros((self.svg_Kg, self.n_steps, self.a_shape))

        self.svg_Kg  = config.svg_Kg
        self.svg_N   = config.svg_N
        self.svg_L   = config.svg_L
        self.svg_eps = config.svg_eps

    def sample_initial_guides(self, rng_key):
        """
        Draw Kg guide particles from N(a_opt, σ_init^2 I).
        Returns Vg of shape [Kg, n_steps, a_shape].
        """
        # scalar σ
        sigma = self.svg_sigma_init
        # draw IID normal noise
        noise = jax.random.normal(rng_key,
                 shape=(self.svg_Kg, self.n_steps, self.a_shape))
        # scale, add to current nominal a_opt
        Vg = jnp.clip(self.a_opt[None] + sigma * noise,
                      -1.0, 1.0)
        # store and return
        self.Vg = Vg
        return Vg
    

    def _one_svgd_step(self, Vg, Sigma_g, env_state, reference_traj, rng_key):
        """Single SVGD update (surrogate), returns new Vg, new Σ_g."""
        # draw N samples around each guide
        Kg, N = self.svg_Kg, self.n_samples
        Σ_diag = Sigma_g               # [T,m]
        # sample noise
        key, sub = jax.random.split(rng_key)
        noise = jax.random.normal(sub, (Kg, N, self.n_steps, self.a_shape))
        V_samples = Vg[:,None] + noise * jnp.sqrt(Σ_diag)

        # rollout all samples
        flat_V = V_samples.reshape(-1, self.n_steps, self.a_shape)
        flat_states = jax.vmap(self.rollout, in_axes=(0,None,None))(
                             flat_V, env_state, key)

        # compute *rewards* (higher is better)
        seq_reward = jax.vmap(self.env.reward_fn_xy,
                              (0,None,None,None,None,None))(
            flat_states, reference_traj,
            self.env.costmap_jax,
            self.env.costmap_origin[0],
            self.env.costmap_origin[1],
            self.env.costmap_resolution
        )  # [Kg*N, T]

        # do MPPI‐style weighting: higher reward ⇒ higher w
        seq_R = jnp.mean(seq_reward, axis=1)           # [Kg*N]
        w_flat = jax.nn.softmax(seq_R / self.temperature)

        # surrogate gradient estimate of reverse‐KL
        score = (V_samples - Vg[:,None]) / Σ_diag
        score = score.reshape(-1, self.n_steps, self.a_shape)
        grad = jnp.tensordot(w_flat, score, axes=(0,0))  # [T,m]

        # drive guides toward high‐reward mode
        Vg_new = Vg - self.svg_eps * grad[None]
        # adapt covariance to weighted var of last batch
        diff2 = (V_samples - Vg_new[:,None])**2
        Sigma_hat = jnp.maximum(
            jnp.tensordot(w_flat, diff2.reshape(-1,self.n_steps,self.a_shape),
                          axes=(0,0)),
            self.svg_sigma_init**2 * 0.01  # floor
        )
        return Vg_new, Sigma_hat, key


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
        self.a_opt, self.a_cov = self.shift_prev_opt(self.a_opt, self.a_cov)

        # ─────── Step 1: Draw initial guide(s) ───────
        rng, sub = jax.random.split(self.jrng.new_key())
        Vg = self.sample_initial_guides(sub)       # [Kg, T, m] control sequences
        # …later we’ll transport these; for now we just visualize Vg[0]…
        
        self.guides_ctrl = [ Vg[0] ]  

        # ─── initialize guide covariance Σ_g for the first SVGD step ───
        Sigma_g = (self.svg_sigma_init**2) * jnp.ones((self.n_steps, self.a_shape))

        # ───(2) run L SVGD steps, publishing each ────
        for l in range(1, self.svg_L+1):
            rng, sub = jax.random.split(rng)
            Vg, Sigma_g, _ = self._one_svgd_step(
                Vg, Sigma_g, env_state, reference_traj, sub)
            self.guides_ctrl.append( Vg[0] )

        # after SVGD, adopt the first guide as MPPI nominal
        self.a_opt = self.guides_ctrl[-1]
        self.a_cov = Sigma_g
        # ------------------------------------------------------




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
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
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