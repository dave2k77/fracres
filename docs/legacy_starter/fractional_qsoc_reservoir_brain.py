class qSOCFractionalReservoir(eqx.Module):
    W_res: jnp.ndarray 
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    
    # qSOC Hyperparameters
    E_crit: float      # Target critical energy
    tau_b: float       # Homeostatic time constant
    gamma: float       # Adaptation gain

    def __init__(self, in_features, res_size, fractional_operator, E_crit, tau_b, gamma, key):
        k1, k2 = jax.random.split(key)
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * 0.95 / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length
        
        self.E_crit = E_crit
        self.tau_b = tau_b
        self.gamma = gamma

    def __call__(self, u_t, x_history, b_t, dt):
        """
        Extended signature to carry the dynamic adaptive threshold b_t.
        """
        # 1. Non-Markovian temporal fractional integration
        fractional_memory = self.fractional_operator(x_history[:-1])
        
        # 2. Current state evaluation
        current_state = x_history[0]
        
        # 3. Macroscopic activation with the dynamic qSOC threshold b_t
        # Firing rate is now regulated by intrinsic homeostatic feedback
        activation = jax.nn.sigmoid(
            jnp.dot(self.W_res, current_state) + jnp.dot(self.W_in, u_t) + b_t
        )
        
        # 4. State step via the fractional operator
        x_next = current_state - fractional_memory + activation
        
        # 5. qSOC Update: Compute instantaneous energy and update threshold
        current_energy = jnp.mean(jnp.square(x_next))
        # Discrete Euler step for the homeostatic threshold ODE
        db = (-b_t + self.gamma * (self.E_crit - current_energy)) / self.tau_b
        b_next = b_t + db * dt
        
        # 6. Update rolling history buffer
        new_history = jnp.roll(x_history, shift=1, axis=0)
        new_history = new_history.at[0].set(x_next)
        
        return x_next, new_history, b_next
    
def simulate_phantom_brain_qsoc(model, U_drive, dt=0.01):
    res_size = model.reservoir.W_res.shape[0]
    
    def step_fn(carry, u_t):
        x_history, b_t = carry
        
        # Process step through qSOC fractional reservoir
        x_next, updated_history, b_next = model.reservoir(u_t, x_history, b_t, dt)
        
        # Project to lower-dimensional output space
        y_hat_t = model.readout(x_next)
        
        return (updated_history, b_next), (x_next, y_hat_t, b_t)

    # Initialize carry states
    init_history = jnp.zeros((model.reservoir.history_length, res_size))
    init_bias = jnp.zeros((res_size,)) # Start with neutral baseline thresholds
    
    _, (X_states, Y_hat, B_thresholds) = jax.lax.scan(
        step_fn, (init_history, init_bias), U_drive
    )
    
    return Y_hat, B_thresholds

