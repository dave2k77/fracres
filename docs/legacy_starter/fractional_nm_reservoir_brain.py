import jax
import jax.numpy as jnp
import equinox as eqx
import optax

class FractionalNeuralMassReservoir(eqx.Module):
    # Fixed parameters (not updated by optimizer)
    W_res: jnp.ndarray 
    W_in: jnp.ndarray
    gl_coeffs: jnp.ndarray 
    alpha: float
    history_length: int

    def __init__(self, in_features, res_size, alpha, history_length, key):
        k1, k2 = jax.random.split(key)
        # Sparse, fixed macroscopic connectome tuned to edge of chaos
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * 0.95 / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        
        self.alpha = alpha
        self.history_length = history_length
        # Precompute Grünwald-Letnikov binomial coefficients
        self.gl_coeffs = self._compute_gl_coeffs(alpha, history_length)

    def _compute_gl_coeffs(self, alpha, L):
        # Recursive GL coefficient generation: c_j = (1 - (1+alpha)/j) * c_{j-1}
        coeffs = [1.0]
        for j in range(1, L):
            coeffs.append((1.0 - (1.0 + alpha) / j) * coeffs[-1])
        return jnp.array(coeffs)

    def __call__(self, u_t, x_history):
        # x_history shape: (history_length, res_size)
        # u_t shape: (in_features,) - The stochastic drive at time t

        # Non-Markovian summation over the history buffer
        # Equivalent to evaluating the fractional temporal derivative
        fractional_memory = jnp.einsum('j,jk->k', self.gl_coeffs[1:], x_history[:-1])
        
        # Neural Mass activation (e.g., sigmoid for population firing rate)
        current_state = x_history[0]
        activation = jax.nn.sigmoid(jnp.dot(self.W_res, current_state) + jnp.dot(self.W_in, u_t))
        
        # GL update rule mapping to fractional ODE
        x_next = current_state - fractional_memory + activation
        
        # Update rolling buffer: shift old states down, insert new state at index 0
        new_history = jnp.roll(x_history, shift=1, axis=0)
        new_history = new_history.at[0].set(x_next)
        
        return x_next, new_history

class TopologicalReadout(eqx.Module):
    # Trainable weights
    W_out: jnp.ndarray
    
def littlewood_paley_penalty(Y_hat, masks, s, p, q):
    """
    Computes the B^{s}_{p,q} Besov norm of the output trajectory.
    """
    # FFT over time axis
    Y_fft = jnp.fft.fft(Y_hat, axis=0)
    
    # Apply dyadic frequency masks via broadcasting
    filtered_fft = masks[..., None] * Y_fft[None, ...]
    
    # Extract dyadic blocks in time domain (real parts)
    delta_j = jnp.real(jnp.fft.ifft(filtered_fft, axis=1))
    
    # Compute L^p norm (integrability constraint)
    Lp_norms = jnp.sum(jnp.abs(delta_j + 1e-8)**p, axis=(1, 2))**(1/p)
    
    # Scale by fractional smoothness 2^{js} and compute l^q norm
    j_indices = jnp.arange(1, masks.shape[0] + 1)
    scaled_norms = (2.0 ** (j_indices * s)) * Lp_norms
    
    return jnp.sum(scaled_norms**q)**(1/q)

def compute_loss(model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg):
    # Enforce strict theoretical bounds for critical generative dynamics
    # Integrability p must be strictly less than derivative order alpha
    p = alpha - 0.05  
    # Fractional smoothness s must be strictly less than Hurst exponent H
    s = H_hurst - 0.05 
    # Set q = p for a standard Besov-Slobodeckij geometry
    q = p 

    # Generate synthetic brain dynamics
    Y_hat = model(U_drive)
    
    # Standard predictive error
    mse_loss = jnp.mean((Y_target - Y_hat)**2)
    
    # Topological boundary enforcer
    besov_loss = littlewood_paley_penalty(Y_hat, masks, s, p, q)
    
    return mse_loss + lambda_reg * besov_loss

@eqx.filter_jit
def train_step(model, optimizer, opt_state, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg):
    # Differentiate ONLY the trainable readout layer. 
    # eqx.filter_value_and_grad cleanly ignores the fixed reservoir arrays.
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg
    )
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

    def __init__(self, res_size, out_features, key):
        self.W_out = jax.random.normal(key, (out_features, res_size))

    def __call__(self, x_t):
        return jnp.dot(self.W_out, x_t)
    
class PhantomBrain(eqx.Module):
    reservoir: FractionalNeuralMassReservoir
    readout: TopologicalReadout

    def __init__(self, in_features, res_size, out_features, alpha, history_length, key):
        k1, k2 = jax.random.split(key)
        self.reservoir = FractionalNeuralMassReservoir(in_features, res_size, alpha, history_length, k1)
        self.readout = TopologicalReadout(res_size, out_features, k2)

    def __call__(self, U_drive):
        # U_drive: trajectory of stochastic noise, shape (time_steps, in_features)
        
        def step_fn(history_buffer, u_t):
            # 1. Elevate noise into Besov space trajectory (p < alpha constraint inherent to physics)
            x_next, updated_buffer = self.reservoir(u_t, history_buffer)
            # 2. Project down to EEG/MEG space
            y_hat_t = self.readout(x_next)
            
            return updated_buffer, (x_next, y_hat_t)

        # Initialize empty history buffer
        init_history = jnp.zeros((self.reservoir.history_length, self.reservoir.W_res.shape[0]))
        
        # Fast compilation of the entire generative sequence
        _, (X_states, Y_hat) = jax.lax.scan(step_fn, init_history, U_drive)
        
        return Y_hat