import jax
import jax.numpy as jnp
import equinox as eqx

class AbstractFractionalKernel(eqx.Module):
    """Base class for all discrete fractional memory kernels."""
    alpha: float
    history_length: int
    weights: jnp.ndarray

    def __call__(self, history_buffer):
        """Applies the non-Markovian memory convolution."""
        # history_buffer shape: (history_length, res_size)
        # Returns the fractional memory vector of shape (res_size,)
        return jnp.einsum('j,jk->k', self.weights, history_buffer)

class GLKernel(AbstractFractionalKernel):
    """Standard Grünwald-Letnikov fractional difference kernel."""
    def __init__(self, alpha, history_length):
        self.alpha = alpha
        self.history_length = history_length
        self.weights = self._compute_weights()

    def _compute_weights(self):
        # Recursive GL coefficient generation: c_j = (1 - (1+alpha)/j) * c_{j-1}
        coeffs = [1.0]
        for j in range(1, self.history_length):
            coeffs.append((1.0 - (1.0 + self.alpha) / j) * coeffs[-1])
        # We slice [1:] because c_0 is applied to the current step, 
        # while the kernel evaluates the history t-1 to t-L.
        return jnp.array(coeffs[1:])

class L1CaputoKernel(AbstractFractionalKernel):
    """
    L1 finite difference scheme for the Caputo fractional derivative.
    Utilizes a different power-law scaling for the memory weights.
    """
    dt: float

    def __init__(self, alpha, history_length, dt=1.0):
        self.alpha = alpha
        self.history_length = history_length
        self.dt = dt
        self.weights = self._compute_weights()

    def _compute_weights(self):
        # L1 scheme weights: b_j = j^{1-alpha} - (j-1)^{1-alpha}
        j_indices = jnp.arange(1, self.history_length + 1)
        b_j = (j_indices)**(1 - self.alpha) - (j_indices - 1)**(1 - self.alpha)
        
        # Scaling factor includes the time step dt and Gamma(2-alpha)
        # For simplicity in this pseudo-code, we isolate the relative weight decay.
        scaling = 1.0 / (jax.scipy.special.gamma(2 - self.alpha) * (self.dt ** self.alpha))
        return b_j[:-1] * scaling
    
    
class ParameterizedFractionalReservoir(eqx.Module):
    W_res: jnp.ndarray 
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel  # Injected dependency
    history_length: int

    def __init__(self, in_features, res_size, fractional_operator, key):
        k1, k2 = jax.random.split(key)
        
        # Initialize macroscopic connectivity near the edge of chaos
        self.W_res = jax.random.normal(k1, (res_size, res_size)) * 0.95 / jnp.sqrt(res_size)
        self.W_in = jax.random.normal(k2, (res_size, in_features))
        
        self.fractional_operator = fractional_operator
        self.history_length = fractional_operator.history_length

    def __call__(self, u_t, x_history):
        # 1. Delegate the non-Markovian temporal integration to the injected operator
        fractional_memory = self.fractional_operator(x_history[:-1])
        
        # 2. Compute the spatial/macroscopic activation
        current_state = x_history[0]
        activation = jax.nn.sigmoid(jnp.dot(self.W_res, current_state) + jnp.dot(self.W_in, u_t))
        
        # 3. State update
        x_next = current_state - fractional_memory + activation
        
        # 4. Update rolling history tensor
        new_history = jnp.roll(x_history, shift=1, axis=0)
        new_history = new_history.at[0].set(x_next)
        
        return x_next, new_history
    
    
class PhantomBrain(eqx.Module):
    reservoir: ParameterizedFractionalReservoir
    readout: TopologicalReadout # As defined previously

    def __init__(self, in_features, res_size, out_features, fractional_operator, key):
        k1, k2 = jax.random.split(key)
        self.reservoir = ParameterizedFractionalReservoir(in_features, res_size, fractional_operator, k1)
        self.readout = TopologicalReadout(res_size, out_features, k2)

    def __call__(self, U_drive):
        def step_fn(history_buffer, u_t):
            x_next, updated_buffer = self.reservoir(u_t, history_buffer)
            y_hat_t = self.readout(x_next)
            return updated_buffer, (x_next, y_hat_t)

        init_history = jnp.zeros((self.reservoir.history_length, self.reservoir.W_res.shape[0]))
        _, (X_states, Y_hat) = jax.lax.scan(step_fn, init_history, U_drive)
        
        return Y_hat

# --- Example Implementation ---
key = jax.random.PRNGKey(42)

# Experiment A: Grünwald-Letnikov Operator
gl_operator = GLKernel(alpha=0.8, history_length=500)
model_gl = PhantomBrain(in_features=1, res_size=1000, out_features=64, fractional_operator=gl_operator, key=key)

# Experiment B: L1 Caputo Scheme
caputo_operator = L1CaputoKernel(alpha=0.8, history_length=500, dt=0.01)
model_caputo = PhantomBrain(in_features=1, res_size=1000, out_features=64, fractional_operator=caputo_operator, key=key)