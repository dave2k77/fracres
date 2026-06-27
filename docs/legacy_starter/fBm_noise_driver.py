import jax
import jax.numpy as jnp
import equinox as eqx

def generate_fbm_increments(time_steps, H, key):
    """
    Generates exact fractional Gaussian noise (fBm increments) 
    using the Davies-Harte method via FFT.
    """
    # 1. Compute the autocovariance sequence of fractional Gaussian noise
    r = jnp.zeros(time_steps)
    idx = jnp.arange(time_steps)
    
    # Fractional covariance formula
    r = r.at[0].set(1.0)
    r = r.at[1:].set(0.5 * ((idx[1:] + 1)**(2*H) - 2*(idx[1:])**(2*H) + (idx[1:] - 1)**(2*H)))
    
    # 2. Construct the circulant matrix eigenvalues via FFT
    c = jnp.concatenate([r, jnp.array([0.0]), r[:0:-1]])
    eigenvalues = jnp.fft.fft(c).real
    # Clip small negative values due to numerical precision limits
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    
    # 3. Standard Gaussian random variables in the frequency domain
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (len(eigenvalues),)) + 1j * jax.random.normal(k2, (len(eigenvalues),))
    
    # 4. Transform back to time domain to get the exact increments
    fgn = jnp.fft.ifft(z * jnp.sqrt(eigenvalues)).real[:time_steps]
    # Standardize variance
    return fgn / jnp.std(fgn)

# --- Execution Simulation ---
time_steps = 2000
res_size = 500
alpha_derivative = 0.75
H_noise = 0.65 # Persistent driving noise

key = jax.random.PRNGKey(2026)
k1, k2 = jax.random.split(key)

# 1. Generate the biologically relevant power-law kernel
biological_kernel = L1CaputoKernel(alpha=alpha_derivative, history_length=200, dt=0.005)

# 2. Instantiate the parameterized reservoir
reservoir = ParameterizedFractionalReservoir(
    in_features=1, 
    res_size=res_size, 
    fractional_operator=biological_kernel, 
    key=k1
)

# 3. Generate the fBm driving signal
fbm_drive = generate_fbm_increments(time_steps, H=H_noise, key=k2)
fbm_drive = fbm_drive[..., None] # Shape: (time_steps, 1)

# 4. Run the simulation over time using jax.lax.scan
def step_fn(history_buffer, u_t):
    x_next, updated_buffer = reservoir(u_t, history_buffer)
    return updated_buffer, x_next

init_history = jnp.zeros((reservoir.history_length, res_size))
_, X_states = jax.lax.scan(step_fn, init_history, fbm_drive)