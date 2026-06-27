# Knowledge Base: Fractional Reservoir Computing for Critical Brain Dynamics
**Theoretical Framework, Function Spaces, and High-Performance Neural Mass Models**

---

## 1. Fundamentals of Reservoir Computing (RC)

Reservoir Computing (RC) is a computational framework derived from recurrent neural network (RNN) theory, primarily conceptualized through Echo State Networks (ESNs) and Liquid State Machines (LSMs). It is designed to process temporal data and model dynamical systems without the computational overhead of training recurrent weights via backpropagation through time (BPTT).

### 1.1 Core Architecture
A standard reservoir computer consists of three distinct components:
1. **Input Layer:** Projects a low-dimensional driving signal into the high-dimensional state space of the reservoir via a fixed random weight matrix $\mathbf{W}_{in}$.
2. **The Reservoir:** A large, sparse, and randomly connected network of non-linear nodes ($\mathbf{W}_{res}$). Its weights are initialized once and remain fixed. The reservoir effectively projects the input history into a high-dimensional space, acting as a temporal kernel.
3. **Readout Layer:** A linear output layer ($\mathbf{W}_{out}$) that maps the high-dimensional transient states of the reservoir to the desired target output. **This is the only part of the network that is trained.**

### 1.2 Mathematical Formulation
Let $\mathbf{u}(t) \in \mathbb{R}^{N_u}$ denote the input vector at discrete time $t$, $\mathbf{x}(t) \in \mathbb{R}^{N_x}$ the state of the reservoir, and $\mathbf{y}(t) \in \mathbb{R}^{N_y}$ the output. The state update of a leaky-integrator reservoir is governed by:

$$\mathbf{x}(t) = (1 - \alpha)\mathbf{x}(t-1) + \alpha f(\mathbf{W}_{in}\mathbf{u}(t) + \mathbf{W}_{res}\mathbf{x}(t-1) + \mathbf{b})$$

Where:
* $\mathbf{W}_{in} \in \mathbb{R}^{N_x \times N_u}$ and $\mathbf{W}_{res} \in \mathbb{R}^{N_x \times N_x}$ are fixed random matrices.
* $\mathbf{b} \in \mathbb{R}^{N_x}$ is a fixed bias vector.
* $f(\cdot)$ is the element-wise non-linear activation function (typically $\tanh$).
* $\alpha \in (0, 1]$ is the leak rate, dictating the timescale of the reservoir dynamics.

The output equation is defined as:

$$\mathbf{y}(t) = \mathbf{W}_{out} [\mathbf{x}(t); \mathbf{u}(t)]$$

### 1.3 The Echo State Property (ESP)
For a reservoir computer to function predictably as a temporal filter, it must satisfy the **Echo State Property (ESP)**. The ESP dictates that the asymptotic state of the reservoir should depend exclusively on the history of the input signal, completely "forgetting" its initial conditions over time. 

A sufficient condition for the ESP is that the spectral radius of the reservoir weight matrix is less than unity:

$$\rho(\mathbf{W}_{res}) < 1$$

Where $\rho(\mathbf{W}_{res})$ is the largest absolute eigenvalue of $\mathbf{W}_{res}$. Scaling $\mathbf{W}_{res}$ near the "edge of chaos" ($\rho \approx 1$) maximizes its fading memory capacity and information processing capability.

### 1.4 Linear Readout Training
Because internal weights are fixed, the states $\mathbf{x}(t)$ can be collected over the entire training sequence into a state matrix $\mathbf{X}$. Training reduces to a simple Ridge Regression (Tikhonov regularization) problem, admitting a fast, closed-form analytic solution:

$$\mathbf{W}_{out} = \mathbf{Y}_{target} \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \beta \mathbf{I})^{-1}$$

Where $\beta$ is a regularization parameter and $\mathbf{I}$ is the identity matrix.

---

## 2. Biological Analogs: From Spiking to Mean-Field Models

### 2.1 Liquid State Machines (LSMs)
While ESNs use continuous-rate artificial neurons, LSMs use biologically realistic Spiking Neural Networks (SNNs)—typically Leaky Integrate-and-Fire (LIF) neurons—as the reservoir. When an input spike train perturbs this "liquid", the network generates high-dimensional transient spatio-temporal patterns of spikes. To perform a computation, a readout mechanism extracts information from the low-pass filtered spike trains or membrane potentials.

### 2.2 Neural Mass and Field Models (NMMs / NFMs)
To model macroscopic brain dynamics (such as EEG, MEG, or fMRI) at a population level without the computational bottleneck of discrete spike events, we can embed continuous Neural Mass Models (e.g., Wilson-Cowan) or Neural Field Models (Amari equations) directly into the reservoir.

A fractional neural mass reservoir consisting of $N$ coupled neural masses describes the interactions between excitatory ($E$) and inhibitory ($I$) subpopulations:

$$\tau_E^\alpha \mathcal{D}^\alpha_t E_i(t) = -E_i(t) + \mathcal{S}_E \left( \sum_{j=1}^N W_{ij}^{EE} E_j(t) - \sum_{j=1}^N W_{ij}^{EI} I_j(t) + W_{in}^{E} u_i(t) \right)$$

$$\tau_I^\alpha \mathcal{D}^\alpha_t I_i(t) = -I_i(t) + \mathcal{S}_I \left( \sum_{j=1}^N W_{ij}^{IE} E_j(t) - \sum_{j=1}^N W_{ij}^{II} I_j(t) \right)$$

Where $\mathcal{D}^\alpha_t$ is a continuous temporal fractional derivative of order $\alpha \in (0,1)$, $\tau$ represents population time constants, and $\mathcal{S}(\cdot)$ is a non-linear sigmoid activation function representing the population firing rate.

---

## 3. Mathematical Framework of the Fractional Reservoir (F-RC)

Embedding fractional dynamics directly into the reservoir nodes shifts the generation of long-range dependence (LRD) from network topology to the fundamental integro-differential operators of the nodes themselves.

### 3.1 Continuous Formulation
Let $\mathbf{x}(t) \in \mathbb{R}^{N_x}$ be the internal state of the reservoir, and $\mathbf{u}(t) \in \mathbb{R}^{N_u}$ be the external stochastic driving input. The internal dynamics are governed by the Caputo fractional derivative $\mathcal{D}^\alpha_t$:

$$\mathcal{D}^\alpha_t \mathbf{x}(t) = -\mathbf{\Lambda} \mathbf{x}(t) + f(\mathbf{W}_{res} \mathbf{x}(t) + \mathbf{W}_{in} \mathbf{u}(t))$$

Where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_{N_x})$ represents node-specific decay rates.

### 3.2 Discrete-Time Approximation
Using the Grünwald-Letnikov (GL) definition, the discrete fractional difference is defined as:

$$\Delta^\alpha \mathbf{x}_k = \frac{1}{h^\alpha} \sum_{j=0}^{k} (-1)^j \binom{\alpha}{j} \mathbf{x}_{k-j}$$

Where $h$ is the integration step, and the generalized binomial coefficients are computed recursively:

$$c_0 = 1, \quad c_j = \left(1 - \frac{1+\alpha}{j}\right) c_{j-1}$$

This yields the non-Markovian discrete state update rule:

$$\mathbf{x}_k = \mathbf{x}_{k-1} - \sum_{j=2}^{k} c_j \mathbf{x}_{k-j} + h^\alpha \left[ -\mathbf{\Lambda} \mathbf{x}_{k-1} + f(\mathbf{W}_{res} \mathbf{x}_{k-1} + \mathbf{W}_{in} \mathbf{u}_{k-1}) \right]$$

### 3.3 The Short Memory Principle (Truncation)
To mitigate the $\mathcal{O}(k^2)$ computational explosion over long time windows, a fixed history window $L$ is applied to truncate the non-local summation:

$$\sum_{j=2}^{\min(k, L)} c_j \mathbf{x}_{k-j}$$

This window can be maintained efficiently as a rolling buffer tensor.

---

## 4. The Functional Analysis Perspective: Besov Space Regularity

When modeling critical brain dynamics driven by a heavy-tailed stochastic process (such as fractional Brownian motion, fBm), the state trajectories live within specific fractional function spaces.

### 4.1 From Sobolev-Slobodeckij to Besov Spaces
The fractional Sobolev-Slobodeckij space $W^{s,p}(\Omega)$ contains functions whose Gagliardo seminorm is finite:

$$[V]_{s,p} = \left( \int_\Omega \int_\Omega \frac{|V(x) - V(y)|^p}{|x - y|^{d + sp}} \, dx \, dy \right)^{1/p}$$

Evaluating this double spatial integral scales quadratically. To optimize this computationally, we utilize the equivalent Besov space $B^s_{p,q}$ via the **Littlewood-Paley decomposition**, transforming the non-local topology evaluation into a dyadic frequency band filtering problem solvable in $\mathcal{O}(N \log N)$ via FFTs.

### 4.2 Strict Trajectory Regularity Bounds
To accurately model scale-free critical dynamics without numerical divergence or trivial states, the parameters of the function space must be strictly bound relative to the system parameters:
1. **Integrability Bound:** $p < \alpha$ (where $\alpha$ is the order of the fractional derivative).
2. **Smoothness Bound:** $s < H$ (where $H$ is the Hurst exponent of the driving fBm noise).

### 4.3 Littlewood-Paley Characterization
Let $\mathbf{\hat{y}}$ be the predicted trajectory. We define a sequence of smooth, band-pass filters $\phi_j(\omega)$ in the frequency domain localized around $|\omega| \sim 2^j$. The dyadic blocks are defined as:

$$\Delta_j \mathbf{\hat{y}} = \mathcal{F}^{-1} (\phi_j \cdot \mathcal{F}(\mathbf{\hat{y}}))$$

The total Besov norm regularizer is then aggregated as:

$$||\mathbf{\hat{y}}||_{B^s_{p,q}} = ||\mathcal{F}^{-1}(\psi \cdot \mathcal{F}(\mathbf{\hat{y}}))||_{L^p} + \left( \sum_{j=1}^{J} \left( 2^{js} ||\Delta_j \mathbf{\hat{y}}||_{L^p} \right)^q \right)^{1/q}$$

---

## 5. Quasi-Self-Organized Criticality (qSOC)

Biological brains exhibit *quasi*-SOC, where continuous feedback mechanisms dynamically pull the system back toward the critical phase transition whenever external non-local driving forces threaten to cause hyper-excitability (supercriticality) or silence (subcriticality).

### 5.1 Activity-Dependent Intrinsic Cellular Regulation
We define the recent macroscopic energy of the reservoir over a homeostatic window $\tau_{soc}$:

$$E_{avg}(t) = \frac{1}{\tau_{soc}} \int_{t-\tau_{soc}}^{t} ||\mathbf{x}(\tau)||^2 d\tau$$

The qSOC mechanism adaptively perturbs the internal threshold bias vector $\mathbf{b}(t)$ of the neural mass populations to actively enforce homeostatic stability:

$$\tau_{b} \frac{d\mathbf{b}(t)}{dt} = -\mathbf{b}(t) + \gamma \left( E_{crit} - ||\mathbf{x}(t)||^2 \right)$$

This non-linear feedback loop stabilizes the trajectory inside the tightest bounds of the Besov space $B^s_{p,q}$, acting as an automatic topological anchor.

---

## 6. High-Performance Computational Architecture (JAX / Equinox)

Below is the complete, modular pseudocode implementation designed for high-performance array operations, compilation (`jit`), and vectorization (`vmap`).

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# ==========================================
# 1. PARAMETERIZED FRACTIONAL KERNELS
# ==========================================

class AbstractFractionalKernel(eqx.Module):
    alpha: float
    history_length: int
    weights: jnp.ndarray

    def __call__(self, history_buffer):
        """Applies the non-Markovian memory convolution matrix multiplication."""
        return jnp.einsum('j,jk->k', self.weights, history_buffer)

class GLKernel(AbstractFractionalKernel):
    """Grünwald-Letnikov power-law memory operator."""
    def __init__(self, alpha, history_length):
        self.alpha = alpha
        self.history_length = history_length
        self.weights = self._compute_weights()

    def _compute_weights(self):
        coeffs = [1.0]
        for j in range(1, self.history_length):
            coeffs.append((1.0 - (1.0 + self.alpha) / j) * coeffs[-1])
        return jnp.array(coeffs[1:])

class L1CaputoKernel(AbstractFractionalKernel):
    """L1 Finite Difference Scheme for the Caputo fractional derivative."""
    dt: float

    def __init__(self, alpha, history_length, dt=1.0):
        self.alpha = alpha
        self.history_length = history_length
        self.dt = dt
        self.weights = self._compute_weights()

    def _compute_weights(self):
        j_indices = jnp.arange(1, self.history_length + 1)
        b_j = (j_indices)**(1 - self.alpha) - (j_indices - 1)**(1 - self.alpha)
        scaling = 1.0 / (jax.scipy.special.gamma(2 - self.alpha) * (self.dt ** self.alpha))
        return b_j[:-1] * scaling

# ==========================================
# 2. FRACTIONAL RESERVOIR WITH qSOC MECHANISM
# ==========================================

class qSOCFractionalReservoir(eqx.Module):
    W_res: jnp.ndarray 
    W_in: jnp.ndarray
    fractional_operator: AbstractFractionalKernel
    history_length: int
    E_crit: float      
    tau_b: float       
    gamma: float       

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
        # Evaluate temporal nonlocality via the memory kernel
        fractional_memory = self.fractional_operator(x_history[:-1])
        current_state = x_history[0]
        
        # Mean-field population activation incorporating adaptive threshold b_t
        activation = jax.nn.sigmoid(
            jnp.dot(self.W_res, current_state) + jnp.dot(self.W_in, u_t) + b_t
        )
        
        # State transition 
        x_next = current_state - fractional_memory + activation
        
        # Continuous qSOC adjustment
        current_energy = jnp.mean(jnp.square(x_next))
        db = (-b_t + self.gamma * (self.E_crit - current_energy)) / self.tau_b
        b_next = b_t + db * dt
        
        # Shift rolling memory tensor
        new_history = jnp.roll(x_history, shift=1, axis=0)
        new_history = new_history.at[0].set(x_next)
        
        return x_next, new_history, b_next

# ==========================================
# 3. READOUT LAYER AND PHANTOM BRAIN MODEL
# ==========================================

class TopologicalReadout(eqx.Module):
    W_out: jnp.ndarray

    def __init__(self, res_size, out_features, key):
        self.W_out = jax.random.normal(key, (out_features, res_size))

    def __call__(self, x_t):
        return jnp.dot(self.W_out, x_t)

class PhantomBrain(eqx.Module):
    reservoir: qSOCFractionalReservoir
    readout: TopologicalReadout

    def __init__(self, in_features, res_size, out_features, fractional_operator, E_crit, tau_b, gamma, key):
        k1, k2 = jax.random.split(key)
        self.reservoir = qSOCFractionalReservoir(in_features, res_size, fractional_operator, E_crit, tau_b, gamma, k1)
        self.readout = TopologicalReadout(res_size, out_features, k2)

    def __call__(self, U_drive, dt=0.01):
        def step_fn(carry, u_t):
            x_history, b_t = carry
            x_next, updated_history, b_next = self.reservoir(u_t, x_history, b_t, dt)
            y_hat_t = self.readout(x_next)
            return (updated_history, b_next), (x_next, y_hat_t, b_t)

        init_history = jnp.zeros((self.reservoir.history_length, self.reservoir.W_res.shape[0]))
        init_bias = jnp.zeros((self.reservoir.W_res.shape[0],))
        
        _, (X_states, Y_hat, B_thresholds) = jax.lax.scan(step_fn, (init_history, init_bias), U_drive)
        return Y_hat

# ==========================================
# 4. TOPOLOGICAL OPTIMIZATION AND TRAINING
# ==========================================

def littlewood_paley_penalty(Y_hat, masks, s, p, q):
    """Computes the exact B^{s}_{p,q} Besov norm regularizer via FFT."""
    Y_fft = jnp.fft.fft(Y_hat, axis=0)
    filtered_fft = masks[..., None] * Y_fft[None, ...]
    delta_j = jnp.real(jnp.fft.ifft(filtered_fft, axis=1))
    
    Lp_norms = jnp.sum(jnp.abs(delta_j + 1e-8)**p, axis=(1, 2))**(1/p)
    j_indices = jnp.arange(1, masks.shape[0] + 1)
    scaled_norms = (2.0 ** (j_indices * s)) * Lp_norms
    return jnp.sum(scaled_norms**q)**(1/q)

def compute_loss(model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg):
    # Enforce strict regularity constraints natively
    p = alpha - 0.05  # p < alpha
    s = H_hurst - 0.05 # s < H
    q = p

    Y_hat = model(U_drive)
    mse_loss = jnp.mean((Y_target - Y_hat)**2)
    besov_loss = littlewood_paley_penalty(Y_hat, masks, s, p, q)
    return mse_loss + lambda_reg * besov_loss

@eqx.filter_jit
def train_step(model, optimizer, opt_state, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, U_drive, Y_target, masks, alpha, H_hurst, lambda_reg
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss