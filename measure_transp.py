####################################################################################################
####################################################################################################
#                                                                                                  #
# importing the functions                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



import numpy as np
import math
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax



####################################################################################################
####################################################################################################
#                                                                                                  #
# problem set-up                                                                                   #
#                                                                                                  #
####################################################################################################
####################################################################################################



def B(t, x):
    """ bod curve with smooth sigmoidal params a(x1), b(x2) """


    x1, x2 = float(x[0]), float(x[1])
    a = 0.4 + 0.4 * (1.0 + math.erf(x1 / math.sqrt(2.0)))
    b = 0.01 + 0.15 * (1.0 + math.erf(x2 / math.sqrt(2.0)))

    return a * (1.0 - np.exp(-b * t))


####################################################################################################



def prior(x):
    """ standard normal prior density in r^2 (up to the exact 1/(2π) constant) """


    return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * np.dot(x, x))




####################################################################################################



def likelihood(x, y, t, sigma_2):
    """g aussian likelihood density (ignoring normalizing consts is fine for mh) """


    resid = y - B(t, x)

    return np.exp(- np.sum(resid**2) / (2.0 * sigma_2))



####################################################################################################



def semi_posterior(x, y, t, sigma_2):
    """ unnormalized posterior density π(x|y) ∝ likelihood * prior """


    return likelihood(x, y, t, sigma_2) * prior(x)



####################################################################################################
####################################################################################################
#                                                                                                  #
# random walk metroplois                                                                           #
#                                                                                                  #
####################################################################################################
####################################################################################################



def rw_metropolis(y, t, sigma_2, n_samples, step_size):
    """ builds full chain and returns acceptance rate"""


    x_chain = np.zeros((n_samples, 2), dtype=float)
    x_current = np.random.normal(0.0, step_size, size=2)
    acc = 0

    p_current = semi_posterior(x_current, y, t, sigma_2)

    for i in range(1, n_samples):
        x_proposal = x_current + np.random.normal(0.0, step_size, size=2)

        p_proposal = semi_posterior(x_proposal, y, t, sigma_2)
        alpha = min(1.0, p_proposal / p_current)

        if np.random.uniform(0.0, 1.0) < alpha:
            x_current = x_proposal
            p_current = p_proposal
            acc += 1

        x_chain[i] = x_current

    return x_chain, acc / n_samples




####################################################################################################



def run_acceptance_table(y, t, sigma_2, step_list, n_steps=25000):
    """ test multiple step sizes """


    rates = []
    for s in step_list:
        _, acc = rw_metropolis(y, t, sigma_2, n_steps, s)
        rates.append(acc)

    return np.array(step_list, dtype=float), np.array(rates, dtype=float)



####################################################################################################
####################################################################################################
#                                                                                                  #
# problem diagnostics                                                                              #
#                                                                                                  #
####################################################################################################
####################################################################################################



def plot_hist_and_trace(samples, start=None, xlim=(-1.0, 2.5), ylim=(-0.5, 4.0)):
    """ 2d histogram + traceplot with consistent axis limits """

    x1, x2 = samples[:,0], samples[:,1]
    plt.figure(figsize=(10, 5.2))

    # histogram
    plt.subplot(1,2,1)
    plt.hist2d(x1, x2, bins=80, range=[xlim, ylim])  # copre tutto il rettangolo
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title("histogram")
    plt.xlim(xlim); plt.ylim(ylim)

    # traceplot
    plt.subplot(1,2,2)
    plt.plot(x1, x2, "-", linewidth=0.3, alpha=0.8)  # linea sottile continua
    if start is not None:
        plt.plot(start[0], start[1], "ro", markersize=6)  # punto rosso grande
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title("traceplot")
    plt.xlim(xlim); plt.ylim(ylim)

    plt.tight_layout()
    plt.show()



####################################################################################################



def ips_asymptotic_variance(phi):
    """ initial positive sequence estimator for var of sample mean """


    phi = np.asarray(phi, dtype=float)
    n = phi.size
    s = phi - phi.mean()
    ac = np.correlate(s, s, mode="full")[n-1:] / n
    K = 0
    while 2*(K+1)+1 < len(ac) and (ac[2*(K+1)] + ac[2*(K+1)+1]) > 0:
        K += 1
    sigma2 = -ac[0] + 2.0 * np.sum(ac[2:2*K+2])
    if sigma2 <= 0: sigma2 = np.var(phi)

    return sigma2



####################################################################################################



def clt_ci_mean(phi, z=1.96):
    """ 95% ci for mean using ips variance """


    n = len(phi)
    m = float(np.mean(phi))
    s2 = float(ips_asymptotic_variance(phi))
    half = z * np.sqrt(s2 / n)

    return m, (m - half, m + half)



####################################################################################################
####################################################################################################
#                                                                                                  #
# polynomial basis                                                                                 #
#                                                                                                  #
####################################################################################################
####################################################################################################



def _poly_basis_1d(x1, deg):
    """ returns [1, x1, x1^2, ..., x1^deg] with shape (B, deg+1) """


    B = x1.shape[0]
    if deg < 0:  # safeguard
        return jnp.ones((B, 1))
    powers = jnp.arange(deg + 1, dtype=jnp.int32)[None, :]

    return jnp.power(x1[:, None], powers)



####################################################################################################
####################################################################################################
#                                                                                                  #
# triangular map 2d                                                                                #
#                                                                                                  #
####################################################################################################
####################################################################################################



@jax.tree_util.register_pytree_node_class
class TriMap2D:
    """
    t1(x1) = exp(s1) * x1 + m1
    t2(x1,x2) = exp(s2(x1)) * x2 + m2(x1)
    """


    def __init__(self, deg: int, key=jax.random.PRNGKey(0), scale=0.01):
        self.deg = int(deg)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        F = self.deg + 1
        self.m1 = scale * jax.random.normal(k1, ())
        self.s1 = scale * jax.random.normal(k2, ())
        self.m2 = scale * jax.random.normal(k3, (F,))
        self.s2 = scale * jax.random.normal(k4, (F,))

    def apply(self, x: jnp.ndarray):
        x1, x2 = x[:, 0], x[:, 1]
        a1 = jnp.exp(self.s1)
        y1 = a1 * x1 + self.m1
        Phi = _poly_basis_1d(x1, self.deg)
        m2x = Phi @ self.m2
        s2x = Phi @ self.s2
        a2  = jnp.exp(s2x)
        y2  = a2 * x2 + m2x
        logdet = jnp.log(a1) + s2x
        y = jnp.stack([y1, y2], axis=1)
        return y, logdet

    # pytree leaves: only floats/arrays (m1,s1,m2,s2); deg is static aux
    def tree_flatten(self):
        children = (self.m1, self.s1, self.m2, self.s2)
        aux = self.deg
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        m1, s1, m2, s2 = children
        obj = cls(aux)
        obj.m1, obj.s1, obj.m2, obj.s2 = m1, s1, m2, s2
        return obj



####################################################################################################
####################################################################################################
#                                                                                                  #
# kl objective and training                                                                        #
#                                                                                                  #
####################################################################################################
####################################################################################################



def kl_objective(params, x_batch, log_g_tilde):
    """ theorem 1.3 objective """


    y, logdet = params.apply(x_batch)

    return jnp.mean(-log_g_tilde(y) - logdet)



####################################################################################################



def train_step(params: TriMap2D, opt_state, x_batch, log_g_tilde, optimizer):
    """ single training step """


    val, grads = jax.value_and_grad(kl_objective)(params, x_batch, log_g_tilde)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, val



####################################################################################################



def empirical_KL(params: TriMap2D, x_all: jnp.ndarray, log_g_tilde):
    """ empirical kl divergence """


    y, logdet = params.apply(x_all)

    return float(jnp.mean(-log_g_tilde(y) - logdet))



####################################################################################################
####################################################################################################
#                                                                                                  #
# bod model in jax                                                                                 #
#                                                                                                  #
####################################################################################################
####################################################################################################



def B_jax(t_scalar, x):
    """ jax translation """


    x1, x2 = x[:, 0], x[:, 1]
    a = 0.4 + 0.4 * (1.0 + jax.scipy.special.erf(x1 / jnp.sqrt(2.0)))
    b = 0.01 + 0.15 * (1.0 + jax.scipy.special.erf(x2 / jnp.sqrt(2.0)))

    return a * (1.0 - jnp.exp(-b * t_scalar))




####################################################################################################
####################################################################################################
#                                                                                                  #
# log g tilde builder                                                                              #
#                                                                                                  #
####################################################################################################
####################################################################################################



def make_log_g_tilde(t_np, y_np, sigma2):
    """ build log g tilde given data """


    t_j = jnp.asarray(t_np)
    y_j = jnp.asarray(y_np)

    def log_g_tilde(x):  # x: (B,2)
        # prior n(0,i) without constant
        log_prior = -0.5 * jnp.sum(x**2, axis=1)
        # gaussian likelihood without constant
        Bvals = jnp.stack([B_jax(ti, x) for ti in t_j], axis=1)  # (B, len(t))
        resid = y_j[None, :] - Bvals
        log_like = - jnp.sum(resid**2, axis=1) / (2.0 * sigma2)
        return log_prior + log_like

    return log_g_tilde



####################################################################################################
####################################################################################################
#                                                                                                  #
# exact inverse & proposal log-density via change of variables                                     #
#                                                                                                  #
####################################################################################################
####################################################################################################



def inverse_apply(params: TriMap2D, y: jnp.ndarray):
    """ y: (B,2) -> x: (B,2) and logdet forward evaluated at x """


    if y.ndim == 1:
        y = y[None, :]
    y1, y2 = y[:, 0], y[:, 1]

    # invert t1: y1 = a1 * x1 + m1
    a1 = jnp.exp(params.s1)                      # scalar
    x1 = (y1 - params.m1) / a1                   # (B,)

    # invert t2: y2 = exp(s2(x1)) * x2 + m2(x1)
    Phi = _poly_basis_1d(x1, params.deg)         # (B, F)
    s2x = Phi @ params.s2                        # (B,)
    m2x = Phi @ params.m2                        # (B,)
    a2  = jnp.exp(s2x)                           # (B,)
    x2  = (y2 - m2x) / a2                        # (B,)

    # logdet forward at x
    logdet = jnp.log(a1) + s2x                   # (B,)

    x = jnp.stack([x1, x2], axis=1)

    return x, logdet



####################################################################################################



def log_g_tilde_proposal(params: TriMap2D, y: jnp.ndarray):
    """ unnormalized log-density of g = T#eta evaluated at y via inverse map """
    

    x, logdet = inverse_apply(params, y)

    return -0.5 * jnp.sum(x**2, axis=1) - logdet   # drop constant of eta



####################################################################################################
####################################################################################################
#                                                                                                  #
# sampler from proposal g                                                                          #
#                                                                                                  #
####################################################################################################
####################################################################################################



def sample_from_proposal(params: TriMap2D, n: int, key):
    """ draw n samples from g = T#eta by pushing forward z ~ n(0,i) """


    z = jax.random.normal(key, (n, 2))
    y, _ = params.apply(z)

    return y  # (n,2) jnp



####################################################################################################
####################################################################################################
#                                                                                                  #
# independence sampler metropolis–hastings (using exact log g~)                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def independence_sampler_MT(params: TriMap2D, log_f_tilde, n_steps=25_000, x0=None, seed=0):
    """
    independence sampler with proposal g = T#eta:
      y' ~ g,  alpha = min(1, f~(y')/f~(x) * g~(x)/g~(y')).
    all logs are used for numerical stability.
    returns chain (n_steps, 2) and acceptance rate
    """


    key = jax.random.PRNGKey(seed)

    # init state: use proposal draw if x0 is none
    if x0 is None:
        key, k = jax.random.split(key)
        x0 = np.array(sample_from_proposal(params, 1, k))[0]
    x = np.asarray(x0, dtype=float)

    # helpers returning python floats
    def _logf(z):
        z = jnp.asarray(z[None, :])
        return float(log_f_tilde(z)[0])

    def _logg(z):
        z = jnp.asarray(z[None, :])
        return float(log_g_tilde_proposal(params, z)[0])

    logf_x = _logf(x)
    logg_x = _logg(x)

    chain = np.zeros((n_steps, 2), dtype=float)
    chain[0] = x
    acc = 0

    for i in range(1, n_steps):
        key, k = jax.random.split(key)
        y_prop = np.array(sample_from_proposal(params, 1, k))[0]
        logf_y = _logf(y_prop)
        logg_y = _logg(y_prop)

        # log acceptance ratio
        log_alpha = (logf_y - logf_x) + (logg_x - logg_y)
        if np.log(np.random.rand()) < min(0.0, log_alpha):
            x, logf_x, logg_x = y_prop, logf_y, logg_y
            acc += 1

        chain[i] = x

    acc_rate = acc / max(n_steps - 1, 1)

    return chain, acc_rate




####################################################################################################
####################################################################################################
#                                                                                                  #
# mixed sampler                                                                                    #
#                                                                                                  #
####################################################################################################
####################################################################################################



def mixed_rw_tmis_sampler(params: TriMap2D, log_f_tilde, step_size=0.1, gamma=0.5,
                          n_steps=25_000, x0=None, seed=0):
    """
    MH with mixed proposal:
      - with prob. gamma: RWM x' = x + N(0, step_size^2 I)
          alpha = min(1, f~(x')/f~(x))
      - with prob. 1-gamma: independence from g = T#eta
          alpha = min(1, f~(y)/f~(x) * g~(x)/g~(y))
    returns: chain, overall_acceptance, rwm_accept, tmis_accept, rwm_tries, tmis_tries
    """


    key = jax.random.PRNGKey(seed)

    # init state
    if x0 is None:
        key, k = jax.random.split(key)
        x0 = np.array(sample_from_proposal(params, 1, k))[0]
    x = np.asarray(x0, dtype=float)

    def _logf(z):
        z = jnp.asarray(z[None, :])
        return float(log_f_tilde(z)[0])

    def _logg(z):
        z = jnp.asarray(z[None, :])
        return float(log_g_tilde_proposal(params, z)[0])

    logf_x = _logf(x)
    logg_x = _logg(x)  # used only for TMIS branch

    chain = np.zeros((n_steps, 2), dtype=float)
    chain[0] = x

    acc_all = 0
    acc_rwm = 0
    acc_tmis = 0
    tries_rwm = 0
    tries_tmis = 0

    for i in range(1, n_steps):
        key, kflip, kprop = jax.random.split(key, 3)
        do_rwm = bool(jax.random.bernoulli(kflip, p=gamma))

        if do_rwm:
            # RWM proposal
            prop = x + np.random.normal(0.0, step_size, size=2)
            logf_prop = _logf(prop)

            log_alpha = logf_prop - logf_x
            tries_rwm += 1
            if np.log(np.random.rand()) < min(0.0, log_alpha):
                x = prop
                logf_x = logf_prop
                # logg_x unchanged (not needed for RWM)
                acc_all += 1
                acc_rwm += 1

        else:
            # TMIS (independence) proposal
            y_prop = np.array(sample_from_proposal(params, 1, kprop))[0]
            logf_y = _logf(y_prop)
            logg_y = _logg(y_prop)

            log_alpha = (logf_y - logf_x) + (logg_x - logg_y)
            tries_tmis += 1
            if np.log(np.random.rand()) < min(0.0, log_alpha):
                x = y_prop
                logf_x = logf_y
                logg_x = logg_y
                acc_all += 1
                acc_tmis += 1

        chain[i] = x

    overall_acc = acc_all / max(n_steps - 1, 1)
    rwm_acc = (acc_rwm / tries_rwm) if tries_rwm > 0 else 0.0
    tmis_acc = (acc_tmis / tries_tmis) if tries_tmis > 0 else 0.0

    return chain, overall_acc, rwm_acc, tmis_acc, tries_rwm, tries_tmis


