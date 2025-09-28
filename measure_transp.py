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

