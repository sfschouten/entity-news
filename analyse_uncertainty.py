import sys

from tqdm import tqdm
from scipy.special import softmax, gammaln, digamma
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dirichlet import mle, dirichlet


def calc_variance(alphas):
    alpha_0 = alphas.sum(axis=-1, keepdims=True)
    alphas_tilde = alphas / alpha_0

    variance = alphas_tilde * (1 - alphas_tilde) / (alpha_0 + 1)

    return variance


def calc_entropy(alphas):
    N, K = alphas.shape
    alpha_0 = alphas.sum(axis=-1, keepdims=True)
   
    psi_alpha_0 = digamma(alpha_0)
    psi_alphas = digamma(alphas)

    def log_multivariate_beta(values):
        return gammaln(values).sum(axis=-1, keepdims=True) - gammaln(values.sum(axis=-1, keepdims=True))

    entropy = log_multivariate_beta(alphas)         \
        + (alpha_0 - K) * psi_alpha_0               \
        - ((alphas - 1) * psi_alphas).sum(axis=-1, keepdims=True)

    return entropy.squeeze()


if __name__ == "__main__":

    logits = []
    for i, arg in enumerate(sys.argv[1:]):
        print(f"loading {arg}")
        logits_ = np.loadtxt(arg)
        logits.append(logits_)

    logits = np.stack(logits, axis=1) # N, M, K
    N, M, K = logits.shape

    print("calculating MLE Dirichlet distribution parameters")
    all_alphas = []
    for M_logits in tqdm(logits):
        # softmax logits
        probs = softmax(M_logits, axis=-1)

        try:
            alphas = mle(probs, method='meanprecision')
            all_alphas.append(alphas)
            continue
        except dirichlet.NotConvergingError:
            print("Did not converge with meanprecision, trying with fixedpoint...")

        try:
            alphas = mle(probs, method="fixedpoint")
            all_alphas.append(alphas)
        except dirichlet.NotConvergingError:
            print("fixedpoint also did not converge, skipping (!) this sample")
            print("logits:")
            print(M_logits)
        else:
            print("success!")

    print("calculating variance and entropy")
    alphas = np.stack(all_alphas)

    variances = calc_variance(alphas)
    entropies = calc_entropy(alphas)

    print(f"Mean variance is {variances.mean()}; with standard deviation of {variances.std()} .")
    print(f"Mean entropy is {entropies.mean()}; with standard deviation of {entropies.std()} .")

    np.save("entropies.npy", entropies)

    sns.kdeplot(data=entropies)
    plt.savefig("entropies.png")




