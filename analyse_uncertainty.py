import sys

from tqdm import tqdm
from scipy.special import softmax, gammaln, digamma, rel_entr

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


def calc_kl_with_uniform(alphas):
    """
    Attempt at implementation of LDDP with uniform Dirichlet as invariant measure.
    Equal to entropy + a constant making it positive (unlike differential entropy).
    """
    
    N, K = alphas.shape

    def kl_divergence_dirichlets(alphas, betas):
        alpha_0 = alphas.sum(axis=-1, keepdims=True)
        beta_0 = betas.sum(axis=-1, keepdims=True)

        return gammaln(alpha_0)                                                                             \
            - (gammaln(alphas) - gammaln(beta_0)).sum(axis=-1, keepdims=True)                               \
            + gammaln(betas).sum(axis=-1, keepdims=True)                                                    \
            + ( (alphas - betas) * (digamma(alphas) - digamma(alpha_0)) ).sum(axis=-1, keepdims=True)

    # the parameters of the uniform Dirichlet
    betas = np.full((1, K), 1)

    return kl_divergence_dirichlets(alphas, betas).squeeze()    


def ml_dirichlet_based_uncertainty(logits):
    """
    """
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
    kl_uniform = calc_kl_with_uniform(alphas)

    print(f"Mean variance is {variances.mean()}; with standard deviation of {variances.std()} .")
    print(f"Mean entropy is {entropies.mean()}; with standard deviation of {entropies.std()} .")
    print(f"Mean KL w/ uniform-dirichlet is {kl_uniform.mean()}; with standard deviation of {kl_uniform.std()} .")

    np.save("entropies.npy", entropies)

    sns.kdeplot(data=entropies)
    plt.savefig("entropies.png")


def lakshminarayanan_uncertainty(logits):
    """
    """

    predictions = softmax(logits, axis=-1)

    # average over model instances to obtain ensemble prediction
    prediction_E = predictions.mean(axis=1)

    # calculate kl divergence between predictions and ensemble prediction 
    divergences = rel_entr(predictions.swapaxes(0,1), prediction_E).sum(-1)

    # average over model instances to obtain disagreements
    disagreements = divergences.mean(axis=0)


    print(f"Mean disagreement is {disagreements.mean()}; with standard deviation of {disagreements.std()}")

    np.save("disagreements.npy", disagreements)

    sns.kdeplot(data=disagreements)
    plt.savefig("disagreements.png")


if __name__ == "__main__":

    logits = []
    for i, arg in enumerate(sys.argv[1:]):
        print(f"loading {arg}")
        logits_ = np.loadtxt(arg)
        logits.append(logits_)

    logits = np.stack(logits, axis=1) # N, M, K
    N, M, K = logits.shape

    lakshminarayanan_uncertainty(logits)
    ml_dirichlet_based_uncertainty(logits)
        


