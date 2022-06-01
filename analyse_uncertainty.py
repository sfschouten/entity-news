import sys

from tqdm import tqdm
from scipy.special import beta, digamma
import numpy as np

from dirichlet import mle


def calc_variance(alphas):
    """
    alphas (N, K)
    """
    
    alpha_0 = alphas.sum(dim=1, keepdims=True)
    alphas_tilde = alphas / alpha_0

    variance = alphas_tilde * (1 - alphas_tilde) / (alpha_0 + 1)

    return variance


def calc_entropy(alphas):
    # TODO
    alpha_0 = alphas.sum(dim=1, keepdims=True)
   
    psi_alpha_0 = digamma(alphas)
    psi_alphas = digamma(alphas)

    np.log(beta(alphas)) + (alpha_0 - K) `* 

    pass


if __name__ == "__main__":

    logits = []
    for i, arg in enumerate(sys.argv):
        #TODO read in csv specified by `arg'
        logits_ = np.loadtxt(arg)
        logits.append(logits)

    #TODO check if stack is the one i want
    logits = np.stack(logits) # N, M, K

    all_alphas = []
    for pred_dist in tqdm(logits):
        # pred_dist (M, K)

        alphas = mle(pred_dist) # K
        all_alphas.append(alphas)

    alphas = np.stack(all_alphas)
    variances = calc_variance(alphas)
    # entropies = calc_entropy(alphas)

    print(variances.mean())
    # print(entropies.mean())




