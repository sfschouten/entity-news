import sys
import argparse
import warnings
import math

from tqdm import tqdm
from scipy.special import softmax, gammaln, digamma, rel_entr, log_softmax, logsumexp
from scipy.stats import pearsonr, multivariate_normal

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

        return gammaln(alpha_0) - gammaln(alphas).sum(axis=-1, keepdims=True)                           \
            -  gammaln(beta_0)  + gammaln(betas).sum(axis=-1, keepdims=True)                            \
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

    r = pearsonr(entropies, kl_uniform) 
    print(r)

    print(f"Mean variance is {variances.mean()}; with standard deviation of {variances.std()} .")
    print(f"Mean entropy is {entropies.mean()}; with standard deviation of {entropies.std()} .")
    print(f"Mean KL w/ uniform-dirichlet is {kl_uniform.mean()}; with standard deviation of {kl_uniform.std()} .")

    np.save("entropies.npy", entropies)

    sns.kdeplot(data=entropies)
    plt.savefig("entropies.png")

def kde_based_uncertainty(logits):
    from KDEpy import NaiveKDE
    N, M, K = logits.shape

    for bw in np.linspace(0.3, 1, num=10):
        print(f"bandwith={bw}")
        all_entropies = []
        for M_logits in tqdm(logits):
            probs = softmax(M_logits, axis=-1)

            kde = NaiveKDE(bw=bw).fit(probs)
            
            # sample around the points on the simplex we know will have a lot of density

            # estimate covariance matrix
            cov_probs = np.cov(probs, rowvar=False)
            mean_probs = np.mean(probs, axis=0)

            # sample Gaussian around probs
            normal_dist = multivariate_normal(mean_probs, cov_probs, allow_singular=True)
            points = normal_dist.rvs(size=256)
            normal_density = normal_dist.pdf(points)

            #p = kde.evaluate(probs)
            ps = kde.evaluate(grid_points=points)

            entropy = -np.sum( ps * np.log(ps) / normal_density ) + gammaln(K)
            all_entropies.append(entropy)

        entropies = np.stack(all_entropies)

        print(f"Mean KDE-based entropy is {entropies.mean()}; with standard deviation of {entropies.std()}")


def lakshminarayanan_uncertainty(logits):
    """
    """
    np.set_printoptions(threshold=sys.maxsize)

    N, M, K = logits.shape

    mask_tokens = np.all(logits[:, 0:1, :] == -100, axis=2, keepdims=True)
    mask_tokens = np.broadcast_to(mask_tokens, (N, M, K))
    logits = logits[~mask_tokens].reshape(-1, M, K)

    logits[logits==-100] = float('-inf')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        predictions = log_softmax(logits, axis=-1)                  # N x M x K

    # average over model instances to obtain ensemble prediction
    prediction_E = logsumexp(predictions, axis=1) - np.log(M)   # N x K

    # calculate kl divergence between predictions and ensemble prediction 
    predictions = predictions.swapaxes(0,1)                     # M x N x K
    divergences = np.sum(np.exp(predictions) * (predictions - prediction_E), axis=-1)   # M x N

    # average over model instances to obtain disagreements
    disagreements = divergences.mean(axis=0)                    # N

    disagreements = disagreements[~np.isnan(disagreements)]

    disagreement_mean = disagreements.mean()
    #disagreement_std = disagreements.std()

    print(f"Mean disagreement is {disagreement_mean}") #; with standard deviation of {disagreement_std}")

    #np.save("disagreements.npy", disagreements)

    #sns.kdeplot(data=disagreements)
    #plt.savefig("disagreements.png")

    return disagreements.sum(), len(disagreements) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--file_format', default='npy', choices=['npy', 'npy-mmap', 'txt'])
    parser.add_argument('--shape', nargs='*', type=int)
    parser.add_argument('--uncertainty_metric', default='lakshminarayanan', choices=['lackshminarayanan', 'ml-dirichlet', 'kde-based'])
    parser.add_argument('--nr_steps', default=10, type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--skip_check', action='store_true')
    args = parser.parse_args()

    def check(logits):
        _, M, _ = logits.shape
        for i in range(M-1):
            arr1 = logits[:, i, :] != -100
            arr2 = logits[:, i+1, :] != -100
            assert np.array_equal(arr1, arr2), \
                    f"Logits from {args.files[i]} & {args.files[i+1]} do not share a padding pattern. " \
                +   f"Nr of incongruent pads: {np.sum(np.logical_xor(arr1, arr2))}."

    def calc(logits):
        if args.uncertainty_metric == 'lakshminarayanan':
            return lakshminarayanan_uncertainty(logits)
        elif args.uncertainty_metric == 'ml-dirichlet':
            return ml_dirichlet_based_uncertainty(logits)
        elif args.uncertainty_metric == 'kde-based':
            return kde_based_uncertainty(logits)

    if args.file_format != 'npy-mmap':
        logits = []
        for i, arg in enumerate(args.files):
            print(f"loading {arg}")
            if args.file_format == 'txt':
                logits_ = np.loadtxt(arg)
            elif args.file_format == 'npy':
                logits_ = np.load(arg)
            logits.append(logits_)

        logits = np.stack(logits, axis=1) # N, M, K
        check(logits)
        calc(logits)
    else:
        nr_samples = args.shape[0]
        if args.step_size is not None:
            step_size = args.step_size
            nr_steps = math.ceil(nr_samples / step_size)
            print(f"nr steps: {nr_steps}")
        else:
            nr_steps = args.nr_steps
            step_size = nr_samples // args.nr_steps
            print(f"step size: {step_size}")

        memmaps = [
            np.memmap(f, mode='r', dtype='float16', shape=tuple(args.shape))
            for f in args.files
        ]

        disagreement_total = 0
        count_total = 0
        for o in tqdm(range(nr_samples-step_size, nr_samples-nr_steps*step_size-1, -step_size)):
            logits = []
            for arr in memmaps:
                logits_ = arr[max(o, 0):o+step_size]
                logits.append(logits_)
            logits = np.stack(logits, axis=1)
           
            if not args.skip_check:
                check(logits)

            disagreement, count = calc(logits)
            disagreement_total += disagreement
            count_total += count

            print(f"New total average: {disagreement_total/count_total}")
        


