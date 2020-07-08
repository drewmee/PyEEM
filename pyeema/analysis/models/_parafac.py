from tensorly.decomposition import parafac

class PARAFAC():
    """[summary]
    """
    def __init__(self):
        return

def perform_parafac(X, rank):
    # Specify the tensor, and the rank (np. of factors)
    X, rank = observed, 3

    # Perform CP decompositon using TensorLy
    # n_iter_max : int - Maximum number of iteration
    # tol : float, optional (Default: 1e-6) - Relative
    # reconstruction error tolerance. The algorithm is
    # considered to have found the global minimum when
    # the reconstruction error is less than tol.
    factors_tl = parafac(X, rank=rank, non_negative=True)

    # Reconstruct M
    M_tl = reconstruct(factors_tl)

    # plot the decomposed factors
    plot_factors(factors_tl)
    return