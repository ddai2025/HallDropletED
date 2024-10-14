"""
Functions to calculate interaction matrix elements.
"""


import sys
from numpy import *
import scipy.special as special
import scipy.sparse as sparse
import scipy.integrate as integrate
import mpmath


"""
Electron-electron matrix elements.
"""
# CHECKED
def my_print(txt: str) -> None:
    """Print without creating new line."""

    print(txt, end=" ")
    sys.stdout.flush()

# CHECKED
def log_fact(n: ndarray) -> ndarray:
    return (special.gammaln(n + 1))

# CHECKED
def log_nCr(n: ndarray, r: ndarray) -> ndarray:
    return (special.gammaln(n + 1) - special.gammaln(r + 1)
            - special.gammaln(n - r + 1))

# CHECKED
def short_range_v_m(lowest_v_m: ndarray, L_cut: int) -> ndarray:
    """Convenience function for v_m nonzero only in the lowest few channels."""

    v_m = zeros(2 * L_cut - 1)
    v_m[:len(lowest_v_m)] = lowest_v_m
    return (v_m)

# CHECKED
def power_law_v_m(alpha: float, L_cut: int) -> ndarray:
    """Calculate v_m for power-law interaction v(r) = r^alpha."""

    m = arange(2 * L_cut - 1)
    log_v_m = log(2) * alpha + special.gammaln(m + 1 + alpha / 2) -\
        special.gammaln(m + 1)
    v_m = exp(log_v_m)
    return (v_m)

# CHECKED
def logarithmic_v_m(L_cut: int) -> ndarray:
    """Calculate v_m for logarithmic interaction v(r) = -ln(r)."""

    m = arange(2 * L_cut - 1)
    v_m = - log(2) - (1 / 2) * special.digamma(1 + m)
    return (v_m)

# CHECKED
def to_COM_basis(n1: int, n2: int) -> ndarray:
    """
    Expand |n1, n2> ~ z1^n1 z2^n2 into rel. Lz basis |M, m> ~ 
    (z1 + z2)^M (z1 - z2)^m. Logarithms used to avoid overflows. 
    See Supplementary Materials for derivation.

    Returns:
    C_m, len=n1 + n2 + 1: Coefficient of rel. Lz m channel.
    """

    # Sometimes hyp2f1 is exactly zero. This confuses mpmath, which tries to
    # converge RELATIVE precision. To fix this, set zero-prec, the tolerance
    # for deciding that a number is exactly zero, to a large finite number such
    # as 1000. Also, SciPy's hyp2f1 produces WRONG values, as can be seen by
    # the violation of normalization in the C_m.
    C_m = zeros(n1 + n2 + 1)
    for m in arange(n1 + n2 + 1):
        if m <= n2:
            log_pref = (1 / 2) * (log_nCr(n1 + n2 - m, n1) + log_nCr(n2, m)
                                  - log(2) * (n1 + n2))
            sign_pref = (-1) ** m
            binom_sum = mpmath.hyp2f1(-m, -n1, 1 - m + n2, -1, zeroprec=1000)
        else:
            log_pref = (1 / 2) * (log_nCr(m, n2) + log_nCr(n1, m - n2)
                                  - log(2) * (n1 + n2))
            sign_pref = (-1) ** n2
            binom_sum = mpmath.hyp2f1(m - n1 - n2, -n2, 1 + m - n2, -1,
                                      zeroprec=1000)
        coefficient = sign_pref * exp(log_pref) * float(binom_sum)
        C_m[m] = coefficient
    return (C_m)

# CHECKED
def check_basis(L_cut: int) -> tuple[ndarray, float]:
    """
    Check rel. Lz C_l precision by checking orthonormality of |n1, n2>.

    Returns:
    error, shape=(L_cut ** 2, L_cut ** 2): Deviation from O.N. for <m'n'|mn>.
    max_error: Maximum error for any |m'n'> and |mn>.
    """

    # calc all rel Lz expansions
    C_m_n_l = zeros((L_cut, L_cut, 2 * L_cut - 1))
    for m, n in ndindex((L_cut, L_cut)):
        C_l = to_COM_basis(m, n)
        C_m_n_l[m, n, :m + n + 1] = C_l

    # calc overlaps
    C_mn_l = C_m_n_l.reshape((L_cut ** 2, 2 * L_cut - 1))
    S_mpnp_mn = C_mn_l @ C_mn_l.T
    m_p, n_p, m, n = mgrid[:L_cut, :L_cut, :L_cut, :L_cut].\
        reshape((4, L_cut ** 2, L_cut ** 2))
    mask = (m_p + n_p) == (m + n)  # only same tot. Lz mix
    S_mpnp_mn = mask * S_mpnp_mn

    # assess error
    error = S_mpnp_mn - eye(L_cut ** 2)
    max_error = amax(abs(error))
    print(f"Maximum orthonormality error: {max_error}.")
    return ((error, max_error))

# CHECKED
def pair_matrix_elements(v_m: ndarray) -> tuple[sparse.csr_matrix,
                                                sparse.csr_matrix]:
    """
    Calcualte <m'n'|V|mn> and <m'n'|V|mn> - <m'n'|V|nm> given Haldane 
    pseudopotentials v_m. Uses index ravelling (m, n) -> m * L_cut + n.

    Params:
    v_m, len=(2 * L_cut - 1): Haldane pseudopotentials.

    Returns:
    H_pair_unsym, shape=(L_cut ** 2, L_cut ** 2): <m'n'|V|mn>.
    H_pair_antisym, shape=(L_cut ** 2, L_cut ** 2)): <m'n'|V|mn> - <m'n'|V|nm>.
    """
    my_print("Computing interaction matrix elements ...")
    L_cut = (len(v_m) + 1) // 2

    # get COM expansion for all pairs
    C_m_n_l = zeros((L_cut, L_cut, 2 * L_cut - 1))
    for m, n in ndindex((L_cut, L_cut)):
        C_l = to_COM_basis(m, n)
        C_m_n_l[m, n, :len(C_l)] = C_l

    # sparse encoding since L_z conservation makes many <m'n'|V|mn> zero
    rows = []
    cols = []
    vals_unsym = []
    vals_antisym = []
    for m, n in ndindex((L_cut, L_cut)):
        m_prime = arange(0, L_cut)  # by default in bounds
        n_prime = (m + n) - m_prime
        keep = where((0 <= n_prime) * (n_prime < L_cut))[0]  # n_prime bounds
        m_prime, n_prime = m_prime[keep], n_prime[keep]
        C_l_prime = C_m_n_l[m_prime, n_prime]
        C_l = C_m_n_l[m, n]
        C_l_exchanged = C_m_n_l[n, m]

        direct = C_l_prime @ (v_m * C_l)
        exchange = C_l_prime @ (v_m * C_l_exchanged)
        V_mpnp_mn_unsym = direct
        V_mpnp_mn_antisym = direct - exchange

        index_prime = m_prime * L_cut + n_prime
        index = m * L_cut + n
        rows.extend(index_prime)
        cols.extend([index] * len(m_prime))
        vals_unsym.extend(V_mpnp_mn_unsym)
        vals_antisym.extend(V_mpnp_mn_antisym)

    H_pair_unsym = sparse.coo_matrix((vals_unsym, (rows, cols)),
                                     shape=(L_cut ** 2, L_cut ** 2)).tocsr()
    H_pair_antisym = sparse.coo_matrix((vals_antisym, (rows, cols)),
                                       shape=(L_cut ** 2, L_cut ** 2)).tocsr()
    print("done!\n")
    return ((H_pair_unsym, H_pair_antisym))