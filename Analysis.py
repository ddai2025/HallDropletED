"""
Analyzer class to calculate pair correlation function and entanglement entropy.
"""


from DropletSystem import *
from scipy.stats import entropy
from sympy.combinatorics.permutations import Permutation
from itertools import permutations
from scipy.special import factorial


# CHECKED
def calc_quartic_EVs(SDs: ndarray, psi: ndarray, N_workers: int) -> ndarray:
    """
    Calculate expectations of fermion quartics <c^+_a c^+_b c_j c_i>
    with respect to many-body state psi.

    Params:
    SDs: Basis Slater determinants (assume pre-sorted).
    psi: Many-body state.

    Returns:
    quartic_EVs, shape=(L_cut, L_cut, L_cut, L_cut): Expectation value of all
                                                     quartics.
    """

    # setup useful constants
    L_tot_dim, N = SDs.shape
    L_cut = amax(SDs) + 1
    to_ID = L_cut ** arange(N)[::-1]
    SD_IDs = SDs @ to_ID

    # enumerate all unique off-diag quartics: a + b = i + j, i < a < b < j
    quartics = []
    for i in range(L_cut):
        for j in range(i + 1, L_cut):
            max_delta = (j - i - 1) // 2
            for m in range(1, max_delta + 1):
                a = i + m
                b = j - m
                quartics.append(array([a, b, i, j]))
    quartics = vstack(quartics)
    quartic_to_ID = L_cut ** arange(4)[::-1]
    quartic_IDs = quartics @ quartic_to_ID
    sorter = argsort(quartic_IDs)
    quartics, quartic_IDs = quartics[sorter], quartic_IDs[sorter]
    N_quartics = len(quartic_IDs)
    assert all(quartic_IDs > 0)  # check overflow
    assert len(unique(quartic_IDs)) == len(quartic_IDs)  # check collision

    # helper to find indices of Slater determinants
    def calc_indices(find_SDs: ndarray) -> ndarray:
        IDs = find_SDs @ to_ID
        inds = searchsorted(SD_IDs, IDs)
        assert ((SDs[inds] == find_SDs).all())
        return (inds)

    # helper to do double replacement (same as bubble_system)
    def double_replace(I: int) -> tuple[ndarray, ndarray, ndarray]:
        SD = SDs[I]
        new_SDs, ab_ij, signs = [], [], []

        # get pairs of occupied LL states to move
        for i in range(N):
            for j in range(i + 1, N):
                l_i, l_j = SD[i], SD[j]

                # the moves have to conserve angular momentum
                max_delta = (l_j - l_i - 1) // 2
                for m in range(1, max_delta + 1):
                    l_a = l_i + m
                    l_b = l_j - m
                    valid = (l_a not in SD) and (l_b not in SD)
                    if valid:
                        new_SD = SD.copy()
                        new_SD[i] = l_a
                        new_SD[j] = l_b

                        # sort new_SD ONLY AFTER recording the sign
                        ab_ij.append([l_a, l_b, l_i, l_j])
                        signs.append(ferm_sign(new_SD))
                        new_SDs.append(sort(new_SD))

        # sometimes, no entries - need to handle specially
        if len(signs) == 0:
            return (None)
        else:
            J = calc_indices(new_SDs)
            ab_ij = array(ab_ij)
            signs = array(signs)
            return (J, ab_ij, signs)

    # helper function
    def calc_quartic_block(col_block: range) -> ndarray:
        """
        Contribution of a few columns of c^+_a c^+_b c_j c_i to the expectation
        value <psi|cccc|psi>. Only do i < a < b < j explicitly, rest by 
        identities.

        Params:
        col_block: Which rows to build.
        
        Returns:
        block_EVs, len=N_quartics: Unique off-diagonal quartics.
        """

        # fermion quartics correspond to double replacement
        my_print(f"({min(col_block)}-{max(col_block) + 1})")
        block_EVs = zeros(N_quartics, dtype=psi.dtype)
        for I in col_block:
            out = double_replace(I)
            if out is None:
                pass
            else:
                J, ab_ijs, signs = double_replace(I)
                inds = searchsorted(quartic_IDs, ab_ijs @ quartic_to_ID)
                assert [quartics[inds] == ab_ijs]
                contributions = psi[J].conj() * signs * psi[I]
                # for given I, all inds unique
                block_EVs[inds] += contributions
        return (block_EVs)

    # parallelize
    t1 = perf_counter()
    divs = linspace(0, L_tot_dim, N_workers + 1).round().astype(int)
    blocks = [arange(divs[w], divs[w + 1]) for w in range(N_workers)]
    my_print("Calculating four-fermion expectations, columns")
    with mp.Pool(N_workers) as pool:
        results = pool.map(calc_quartic_block, blocks)
    print("done!")
    t2 = perf_counter()
    print(f"Off-diagonal time: {t2 - t1}.")
    a, b, i, j = quartics.T
    EVs = vstack(results).sum(axis=0)

    # on diagonal part: c^+_m c^+_n c_n c_m = (m_present) * (n_present) * (m !+ n)
    n_mask = zeros((L_tot_dim, L_cut, L_cut))
    dummy = tile(arange(L_tot_dim)[:, None], (1, N))
    n_mask[dummy, SDs] = 1
    m_mask = swapaxes(n_mask, 1, 2)
    nm_mask = n_mask * m_mask
    nm_mask[:, range(L_cut), range(L_cut)] = 0
    diag_EVs = einsum("c,cij->ij", abs(psi) ** 2, nm_mask).flatten()
    n, m = mgrid[:L_cut, :L_cut].reshape(2, L_cut ** 2)

    # use antisymmetry and Hermiticity to complete
    all_a = concatenate((a, a, b, b, i, j, i, j, n, n))
    all_b = concatenate((b, b, a, a, j, i, j, i, m, m))
    all_i = concatenate((i, j, i, j, a, a, b, b, n, m))
    all_j = concatenate((j, i, j, i, b, b, a, a, m, n))
    all_EVs = concatenate((EVs, -EVs, -EVs, EVs, EVs.conj(), -EVs.conj(),
                           -EVs.conj(), EVs.conj(), diag_EVs, -diag_EVs))
    quartic_EVs = zeros((L_cut, L_cut, L_cut, L_cut), dtype=psi.dtype)
    quartic_EVs[all_a, all_b, all_i, all_j] = all_EVs
    return (quartic_EVs)

# MODIFIED (PHASE CONVENTION)
def phi_vals(L_cut: int, z_grid: ndarray) -> ndarray:
    """
    Calc symmetric gauge phi_l(z) on grid.

    Params:
    z_grid, len=#: Positions in complex plane.

    Returns:
    phi_l_z, shape=(L_cut, #): z^l / sqrt{2 \pi l! 2^l} \exp(-|z|^2 / 4)
    """

    z = tile(z_grid[None, :], (L_cut, 1))
    l = tile(arange(L_cut)[:, None], (1, len(z_grid)))
    log_normalizer = (log(2 * pi) + special.gammaln(l + 1) + l * log(2)) / 2
    envelope = -abs(z_grid) ** 2 / 4
    phi_l_z = ((1j * z) ** l) * exp(envelope - log_normalizer)
    return (phi_l_z)

# CHECKED
def calc_n_z(n_l: ndarray, z_grid: ndarray) -> ndarray:
    """
    Calculate total electron density n(z).

    Params:
    n_l, shape=(#1, L_cut) or (L_cut): <c^+_l c_l>
    z_grid, len=#2: Points to calculate density at.

    Returns:
    n_z, shape=(#1, #2) or (#2)
    """

    L_cut = n_l.shape[-1]
    phi_z = phi_vals(L_cut, z_grid)
    phi_z_abs_sq = abs(phi_z) ** 2
    n_tot_z = n_l @ phi_z_abs_sq
    return (n_tot_z)

# CHECKED
def calc_g_z1_z2(n_l: ndarray, quartic_EVs: ndarray, z1: ndarray,
                 z2: ndarray) -> ndarray:
    """
    Computes pair correlation function

    Params:
    quartic_EVs, shape=(L_cut, L_cut, L_cut, L_cut): <c^+_m' c^+_n' c_n c_m>.
    z1, len=#1: Coordinates of test charge. 
    z2, len=#2: Position of the rest of the electrons.

    Returns:
    g_z1_z2, shape=(#1, #2): <:n(z1)n(z2):>,, note that this is real.
    """

    L_cut = quartic_EVs.shape[0]
    phi_z1 = phi_vals(L_cut, z1)
    phi_z2 = phi_vals(L_cut, z2)
    n_z1 = calc_n_z(n_l, z1)
    n_z1_n_z2 = einsum("abcd,ai,bj,ci,dj->ij", quartic_EVs, phi_z1.conj(),
                       phi_z2.conj(), phi_z1, phi_z2, optimize="greedy").real
    g_z1_z2 = einsum("i,ij->ij", 1 / n_z1, n_z1_n_z2)
    return (g_z1_z2)


class ED_Results:
    """
    Container for results from finished ED calculation.
    """

    def __init__(self, N: int, L_tot: int, N_workers: int, suffix: str,
                 tag: str, load_energies=True, load_states=True):
        """
        Load ED resutls for electron droplet in the lowest Landau level,
        symmetry sector (N, L_tot, L_CM = 0).
        
        Params:
        N: Number of particles.
        L_tot: Total angular momentum.
        N_workers: Number of workers for parallelization.
        suffix: Read data from f"{N}_{L_tot}_{suffix}", if None use f"{N}_{L_tot}".
        tag: String to append to plots, for example clarifies the interaction type.

        Kwargs:
        load_states: Whether to load eigenstates (large memory usage).
        """

        # setup basis
        poly_tree, L_ints_d_ks = enum_partitions(N, L_tot)
        SDs, SD_IDs = enum_SDs(L_ints_d_ks)
        L_tot_dim = len(SDs)
        L_int_dim = L_ints_d_ks[-1].shape[0]
        L_cut = amax(SDs) + 1
        L_min = (N * (N - 1)) // 2
        L_ex = L_tot - L_min

        # set attributes
        self.N = N
        self.L_tot = L_tot
        self.L_cut = L_cut
        self.L_ex = L_ex
        self.N_workers = N_workers
        self.SDs = SDs
        self.SD_IDs = SD_IDs
        self.to_ID = L_cut ** arange(N)[::-1]
        self.L_tot_dim = L_tot_dim
        self.L_int_dim = L_int_dim

        # report basis information
        print(f"Electrons: {N}.")
        print(f"Total Lz: {L_tot}.")
        print(f"Total Lz dimension: {L_tot_dim}")
        print(f"Internal Lz dimension: {L_int_dim}\n")

        # setup descriptors
        rel_folder = f"Results/{N}_{L_tot}_{suffix}"
        folder = os.path.join(os.getcwd(), rel_folder)
        self.folder = folder
        self.suffix = suffix
        self.tag = tag

        # load eigenbasis
        my_print("Loading eigenbasis ...")
        vals_path = os.path.join(self.folder, f"vals")
        vecs_path = os.path.join(self.folder, f"vecs")
        if load_energies:
            self.E_eigs = loadtxt(vals_path)
        if load_states:
            self.eigbasis = loadtxt(vecs_path)
        print("done!\n")

        # convert SDs to Fock state (0 / 1 occupancy for each orbital)
        SDs_n_l = zeros((L_tot_dim, L_cut), dtype=int)
        i = arange(L_tot_dim)
        SDs_n_l[outer(i, ones(N, dtype=int)), self.SDs[i]] = 1
        self.SDs_n_l = SDs_n_l

        # calculate average orbital occupancies and nonuniformities S_rho1
        if load_states:
            assert type(self.eigbasis) != matrix  # issue with ** 2 overloading
            eigbasis_n_l = abs(self.eigbasis) ** 2 @ SDs_n_l
            self.eigbasis_n_l = eigbasis_n_l
            self.eigbasis_S_rho1 = entropy(eigbasis_n_l, axis=1)


def calc_S_ent(results: ED_Results, L_A: int, save_intermediates: bool = False) -> None:
    """
    Calculate the subsystem entanglement entropy.

    Params:
    L_A: Angular momentum cutoff for inner disk.

    Kwargs:
    save_intermediates: Whether to save intermediate quantities used to calculate
        S_ent. Helpful for debugging but hogs memory.
    """

    SDs_n_l = results.SDs_n_l
    N = results.N
    L_cut = results.L_cut
    L_tot = results.L_tot
    L_int_dim = results.L_int_dim
    eigbasis = results.eigbasis
    SD_IDs = results.SD_IDs
    to_ID = results.to_ID

    # subsystem fock states for each SD of full Hilbert space
    SDs_A_n_l = SDs_n_l[:, :L_A]
    SDs_B_n_l = SDs_n_l[:, L_A:]

    # get unique subsystem fock states and IDs
    A_n_ls = unique(SDs_A_n_l, axis=0)  # also sorts before returning
    B_n_ls = unique(SDs_B_n_l, axis=0)
    A_to_ID = 2 ** arange(L_A, dtype=object)[::-1]
    B_to_ID = 2 ** arange(L_cut - L_A, dtype=object)[::-1]
    A_IDs = A_n_ls @ A_to_ID  # already in lex. ord
    B_IDs = B_n_ls @ B_to_ID
    dim_A = len(A_IDs)
    dim_B = len(B_IDs)
    print(f"Inner disk A H_dim: {dim_A}")
    print(f"Outer annulus B H_dim: {dim_B}")
    assert (sorted(A_IDs) == A_IDs).all()
    assert (sorted(B_IDs) == B_IDs).all()

    # split each SD into tensor product of A and B states
    SD_A_IDs = SDs_A_n_l @ A_to_ID
    SD_B_IDs = SDs_B_n_l @ B_to_ID
    SD_A_inds = searchsorted(A_IDs, SD_A_IDs)
    SD_B_inds = searchsorted(B_IDs, SD_B_IDs)

    # calculate symmetry numbers
    N_As = sum(A_n_ls, axis=1)
    N_Bs = sum(B_n_ls, axis=1)
    L_tot_As = A_n_ls @ arange(L_A)
    L_tot_Bs = B_n_ls @ arange(L_A, L_cut)
    base_A = amax(L_tot_As) + 1
    base_B = amax(L_tot_Bs) + 1
    combined_A = N_As * base_A + L_tot_As  # combine into single index
    combined_B = N_Bs * base_B + L_tot_Bs

    # split into symmetry sectors
    sectors_A_inds = []
    sectors_B_inds = []
    for sector_A in unique(combined_A):
        N_A, L_tot_A = sector_A // base_A, sector_A % base_A
        N_B, L_tot_B = N - N_A, L_tot - L_tot_A
        sector_B = N_B * base_B + L_tot_B

        # asdf
        sector_inds_A = where(combined_A == sector_A)[0]
        sector_inds_B = where(combined_B == sector_B)[0]
        sectors_A_inds.append(sector_inds_A)
        sectors_B_inds.append(sector_inds_B)

    # subsystem density matrix and eigenvalues for each symmetry sector
    sectors_lambdas = []
    for inds_A, inds_B in zip(sectors_A_inds, sectors_B_inds):
        d_A, d_B = len(inds_A), len(inds_B)
        i, j = mgrid[:d_A, :d_B]
        sector_A_n_ls = A_n_ls[inds_A][i]
        sector_B_n_ls = B_n_ls[inds_B][j]
        sector_n_ls = concatenate((sector_A_n_ls, sector_B_n_ls), axis=2)
        sector_SDs = apply_along_axis(where, 2, sector_n_ls)[:, :, 0, :]
        sector_SD_IDs = einsum("abi,i->ab", sector_SDs, to_ID)
        sector_inds = searchsorted(SD_IDs, sector_SD_IDs)

        # sector density matrices
        sector_psi_ABs = eigbasis[:, sector_inds]
        if dim_A < dim_B:
            contraction = "iae,ibe->iab"
        else:
            contraction = "iea,ieb->iab"
        sector_rhos = einsum(contraction, sector_psi_ABs, sector_psi_ABs,
                             optimize="greedy")
        sectors_lambdas.append(nla.eigvals(sector_rhos).real)

    # rejoin sectors and calculate entanglement
    eigbasis_lambdas = concatenate(sectors_lambdas, axis=1)
    eigbasis_lambdas = eigbasis_lambdas * \
        (eigbasis_lambdas > 0)  # kill small neg part
    eigbasis_S_ent = entropy(eigbasis_lambdas, axis=1)
    return (eigbasis_S_ent)
