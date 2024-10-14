"""
Functions for generating the exact diagonalization basis.

Notes:
    Slater determinants are lists of occupied orbitals, ALWAYS SORTED by 
    increasing Lz.

    You could generate the Slater determinants using Python's itertools, but 
    empirically my way using the 1D boson-fermion correspondence is faster. It
    also tells you the dimension of the center-of-mass angular momentum
    sectors.

Standard abbreviations / variable names:
    SD: Slater determinant.
    L_tot: Total angular momentum.
    L_CM: Angular momentum of the center of mass.
    L_int: Internal angular momentum = L_tot - L_CM.
    L_min: N * (N - 1) / 2, minimum L_tot for N fermions.
    L_ex: L_tot - L_min.
"""


import sys
from time import perf_counter
import numpy
from numpy import *
import sympy.combinatorics.permutations as sympy_permutations
from treelib import Tree
import scipy.sparse as sparse
import multiprocess as mp
import matplotlib.pyplot as plt


# CHECKED
def my_print(txt: str) -> None:
    """Print without creating new line."""

    print(txt, end=" "); sys.stdout.flush()

# CHECKED
def ferm_sign(state: ndarray):
    """Sign flip from sorting orbitals into ascending Lz."""
    
    permutation = argsort(state)
    sign = sympy_permutations.Permutation(permutation).signature()
    return(sign)

# CHECKED
def enum_partitions(N: int, L_tot: int) -> tuple[Tree, ndarray]:
    """
    Enumerate all ways of summing 2, 3, ..., N to L_ex or less, i.e. d_k such
    that \sum_{k=0}^{N-2} (k + 2) * d_k = L_ex, L_ex - 1, L_ex - 2, ... , 0.
    L_ex = L_sec - N * (N - 1) / 2 is the excess Lz above the v = 1 state.

    Params:
    N: Number of electrons.
    L: Excess total Lz (on top of the v=1 IQH state's total Lz).
    
    Returns:
    (d_ks, int, shape = (#, N): \sum_{k=0}^{N-2} (k + 2) * d_k = L_int)
    tree: Tree of partitions giving the order in which they were created.
    L_ints_d_ks, object, len = L_tot + 1: Partitions for L_ints in [0, L_tot].
    """

    L_min = (N * (N - 1)) // 2
    L_ex = L_tot - L_min
    base = L_ex # to assign IDs, interpret d_k as base L number
    to_ID = base ** arange(N - 1, dtype=int)[::-1]

    # Initialize tree.
    tree = Tree()
    d_k = zeros(N - 1, dtype=int)
    ID = d_k @ to_ID
    tree.create_node(tag=str(d_k), identifier=ID, data=d_k)

    # Grow partitions by increasing some element of leaf d_k by 1.
    my_print("Building partition tree, layer")
    layers = 0
    while True:
        my_print(layers)
        done = True
        leaves = tree.leaves()
        for leaf in leaves:
            leaf_d_k = leaf.data
            leaf_sum = leaf_d_k @ arange(2, N + 1)

            # Check if partition is complete.
            if leaf_sum < L_ex - 1:
                done = False
                delta = L_ex - leaf_sum

                # Only increment to the left.
                if (leaf_d_k == zeros(N - 1, dtype=int)).all():
                    leaf_min_k = N - 2
                else:
                    leaf_min_k = amin(where(leaf_d_k)[0])
                k_incs = arange(min([delta - 1, leaf_min_k + 1]))
                
                # Create children.
                for k_inc in k_incs:
                    child_d_k = leaf_d_k.copy()
                    child_d_k[k_inc] += 1
                    ID = child_d_k @ to_ID
                    tree.create_node(tag=str(child_d_k), identifier=ID, 
                                     data=child_d_k, parent=leaf)
        layers += 1
        if done:
            print("done!\n")
            break
            
    # Get states for each internal Lz sector.
    all_d_ks = vstack([node.data for node in tree.all_nodes()])
    node_sums = all_d_ks @ arange(2, N + 1)
    L_ints_d_ks = empty(L_tot + 1, dtype=object)
    for L_int in range(L_tot + 1):
        L_int_inds = where(node_sums == (L_int - L_min))[0]
        L_int_d_ks = all_d_ks[L_int_inds]
        L_ints_d_ks[L_int] = L_int_d_ks

    # Return both the full tree and the partitions for each sector.
    return(tree, L_ints_d_ks)

# CHECKED
def enum_SDs(L_ints_d_ks: ndarray) -> tuple[ndarray, ndarray]:
    """
    Enumerate all Slater determinants in the total angular momentum sector
    given the integer partitions using 1D boson-fermion duality.

    Params:
    (d_ks, int, shape = (#, N): \sum_{k=0}^{N-2} (k + 2) * d_k = L_int)
    L_ints_d_ks, object, len = L_tot + 1: Partitions for L_ints in [0, L_tot].

    Returns:
    SDs, shape=(L_tot_dim, N): All possible sorted lists of occupied orbitals.
    SD_IDs, len=L_tot_dim: Get ID by interpreting SD as base L_cut integer.
    """

    all_d_ks = vstack(L_ints_d_ks)
    L_tot_dim = all_d_ks.shape[0] # dim of total Lz sector
    N = all_d_ks.shape[1] + 1
    L_min = (N * (N - 1)) // 2
    L_ints = L_min + all_d_ks @ arange(2, N + 1)
    L_tot = len(L_ints_d_ks) - 1

    # Get boson occupation numbers (k = 1 to N now, NOT 2 to N).
    n_ks = zeros(all_d_ks.shape + array([0, 1]), dtype=int)
    n_ks[:, 1:] = all_d_ks
    n_ks[:, 0] = L_tot - L_ints

    # Apply boson-fermion duality.
    lifter = zeros((N, N), dtype=int)
    i, j = mgrid[:N, :N]
    lifter = ((i + j) >= (N - 1)).astype(int)
    increments = n_ks @ lifter
    SDs = tile(arange(N), (L_tot_dim, 1)) + increments

    # Calc IDs by interpreting SD as base L_cut int, sort into lex. order.
    L_cut = amax(SDs) + 1
    to_ID = L_cut ** arange(N, dtype=int)[::-1]
    SD_IDs = SDs @ to_ID
    sorter = argsort(SD_IDs)
    SDs, SD_IDs = SDs[sorter], SD_IDs[sorter]
    assert all(SD_IDs > 0) # check overflow
    assert len(unique(SD_IDs)) == len(SD_IDs) # check collision
    return(SDs, SD_IDs)

# CHECKED
def build_Zk(row_SDs: ndarray, col_SDs: ndarray, N_workers: int) ->\
    sparse.csr_matrix:
    """
    Operator representation of Z_k = \sum_i z_i^k / N. Infer k automatically.

    Params:
    row_SDs: Output vector space.
    col_SDs: Input vector space.

    Returns:
    Zk: Symmetric polynomial operator.
    """

    row_dim, N = row_SDs.shape
    col_dim: int = col_SDs.shape[0]
    row_sec = sum(row_SDs[0])
    col_sec = sum(col_SDs[0])    
    k = row_sec - col_sec

    # Calc and store IDs.
    row_L_cut = amax(row_SDs) + 1
    row_to_ID = row_L_cut ** arange(N, dtype=int)[::-1]
    row_IDs = row_SDs @ row_to_ID

    # Helper to find ID of row state.
    def calc_row_inds(SDs: ndarray) -> ndarray:
        IDs = SDs @ row_to_ID
        inds = searchsorted(row_IDs, IDs)
        assert all(row_SDs[inds] == SDs)
        return(inds)

    # Helper to calculate columns of Zk.
    def build_Zk_cols(col_block: range) -> tuple[list, list, list]:
        """
        Builds a group of columns of the Z_k operator.

        Params:
        col_block: Which rows to build.
        
        Returns:
        rows, cols, vals: For COO format.
        """

        my_print(f"({min(col_block)}-{max(col_block) + 1})")
        N_cols = len(col_block)
        # At most N_cols * N nonzero entries.
        rows = zeros((N_cols, N), dtype=uint32)
        cols = zeros((N_cols, N), dtype=uint32)
        vals = zeros((N_cols, N), dtype=float)

        for j, I in enumerate(col_block):
            # Raise any of the N filled orbitals.
            SD = col_SDs[I]
            for i in range(N):
                l_i = SD[i]
                l_i_prime = l_i + k

                # Avoid double occupancy (OK to "skip over", i.e. 123->234).
                if l_i_prime not in SD:
                    new_SD = copy(SD)
                    new_SD[i] = l_i_prime
                    val = sqrt(prod(arange(l_i + 1, l_i_prime + 1))) *\
                        ferm_sign(new_SD) / N
                    new_SD = sort(new_SD)
                    J = calc_row_inds(new_SD)

                    # Use preallocated arrays to avoid Python lists.
                    rows[j, i] = J
                    cols[j, i] = I
                    vals[j, i] = val

        # We may not have used all of the preallocated arrays.
        rows, cols, vals = rows.flatten(), cols.flatten(), vals.flatten()
        keep = where(abs(vals) > 1e-16)
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        return(rows, cols, vals)

    # Parallelize using the column builder.
    t1 = perf_counter()
    divs = linspace(0, col_dim, N_workers + 1).round().astype(int)
    blocks = [arange(divs[w], divs[w + 1]) for w in range(N_workers)]
    my_print(f"Building Z_{k}({row_sec}<-{col_sec}), columns")
    with mp.Pool(N_workers) as pool:
        results = pool.map(build_Zk_cols, blocks)
    print("done!")

    # Assemble into CSR for fast mat-vec.
    rows = hstack([result[0] for result in results])
    cols = hstack([result[1] for result in results])
    vals = hstack([result[2] for result in results])
    Zk = sparse.csr_matrix((vals, (rows, cols)), shape=(row_dim, col_dim))
    t2 = perf_counter(); print(f"Time: {t2 - t1}\n")
    return(Zk)

# CHECKED
def build_l_seq(L_ex: int, thresh: float) -> ndarray:
    """
    Determine optimum sequence of matrix multiplications (I - L_CM / l) to 
    project into the kernel.

    Params:
    L_ex: Angular momentum above v=1, which is the maximum eigenvalue of L_CM.
    thresh: When the error is small enough to halt.

    Returns:
    l_seq: List of (I - LCM / l) to apply.
    """

    w = ones(L_ex) # weight of each eigenspace
    lambdas = arange(1, L_ex + 1) # possible L_CM eigenvalues

    l_seq = []
    for l in lambdas:
        w_block = ones(L_ex) * (1 - lambdas / l)
        while(amax(abs(w_block)) > 1):
            w_block = w_block * (1 - lambdas / L_ex)
            l_seq.append(L_ex)
        l_seq.append(l)
        w = w * w_block
        if amax(abs(w)) < thresh:
            break
    return(l_seq)

# CHECKED
def test_l_seq(l_seq: list, noise: float) -> None:
    """
    Test the decay sequence with some simulated noise.
    """

    L_ex = amax(l_seq)
    lambdas = arange(0, L_ex + 1)
    w = ones(L_ex + 1) # running weight of each eigenspace
    amp_i = empty(len(l_seq)) # maximum amplification at each iteration
    w_i = empty((len(l_seq), L_ex + 1)) # weight at each iteration
    for i, l in enumerate(l_seq):
        w = w + noise * random.normal(size=w.shape)
        w = (1 - lambdas / l) * w
        amp = amax(abs(w[1:]))
        w_i[i] = w
        amp_i[i] = amp

    # Plot progress on each subspace versus steps.
    tol = noise
    fig, ax = plt.subplots(figsize=(5, 3))
    ax : plt.Axes = ax
    im = ax.imshow(log10(abs(w_i[:, :]) + tol).T, cmap="hot", origin="lower")
    corners = ax.get_position().get_points()
    height = corners[1][1] - corners[0][1]
    width = (corners[1][0] - corners[0][0]) / 20
    left = corners[1][0] + width
    bottom = corners[0][1]
    cbar_ax : plt.Axes = fig.add_axes([left, bottom, width, height])
    plt.colorbar(im, cax=cbar_ax)
    ax.set_title("$L_{CM}$ Eigenspace Weight")
    ax.set_xlabel("Step")
    ax.set_ylabel("Eigenvalue")

    # Plot maximum weight outside kernel for each step.
    fig, ax = plt.subplots(figsize=(5, 3))
    ax : plt.Axes = ax
    ax.plot(log10(amp_i))
    ax.set_title("Max. Weight Outside Kernel")
    ax.set_xlabel("Step")
    ax.set_ylabel("Log Weight")
    print(f"Max Amplification: {max(amp_i)}")
    print(f"Final: {amp} (noise level is {noise})\n")