"""
Main ED module: Droplet class contains functions for building the hamiltonian,
exact diagonalization, and exact time evolution.

Notes:
    Slater determinants are lists of occupied orbitals, ALWAYS SORTED by increasing Lz.
    Eigenvectors are STORED AS ROWS, i.e. eigbasis[0, :] is the ground state.

Common abbreviations / variable names:
    SD: Slater determinant.
    I, J ...: Variable representing an SD.
    ID: Identification number.
    L_tot: Total angular momentum.
    L_CM: Angular momentum of the center of mass.
    L_int: Internal angular momentum = L_tot - L_CM.
    L_min: N * (N - 1) / 2, minimum L_tot for N fermions.
    L_ex: L_tot - L_min.
    #: Arbitrary but consistent integer (for indefinite array dimensions).
"""


import sys
import os
import numpy.linalg as nla
from BasisGeneration import *
from MatrixElements import *


class Droplet():
    """
    Droplet of interacting electrons in a Landau level.
    """

    # CHECKED
    def __init__(self, N: int, L_tot: int, N_workers: int, interaction,
                 suffix: str):
        """
        Electron droplet in symmetry sector (N, L_tot, L_CM = 0).
        
        Params:
        N: Number of particles.
        L_tot: Total angular momentum.
        N_workers: Number of workers for parallelization.
        interaction: If float a, assume power law v(r) = r^a with convention
                     that a = 0 means logarithmic interaction v(r) = -ln(r).
                     If array, assume short-range pseudopotential.
        suffix: Save data to f"{N}_{L_tot}_{suffix}"
        """

        # setup basis
        partition_tree, L_ints_d_ks = enum_partitions(N, L_tot)
        SDs, SD_IDs = enum_SDs(L_ints_d_ks)
        L_tot_dim = len(SDs)
        L_int_dim = L_ints_d_ks[-1].shape[0]
        L_cut = amax(SDs) + 1
        L_min = (N * (N - 1)) // 2
        L_ex = L_tot - L_min

        # report basis information
        print(f"Electrons: {N}.")
        print(f"Total Lz: {L_tot}.")
        print(f"Total Lz dimension: {L_tot_dim}")
        print(f"Internal Lz dimension: {L_int_dim}\n")

        # build center-of-mass angular momentum
        lowered_SDs, lowered_SD_IDs = enum_SDs(L_ints_d_ks[:-1])
        Z1 = build_Zk(SDs, lowered_SDs, N_workers)

        # set object data
        self.N = N
        self.L_tot = L_tot
        self.L_cut = L_cut
        self.L_ex = L_ex
        self.N_workers = N_workers
        self.suffix = suffix
        self.SDs = SDs
        self.SD_IDs = SD_IDs
        self.to_ID = L_cut ** arange(N)[::-1]
        self.L_tot_dim = L_tot_dim
        self.L_int_dim = L_int_dim
        self.b1 = sqrt(N) * Z1.T  # L_CM too big, faster to do b1.H @ (b1 @ v)
        self.interaction = interaction

    """
    Methods to build Hamiltonian.
    """
    # CHECKED

    def calc_inds(self, SDs: ndarray) -> ndarray:
        """
        Finds indices of given Slater determinants in the basis.

        Params:
        SDs, shape=(#, N): Each row is a list of occupied orbitals.

        Returns:
        inds, len=#: Indices of Slater determinants.
        """

        IDs = SDs @ self.to_ID
        inds = searchsorted(self.SD_IDs, IDs)
        assert ((self.SDs[inds] == SDs).all())
        return (inds)

    # CHECKED
    def double_replace(self, I: int) -> tuple[ndarray, ndarray, ndarray]:
        """
        Given state I, return all states J related by double replacement
        (i, j, ...) -> (a = i + m, b = j - m, ...). It is sufficient to 
        do i < a < b < j explicitly and relate the rest by Hermiticity.
        
        Params:
        I: Index of Slater determinant.

        Returns:
        J, len=#: Indices of SDs connected by double replacmeent.
        ab_ij, shape=(#, 4): Double replacements (i, j) -> 
                             (a = i + m, b = j - m).
        signs, shape=#: Signs from fermion exchange to sort SD after 
                             replacement.
        """

        N = self.N
        SD = self.SDs[I]
        new_SDs, ab_ij, signs = [], [], []

        # get pairs of occupied LL states to move, need i < j and a < b
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
            J = self.calc_inds(new_SDs)
            ab_ij = array(ab_ij)
            signs = array(signs)
            return (J, ab_ij, signs)

    # CHECKED
    def calc_H_col(self, col_block: range) -> tuple[list, list, list]:
        """
        Builds off-diagonal part of the the Hamiltonian for a block of columns.
        Helper function for later parallelization. Only does half the entries,
        do other half by Hermiticity later.

        Params:
        col_block: Which rows to build.
        
        Returns:
        rows, cols, vals: For COO format.
        """

        my_print(f"({min(col_block)}-{max(col_block) + 1})")
        N_cols = len(col_block)
        rows = empty(N_cols, dtype=object)
        cols = empty(N_cols, dtype=object)
        vals = empty(N_cols, dtype=object)

        for i, I in enumerate(col_block):
            # get neighboring Slater determinants, ab_ij data, and fermion sign
            out = self.double_replace(I)
            if out is None:
                rows[i] = array([], dtype=uint32)
                cols[i] = array([], dtype=uint32)
                vals[i] = array([])
            else:
                J, ab_ij, signs = out
                ab = ab_ij[:, 0] * self.L_cut + ab_ij[:, 1]
                ij = ab_ij[:, 2] * self.L_cut + ab_ij[:, 3]
                both = array(self.H_pair_antisym[ab, ij])[
                    0]  # weird matrix slicing

                # use preallocated obj array to avoid python list hogging memory
                rows[i] = J.astype(uint32)
                cols[i] = tile(I, len(J)).astype(uint32)
                vals[i] = signs * both

        rows = hstack(rows)
        cols = hstack(cols)
        vals = hstack(vals)
        return (rows, cols, vals)

    # CHECKED
    def build_H(self):
        """
        Calculates the Hamiltonian.
        """

        N = self.N
        L_cut = self.L_cut
        SDs = self.SDs
        L_tot_dim = self.L_tot_dim
        N_workers = self.N_workers

        # calculate matrix elements
        interaction = self.interaction
        if type(interaction) == float or type(interaction) == int:
            if interaction == 0:
                print(f"Using logarithmic interaction v(r) = - ln(r).")
                v_m = logarithmic_v_m(L_cut)
            else:
                print(f"Using power law interaction v(r) = r^({interaction}).")
                v_m = power_law_v_m(interaction, L_cut)
        elif type(interaction) == ndarray or type(interaction) == list:
            print(f"Using pseudopotential v_m = {interaction}.")
            v_m = short_range_v_m(interaction, L_cut)
        else:
            assert False
        v_m[::2] = 0  # even angular momentum channels are irrelevant
        H_pair_unsym, H_pair_antisym = pair_matrix_elements(v_m)
        self.H_pair_antisym = H_pair_antisym

        # on-diagonal matrix elements: <I|V|I> = (<ij|V|ij> - <ij|V|ji>) / 2
        s_i = einsum("ci,j->cij", SDs, ones(N, dtype=int))
        s_j = einsum("i,cj->cij", ones(N, dtype=int), SDs)
        s_ij = (s_i * L_cut + s_j).reshape((L_tot_dim, N ** 2))  # occ. pairs
        both = array(self.H_pair_antisym[s_ij, s_ij].todense())  # avoid matrix
        diag_vals = (1 / 2) * both.sum(axis=1)
        diag_rows = arange(L_tot_dim, dtype=uint32)
        diag_cols = arange(L_tot_dim, dtype=uint32)

        # double-replacement: <I^{ab}_{ij}|H|I> = <ab|V|ij> - <ab|V|ji>
        t1 = perf_counter()
        divs = linspace(0, L_tot_dim, N_workers + 1).round().astype(int)
        blocks = [arange(divs[w], divs[w + 1]) for w in range(N_workers)]
        my_print("Building H, columns")
        with mp.Pool(N_workers) as pool:
            results = pool.map(self.calc_H_col, blocks)
        print("done!")
        t2 = perf_counter()
        print(f"Off-diagonal creation time: {t2 - t1}\n")

        # final assembly
        off_diag_rows = concatenate([result[0] for result in results])
        off_diag_cols = concatenate([result[1] for result in results])
        off_diag_vals = concatenate([result[2] for result in results])
        all_rows = concatenate([diag_rows, off_diag_rows, off_diag_cols])
        all_cols = concatenate([diag_cols, off_diag_cols, off_diag_rows])
        all_vals = concatenate([diag_vals, off_diag_vals, off_diag_vals])
        H = sparse.csr_matrix((all_vals, (all_rows, all_cols)),
                              shape=(L_tot_dim, L_tot_dim))
        self.H = H

    """
    Methods for diagonalization, real-time dynamics, and data transfer.
    """
    # CHECKED
    def build_L_CM_basis(self, l_seq: list, cycles=1):
        """
        Build the basis of states with L_CM = 0 by repeatedly applying
        the operator [1 + a * float]^(-1).

        Kwargs:
        cycles: How many times to apply kernel projection (1 usually OK).
        """

        L_tot_dim = self.L_tot_dim
        L_int_dim = self.L_int_dim
        b1 = self.b1

        # random starting vectors, orthogonalize,
        basis = random.normal(size=(L_tot_dim, L_int_dim))
        my_print("Orthogonalizing starting vectors ...")
        basis = nla.qr(basis).Q
        print("done!")

        # apply kernel projection
        for cycle in arange(cycles):
            my_print("Kernel projection sequence ...")
            for i, l in enumerate(l_seq):
                my_print(f"{i + 1}/{len(l_seq)}")
                basis = basis - b1.T @ (b1 @ basis) / l
            print("done!")

            # check for loss of independence before orthogonalizing
            S = basis.T @ basis
            plt.matshow(abs(S), vmin=0, cmap="hot")
            my_print("Orthogonalizing final vectors ...")
            basis = nla.qr(basis).Q
            print("done!")

        # check deviation from L_CM @ basis = 0 after orthogonalization
        my_print("Checking kernel error ...")
        error = amax(abs((b1 @ basis).T @ (b1 @ basis)))
        print(f"{error}\n")
        self.basis = basis  # USUAL CONVENTION REVERSED (COL IS VECTOR)

    # CHECKED
    def diagonalize(self, N_check: int = 50):
        """
        Fully diagonalize the Hamiltonian in the L_CM = 0 sector..

        Kwargs:
        N_check: Number of eigenvectors to randomly check.
        
        My convention is always that the eigenvectors are STORED AS ROWS, i.e.
        eigbasis[0, :] is the ground state.
        """

        # project into L_CM = 0 subspace
        basis = self.basis
        H = self.H
        my_print("Transforming into L_CM sector ...")
        H_sector = array(basis.T @ H @ basis)
        print("done!")

        # diagonalize in L_CM kernel
        my_print("Diagonalizing ...")
        E_eigs, eigbasis = nla.eigh(H_sector)
        E_eigs, eigbasis = array(E_eigs), array(eigbasis)  # back to numpy
        print("done!")

        # transform back to Slater determinants
        my_print("Transforming into back Slater determinants ...")
        eigbasis = (basis @ eigbasis).T  # back into SD basis, flip to rows
        print("done!")
        self.E_eigs = E_eigs
        self.eigbasis = eigbasis

        # randomly check
        N_check = min([N_check, self.L_int_dim])
        checks = random.choice(arange(eigbasis.shape[0]), N_check, False)
        max_error = 0
        for i in checks:
            error = nla.norm(H @ eigbasis[i] - eigbasis[i] * E_eigs[i])
            max_error = max([error, max_error])
        print(f"Max. error: {max_error}")

    # CHECKED
    def save_data(self):
        """
        Save energies and eigenstates.
        """

        # create folder for results
        rel_folder = f"Results/{self.N}_{self.L_tot}_{self.suffix}"
        folder = os.path.join(os.getcwd(), rel_folder)
        if not os.path.exists(folder):
            os.mkdir(folder)

        my_print("Saving data ...")
        vals_path = os.path.join(folder, f"vals")
        vecs_path = os.path.join(folder, f"vecs")
        savetxt(vals_path, self.E_eigs)
        savetxt(vecs_path, self.eigbasis)
        print("done!")