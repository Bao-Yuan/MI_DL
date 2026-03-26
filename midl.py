"""
Mutual-information-based dimensional learning (MIDL).
This module implements the core pipeline described in: Zhang & He (2025).
"""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import differential_evolution
from sklearn.feature_selection import mutual_info_regression

###############################################
# Basis computation (Null space of D_in)
###############################################
def calc_basis(D_in, tol=1e-12):

    D_in = np.asarray(D_in, dtype=float)
    if D_in.ndim != 2:
        raise ValueError("D_in must be 2D.")
    # SVD
    U, S, Vt = np.linalg.svd(D_in)
    # rank
    r = np.sum(S > tol)
    # null space (columns of V corresponding to zero singular values)
    V = Vt.T
    null_space_cols = V[:, r:]   # shape: (m, k=n-r); basis vectors are columns
    basis_matrices = null_space_cols.T  # shape: (k, m); 

    return basis_matrices, r


class MIDL:
    """Mutual-Information-based Dimensional Learning.

    Steps:
    1) log transform independent dimensionless quantities
    2) sequentially optimize directions maximizing MI
    3) enforce orthogonality by searching in null-space complements
    4) infer dominant count by largest adjacent MI drop
    """

    def __init__(
        self,
        k_neighbors: int = 6,
        de_maxiter: int = 300,
        de_popsize: int = 20,
        random_state: int = 42,
        eps_log: float = 1e-12,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.random_state = random_state
        self.eps_log = eps_log

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            return v
        return v / norm

    def _estimate_mi(self, y, x1d):
        x_feature = x1d.reshape(-1, 1)
        mi = mutual_info_regression(
            x_feature,
            y,
            discrete_features=False,
            n_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )[0]
        return float(mi)

    def _optimize_direction_in_subspace(self, X, y, B):
        """Optimize w = B @ u with ||u|| = 1 to maximize I(y, Xw)."""

        d_sub = B.shape[1]

        def objective(u_raw):
            u = self._normalize(np.asarray(u_raw, dtype=float))
            w = B @ u
            xhat = X @ w 
            mi = self._estimate_mi(y, xhat)
            return -mi

        bounds = [(-1.0, 1.0)] * d_sub
        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=self.de_maxiter,
            popsize=self.de_popsize,
            seed=self.random_state,
            polish=True,
            workers=1,
            updating="deferred",
        )
        u_best = self._normalize(result.x)
        w_best = self._normalize(B @ u_best)
        mi_best = -float(result.fun)
        return w_best, mi_best

    def fit(self, Pi_independent, pi_dependent):
        """Fit MIDL model.

        Args:
            Pi_independent: shape (n_samples, n_features), each entry must be > 0.
            pi_dependent: shape (n_samples,), dependent dimensionless quantity.
        """

        Pi_independent = np.asarray(Pi_independent, dtype=float)
        pi_dependent = np.asarray(pi_dependent, dtype=float).ravel()

        if Pi_independent.ndim != 2:
            raise ValueError("Pi_independent must be 2D.")
        if pi_dependent.ndim != 1:
            raise ValueError("pi_dependent must be 1D.")
        if Pi_independent.shape[0] != pi_dependent.shape[0]:
            raise ValueError("Sample sizes do not match.")
        if np.any(Pi_independent <= 0.0):
            raise ValueError("All Pi_independent values must be > 0 for log transform.")

        n_features = Pi_independent.shape[1]

     # Log-transform of Pi (to linearize multiplicative relations)
        X = np.log(Pi_independent + self.eps_log)
        y = pi_dependent

        W_list = []   # List to store projection directions
        mi_list = []  # List to store mutual information values

        threshold = 10  # Early stopping threshold for MI drop ratio

        for s in range(n_features):

            # =====  Construct subspace =====
            if s == 0:
                # First direction: search in full space
                B = np.eye(n_features)
            else:
                # Subsequent directions: search in orthogonal complement
                D_s = np.column_stack(W_list)
                B = null_space(D_s.T)

                # If no remaining subspace, stop
                if B.shape[1] == 0:
                    break

            # ===== Optimize direction in subspace =====
            w_s, mi_s = self._optimize_direction_in_subspace(X, y, B)

            # =====  Re-orthogonalization (Gram-Schmidt) =====
            if W_list:
                for w_prev in W_list:
                    w_s = w_s - np.dot(w_s, w_prev) * w_prev
                w_s = self._normalize(w_s)

            W_list.append(w_s)
            mi_list.append(mi_s)

            print(f"[Step {s+1}] MI = {mi_s:.6f}")

            # ===== Early stopping based on MI drop =====
            if s >= 1:
                ratio = mi_list[-2] / (mi_list[-1] + 1e-12)
                print(f"   ratio = {ratio:.3f}")

                if ratio > threshold:
                    print(f"   >>> Early stopping triggered at step {s+1}")
                    dominant_q = s
                    drop_ratios = ratio
                    break

                dominant_q = s+1
                drop_ratios = None

        # ===== Step 5: Assemble results =====
        W = np.column_stack(W_list)
        mi_scores = np.array(mi_list)

        # Low-dimensional representation in log-space
        xhat = X @ W

        # ===== Final output =====
        return {
            "W": W,
            "mi_scores": mi_scores,
            "xhat": xhat,
            "dominant_q": dominant_q,
            "drop_ratios": drop_ratios,
        }

    @staticmethod
    def compose_new_pi(Pi_independent, W):
        """Compute transformed dimensionless quantities Pi_hat = exp(log(Pi) @ W)."""
        X = np.log(np.asarray(Pi_independent, dtype=float))
        return np.exp(X @ W)
