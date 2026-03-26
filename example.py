"""Minimal runnable example for MIDL."""

import numpy as np
import midl
from midl import MIDL

def run_demo() -> None:
    # Synthetic example: u(x, y, z) = sin(xy) * cos(z).
    rng = np.random.default_rng(123)
    n_samples = 200
    x = rng.uniform(1.0, 5.0, n_samples)
    y = rng.uniform(1.0, 5.0, n_samples)
    z = rng.uniform(1.0, 5.0, n_samples)
    L = np.ones(n_samples)
    X_raw = np.column_stack([x, y, z,L])
    u = np.sin(x*y/L/L) * np.cos(z/L)

    # Define dimension matrix
    D_in = np.matrix([1, 1, 1, 1])
    print("Dimension matrix D_in:\n", D_in)
    # Compute basis and rank
    basis, r = midl.calc_basis(D_in)
    print("Basis vectors dimension:", D_in.shape[1] - r)
    print("Basis vectors (π exponents):\n", basis)
    # Transform X_raw to Pi
    Pi = np.exp(np.log(X_raw) @ basis)
 

    model = MIDL(
        k_neighbors=6,
        de_maxiter=200,
        de_popsize=15,
        random_state=42,
    )
    result = model.fit(Pi_independent=Pi, pi_dependent=u)
    pi_hat = MIDL.compose_new_pi(Pi, result["W"])

    print("=== MIDL Results ===")
    print("MI scores:", result["mi_scores"])
    print("W (columns are w_i):")
    print(result["W"])
    print("dominant_q:", result["dominant_q"])
    print("drop ratios I_i / I_(i+1):", result["drop_ratios"])
    alpha = basis @ result["W"]
    print("\n=== Recovered exponents ===")
    print(alpha)





if __name__ == "__main__":
    run_demo()
