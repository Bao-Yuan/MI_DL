import numpy as np
from scipy.special import erf, psi
from numpy.linalg import inv, matrix_rank
import scipy.spatial as scispa
import warnings
import random
from cma import CMAEvolutionStrategy
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shutil
plt.rcParams['font.family'] = 'Times New Roman'  # Set the font to Times New Roman
# Use LaTeX for all text rendering when available; otherwise fall back gracefully.
plt.rcParams['text.usetex'] = shutil.which('latex') is not None

###############################################
# Basis computation (Null space of D_in)
###############################################
def calc_basis(D_in, col_range):
    num_rows = np.shape(D_in)[0]
    Din1, Din2 = D_in[:, :num_rows], D_in[:, num_rows:]
    basis_matrices = []
    for i in range(col_range):
        x2 = np.zeros((col_range, 1))
        x2[i, 0] = -1
        x1 = -inv(Din1) * Din2 * x2
        basis_matrices.append(np.vstack((x1, x2)))
    return np.asmatrix(np.array(basis_matrices))

###############################################
# Construct dimensionless variables
###############################################
def calc_pi(c, basis_matrices, X):
    coef_pi = np.dot(c, basis_matrices)
    coef_pi = np.asarray(coef_pi)
    if coef_pi.ndim == 1:
        coef_pi = coef_pi.reshape(1, -1)
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(coef_pi.shape[1]):
        tmp = X[:, i] ** coef_pi[:, i]
        pi_mat = np.multiply(pi_mat, tmp.reshape(-1, 1))
    return pi_mat

def calc_pi_omega(coef_pi, X):
    coef_pi = np.asarray(coef_pi)
    if coef_pi.ndim == 1:
        coef_pi = coef_pi.reshape(1, -1)
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(coef_pi.shape[1]):
        tmp = X[:, i] ** coef_pi[:, i]
        pi_mat = np.multiply(pi_mat, tmp.reshape(-1, 1))
    return pi_mat

###############################################
# Mutual information estimators
###############################################
def MI_d_binning(input, output, num_bins):
    def entropy_bin(X, num_bins):
        N, D = X.shape
        bins = [num_bins] * D
        hist, _ = np.histogramdd(X, bins=bins)
        hist = hist / np.sum(hist)
        positive_indices = hist > 0
        return -np.sum(hist[positive_indices] * np.log(hist[positive_indices]))
    mi = entropy_bin(input, num_bins) + entropy_bin(output, num_bins) - entropy_bin(np.hstack([input, output]), num_bins)
    return mi

def KraskovMI1_nats(x, y, k=1):
    N, dim = x.shape
    V = np.hstack([x, y])
    kdtree = scispa.KDTree(V)
    ei, _ = kdtree.query(V, k+1, p=np.inf)
    dM = ei[:, -1]
    nx = scispa.KDTree(x).query_ball_point(x, dM, p=np.inf, return_length=True)
    ny = scispa.KDTree(y).query_ball_point(y, dM, p=np.inf, return_length=True)
    ave = (psi(nx) + psi(ny)).mean()
    return psi(k) - ave + psi(N)

###############################################
# Objective function for CMA-ES
###############################################
def MI_input_output(
    para,
    para_o,
    basis_matrices,
    X,
    Y,
    num_basis,
    num_inputs,
    estimator="binning",
    estimator_params=None,
    optimize_output=False
):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        a_list = [tuple(para[i*num_basis:(i+1)*num_basis]) for i in range(num_inputs)]
        try:
            pi_list = [calc_pi(a, basis_matrices, X) for a in a_list]
            pi = np.column_stack(pi_list)
            if optimize_output and para_o is not None:
                a_o_list = [tuple(para_o)]
                pi_o = np.squeeze(np.array([calc_pi(a_o, basis_matrices, X) for a_o in a_o_list]))
                pi_o = pi_o.reshape(-1, 1)
                Y = Y.reshape(-1, 1)
                Y = Y * pi_o
            else:
                Y = Y.reshape(-1, 1)
        except RuntimeWarning:
            return random.uniform(1e6, 1e10)

    if np.any(np.isnan(pi)):
        return random.uniform(1e6, 1e10)

    if estimator == "binning":
        num_bins = estimator_params.get("num_bins", 50)
        mi = MI_d_binning(np.array(pi), np.array(Y), num_bins)
    elif estimator == "kraskov":
        k = estimator_params.get("k", 5)
        mi = KraskovMI1_nats(np.array(pi), np.array(Y), k)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'.")

    return -mi


###############################################
# Labels generation
###############################################
def create_labels(omega, variables):
    labels = []
    for row in omega:
        positive_part = ''
        negative_part = ''
        for i, value in enumerate(row):
            value = float(value)  # Safe scalar cast
            value = np.round(value, 2)  # Round to two decimal places
            if value > 0:
                if positive_part == '':
                    positive_part = f"{variables[i]}^{{{value}}}"
                else:
                    positive_part += f" \\cdot {variables[i]}^{{{value}}}"
            elif value < 0:
                if negative_part == '':
                    negative_part = f"{variables[i]}^{{{-value}}}"
                else:
                    negative_part += f" \\cdot {variables[i]}^{{{-value}}}"
        if negative_part == '':
            labels.append(f"${positive_part}$")
        elif positive_part == '':
            labels.append(f"$\\frac{{1}}{{{negative_part}}}$")
        else:
            labels.append(f"$\\frac{{{positive_part}}}{{{negative_part}}}$")
    return labels

###############################################
# Compute irreducible error and uncertainty
###############################################
def calculate_bound_and_uq(input_data, output_data, num_trials):
    mi_full = KraskovMI1_nats(input_data, output_data, 5)
    epsilon_full = np.exp(-mi_full)
    epsilon_half_values = []
    for _ in range(num_trials):
        idx = np.random.choice(input_data.shape[0], input_data.shape[0]//2, replace=False)
        mi_half = KraskovMI1_nats(input_data[idx], output_data[idx], 5)
        epsilon_half_values.append(np.exp(-mi_half))
    epsilon_half_mean = np.mean(epsilon_half_values)
    uq = abs(epsilon_full - epsilon_half_mean)
    return epsilon_full, uq


###############################################
# Plot the results
###############################################
def plot_scatter(input_PI, Y):
    """
    Scatter plot of output vs. optimal dimensionless input (only for scalar input_PI)
    """
    if input_PI.shape[1] == 1:
        plt.figure(figsize=(4, 4))
        plt.scatter(input_PI, Y, alpha=0.5)
        plt.xlabel(r'$\Pi^*$', fontsize=20, labelpad=10)
        plt.ylabel(r'$\Pi_o^*$', fontsize=20, labelpad=10)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()
        
def plot_error_bars(input_PI, epsilon, uq):
    """
    Bar plot of irreducible error with uncertainty error bars
    """

    if input_PI.shape[1] == 1:
        x_labels = [r'${\Pi}^*$']
    else:
        x_labels = [fr'$\Pi_{{{i+1}}}^*$' for i in range(input_PI.shape[1])]
        x_labels.append(r'$\mathbf{\Pi}^*$')  # For the joint set

    print("x_labels:", x_labels)
    plt.figure(figsize=(4, 2))
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.bar(x_labels, epsilon, yerr=uq, capsize=5, edgecolor='black', color='red')
    plt.xticks(fontsize=20)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylabel(r'$\tilde{\epsilon}_{L B}$', fontsize=20, labelpad=15)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('UQ.eps', dpi=300, bbox_inches='tight',transparent=True)
    plt.show()
    
###############################################
#Region identification
###############################################


def partition_space(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    regions = kmeans.fit_predict(X)
    return regions

def analyze_regions(X1, X2, Y, regions):
    ratio_X1 = np.zeros_like(X1)  
    ratio_X2 = np.zeros_like(X2)  
    results = []
    for region in np.unique(regions):
        region_indices = (regions == region)
        X1_region = X1[region_indices]
        X2_region = X2[region_indices]
        Y_region = Y[region_indices]
        # Calculate mutual information
        mi_X1_Y = KraskovMI1_nats(X1_region.reshape(-1, 1), Y_region, 5)
        mi_X2_Y = KraskovMI1_nats(X2_region.reshape(-1, 1), Y_region, 5)
        mi_X1_X2_Y = KraskovMI1_nats(np.hstack([X1_region.reshape(-1, 1), X2_region.reshape(-1, 1)]), Y_region, 5)
        #ratio_X1_region = np.exp(-mi_X1_X2_Y)/np.exp(-mi_X1_Y) if mi_X1_X2_Y != 0 else 0
        #ratio_X2_region = np.exp(-mi_X1_X2_Y)/np.exp(-mi_X2_Y) if mi_X1_X2_Y != 0 else 0
        ratio_X1_region = mi_X1_Y/ mi_X1_X2_Y if mi_X1_X2_Y != 0 else 0
        ratio_X2_region = mi_X2_Y/mi_X1_X2_Y if mi_X1_X2_Y != 0 else 0
        ratio_X1[region_indices] = ratio_X1_region 
        ratio_X2[region_indices] = ratio_X2_region
        
        threshold = 0.9
        if ratio_X1_region > threshold:
            result = "X1"
        elif ratio_X2_region > threshold:
            result = "X2"
        else:
            result = "Both"
        results.append((region, result))
    return results, ratio_X1, ratio_X2

###############################################
# Main function
# X: Input data matrix 
# Y: Output data vector 
# D_in: Dimension matrix for input variables
# num_input: Number of input variables (default is 1)
# popsize: Population size for CMA-ES (default is 300)
# maxiter: Maximum number of iterations for CMA-ES (default is 50000)
# num_trials: Number of trials for uncertainty calculation (default is 10)
# estimator: Mutual information estimator to use ('binning' or 'kraskov', default is 'binning')
# estimator_params: Dictionary of hyperparameters for the estimator (default is None, which uses default values)    
# Returns labels for optimal dimensionless variables, irreducible error, and uncertainty.
###############################################
def main(
    X,
    Y,
    basis_matrices = None,
    num_input=1,
    popsize=300,
    maxiter=50000,
    num_trials=10,
    estimator="binning",
    estimator_params=None,
    seed=None,
    optimize_output=False
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    num_basis = basis_matrices.shape[0]
    print('-'*60)

    num_para_input = num_basis * num_input
    num_para_output = num_basis if optimize_output else 0
    num_para_total = num_para_input + num_para_output
    print('num of parameters:', num_para_total)

    bounds = [[-2] * num_para_total, [2] * num_para_total]
    options = {
        'bounds': bounds,
        'maxiter': maxiter,
        'tolx': 1e-8,
        'tolfun': 1e-8,
        'popsize': popsize,
        'seed': seed if seed is not None else random.randint(0, 10000),
    }

    if estimator_params is None:
        estimator_params = {"num_bins": 50} if estimator == "binning" else {"k": 5}
    print(f"\nUsing estimator: '{estimator}' with hyperparameters: {estimator_params}\n")

    def objective_fn(params):
        input_params = params[:num_para_input]
        output_params = params[num_para_input:] if optimize_output else None
        return MI_input_output(
            input_params,
            output_params,
            basis_matrices,
            X,
            Y,
            num_basis,
            num_input,
            estimator=estimator,
            estimator_params=estimator_params,
            optimize_output=optimize_output
        )

    es = CMAEvolutionStrategy([0.1]*num_para_total, 0.5, options)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [objective_fn(x) for x in solutions])
        es.disp()
    es.result_pretty()

    optimized_params = es.result.xbest
    optimized_MI = es.result.fbest
    print('Optimized_params:', optimized_params)
    print('Optimized_MI:', optimized_MI)
    print('-'*60)

    a_list = [tuple(optimized_params[i*num_basis:(i+1)*num_basis]) for i in range(num_input)]
    coef_pi_list = [np.dot(a, basis_matrices) for a in a_list]
    normalized_coef_pi_list = []
    for coef_pi in coef_pi_list:
        max_abs_value = np.max(np.abs(coef_pi))
        normalized_coef_pi = coef_pi / max_abs_value
        normalized_coef_pi_list.append(np.round(normalized_coef_pi, 2))
        print('coef_pi:', normalized_coef_pi)

    #omega = np.array(normalized_coef_pi_list).reshape(-1, len(variables))
    # optimal_pi_lab = create_labels(omega, variables)
    # for j, label in enumerate(optimal_pi_lab):
    #     print(f'Optimal_pi_lab[{j}] = {label}')

    input_list = [calc_pi_omega(np.array(omega), X) for omega in normalized_coef_pi_list]
    input_PI = np.column_stack(input_list)

    output_PI = Y
    if optimize_output:
        normalized_a_list = []
        for a in a_list:
            normalized_a = tuple(x / a[2] for x in a) if len(a) > 2 and a[2] != 0 else tuple(a)
            normalized_a = np.round(normalized_a,2)
            normalized_a_list.append(normalized_a)
        print('a_list:',normalized_a_list)
        normalized_coef_pi_list = [np.dot(a, basis_matrices) for a in normalized_a_list]
        input_list   = [calc_pi(a, basis_matrices, X) for a in normalized_a_list]
        input_PI        = np.column_stack(input_list)

        a_list_o = [np.round(tuple(optimized_params[i*num_basis+num_para_input:(i+1)*num_basis+num_para_input]), 1) for i in range(num_input)]
        #print('a_list_o =', a_list_o)

        modified_a_list_o = []
        first_norm_a = normalized_a_list[0]
        for a_o in a_list_o:
            ratio = a_o[2] / first_norm_a[2] if first_norm_a[2] != 0 else 1.0
            modified_a_o = tuple(a_o[i] - first_norm_a[i] * ratio for i in range(len(a_o)))
            modified_a_list_o.append(modified_a_o)
        a_list_o = modified_a_list_o

        coef_pi_o = np.dot(a_list_o[0], basis_matrices)
        coef_pi_o = coef_pi_o / np.max(np.abs(coef_pi_o))
        coef_pi_o = np.round(coef_pi_o, 2)
        print('coef_pi_o:', coef_pi_o)
        print("a_list_o:", a_list_o)
        output_PI = np.asarray(np.column_stack([calc_pi(a, basis_matrices, X) for a in modified_a_list_o])).reshape(-1, 1) * Y


    epsilon_values = []
    uq_values = []
    for j in range(input_PI.shape[1]):
        epsilon_full, uq = calculate_bound_and_uq(
            input_PI[:, j].reshape(-1, 1), output_PI, num_trials
        )
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)

    if input_PI.shape[1] > 1:
        epsilon_full, uq = calculate_bound_and_uq(input_PI, output_PI, num_trials)
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)

    return {
        "input_coef": normalized_coef_pi_list,
        "output_coef": coef_pi_o if optimize_output else None,
        "input_coef_basis": normalized_a_list if optimize_output else a_list,
        "output_coef_basis": a_list_o if optimize_output else None,
        "irreducible_error": epsilon_values,
        "uncertainty": uq_values,
        "input_PI": input_PI,
        "output_PI": output_PI,
        "optimized_params": optimized_params,
    }