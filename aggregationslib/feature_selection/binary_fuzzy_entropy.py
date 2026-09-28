import numpy as np
from math import comb
from scipy.spatial.distance import cdist

def gaussian_similarity_matrix(X, delta):
    """Oblicza macierz rozmytej podobieństwa Gaussa dla 1D lub 2D"""
    dist_sq = cdist(X, X, metric='sqeuclidean')
    return np.exp(-dist_sq / (2 * delta ** 2))

def lambda_relation_matrix(sim_matrix, lam=0.5):
    """Tworzy macierz 0/1 w zależności od progu λ"""
    return (sim_matrix >= lam).astype(int)

def fuzzy_class_sizes(bin_matrix):
    """Zlicza liczbę podobnych obiektów dla każdego xᵢ"""
    return bin_matrix.sum(axis=1)

def fuzzy_conditional_entropy(f_sizes, joint_sizes, n):
    total_pairs = comb(n, 2)
    acc = 0
    for i in range(n):
        if f_sizes[i] - int(f_sizes[i]) > 0:
            raise Exception("Not Int value")
        if joint_sizes[i] - int(joint_sizes[i]) > 0:
            raise Exception("Not Int value")
        a = comb(int(f_sizes[i]), 2) if f_sizes[i] >= 2 else 0
        b = comb(int(joint_sizes[i]), 2) if joint_sizes[i] >= 2 else 0
        acc += (a - b) / total_pairs
    return acc / n

def ICE(fi, y, delta, lam=0.5):
    """Oblicza ICE(fi ; D) = ~CE(D) - ~CE(D | fi)"""
    n = len(fi)
    fi_ = fi.reshape(-1, 1)
    y_ = y.reshape(-1, 1)

    sim_fi = gaussian_similarity_matrix(fi_, delta)
    sim_y = gaussian_similarity_matrix(y_, delta)
    sim_fi_y = gaussian_similarity_matrix(np.hstack([fi_, y_]), delta)

    bin_fi = lambda_relation_matrix(sim_fi, lam)
    bin_y = lambda_relation_matrix(sim_y, lam)
    bin_fi_y = lambda_relation_matrix(sim_fi_y, lam)

    size_fi = fuzzy_class_sizes(bin_fi)
    size_y = fuzzy_class_sizes(bin_y)
    size_joint = fuzzy_class_sizes(bin_fi_y)

    ce_d = fuzzy_conditional_entropy(np.ones(n) * n, size_y, n)
    ce_d_given_fi = fuzzy_conditional_entropy(size_fi, size_joint, n)

    return ce_d - ce_d_given_fi

def ICE_cond(f_target, f_cond, y, delta, lam=0.5):
    """ICE(f_target ; D | f_cond) = ~CE(D | f_cond) - ~CE(D | f_target ∪ f_cond)"""
    n = len(f_target)
    f_target = f_target.reshape(-1, 1)
    f_cond = f_cond.reshape(-1, 1)
    y_ = y.reshape(-1, 1)

    sim_cond = gaussian_similarity_matrix(f_cond, delta)
    bin_cond = lambda_relation_matrix(sim_cond, lam)
    size_cond = fuzzy_class_sizes(bin_cond)

    sim_y = gaussian_similarity_matrix(y_, delta)
    size_y = fuzzy_class_sizes(lambda_relation_matrix(sim_y, lam))

    ce_d_given_cond = fuzzy_conditional_entropy(size_cond, size_y, n)

    joint = np.hstack([f_target, f_cond])
    sim_joint = gaussian_similarity_matrix(np.hstack([joint, y_]), delta)
    bin_joint = lambda_relation_matrix(sim_joint, lam)
    size_joint = fuzzy_class_sizes(bin_joint)

    ce_d_given_joint = fuzzy_conditional_entropy(size_cond, size_joint, n)

    return ce_d_given_cond - ce_d_given_joint

def MGRMLR(fi, S, y, lam=0.5):
    """Oblicza wartość MGRMLR dla cechy fi względem zbioru cech S"""
    delta = np.std(fi)

    J_grel = ICE(fi, y, delta, lam)
    J_nrel = 0
    J_rrel = 0

    for s in S:
        J_nrel += ICE_cond(fi, s, y, delta, lam)
        J_rrel += ICE_cond(s, fi, y, delta, lam)

    return J_grel + J_nrel + J_rrel


if __name__ == "__main__":
    # Dane z pracy
    a1 = np.array([0.3, 0.4, 0.7, 1.0])
    a2 = np.array([0.1, 0.2, 0.6, 0.9])
    a3 = np.array([0.1, 0.3, 0.7, 1.0])
    a4 = np.array([0.2, 0.4, 0.8, 1.0])
    D = np.array([1, 1, 2, 3])

    # Przykład: oblicz MGRMLR(a1) bez kontekstu (S = [])
    result = MGRMLR(a1, [a3], D)
    print(f"MGRMLR(a1) = {result:.4f}")
