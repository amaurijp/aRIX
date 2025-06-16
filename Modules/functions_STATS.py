#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Tuple, List, Dict
from statsmodels.stats.multitest import multipletests
import time


#------------------------------
def bootstrap_test(x: List[float], y: List[float], statistic: Callable[[np.ndarray, np.ndarray], float] = None,
                   n_resamples: int = 10000, ci: float = 0.95, random_state: int = None, ) -> float:
    """
    Returns observed statistic, bootstrap 95 % CI, and
    a two-sided p-value against H0: statistic == 0
    (based on the bootstrap distribution).
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)
    if statistic is None:
        statistic = lambda a, b: np.mean(a) - np.mean(b)

    stat_obs = statistic(x, y)

    boot_stats = np.empty(n_resamples, dtype = float)
    for i in range(n_resamples):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        boot_stats[i] = statistic(bx, by)

    # Percentile confidence interval
    alpha = 1.0 - ci
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value for H0: stat == 0
    p_val = np.mean(np.abs(boot_stats) >= abs(stat_obs))

    #print(f"statistic : {stat_obs}, {int(ci*100)}%_CI_low : {lo}, {int(ci*100)}%_CI_high : {hi}, p_value : {p_val}")

    return p_val


#------------------------------
def calculate_crossentropy(prob_vector, label_prob_vector):

    import math
    
    #print('prob_func.sum() = ', prob_func.sum())
    #print('base_prob_func.sum() = ', base_prob_func.sum())
    
    cross_entropy = 0
    
    for i in range(len(label_prob_vector)):            
        p_label = label_prob_vector[i]
        p_score = prob_vector[i]
        if p_label == 0 or p_score == 0:
            continue        
        val = - ( p_label * math.log(p_score))
        cross_entropy += val
    
    return cross_entropy


#------------------------------
def calculate_prob_dist_RSS(prob_vector, label_prob_vector):    
    return ( ( prob_vector.cumsum() - label_prob_vector.cumsum() )**2 ).sum()


#------------------------------
def diff_of_means(x, y):
    return np.mean(x) - np.mean(y)


#------------------------------
def diff_of_medians(x, y):
    return np.median(x) - np.median(y)


#------------------------------
def diff_of_trimmed_means(x, y):
    return trimmed_mean(x) - trimmed_mean(y)


#------------------------------
def diff_of_welch_t_means(x, y, trim_perc = None):

    if trim_perc is not None and type(trim_perc) == float:
        x = trim_array(x, trim_perc)
        y = trim_array(y, trim_perc)

    na, nb = len(x), len(y)
    va, vb = x.var(ddof=1), y.var(ddof=1) #ddof = 1 for Bessel’s correction
    return (x.mean() - y.mean()) / np.sqrt(va/na + vb/nb)


#------------------------------
def kruskal_wallis_test(*samples):
    
    """
    Parameters
    ----------
    *samples : array-like
        Any number (≥2) of 1-D arrays or lists of numeric data.

    Returns
    -------
    H : float
        Kruskal–Wallis test statistic (tie-corrected).
    p_value : float or None
        Two-sided p-value from the chi-square distribution with k-1 df.
        Returned as None if SciPy is unavailable.
    df : int
        Degrees of freedom (= k-1).
    """
    
    # ---------- 1. prepare data ----------
    arrays = [np.asarray(a, dtype=float).ravel() for a in samples]
    if len(arrays) < 2:
        raise ValueError("Need at least two groups")
    if any(len(a) == 0 for a in arrays):
        raise ValueError("All groups must contain data")

    lens = np.array([len(a) for a in arrays])
    n_groups, n_vals = len(arrays), lens.sum()

    # ---------- 2. pooled mid-ranks with tie handling ----------
    pooled = np.concatenate(arrays)
    order  = np.argsort(pooled, kind="mergesort")
    ranks  = np.empty(n_vals, dtype=float)
    ranks[order] = np.arange(1, n_vals + 1)          # preliminary ranks 1 … N

    # assign average rank for ties
    sorted_vals_per_rank = pooled[order]
    tie_starts = np.flatnonzero(np.diff(sorted_vals_per_rank)) + 1
    tie_starts = np.r_[0, tie_starts, n_vals]        # edges of unique blocks
    for i in range(len(tie_starts) - 1):
        a, b = tie_starts[i], tie_starts[i + 1]
        if b - a > 1:                           # a tie block
            ranks[order[a:b]] = ranks[order[a:b]].mean()

    # ---------- 3. compute H ----------
    idx = np.cumsum(lens)[:-1]                  # split points
    rank_groups = np.split(ranks, idx)
    R_means = np.array([rg.mean() for rg in rank_groups])

    H = ( 12 / ( n_vals * ( n_vals + 1 ) ) )  * np.sum( lens * ( R_means - ( ( n_vals + 1 ) / 2 ) )**2 )

    # ---------- 4. tie correction ----------
    # T = Σ(t³ – t) over all unique tie group sizes
    _, tie_counts = np.unique(pooled, return_counts=True)
    T = np.sum(tie_counts**3 - tie_counts)
    if T > 0:
        H /= 1.0 - T / (n_vals**3 - n_vals)

    df = n_groups - 1

    # ---------- 5. p-value from chi-square (SciPy if available) ----------
    try:
        from scipy.stats import chi2
        p_value = chi2.sf(H, df)
    except ModuleNotFoundError:
        p_value = None   # install SciPy for the p-value

    print(H, df)
    return p_value


#------------------------------
def p_holm_bh_correction(cat_cat_p_vals: np.array) -> np.array:
    
    # 1) Holm–Bonferroni  (family-wise error rate, alpha = 0.05)
    _, p_holm, _, _ = multipletests(cat_cat_p_vals[ : , 2], alpha=0.05, method='holm')

    # 2) Benjamini–Hochberg  (false-discovery rate, q = 0.05)
    _, p_bh, _, _   = multipletests(cat_cat_p_vals[ : , 2], method='fdr_bh')

    # ---- write adjusted p-values back into the dict ----
    p_corrected = []
    for (p, pH, pBH) in zip(cat_cat_p_vals[:, 2], p_holm, p_bh):
        p_corrected.append( max(p, pH, pBH) )
        #print(p, pH, pBH)

    cat_cat_p_vals[ : , 2] = p_corrected
    
    return cat_cat_p_vals


#------------------------------
def permutation_test_AB(x: List[float], y: List[float], statistic: Callable[[np.ndarray, np.ndarray], float] = None,
                        n_resamples: int = 10000, alternative: str = "two-sided", random_state: int = None) -> float:
    
    """
    Perform a permutation test between two independent samples.

    Parameters
    ----------
    x, y : list-like
        Samples for the two groups.
    statistic : callable, optional
        Function to compute the test statistic. Defaults to absolute difference of means.
    n_resamples : int, optional
        Number of random permutations. Defaults to 10,000.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Defaults to 'two-sided'.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    p_value : float
        Estimated p-value.
    """

    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    if statistic is None:
        statistic = lambda a, b: np.mean(a) - np.mean(b)

    stat_obs = statistic(x, y)

    combined = np.concatenate([x, y])
    n_x = len(x)

    perm_stats = np.empty(n_resamples, dtype = float)

    for i in range(n_resamples):
        rng.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]
        perm_stats[i] = statistic(perm_x, perm_y)

    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_stats) >= np.abs(stat_obs))
    elif alternative == "greater":
        p_value = np.mean(perm_stats >= stat_obs)
    elif alternative == "less":
        p_value = np.mean(perm_stats <= stat_obs)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return p_value


#------------------------------
def permutation_test_welch_anova(vals_groups: np.array, n_resamples: int = 10000, random_state: int = None) -> float:

    rng = np.random.default_rng(random_state)

    # observed statistic (Welch F)
    F_obs = welch_F(vals_groups)

    #avaliando ao valor de F da reamostragem
    n_i = [len(vals) for vals in vals_groups]
    combined = np.concatenate( vals_groups )
    perm_F_vals = np.empty(n_resamples, dtype = float)
    #print(vals_groups)
    #print(combined)

    for i in range(n_resamples):
        
        rng.shuffle(combined)
        
        relabed_vals_groups = np.split(combined, np.cumsum(n_i)[:-1])
        perm_F_vals[i] = welch_F(relabed_vals_groups)
        
    p_value = np.mean(np.abs(perm_F_vals) >= np.abs(F_obs))

    return p_value


#------------------------------
def trim_array(arr: np.array, p_cut : float = 0.1):

    if not 0 <= p_cut < 0.5:
        raise ValueError("proportion_to_cut must be in [0, 0.5).")

    # Sort along the specified axis
    arr = np.sort(arr.copy())
    
    #print(arr)
    
    n = len(arr)
    k = int(np.floor(p_cut * n))          # number to trim from each end
    if 2 * k >= n:
        print("> func trim_array error! p_cut too large for sample size.")
        return arr

    # Slice to keep the central part only
    #print(arr[k : n - k])
    return arr[k : n - k]


#------------------------------
def trimmed_mean(x, proportion_to_cut = 0):

    return trim_array( np.asarray(x), p_cut = proportion_to_cut ).mean()


#------------------------------
def welch_F(groups):
    
    n_i = np.array([len(g) for g in groups], dtype=float)
    s2_i = np.array([g.var(ddof=1) for g in groups])
    w_i = n_i / s2_i
    
    x_mean_i = np.array([g.mean() for g in groups])    
    w_sum = w_i.sum()    
    x_mean_dot  = np.sum(w_i * x_mean_i) / w_sum

    #numerator
    num   = np.sum( w_i * (x_mean_i - x_mean_dot)**2 ) / (len(groups) - 1)
    
    #denominator
    den_aux = np.sum( (1 - w_i / w_sum) **2 / (n_i - 1) )
    den = 1 + ( 2 * (len(groups) -2 ) / (len(groups)**2 - 1) ) * den_aux
    
    return num / den