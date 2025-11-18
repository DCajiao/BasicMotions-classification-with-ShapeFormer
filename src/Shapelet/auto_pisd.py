import numpy as np
import scipy.stats as stats


def auto_piss_extractor(i, time_series=None, num_pip=0.2, j=0, return_pip=0):
    """
    Extracts perceptually important subsequences (PIS) from a time series using PIP-based segmentation.

    Args:
        i (int): Index of the time series to process.
        time_series (list of np.ndarray): List of time series to extract from.
        num_pip (float): Proportion of PIPs to extract (default = 0.2).
        j (int): Auxiliary index for logging/debugging.
        return_pip (bool): If True, returns list of PIP indices instead of PIS intervals.

    Returns:
        np.ndarray: Array of shape (n_intervals, 2) with PIS intervals, or
                    array of PIP indices if `return_pip` is True.
    """

    def pd_distance(parameter):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = parameter[0], parameter[1], parameter[2], \
            parameter[3], parameter[4], parameter[5]
        b = -1
        a = (p2_y - p3_y) / (p2_x - p3_x)
        c = p2_y - p2_x * a
        return abs(b * p1_y + a * p1_x + c) / ((a ** 2 + b ** 2)**0.5)
    print("extract - %s - %s - %s" % (j, i, num_pip))
    ts = time_series[i]
    max_no_pip = int(num_pip*len(ts))
    if max_no_pip < 5:
        max_no_pip = 5
    # Init list of index of time_series with z_norm
    list_index_z = stats.zscore(list(range(len(ts))))

    piss = []
    pips = [0, (len(ts) - 1)]
    remain_pos = list(range(1, (len(ts) - 1)))
    for i in range(max_no_pip - 2):
        biggest_dist = -1
        biggest_dist_pos = -1
        for j in remain_pos:
            p_y = ts[j]
            p_x = list_index_z[j]
            # Find the adjective point (p2,p3) of p in pips
            p2_x = p2_y = p3_x = p3_y = None
            for g in range(len(pips)):
                end_pos = pips[g]
                if j < end_pos:
                    p2_y, p2_x = ts[end_pos], list_index_z[end_pos]
                    start_pos = pips[g - 1]
                    p3_y, p3_x = ts[start_pos], list_index_z[start_pos]
                    break
            # Calculate the distance of p to p2,p3
            distance = pd_distance(
                parameter=[p_x, p_y, p2_x, p2_y, p3_x, p3_y])
            if distance > biggest_dist:
                biggest_dist, biggest_dist_pos = distance, j
        # Add biggest_dist_pos into pips
        if biggest_dist_pos == -1:
            print("-dist errpr-")

        pips.append(biggest_dist_pos)
        pips.sort()
        # Remove biggest_dist_pos from remain_pos
        remain_pos.remove(biggest_dist_pos)

        # Append new piss
        pos = pips.index(biggest_dist_pos)
        piss.append(np.asarray([pips[pos - 1], pips[pos + 1]]))
        if pos > 1:
            piss.append(np.asarray([pips[pos-2], pips[pos]]))
        if pos < len(pips) - 2:
            piss.append(np.asarray([pips[pos], pips[pos+2]]))
        # print(len(piss))
    if return_pip:
        return np.asarray(pips)
    return np.asarray(piss)


def auto_ci_extractor(time_series, piss):
    """
    Computes the contextual interval (CI) weights for a batch of time series and their extracted PIS.

    Args:
        time_series (list of np.ndarray): List of time series.
        piss (list of np.ndarray): List of PIS intervals per time series.

    Returns:
        tuple:
            - ts_ci (list of np.ndarray): CI vectors for each time series.
            - list_ci_piss (list of list of float): CI weights for each PIS interval.
    """

    def ci_calculate(time_series):
        return (time_series[:-1] - time_series[1:]) ** 2

    ts_ci = []
    list_ci_piss = [[] for i in range(len(piss))]
    for i in range(len(piss)):
        ts_ci.append(ci_calculate(time_series[i]))
        sc = piss[i]
        for j in range(len(sc)):
            ci = (np.sum(ts_ci[sc[j][0]:sc[j][1] - 1]) + 0.001)**0.5
            list_ci_piss[i].append(ci)
    return ts_ci, list_ci_piss


def pcs_extractor(pis, w, len_of_ts):
    """
    Expands a given PIS region by a window size `w` to form a PCS (Potential Candidate Segment).

    Args:
        pis (list or np.ndarray): [start, end] interval of PIS.
        w (int): Expansion window.
        len_of_ts (int): Total length of the time series.

    Returns:
        list: [start, end] of the extended segment.
    """
    if w == 0:
        return pis

    start_pos = pis[0] - w + 1
    start_pos = start_pos if start_pos >= 0 else 0
    end_pos = pis[1] + w
    end_pos = end_pos if end_pos < len_of_ts else len_of_ts

    return [start_pos, end_pos]


def ci_extractor(time_series, list_important_point):
    """
    Computes contextual interval (CI) weights for a series of important points.

    Args:
        time_series (np.ndarray): A single time series.
        list_important_point (list): List of PIP indices.

    Returns:
        tuple:
            - np.ndarray: Vector of CI weights for the full time series.
            - list of float: CI values for each PIS interval.
    """

    def ci(time_series):
        return (time_series[:-1] - time_series[1:]) ** 2

    ts_ci = ci(time_series)
    list_ci_piss = []
    for j in range(len(list_important_point)-2):
        ci = (np.sum(ts_ci[list_important_point[j]
              :list_important_point[j+2] - 1]) + 0.001)**0.5
        list_ci_piss.append(ci)
    return ts_ci, list_ci_piss


def find_min_dist(pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list):
    """
    Computes the minimum dissimilarity between a PIS and a sliding PCS using contextual information.

    Args:
        pis (list): [start, end] interval of PIS.
        pcs (list): [start, end] interval of extended PCS.
        matrix_t1 (np.ndarray): Precomputed distance matrix.
        list_start_pos (list): Start positions for sliding comparisons.
        list_end_pos (list): End positions for sliding comparisons.
        pis_ci (float): CI value of the PIS.
        pcs_ci_list (list): CI values for the candidate PCS.

    Returns:
        float: Minimum distance between PIS and PCS under CI weighting.
    """

    sdist = (np.sum(
        matrix_t1[pis[0]:pis[1], list_start_pos[pis[0]]:list_end_pos[pis[1]-1]], axis=0))**0.5

    pcs_cs = np.cumsum(pcs_ci_list)
    w = pcs[1] - pcs[0] - pis[1] + pis[0]
    d_ci = pcs_cs[-w-1:]
    d_ci[-w:] = d_ci[-w:] - pcs_cs[:w]
    d_ci = (d_ci + 0.001) ** 0.5
    d_ci = np.maximum(d_ci, pis_ci)/np.minimum(d_ci, pis_ci)

    return np.min(np.multiply(sdist, d_ci))


def find_min_dist_s1(pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list):
    """
    Simplified variant of `find_min_dist` to compute CI-weighted distance between PIS and PCS.

    Args:
        pis (list): [start, end] of perceptually important segment.
        pcs (list): [start, end] of extended windowed segment.
        matrix_t1 (np.ndarray): Precomputed squared distance matrix.
        list_start_pos (list): Start index per time step.
        list_end_pos (list): End index per time step.
        pis_ci (float): Contextual interval weight for PIS.
        pcs_ci_list (list of float): CI values for the full candidate segment.

    Returns:
        float: Minimum CI-weighted dissimilarity value.
    """

    # sdist = calculate_subdist(matrix_t1, pis, list_start_pos, list_end_pos)
    sdist = (np.sum(
        matrix_t1[pis[0]:pis[1], list_start_pos[pis[0]]:list_end_pos[pis[1]-1]], axis=0))**0.5

    # return np.min(sdist)
    # len_pis = pis[1] - pis[0]
    # len_pcs = pcs[1] - pcs[0]
    # pcs_ci_temp = np.sum(pcs_ci_list[:len_pis-1])
    # pcs_ci = (pcs_ci_temp + 0.001)**0.5
    # sdist[0] = sdist[0]*(max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci))
    # for g in range(0, len_pcs - len_pis):
    #     pcs_ci_temp = pcs_ci_temp + pcs_ci_list[g + len_pis - 1] - pcs_ci_list[g]
    #     pcs_ci = (pcs_ci_temp + 0.001)**0.5
    #     sdist[g+1] = sdist[g+1]*(max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci))

    # d_ci = [pcs_ci_temp]
    # for g in range(0, len_pcs - len_pis):
    #     pcs_ci_temp = pcs_ci_temp + pcs_ci_list[g + len_pis - 1] - pcs_ci_list[g]
    #     d_ci.append(pcs_ci_temp)
    # d_ci = np.asarray(d_ci)

    # return np.min(sdist)

    pcs_cs = np.cumsum(pcs_ci_list)
    w = pcs[1] - pcs[0] - pis[1] + pis[0]
    d_ci = pcs_cs[-w-1:]
    d_ci[-w:] = d_ci[-w:] - pcs_cs[:w]
    d_ci = (d_ci + 0.001) ** 0.5
    d_ci = np.maximum(d_ci, pis_ci)/np.minimum(d_ci, pis_ci)

    return np.min(np.multiply(sdist, d_ci))


def find_min_dist_vec(pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list):
    """
    Vectorized version of `find_min_dist` returning all CI-weighted distances across a sliding window.

    Args:
        pis (list): [start, end] indices of PIS.
        pcs (list): [start, end] indices of candidate PCS.
        matrix_t1 (np.ndarray): Precomputed distance matrix.
        list_start_pos (list): Start positions per row.
        list_end_pos (list): End positions per row.
        pis_ci (float): CI of PIS.
        pcs_ci_list (list): CI weights for PCS region.

    Returns:
        np.ndarray: Array of weighted distances for each alignment window.
    """
    sdist = calculate_subdist(matrix_t1, pis, list_start_pos, list_end_pos)

    len_pis = pis[1] - pis[0]
    len_pcs = pcs[1] - pcs[0]
    pcs_ci_temp = np.sum(pcs_ci_list[:len_pis-1])
    pcs_ci = (pcs_ci_temp + 0.001)**0.5
    d_ci = [(max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci))]
    for g in range(0, len_pcs - len_pis):
        pcs_ci_temp += pcs_ci_list[g + len_pis - 1]
        pcs_ci_temp -= pcs_ci_list[g]
        pcs_ci = (pcs_ci_temp + 0.001)**0.5
        d_ci.append((max(pis_ci, pcs_ci) / min(pis_ci, pcs_ci)))
    d_ci = np.asarray(d_ci)

    return np.multiply(sdist, d_ci)


def PISD(ts_1, ts_1_piss, ts_1_ci, ts_1_ci_piss,
         ts_2, ts_2_piss, ts_2_ci, ts_2_ci_piss,
         list_start_pos, list_end_pos, w, min_dist):
    """
    Computes the full PISD (Perceptually Important Subsequence Distance) between two time series.

    Args:
        ts_1 (np.ndarray): First time series.
        ts_1_piss (list): PIS intervals of ts_1.
        ts_1_ci (np.ndarray): CI vector of ts_1.
        ts_1_ci_piss (list): CI weights for PIS intervals in ts_1.
        ts_2 (np.ndarray): Second time series.
        ts_2_piss (list): PIS intervals of ts_2.
        ts_2_ci (np.ndarray): CI vector of ts_2.
        ts_2_ci_piss (list): CI weights for PIS intervals in ts_2.
        list_start_pos (list): Start positions for sliding window.
        list_end_pos (list): End positions for sliding window.
        w (int): Sliding window size.
        min_dist (float): Early stopping threshold.

    Returns:
        float: Normalized shapelet-based dissimilarity score between ts_1 and ts_2.
    """

    pisd = 0
    sum_len = calculate_sum_of_len(ts_1_piss, ts_2_piss)
    # Calculate matrix of t1 and t2
    matrix_t1, matrix_t2 = calculate_matrix(ts_1, ts_2, w)

    # Calculate distance from ts_1_piss to ts_1_piss
    for k in range(len(ts_1_piss)):
        pis = ts_1_piss[k]
        len_real_pis = pis[1] - pis[0]
        pcs = pcs_extractor(pis, w, len(ts_1))
        pis_ci = ts_1_ci_piss[k]
        pcs_ci_list = ts_2_ci[pcs[0]:pcs[1]-1]
        d_dist = find_min_dist(
            pis, pcs, matrix_t1, list_start_pos, list_end_pos, pis_ci, pcs_ci_list)
        pisd += len_real_pis * d_dist
        if pisd / sum_len >= min_dist:
            break

    # Calculate distance from ts_2_piss to ts_2_piss
    for k in range(len(ts_2_piss)):
        pis = ts_2_piss[k]
        len_real_pis = pis[1] - pis[0]
        pcs = pcs_extractor(pis, w, len(ts_2))
        pis_ci = ts_2_ci_piss[k]
        pcs_ci_list = ts_1_ci[pcs[0]:pcs[1]-1]
        d_dist = find_min_dist(
            pis, pcs, matrix_t2, list_start_pos, list_end_pos, pis_ci, pcs_ci_list)
        pisd += len_real_pis * d_dist
        if pisd / sum_len >= min_dist:
            break
    # print(min_dist)
    return pisd / sum_len


def calculate_sum_of_len(train_ts_sq, test_ts_sq):
    """
    Computes total length of all PIS intervals from two time series.

    Args:
        train_ts_sq (list of intervals): PIS intervals for first time series.
        test_ts_sq (list of intervals): PIS intervals for second time series.

    Returns:
        int: Total length of intervals.
    """

    sum_len = 0
    for k in range(len(train_ts_sq)):
        sum_len += train_ts_sq[k][1] - train_ts_sq[k][0]
    for k in range(len(test_ts_sq)):
        sum_len += test_ts_sq[k][1] - test_ts_sq[k][0]
    return sum_len


def calculate_matrix(ts_1, ts_2, w):
    """
    Constructs a pair of sliding distance matrices for two time series with window size w.

    Args:
        ts_1 (np.ndarray): First time series.
        ts_2 (np.ndarray): Second time series.
        w (int): Window size for alignment.

    Returns:
        tuple of np.ndarray:
            - matrix_1: Distance matrix with ts_1 as reference.
            - matrix_2: Distance matrix with ts_2 as reference.
    """

    matrix_1 = np.ones((len(ts_1), 2 * w + 1))*np.inf
    matrix_2 = np.ones((len(ts_2), 2 * w + 1))*np.inf

    list_dist = (ts_1 - ts_2) ** 2
    matrix_1[:, w] = list_dist
    matrix_2[:, w] = list_dist

    for i in range(w):
        list_dist = (ts_1[(i + 1):] - ts_2[:-(i + 1)]) ** 2
        matrix_1[(i + 1):, w - (i + 1)] = list_dist
        matrix_2[:-(i + 1), w + (i + 1)] = list_dist
        list_dist = (ts_2[(i + 1):] - ts_1[:-(i + 1)]) ** 2
        matrix_2[(i + 1):, w - (i + 1)] = list_dist
        matrix_1[:-(i + 1), w + (i + 1)] = list_dist

    return matrix_1, matrix_2


def calculate_subdist(matrix, pis, list_start_pos, list_end_pos):
    """
    Computes subdistance by summing a matrix slice corresponding to a PIS alignment.

    Args:
        matrix (np.ndarray): Precomputed squared distance matrix.
        pis (list): [start, end] of PIS region.
        list_start_pos (list): Start indices per matrix row.
        list_end_pos (list): End indices per matrix row.

    Returns:
        np.ndarray: 1D vector of distances over the alignment window.
    """

    # sub_matrix = matrix[pis[0]:pis[1], list_start_pos[pis[0]]:list_end_pos[pis[1]-1]]
    # list_dist = np.sqrt(np.sum(sub_matrix, axis=0))
    return (np.sum(matrix[pis[0]:pis[1], list_start_pos[pis[0]]:list_end_pos[pis[1]-1]], axis=0))**0.5


def subdist(t1, t2):
    """
    Computes CI-weighted distance between two time series using a sliding window approach.

    Args:
        t1 (np.ndarray): Query time series.
        t2 (np.ndarray): Candidate time series (must be ≥ t1 in length).

    Returns:
        float: Minimum distance between t1 and all aligned windows in t2.
    """

    len_t1, len_t2 = len(t1), len(t2)
    diff_len = len_t2 - len_t1 + 1
    min_dist = np.inf
    ci_t1 = (np.sum((t1[:-1] - t1[1:]) ** 2) + 0.001)**0.5
    temp_t2 = (t2[:-1] - t2[1:]) ** 2
    first_sum = np.sum(temp_t2[:len_t1 - 1])
    for i in range(diff_len):
        diff_ts = (np.sum((t1 - t2[i:i + len_t1]) ** 2))**0.5
        ci_sq_t2 = (first_sum + 0.001)**0.5
        if i < diff_len - 1:
            first_sum += temp_t2[i + len_t1 - 1]
            first_sum -= temp_t2[i]

        dist = diff_ts * (max(ci_t1, ci_sq_t2) / min(ci_t1, ci_sq_t2))
        min_dist = min(dist, min_dist)

    return min_dist


def subdist_vec(t1, t2):
    """
    Vectorized CI-weighted distance computation across all sliding windows of t2 relative to t1.

    Args:
        t1 (np.ndarray): Query time series.
        t2 (np.ndarray): Candidate time series (must be ≥ t1 in length).

    Returns:
        np.ndarray: Array of distance values for each aligned window.
    """

    len_t1, len_t2 = len(t1), len(t2)
    diff_len = len_t2 - len_t1 + 1
    list_dict = []
    ci_t1 = (np.sum((t1[:-1] - t1[1:]) ** 2) + 0.001)**0.5
    temp_t2 = (t2[:-1] - t2[1:]) ** 2
    first_sum = np.sum(temp_t2[:len_t1 - 1])
    for i in range(diff_len):
        diff_ts = (np.sum((t1 - t2[i:i + len_t1]) ** 2))**0.5
        ci_sq_t2 = (first_sum + 0.001)**0.5
        if i < diff_len - 1:
            first_sum += temp_t2[i + len_t1 - 1]
            first_sum -= temp_t2[i]

        dist = diff_ts * (max(ci_t1, ci_sq_t2) / min(ci_t1, ci_sq_t2))
        list_dict.append(dist)

    return np.asarray(list_dict)


if __name__ == '__main__':
    len_pcs = 10
    len_pis = 5
    list_a = np.arange(20)
    c = np.cumsum(list_a)
    r = c[5:10] - c[0:5]
