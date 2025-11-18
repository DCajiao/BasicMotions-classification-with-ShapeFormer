import numpy as np


def entropy(num_of_pos, num_of_neg):
    """
    Computes the binary entropy for a split with given counts of positive and negative samples.

    This is the standard Shannon entropy used in information theory:
        H(p) = -p*log2(p) - (1-p)*log2(1-p)

    Args:
        num_of_pos (int): Number of positive instances.
        num_of_neg (int): Number of negative instances.

    Returns:
        float: Entropy value of the current split.
    """

    total_len = num_of_pos + num_of_neg
    result = 0
    if num_of_pos / total_len > 0: result -= (num_of_pos / total_len) * np.log2(num_of_pos / total_len)
    if num_of_neg / total_len > 0: result -= (num_of_neg / total_len) * np.log2(num_of_neg / total_len)
    return result


def binary_inforgain(g1_num_of_pos, g1_num_of_neg, g2_num_of_pos, g2_num_of_neg):
    """
    Calculates the binary information gain for a proposed split of time series instances.

    Information gain is defined as the reduction in entropy achieved by partitioning
    the dataset into two groups (g1 and g2).

    Args:
        g1_num_of_pos (int): Number of positives in group 1.
        g1_num_of_neg (int): Number of negatives in group 1.
        g2_num_of_pos (int): Number of positives in group 2.
        g2_num_of_neg (int): Number of negatives in group 2.

    Returns:
        float: Information gain from the binary split.
    """

    total_len = g1_num_of_pos + g1_num_of_neg + g2_num_of_pos + g2_num_of_neg
    g1_len = g1_num_of_pos + g1_num_of_neg
    g2_len = g2_num_of_pos + g2_num_of_neg
    init_num_of_pos = g1_num_of_pos + g2_num_of_pos
    init_num_of_neg = g1_num_of_neg + g2_num_of_neg

    return entropy(init_num_of_pos, init_num_of_neg)\
           -(g1_len/total_len)*entropy(g1_num_of_pos, g1_num_of_neg)\
           -(g2_len/total_len)*entropy(g2_num_of_pos, g2_num_of_neg)


def sort_by_sub_dist(list_sub_dist, input_label):
    """
    Sorts the labels and distances based on sub-distance values in ascending order.

    This function is used to align labels with their respective sub-distance values
    to evaluate thresholds for potential split points.

    Args:
        list_sub_dist (list or np.ndarray): List of sub-distances.
        input_label (list or np.ndarray): Corresponding class labels.

    Returns:
        tuple:
            - np.ndarray: Sorted labels based on sub-distances.
            - np.ndarray: Sorted sub-distances.
    """
    list_sd_table = np.array([input_label, list_sub_dist]).transpose()
    list_sd_table = list_sd_table[list_sd_table[:, 1].argsort()]

    list_label_table = list_sd_table[:, 0]
    list_subdist_table = list_sd_table[:, 1]

    return list_label_table, list_subdist_table


def find_best_split_point_and_info_gain(list_sub_dist, input_label, target_class):
    """
    Finds the best split point for a candidate shapelet based on maximum information gain.

    Iteratively examines thresholds across sorted sub-distances and evaluates
    the binary information gain for each. Returns the highest IG score.

    Args:
        list_sub_dist (list or np.ndarray): Sub-distances of the candidate shapelet to all series.
        input_label (list or np.ndarray): Ground truth class labels.
        target_class (int or str): The label considered as the positive class for entropy.

    Returns:
        float: Maximum information gain found across all valid thresholds.
    """

    list_label_table, list_subdist_table = sort_by_sub_dist(list_sub_dist=list_sub_dist,
                                                            input_label=input_label)

    best_ig = -1
    g1_pos = 0
    g1_neg = 0
    g2_pos = 0
    g2_neg = 0
    for i in list_label_table:
        if i == target_class:
            g2_pos += 1
        else:
            g2_neg += 1

    t_pos = g2_pos

    rate = 0.2
    for i in range(len(list_label_table)-1):
        if list_label_table[i] == target_class:
            g1_pos += 1
            g2_pos -= 1
        else:
            g1_neg += 1
            g2_neg -= 1
        if round(g1_pos/t_pos,1) == rate:
            inforgain = binary_inforgain(g1_num_of_pos=g1_pos, g1_num_of_neg=g1_neg,
                                          g2_num_of_pos=g2_pos, g2_num_of_neg=g2_neg)
            best_ig = max(best_ig, inforgain)
            if rate < 1:
                rate += 0.2
            else:
                break

    return best_ig