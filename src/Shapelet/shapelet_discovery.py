import numpy as np
import Shapelet.auto_pisd as auto_pisd
import Shapelet.pst_support_method as pstsm
import Shapelet.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial


class ShapeletDiscover():
    """
    Shapelet discovery module for extracting discriminative time-series subsequences (shapelets).

    This class implements a full discovery pipeline:
    - Extract PIS (Perceptually Important Segments)
    - Compute CI (Complexity-Invariance) weights
    - Compute distances between all candidates and all series
    - Evaluate discriminative power via Information Gain
    - Select top shapelets per class

    Args:
        window_size (int): Window radius used for local alignment.
        num_pip (float): Proportion of perceptually important points to extract.
        processes (int): Number of processes for multiprocessing.
        len_of_ts (int): Length of each time series.
        dim (int): Number of dimensions per time series (default: 1).
    """

    def __init__(self, window_size=20, num_pip=0.4, processes=4, len_of_ts=None, dim=1):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = len_of_ts
        self.dim = dim
        self.list_labels = None
        self.processes = processes

    def set_window_size(self, window_size):
        """
        Update the window size used for PISD distance and shapelet extraction.

        Args:
            window_size (int): New window radius.
        """
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet):
        """
        Selects and returns the top shapelets across all class groups.

        Converts the requested proportion into an absolute number of shapelets per class,
        then retrieves the highest-scoring shapelets (based on information gain)
        and concatenates them.

        Args:
            number_of_shapelet (float): Proportion of the time series length to use as
                                        the total number of shapelets.

        Returns:
            np.ndarray: Array of selected shapelets sorted by their start position.
        """

        number_of_shapelet = int(number_of_shapelet * self.len_of_ts)
        number_of_shapelet = round(number_of_shapelet / len(self.list_labels))
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        list_group_shapelet = pstsm.find_c_shapelet(self.list_group_ppi[0], number_of_shapelet)
        list_group_shapelet = np.asarray(list_group_shapelet)
        list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
        list_shapelet = list_group_shapelet
        for i in range(1, len(self.list_group_ppi)):
            list_ppi = self.list_group_ppi[i]
            list_group_shapelet = pstsm.find_c_shapelet(list_ppi, number_of_shapelet)
            list_group_shapelet = np.asarray(list_group_shapelet)
            list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
            list_shapelet = np.concatenate((list_shapelet,list_group_shapelet),axis=0)

        return list_shapelet

    def find_ppi(self, i, l):
        """
        Computes PPI (Perceptually Important Intervals) for a single time series within a label group.

        For each PIS interval of the current series:
            - Computes PISD distance to every other series
            - Computes Information Gain of using this interval as a split rule
            - Returns shapelet candidates with metadata:
            [ts_index, start, end, IG, label]

        Args:
            i (int): Index of the time series inside the label group.
            l (int): Index of the label group.

        Returns:
            list: List of PPI candidates with associated metadata.
        """

        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        pdm = {}
        t1 = self.group_train_data[l][i]
        pdm[i * 10000 + i] = np.zeros((0, 0))
        for p in range(len(self.train_data)):
            t2 = self.train_data[p]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 10000 + p] = matrix_1

        for j in range(len(self.group_train_data_piss[l][i])):
            ts_pis = self.group_train_data_piss[l][i][j]
            ts_ci_pis = self.group_train_data_ci_piss[l][i][j]
            # Calculate subdist with all time series
            list_dist = []
            for p in range(len(self.train_data)):
                if p == ts_pos:
                    list_dist.append(0)
                else:
                    matrix = pdm[ts_pos * 10000 + p]
                    ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                    ts_2_ci = self.train_data_ci[p]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                   self.list_end_pos, ts_ci_pis, pcs_ci_list)
                    list_dist.append(dist)

            # Calculate best information gain
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            ppi = np.asarray([ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l]])
            list_result.append(ppi)
        return list_result

    def extract_candidate(self, train_data):
        """
        Extracts PIS (Perceptually Important Segments) and CI (Complexity Invariance)
        information for each time series.

        Steps performed:
            1. Extract perceptually important segments (PIS) per series.
            2. Compute CI profiles for each segment.
            3. Store PIS and CI for later shapelet evaluation.

        Args:
            train_data (list or np.ndarray): Collection of training time series.

        Prints:
            Execution time for extraction.
        """

        # Extract shapelet candidate
        time1 = time.time()
        self.train_data_piss = np.asarray(
            [auto_pisd.auto_piss_extractor(train_data[i], num_pip=self.num_pip) for i in range(len(train_data))])
        ci_return = [auto_pisd.auto_ci_extractor(train_data[i], self.train_data_piss[i]) for i in range(len(train_data))]
        self.train_data_ci = np.asarray([ci_return[i][0] for i in range(len(ci_return))])
        self.train_data_ci_piss = np.asarray([ci_return[i][1] for i in range(len(ci_return))])
        time1 = time.time() - time1
        print("extracting time: %s" % time1)


    def discovery(self, train_data, train_labels):
        """
        Main shapelet discovery pipeline.

        Performs the following:
            - Stores training data and labels
            - Computes sliding window start/end indexing
            - Groups time series by class label
            - Computes PPI candidates for each group
            - Aggregates all candidates for each label
            - Stores discovered PPI groups for later shapelet selection

        Args:
            train_data (list or np.ndarray): List of time series.
            train_labels (list or np.ndarray): Corresponding class labels.

        Prints:
            Total execution time for the discovery process.
        """

        time2 = time.time()
        self.train_data = train_data
        self.train_labels = train_labels

        self.len_of_ts = len(train_data[0])
        self.list_labels = np.unique(train_labels)

        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1

        # Divide time series into group of label
        group_train_data = [[] for i in self.list_labels]
        group_train_data_pos = [[] for i in self.list_labels]
        group_train_data_piss = [[] for i in self.list_labels]
        group_train_data_ci_piss = [[] for i in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(train_data)):
                if train_labels[i] == self.list_labels[l]:
                    group_train_data[l].append(train_data[i])
                    group_train_data_pos[l].append(i)
                    group_train_data_piss[l].append(self.train_data_piss[i])
                    group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        self.group_train_data = np.asarray(group_train_data)
        self.group_train_data_pos = np.asarray(group_train_data_pos)
        self.group_train_data_piss = np.asarray(group_train_data_piss)
        self.group_train_data_ci_piss = np.asarray(group_train_data_ci_piss)

        # Select shapelet
        self.list_group_ppi = []
        for l in range(len(self.list_labels)):
            p = multiprocessing.Pool(processes=self.processes)
            temp_ppi = p.map(partial(self.find_ppi, l=l), range(len(self.group_train_data[l])))

            list_ppi = []
            for i in range(len(self.group_train_data[l])):
                pii_in_i = temp_ppi[i]
                for j in range(len(pii_in_i)):
                    list_ppi.append(pii_in_i[j])
            list_ppi = np.asarray(list_ppi)
            self.list_group_ppi.append(list_ppi)
        time2 = time.time() - time2
        print("window_size: %s - evaluating_time: %s" % (self.window_size, time2))

