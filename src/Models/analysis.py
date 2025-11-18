"""
Collection of functions which enable the evaluation of a classifier's performance,
by showing confusion matrix, accuracy, recall, precision etc.
"""

import numpy as np
import sys

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from tabulate import tabulate
import math
import logging
from datetime import datetime


def acc_top_k(predictions, y_true):
    """
    Computes the cumulative accuracy where the correct class is considered correctly predicted
    if it appears in the top-k ranked predictions.

    Args:
        predictions (np.ndarray): Array of shape (N, k) with top-k predicted class indices per sample.
        y_true (np.ndarray): Array of shape (N,) with ground truth labels.

    Returns:
        np.ndarray: 1D array of shape (k,) where each entry i is the accuracy when allowing the correct
                    label to be among the top-(i+1) predictions.
    """

    y_true = y_true[:, np.newaxis]

    # Create upper triangular matrix of ones, to be used in construction of V
    building_blocks = np.zeros((predictions.shape[1], predictions.shape[1]))
    building_blocks[np.triu_indices(predictions.shape[1])] = 1

    # A matrix of the same shape as predictions. For each sample, the index corresponding
    # to a correct prediction is 1, as well as all following indices.
    # Example: y_true = [1,0], predictions = [[1 5 4],[2 0 3]]. Then: V = [[1 1 1],[0 1 1]]
    V = np.zeros_like(predictions, dtype=int)  # validity matrix
    sample_ind, rank_ind = np.where(predictions == y_true)

    V[sample_ind, :] = building_blocks[rank_ind, :]

    return np.mean(V, axis=0)


def accuracy(y_pred, y_true, excluded_labels=None):
    """
    Computes accuracy, optionally excluding specific labels from the evaluation.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): Ground truth labels.
        excluded_labels (list[int], optional): Labels to exclude from accuracy computation.

    Returns:
        float: Accuracy score, ignoring excluded labels if specified.
    """

    if excluded_labels is None:
        return np.mean(y_pred == y_true)
    else:
        included = (y_pred != excluded_labels[0]) & (
            y_true != excluded_labels[0])
        # The following extra check (rather than initializing with an array of ones)
        # is done because a single excluded label is the most common case
        if len(excluded_labels) > 1:
            for label in excluded_labels[1:]:
                included &= (y_pred != label) & (y_true != label)

        return np.mean(y_pred[included] == y_true[included])


def precision(y_true, y_pred, label):
    """
    Computes precision for a given class.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        label (int): Class label for which to compute precision.

    Returns:
        float: Precision score for the given class.
    """

    predicted_in_C = (y_pred == label)
    num_pred_in_C = np.sum(predicted_in_C)
    if num_pred_in_C == 0:
        return 0
    return np.sum(y_true[predicted_in_C] == label) / num_pred_in_C


def recall(y_true, y_pred, label):
    """
    Computes recall for a given class.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        label (int): Class label for which to compute recall.

    Returns:
        float: Recall score for the given class.
    """

    truly_in_C = (y_true == label)
    num_truly_in_C = np.sum(truly_in_C)
    if num_truly_in_C == 0:
        return 0  # or NaN?
    return np.sum(y_pred[truly_in_C] == label) / num_truly_in_C


def limiter(metric_functions, y_true, y_pred, y_scores, score_thr, label):
    """
    Applies a list of metric functions (e.g. precision/recall) while filtering predictions
    below a score threshold for a specific class.

    Args:
        metric_functions (list[Callable]): List of metric functions to apply.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_scores (np.ndarray): Confidence scores for predictions.
        score_thr (float): Threshold to ignore low-confidence predictions.
        label (int): Target class to apply filtering.

    Returns:
        list: List of metric values corresponding to input metric functions.
    """

    ltd_pred = np.copy(y_pred)
    ltd_pred[(ltd_pred == label) & (y_scores < score_thr)] = -1

    output = [func(y_true, ltd_pred, label) for func in metric_functions]

    return output


def prec_rec_parametrized_by_thr(y_true, y_pred, y_scores, label, Npoints, min_score=None, max_score=None):
    """
    Computes precision and recall values for a class label as a function of the score threshold.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_scores (np.ndarray): Confidence scores for predictions.
        label (int): Class of interest.
        Npoints (int): Number of threshold points to evaluate.
        min_score (float, optional): Minimum threshold value. If None, derived from data.
        max_score (float, optional): Maximum threshold value. If None, derived from data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple with (1) precision-recall values (Npoints × 2) and
                                       (2) the threshold grid.
    """

    if (min_score is None) or (max_score is None):
        predicted_in_C = (y_pred == label)
        # guarantees that all predictions are kept
        min_score = 0.99 * np.amin(y_scores[predicted_in_C])
        # guarantees that no prediction is kept
        max_score = 1.01 * np.amax(y_scores[predicted_in_C])

    grid = np.linspace(min_score, max_score, Npoints)

    def measure(x): return limiter(
        [precision, recall], y_true, y_pred, y_scores, x, label)

    return np.array(map(measure, grid)), grid


'''
def plot_prec_vs_rec(score_grid, rec, prec, prec_requirement=None, thr_opt=None, title=None, show=True, save_as=None):
    """Plots a figure depicting precision and recall as a function of the score threshold.
    Optionally also depicts an imposed precision requirement and a chosen score threshold value."""

    if not (thr_opt is None):
        thr_opt = thr_opt if not (math.isinf(thr_opt)) else None

    plt.figure()
    if title:
        plt.suptitle(title)

    # Recall and Precision vs. Score Threshold
    plt.subplot(211)
    l_rec, = plt.plot(score_grid, rec, '.-')

    plt.hold(True)
    l_prec, = plt.plot(score_grid, prec, 'g.-')
    plt.ylim((0, 1.01))
    plt.xlabel('score threshold')

    legend_lines = [l_rec, l_prec]
    legend_labels = ['recall', 'precision']

    if prec_requirement:
        l_prec_req = plt.axhline(prec_requirement, color='r', linestyle='--')
        legend_lines.append(l_prec_req)
        legend_labels.append('prec. req.')

    if not (thr_opt is None):
        l_score_thr = plt.axvline(thr_opt, color='r')
        legend_lines.append(l_score_thr)
        legend_labels.append('opt. thr.')

    plt.legend(legend_lines, legend_labels, loc='lower right', fontsize=10)

    # Recall vs. Precision
    plt.subplot(212)
    plt.plot(prec, rec, '.-')

    plt.ylim((0, 1.01))
    plt.xlim((0, 1.01))
    plt.ylabel('recall')
    plt.xlabel('precision')

    if prec_requirement:
        l_prec_req = plt.axvline(prec_requirement, color='r', linestyle='--')
        plt.legend([l_prec_req], ['precision req.'], loc='lower left', fontsize=10)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', format='pdf')

    if show:
        plt.tight_layout()
        plt.show(block=False)


def plot_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """Plot confusion matrix in a separate window"""
    plt.imshow(ConfMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_strings:
        tick_marks = np.arange(len(label_strings))
        plt.xticks(tick_marks, label_strings, rotation=90)
        plt.yticks(tick_marks, label_strings)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''


def print_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix'):
    """
    Prints a formatted confusion matrix to the terminal.

    Args:
        ConfMat (np.ndarray): Confusion matrix (2D array).
        label_strings (list[str], optional): Class labels to display.
        title (str, optional): Title of the matrix.
    """

    if label_strings is None:
        label_strings = ConfMat.shape[0] * ['']

    print(title)
    print(len(title) * '-')
    # Make printable matrix:
    print_mat = []
    for i, row in enumerate(ConfMat):
        print_mat.append([label_strings[i]] + list(row))
    print(tabulate(print_mat, headers=[
          'True\Pred'] + label_strings, tablefmt='orgtbl'))


class Analyzer(object):

    def __init__(self, maxcharlength=35, plot=False, print_conf_mat=False, output_filepath=None):
        """
        Initializes the Analyzer class with optional settings for verbosity and output.

        Args:
            maxcharlength (int): Maximum number of characters to display per class name.
            plot (bool): Whether to enable plotting of results (e.g., confusion matrices, histograms).
            print_conf_mat (bool): Whether to print confusion matrices to the terminal.
            output_filepath (str, optional): If provided, logs will be saved to this file.
        """

        self.maxcharlength = maxcharlength
        self.plot = plot
        self.print_conf_mat = print_conf_mat

        # create logger
        self.logID = str(
            # this is to enable individual logging configuration between different instances
            datetime.now())
        self.logger = logging.getLogger(self.logID)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if output_filepath:
            # create file handler
            fh = logging.FileHandler(output_filepath)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def show_acc_top_k_improvement(self, y_pred, y_true, k=5, inp='scores'):
        """
        Displays how accuracy improves as the top-k prediction window increases.

        Args:
            y_pred (np.ndarray): Either:
                - If inp == 'scores': array of shape (N, C) with class scores per sample.
                - If inp == 'indices': array of shape (N, k) with precomputed top-k class indices.
            y_true (np.ndarray): Ground truth labels.
            k (int): Maximum number of predictions to consider per sample.
            inp (str): Input type ('scores' or 'indices').

        Returns:
            np.ndarray: Array of top-k cumulative accuracy values.
        """

        print('How accuracy improves when allowing correct result being in the top 1, 2, ..., k predictions:\n')

        if inp == 'scores':
            predictions = np.argsort(y_pred, axis=1)[
                :, ::-1]  # sort in descending order
        else:
            predictions = y_pred

        predictions = predictions[:, :min(
            k, predictions.shape[1])]  # take top k

        accuracy_per_rank = acc_top_k(predictions, y_true)

        row1 = ['k'] + range(1, len(accuracy_per_rank) + 1)
        row2 = ['Accuracy'] + list(accuracy_per_rank)
        print(tabulate([row1, row2], tablefmt='orgtbl'))

        if self.plot:
            from matplotlib.ticker import MaxNLocator

            ax = plt.figure().gca()
            plt.plot(np.arange(1, k + 1, dtype=int), accuracy_per_rank, '.-')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Number of allowed predictions (k)')
            plt.ylabel(
                'Cumulative accuracy\n(prob. of correct result being in top k pred.)')
            plt.title('Cumulative Accuracy vs Number of allowed predictions')

            plt.show(block=False)

        return accuracy_per_rank

    def generate_classification_report(self, digits=3, number_of_thieves=2, maxcharlength=35):
        """
        Generates a formatted classification report for each class, including metrics and top confusion sources.

        Args:
            digits (int): Number of decimal places to include in the report.
            number_of_thieves (int): Number of "thief" classes (confused predictions) to include.
            maxcharlength (int): Maximum character length for class names.

        Returns:
            str: A multi-line formatted string report with precision, recall, f1-score, and confusion insights.
        """

        # relative frequencies of each class in the true lables
        relative_freq = self.support / np.sum(self.support)
        sorted_class_indices = np.argsort(relative_freq)[
            ::-1]  # sort by "importance" of classes (i.e. occurance frequency)

        last_line_heading = 'avg / total'

        width = max(len(cn) for cn in self.existing_class_names)
        width = max(width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score",
                   "rel. freq.", "abs. freq.", "biggest thieves"]
        fmt = '%% %ds' % width  # first column: class name
        fmt += '  '
        fmt += ' '.join(['% 10s' for _ in headers[:-1]])
        fmt += '|\t % 5s'
        fmt += '\n'

        headers = [""] + headers
        report = fmt % tuple(headers)
        report += '\n'

        for i in sorted_class_indices:
            values = [self.existing_class_names[i]]
            for v in (self.precision[i], self.recall[i], self.f1[i],
                      # v is NOT a tuple, just goes through this list 1 el. at a time
                      relative_freq[i]):
                values += ["{0:0.{1}f}".format(v, digits)]
            values += ["{}".format(self.support[i])]
            thieves = np.argsort(self.ConfMatrix_normalized_row[i, :])[::-1][
                # other class indices "stealing" from class. May still contain self
                :number_of_thieves + 1]
            thieves = thieves[thieves != i]  # exclude self at this point
            steal_ratio = self.ConfMatrix_normalized_row[i, thieves]
            thieves_names = [
                self.existing_class_names[thief][:min(maxcharlength, len(self.existing_class_names[thief]))] for thief
                in thieves]  # a little inefficient but inconsequential
            string_about_stealing = ""
            for j in range(len(thieves)):
                string_about_stealing += "{0}: {1:.3f},\t".format(
                    thieves_names[j], steal_ratio[j])
            values += [string_about_stealing]

            report += fmt % tuple(values)

        report += '\n' + 100 * '-' + '\n'

        # compute averages/sums
        values = [last_line_heading]
        for v in (np.average(self.precision, weights=relative_freq),
                  np.average(self.recall, weights=relative_freq),
                  np.average(self.f1, weights=relative_freq)):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ['{0}'.format(np.sum(relative_freq))]
        values += ['{0}'.format(np.sum(self.support))]
        values += ['']

        # make last ("Total") line for report
        report += fmt % tuple(values)

        return report

    def get_avg_prec_recall(self, ConfMatrix, existing_class_names, excluded_classes=None):
        """
        Calculates average precision and recall across classes, optionally excluding specific ones.

        Args:
            ConfMatrix (np.ndarray): Confusion matrix of shape (C, C).
            existing_class_names (list[str]): List of class names.
            excluded_classes (list[str], optional): Classes to exclude from the averaging.

        Returns:
            Tuple[float, float]: Weighted average precision and recall.
        """

        class2ind = dict(
            zip(existing_class_names, range(len(existing_class_names))))
        included_c = np.full(len(existing_class_names), 1, dtype=bool)

        if not (excluded_classes is None):
            excl_ind = [class2ind[excl_class]
                        for excl_class in excluded_classes]
            included_c[excl_ind] = False

        pred_per_class = np.sum(ConfMatrix, axis=0)
        nonzero_pred = (pred_per_class > 0)

        included = included_c & nonzero_pred
        support = np.sum(ConfMatrix, axis=1)
        weights = support[included] / np.sum(support[included])

        prec = np.diag(ConfMatrix[included, :]
                       [:, included]) / pred_per_class[included]
        prec_avg = np.dot(weights, prec)

        # rec = np.diag(ConfMatrix[included_c,:][:,included_c])/support[included_c]
        rec_avg = np.trace(
            ConfMatrix[included_c, :][:, included_c]) / np.sum(support[included_c])

        return prec_avg, rec_avg

    def prec_rec_histogram(self, precision, recall, binedges=None):
        """
        Displays a histogram of the distribution of classes by their precision and recall scores.

        Args:
            precision (np.ndarray): Precision values per class.
            recall (np.ndarray): Recall values per class.
            binedges (np.ndarray, optional): Array of bin edges to use. If None, a default scheme is used.

        Returns:
            None
        """

        if binedges is None:
            binedges = np.concatenate(
                (np.arange(0, 0.6, 0.2), np.arange(0.6, 1.01, 0.1)), axis=0)
            # add 1 extra bin at the end for >= 1
            binedges = np.append(binedges, binedges[-1] + 0.1)

        hist_precision, binedges = np.histogram(precision, binedges)
        hist_recall, binedges = np.histogram(recall, binedges)

        print("\n\nDistribution of classes with respect to PRECISION: ")
        for b in range(len(binedges) - 1):
            print("[{:.1f}, {:.1f}): {}".format(
                binedges[b], binedges[b + 1], hist_precision[b]))

        print("\n\nDistribution of classes with respect to RECALL: ")
        for b in range(len(binedges) - 1):
            print("[{:.1f}, {:.1f}): {}".format(
                binedges[b], binedges[b + 1], hist_recall[b]))

        if self.plot:
            plt.figure()
            plt.subplot(121)
            widths = np.diff(binedges)
            plt.bar(binedges[:-1], hist_precision, width=widths, align='edge')
            plt.xlim(0, 1)
            ax = plt.gca()
            ax.set_xticks(binedges)
            plt.xlabel('Precision')
            plt.ylabel('Number of classes')
            plt.title("Distribution of classes with respect to precision")

            plt.subplot(122)
            widths = np.diff(binedges)
            plt.bar(binedges[:-1], hist_recall, width=widths, align='edge')
            plt.xlim(0, 1)
            ax = plt.gca()
            ax.set_xticks(binedges)
            plt.xlabel('Recall')
            plt.ylabel('Number of classes')
            plt.title("Distribution of classes with respect to recall")

            plt.show(block=False)

    def analyze_classification(self, y_pred, y_true, class_names, excluded_classes=None):
        """
        Performs a full evaluation of classifier performance, including confusion matrix,
        accuracy, per-class metrics, and optional visualizations.

        Args:
            y_pred (np.ndarray): Predicted class labels.
            y_true (np.ndarray): Ground truth class labels.
            class_names (list[str] or list[int]): List of class names or indices.
            excluded_classes (list[str], optional): Class names to exclude from averaged metrics.

        Returns:
            dict: Dictionary with keys:
                - "total_accuracy"
                - "precision"
                - "recall"
                - "f1"
                - "support"
                - "prec_avg"
                - "rec_avg"
                - "ConfMatrix"
        """

        # Trim class_names to include only classes existing in y_pred OR y_true
        in_pred_labels = set(list(y_pred))
        in_true_labels = set(list(y_true))

        self.existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
        # needed in case `class_names` elements are not strings
        class_strings = [str(name) for name in class_names]
        self.existing_class_names = [class_strings[ind][:min(self.maxcharlength, len(class_strings[ind]))] for ind in
                                     self.existing_class_ind]  # a little inefficient but inconsequential

        # Confusion matrix
        ConfMatrix = metrics.confusion_matrix(y_true, y_pred)

        if self.print_conf_mat:
            print_confusion_matrix(
                ConfMatrix, label_strings=self.existing_class_names, title='Confusion matrix')
            print('\n')
        if self.plot:
            plt.figure()
            plot_confusion_matrix(ConfMatrix, self.existing_class_names)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        self.ConfMatrix_normalized_row = ConfMatrix.astype(
            'float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

        if self.print_conf_mat:
            print_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
                                   title='Confusion matrix normalized by row')
            print('\n')
        if self.plot:
            plt.figure()
            plot_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
                                  title='Confusion matrix normalized by row')

            plt.show(block=False)

        # Analyze results
        self.total_accuracy = np.trace(ConfMatrix) / len(y_true)
        # print('Overall accuracy: {:.3f}\n'.format(self.total_accuracy))

        # returns metrics for each class, in the same order as existing_class_names
        self.precision, self.recall, self.f1, self.support = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                                                     labels=self.existing_class_ind)

        # Print report
        if self.print_conf_mat:
            print(self.generate_classification_report())

        # Calculate average precision and recall
        self.prec_avg, self.rec_avg = self.get_avg_prec_recall(
            ConfMatrix, self.existing_class_names, excluded_classes)
        if excluded_classes:
            print(
                "\nAverage PRECISION: {:.2f}\n(using class frequencies as weights, excluding classes with no predictions and predictions in '{}')".format(
                    self.prec_avg, ', '.join(excluded_classes)))
            print(
                "\nAverage RECALL (= ACCURACY): {:.2f}\n(using class frequencies as weights, excluding classes in '{}')".format(
                    self.rec_avg, ', '.join(excluded_classes)))

        # Make a histogram with the distribution of classes with respect to precision and recall
        # self.prec_rec_histogram(self.precision, self.recall)

        return {"total_accuracy": self.total_accuracy, "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "support": self.support, "prec_avg": self.prec_avg, "rec_avg": self.rec_avg,
                "ConfMatrix": ConfMatrix}
