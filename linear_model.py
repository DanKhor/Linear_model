import numpy as np
from scipy.special import expit
import time
from collections import defaultdict


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

        self.weights = None


    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        history = defaultdict(list)

        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        if self.batch_size is None:
            w_prev = None
            w_cur = w_0
            for i in range(self.max_iter):
                epoch_start = time.time()
                w_prev = w_cur
                w_cur = w_prev - (self.step_alpha/(i+1)**self.step_beta)*self.loss_function.grad(X, y, w_prev)
                epoch_end = time.time()
                if trace:
                    history['dif'].append(np.sum(np.square(w_cur - w_prev)))
                    history['time'].append(round(epoch_end - epoch_start, 4))
                    history['func'].append(self.loss_function.func(X, y, w_cur))
                    if (X_val is not None) and (y_val is not None):
                        history['func_val'].append(self.loss_function.func(X_val, y_val, w_cur))
                # остановка
                if np.sum(np.square(w_cur - w_prev)) < self.tolerance**2:
                    self.weights = w_cur
                    return dict(history)
            print('[GD]Max iterations reached...')
        else:
            w_prev = None
            w_cur = w_0
            for i in range(self.max_iter):
                epoch_start = time.time()
                # Генерируем перестановку объектов обучающей выборки
                index_perm = np.random.permutation(X.shape[0])
                for index_batch in np.split(index_perm,
                                            np.arange(self.batch_size, index_perm.shape[0], self.batch_size)):
                    w_prev = w_cur
                    w_cur = w_prev - (self.step_alpha/(i+1)**self.step_beta)*self.loss_function.grad(X[index_batch],
                                                                                                     y[index_batch],
                                                                                                     w_prev)
                    if np.sum(np.square(w_cur - w_prev)) < self.tolerance ** 2:
                        self.weights = w_cur
                        epoch_end = time.time()
                        if trace:
                            history['time'].append(round(epoch_end - epoch_start, 4))
                            history['func'].append(self.loss_function.func(X, y, w_cur))
                            if (X_val is not None) and (y_val is not None):
                                history['func_val'].append(self.loss_function.func(X_val, y_val, w_cur))
                        return dict(history)
                epoch_end = time.time()
                if trace:
                    history['time'].append(round(epoch_end - epoch_start, 4))
                    history['func'].append(self.loss_function.func(X, y, w_cur))
                    if (X_val is not None) and (y_val is not None):
                        history['func_val'].append(self.loss_function.func(X_val, y_val, w_cur))
            print('[SGD]Max iterations reached...')


    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        weights = self.get_weights()
        test_scores = X.dot(weights)
        return np.where(test_scores < threshold, -1, 1)

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threshold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.weights

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        w = self.get_weights()
        return self.loss_function.func(X, y, w)