import numpy as np
import scipy
from scipy import stats
from curves_to_parameters import Data
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)




def compare_range_of_Values(lowerbound : int, upperbound : int, model, X, Y):
    Y_pred = model.predict(X[lowerbound:upperbound,:,:])
    return Y[lowerbound:upperbound,:], Y_pred


def compare_range_of_new_values(range_size : int, model, sin_gen):
    X_test_range, Y_test_range = sin_gen.generate_testing_batch(range_size)
    Y_pred = model.predict(X_test_range)
    return Y_test_range, Y_pred



class Vizualisation:

    def __init__(self,
                 name,
                 model,
                 sin_gen
                 ):
        super().__init__()
        self.name = name
        self.model = model
        self.sin_gen = sin_gen

    def calculate_mean_std_error(self, testing_size, on_train_dataset=False, no_abs=False, X=None, Y=None):

        if on_train_dataset == False:
            Y_sample, Y_pred = compare_range_of_new_values(testing_size, self.model, self.sin_gen)
        else:
            Y_sample, Y_pred = compare_range_of_Values(0, testing_size, self.model, X, Y)

        Y_sample_omega = Y_sample[:, 0]
        Y_pred_sample_omega = Y_pred[:, 0]

        Y_sample_A = Y_sample[:, 1]
        Y_pred_sample_A = Y_pred[:, 1]

        if no_abs == True:

            omega_mean_error = np.mean(Y_sample_omega - Y_pred_sample_omega)
            omega_std_error = np.std(Y_sample_omega - Y_pred_sample_omega)
            A_mean_error = np.mean(Y_sample_A - Y_pred_sample_A)
            A_std_error = np.std(Y_sample_A - Y_pred_sample_A)

        else:
            omega_mean_error = np.mean(np.abs(Y_sample_omega - Y_pred_sample_omega))
            omega_std_error = np.std(np.abs(Y_sample_omega - Y_pred_sample_omega))
            A_mean_error = np.mean(np.abs(Y_sample_A - Y_pred_sample_A))
            A_std_error = np.std(np.abs(Y_sample_A - Y_pred_sample_A))

        return omega_mean_error, omega_std_error, A_mean_error, A_std_error

    def plot_regression_type_viz(self, testing_size, on_train_dataset=False, X=None, Y=None):
        fig, ax = plt.subplots(2)
        fig.set_figheight(20)
        fig.set_figwidth(10)

        if on_train_dataset == False:
            Y_sample, Y_pred = compare_range_of_new_values(testing_size, self.model, self.sin_gen)
        else:
            Y_sample, Y_pred = compare_range_of_Values(0, testing_size, self.model, X, Y)

        Y_sample_omega = Y_sample[:, 0]
        Y_pred_sample_omega = Y_pred[:, 0]

        Y_sample_A = Y_sample[:, 1]
        Y_pred_sample_A = Y_pred[:, 1]

        ax[0].scatter(Y_sample_omega, Y_pred_sample_omega)
        ax[0].plot([0.5, 0.8], [0.5, 0.8], 'r')
        ax[1].scatter(Y_sample_A, Y_pred_sample_A)
        ax[1].plot([0.5, 1], [0.5, 1], 'r')

    def plot_error_distributions(self, testing_size, on_train_dataset=False, X=None, Y=None):

        fig, ax = plt.subplots(2)
        fig.set_figheight(20)
        fig.set_figwidth(10)

        if on_train_dataset == False:
            Y_sample, Y_pred = compare_range_of_new_values(testing_size, self.model, self.sin_gen)
        else:
            Y_sample, Y_pred = compare_range_of_Values(0, testing_size, self.model, X, Y)

        Y_sample_omega = Y_sample[:, 0]
        Y_pred_sample_omega = Y_pred[:, 0]

        Y_sample_A = Y_sample[:, 1]
        Y_pred_sample_A = Y_pred[:, 1]

        error_omega = Y_sample_omega - Y_pred_sample_omega
        error_A = Y_sample_A - Y_pred_sample_A

        center_omega, std_omega, center_A, std_A = self.calculate_mean_std_error(testing_size, on_train_dataset=False,
                                                                                 no_abs=True)

        _, bins_omega, _ = ax[0].hist(error_omega, bins=100)
        best_fit_omega = scipy.stats.norm.pdf(bins_omega, center_omega, std_omega)

        ax[0].plot([center_omega, center_omega], [0, 30], label="Omega error distribution center")
        ax[0].plot(bins_omega, best_fit_omega, label="Normal distribution")

        _, bins_A, _ = ax[1].hist(error_A, bins=100)
        best_fit_A = scipy.stats.norm.pdf(bins_A, center_A, std_A)

        ax[1].plot([center_A, center_A], [0, 30], 'r', label="Mean")
        ax[1].plot(bins_A, best_fit_A, label="Normal distribution")

        plt.legend()

    def plot_sinusoids_comparison(self, nb_subs, testing_size, on_train_dataset=False, X=None, Y=None):
        fig, ax = plt.subplots(nb_subs, sharex=True, sharey=True)
        fig.set_figheight(20)
        fig.set_figwidth(10)

        if on_train_dataset == False:
            Y_sample, Y_pred = compare_range_of_new_values(testing_size, self.model, self.sin_gen)
        else:
            Y_sample, Y_pred = compare_range_of_Values(0, testing_size, self.model, X, Y)

        sequenceLength = 100
        step_duration = 0.025
        for i in range(nb_subs):
            ax[i].plot(np.linspace(0, sequenceLength * step_duration, sequenceLength),
                       Data.sequence_sin(Y_sample[i, 1], Y_sample[i, 0], sequenceLength, step_duration),
                       label="real sinusoid")
            ax[i].plot(np.linspace(0, sequenceLength * step_duration, sequenceLength),
                       Data.sequence_sin(Y_sample[i, 1], Y_sample[i, 0], sequenceLength, step_duration),
                       label="predicted sinusoid")

        plt.legend()
        plt.show()
