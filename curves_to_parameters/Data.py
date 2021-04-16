import numpy as np
import math
from typing import List
import random

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)





# generate sinusoids and process them into sequences/labels instances for training


def sinus(A: float, omega: float, t: float) -> float:
    return A * math.sin(omega * t - math.pi / 2) + A


def sequence_sin(A: float, omega: float, length: int, step_duration: float) -> List[float]:
    return [sinus(A, omega, t * step_duration) for t in range(length)]


def add_noise(sequence, noise, shape):
    noise = np.random.normal(-noise, +noise, shape)
    return sequence + noise



class Sinusoids_Generator:
    def __init__(self,
                 gen_parameters=[200000, 0.5, 1, 0.5, 0.8, 100, 0.025],
                 on_noise=False,
                 noise=None
                 ):
        super().__init__()
        self.gen_parameters = gen_parameters
        self.on_noise = on_noise
        self.noise = noise

    def get_batch(self, gen_size) -> List[np.ndarray]:
        X = []
        Y = []
        _, minA, maxA, minOmega, maxOmega, sequence_length, step_duration = self.gen_parameters
        for _ in range(gen_size):
            omega = random.uniform(minOmega, maxOmega)
            A = random.uniform(minA, maxA)
            sequence = sequence_sin(A, omega, sequence_length, step_duration)
            if self.on_noise:
                sequence = add_noise(sequence, self.noise, [sequence_length, ])
            X.append(sequence)
            Y.append([omega, A])
        return [np.array(X), np.array(Y)]

    def generate_training_dataset(self):
        X, Y = self.get_batch(self.gen_parameters[0])
        return X.reshape([self.gen_parameters[0], self.gen_parameters[5], 1]), Y.reshape([self.gen_parameters[0], 2])

    def generate_testing_batch(self, test_size: int):
        X_test, Y_test = self.get_batch(test_size)
        return X_test.reshape([test_size, self.gen_parameters[5], 1]), Y_test.reshape([test_size, 2])

    def print_sin(self, nb_sub: int, nb_per_sub: int, dataset):
        fig, ax = plt.subplots(nb_sub, sharex=True)
        fig.set_figheight(20)
        fig.set_figwidth(10)

        for i in range(nb_sub):
            for j in range(nb_per_sub):
                ax[i].plot(np.linspace(0, 100 * 0.025, 100), dataset[0][j * (i + 1)])

        plt.show()
