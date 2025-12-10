import numpy as np
from numpy import typing as npt
from numpy import random as rnd
from collections.abc import Callable, Sequence
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# ##############################################################################
# Accuracy register code
#
# The below code is used to keep track of accuracy during training, as well as
# for plotting accuracy evolution.
# ##############################################################################


def calc_accuracy(expected: npt.NDArray, actual: npt.NDArray) -> float:
    """Determine accuracy (0-1)."""
    correct = np.sum(np.all(actual == expected, axis=1))
    return correct / expected.shape[0]


class AbstractRegister(ABC):
    @abstractmethod
    def register(
        self,
        nn: Sequence[npt.NDArray],
        linfer: Callable[[npt.NDArray, Sequence[npt.NDArray]], Sequence[npt.NDArray]],
    ):
        pass


class AccuracyRegister(AbstractRegister):
    """Use this for keeping track of accuracy evolution during training."""

    def __init__(
        self,
        test_data: npt.NDArray,
        expect_out: npt.NDArray,
        epochs: int,
        node_off: float,
        node_on: float,
    ):
        self.test_data = test_data
        self.expect_out = expect_out
        self.epoch = 0
        self.node_off = node_off
        self.node_on = node_on
        self.accuracies = np.zeros(epochs)
        self.losses = np.zeros(epochs)

    def register(
        self,
        nn: Sequence[npt.NDArray],
        linfer: Callable[[npt.NDArray, Sequence[npt.NDArray]], Sequence[npt.NDArray]],
    ):
        """Determine accuracy after current epoch."""
        actual_out = self.node_off * np.ones(self.expect_out.shape)
        for j in range(actual_out.shape[0]):
            result = linfer(self.test_data[j, :], nn)[-1]
            actual_out[j, np.argmax(result)] = self.node_on

        # A little hack to also calculate the loss (this is the MSE,
        # for easier comparison with Keras)
        self.losses[self.epoch] = np.sum((self.expect_out - actual_out) ** 2) / len(
            self.expect_out
        )

        self.accuracies[self.epoch] = calc_accuracy(self.expect_out, actual_out)

        self.epoch += 1


def plot_metric(
    epochs: int,
    accuracies: dict[str, npt.NDArray],
    metric: str = "Metric",
    title: str = "",
):
    """Plot accuracy during training."""
    fig, ax = plt.subplots()
    for label, fits in accuracies.items():
        ax.plot(range(0, epochs), fits, label=label)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)
    ax.legend()
    if len(title) > 0:
        fig.suptitle(title)


# ##############################################################################
# Loop-based versions
#
# The functions below are slower loop-based versions, but easier to understand,
# as they follow the algorithms exactly.
# ##############################################################################


def sigm(x: float):
    """Standard logistic function."""
    return 1.0 / (1 + np.exp(np.clip(-x, -88.72, 88.72)))


def infer(input: npt.NDArray, nn: Sequence[npt.NDArray]) -> Sequence[npt.NDArray]:
    """Use a neural network to infer a classification (loop-based version)."""
    output = [input]

    for layer in range(len(nn)):
        # Don't forget about the bias
        input = np.hstack((1, output[layer]))

        # For each node
        nodes_output = np.zeros(nn[layer].shape[1])
        for i in range(nn[layer].shape[1]):
            o = 0
            # For each input to that node
            for j in range(nn[layer].shape[0]):

                # Multiply input by respective weight
                o += input[j] * nn[layer][j, i]

            # Rectify and keep output
            nodes_output[i] = sigm(o)

        output.append(nodes_output)

    return output


def backpropagation(
    train_input: npt.NDArray,
    train_output: npt.NDArray,
    topol: Sequence[int],
    epochs: int,
    eta: float,
    momentum: float = 0,
    hooks: Sequence[AbstractRegister] = [],
    seed: int = 123,
):
    """Backpropagation training algorithm, loop-based version."""
    rng = rnd.default_rng(seed)

    # Create fully connected feed-forward network
    nn = []
    dw = []  # Momentum
    for j in range(len(topol) - 1):
        nn.append(rng.uniform(low=-0.05, high=0.05, size=(topol[j] + 1, topol[j + 1])))
        dw.append(np.zeros((topol[j] + 1, topol[j + 1])))

    # Train!
    for _ in range(epochs):
        # Cycle through training data
        for j in range(train_input.shape[0]):
            err = []

            # Propagate the input forward through the network
            output = infer(train_input[j, :], nn)

            # Propagate the errors back through the network

            # Determine the errors for each output unit
            e = np.zeros(topol[-1])
            for k in range(topol[-1]):
                ok = output[-1][k]
                e[k] = ok * (1 - ok) * (train_output[j, k] - ok)
            err.insert(0, e)

            # For each hidden layer...
            for layer in np.arange(len(nn) - 1, 0, -1, dtype=int):
                # ...determine the error of each of its units
                e = np.zeros(topol[layer])
                for h in range(topol[layer]):
                    oh = output[layer][h]
                    e[h] = oh * (1 - oh) * np.sum(nn[layer][h + 1, :] * err[0])
                err.insert(0, e / topol[layer])

            # Update the network weights
            for layer in range(len(nn)):
                for i_ji in range(nn[layer].shape[0]):
                    for j_ji in range(nn[layer].shape[1]):
                        dw[layer][i_ji, j_ji] = (
                            eta * err[layer][j_ji] * np.hstack((1, output[layer]))[i_ji]
                            + momentum * dw[layer][i_ji, j_ji]
                        )
                        nn[layer][i_ji, j_ji] += dw[layer][i_ji, j_ji]

        # Invoke hooks
        for hook in hooks:
            hook.register(nn, infer)

    return nn


# ##############################################################################
# Vectorized versions
#
# The functions below are faster versions vectorized with NumPy, which might not
# be so easy to follow.
# ##############################################################################


# Vectorized standard logistic function
vsigm = np.vectorize(sigm)


def vinfer(input: npt.NDArray, nn: Sequence[npt.NDArray]) -> Sequence[npt.NDArray]:
    """Use a neural network to infer a classification (vectorized version)."""
    output = [input]

    for layer in range(len(nn)):

        # Don't forget about the bias
        input = np.hstack((1, output[layer]))

        output.append(vsigm(input @ nn[layer]))

    return output


def vbackpropagation(
    train_input: npt.NDArray,
    train_output: npt.NDArray,
    topol: Sequence[int],
    epochs: int,
    eta: float,
    momentum: float = 0,
    hooks: Sequence[AbstractRegister] = [],
    seed: int = 123,
):
    """Backpropagation training algorithm, vectorized version."""
    rng = rnd.default_rng(seed)

    # Create fully connected feed-forward network
    nn = []
    dw = []  # Momentum
    for j in range(len(topol) - 1):
        nn.append(rng.uniform(low=-0.05, high=0.05, size=(topol[j] + 1, topol[j + 1])))
        dw.append(np.zeros((topol[j] + 1, topol[j + 1])))

    # Train!
    for _ in range(epochs):
        # Cycle through training data
        for j in range(train_input.shape[0]):
            err = []

            # Propagate the input forward through the network
            output = vinfer(train_input[j, :], nn)

            # Propagate the errors back through the network

            # Determine the errors for each output unit
            err.insert(
                0, output[-1] * (1 - output[-1]) * (train_output[j, :] - output[-1])
            )

            # For each hidden layer...
            for layer in np.arange(len(nn) - 1, 0, -1, dtype=int):
                # ...determine the error of each of its units
                o = output[layer]
                sumult = nn[layer][1:, :] @ err[0]
                err.insert(0, o * (1 - o) * (sumult / len(o)))

            # Update the network weights
            for layer in range(len(nn)):
                dw[layer] = (
                    eta
                    * (
                        np.hstack((1, output[layer])).reshape((-1, 1))
                        @ err[layer].reshape((1, -1))
                    )
                    + momentum * dw[layer]
                )
                nn[layer] += dw[layer]

        # Invoke hooks
        for hook in hooks:
            hook.register(nn, vinfer)

    return nn


# ##############################################################################
# Encoding conversion functions
# ##############################################################################


def categ2oneofn(
    categ_data: npt.NDArray, node_off: float, node_on: float
) -> tuple[npt.NDArray, dict]:
    """Convert categorical labels to 1-of-n output encoding format."""
    categs = np.unique(categ_data)
    ncategs = len(categs)

    mapping = {}
    for i in range(ncategs):
        mapping[categs[i]] = node_off * np.ones(ncategs)
        mapping[categs[i]][i] = node_on

    nn_output = np.array([mapping[categ] for categ in categ_data])

    return nn_output, mapping


def oneofn2categ(oneofn: npt.NDArray, mapping: dict, node_off: float, node_on: float):
    """Return a category given a 1-of-n output encoding."""
    oneofn_fix = node_off * np.ones(len(oneofn))
    oneofn_fix[np.argmax(oneofn)] = node_on

    for categ, encod in mapping.items():
        if np.all(encod == oneofn_fix):
            return categ

    return None
