import numpy as np
import alive_progress as ap
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tabulate import tabulate

from perceptron.layer import Layer
from perceptron.optimizer import create_optimizer
from terminal.arg_parsing import arg_parsing


def one_hot(a: np.ndarray, num_classes: int):
    """Convert an array of integers into a one-hot encoded matrix."""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).astype(int)


class Network:

    def __init__(
        self,
        epoch: int = 200,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        patience: int = 10,
    ) -> None:
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers: list[Layer] = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.f1_scores = []
        self.recalls = []
        self.precisions = []
        self.patience = patience  # early stopping
        self.best_network = None

    def add_layer(
        self,
        Next: int = 0,
        f_activation: str = "sigmoid",
        weights_init: str = "heUniform",
    ) -> None:
        if len(self.layers) == 0:
            layer = Layer(Next, 0, f_activation, weights_init)
        else:
            layer = Layer(Next, self.layers[-1].Next, f_activation, weights_init)
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the network network.

        Args:
            X (np.ndarray): The Z data.

        Returns:
            np.ndarray: The output of the network network."""

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, P: np.ndarray, Y: np.ndarray) -> float:
        """Computes the loss function for the network network.
            Binary cross-entropy loss.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels.

        Returns:
            float: The loss of the network network."""

        epsilon = 1e-16  # to avoid log(0)

        loss = -np.mean(
            Y * np.log(P + epsilon) + (1 - Y) * np.log(1 - P + epsilon),
        )
        return loss

    def backward(self, P: np.ndarray, Y: np.ndarray, opt: str) -> None:
        """Backward propagation through the network network.

        Args:
            P (np.ndarray): The predicted labels.
            Y (np.ndarray): The true labels."""

        Yreshape = one_hot(Y, 2)
        gradient = P - Yreshape
        optimizer = create_optimizer(opt, self.learning_rate)

        for layer in reversed(self.layers):
            gradient = layer.backward(optimizer, gradient)

    def visualize(
        self, X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray
    ) -> None:
        """Visualize the network network."""

        bench = arg_parsing().benchmark

        epoch = len(self.losses)
        if epoch == 0 and not bench:
            print("x_train shape : ", X.shape)
            print("x_test shape : ", X_test.shape)

        # loss train
        P = self.forward(X)
        Y_one = one_hot(Y, 2)

        # loss test
        Y_test = Y_test.astype(int)
        Y_test_one = one_hot(Y_test, 2)

        if not bench:
            print(
                "epoch ",
                epoch + 1,
                "/",
                self.epoch,
                "- loss",
                "{:.4f}".format(self.loss(P, Y_one)),
                "- val_loss",
                "{:.4f}".format(self.loss(self.forward(X_test), Y_test_one)),
            )

        self.losses.append(self.loss(P, Y_one))
        self.val_losses.append(self.loss(self.forward(X_test), Y_test_one))

        self.f1_scores.append(f1_score(Y, np.argmax(P, axis=1)))
        self.accuracies.append(accuracy_score(Y, np.argmax(P, axis=1)))
        self.val_accuracies.append(
            accuracy_score(Y_test, np.argmax(self.forward(X_test), axis=1))
        )
        self.recalls.append(recall_score(Y, np.argmax(P, axis=1)))
        self.precisions.append(precision_score(Y, np.argmax(P, axis=1)))

    def get_batches(self, X: np.ndarray, Y: np.ndarray):
        """Get the batches of data.

        Args:
            X (np.ndarray): The X data.
            Y (np.ndarray): The true labels.
            batch_size (int): The size of the batch.

        Returns:
            tuple[np.ndarray, np.ndarray]: The batch of data and labels."""
        X, Y = shuffle(X, Y, random_state=41)
        for i in range(0, len(X), self.batch_size):
            yield X[i : i + self.batch_size], Y[i : i + self.batch_size]

    def keep_best_network(self, eS) -> bool:
        """Keep the best network network.

        Args:
            network (Network): The network network to keep.

        Returns:
            bool: If the network network should stop early.
        """
        if arg_parsing().early_stop:
            if eS.early_stop(self, self.val_losses):
                self.best_network = eS.best_network
                print("Early stopping")
                return True
        return False

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        opt: str = "SGD",
    ) -> None:
        """Train the network network.

        Args:
            X (np.ndarray): The Z data.
            Y (np.ndarray): The true labels.
            X_test (np.ndarray): The Z test data.
            Y_test (np.ndarray): The true test labels.

        Returns:
            None: None.
        """

        from perceptron.optimizer import EarlyStop

        # Convert Y["0", "1", ...] into float
        Y = Y.astype(int)
        eS = EarlyStop(self.patience)

        with ap.alive_bar(self.epoch, title="Training", enrich_print=False) as bar:
            for _ in range(self.epoch):
                for X_batch, Y_batch in self.get_batches(X, Y):
                    P = self.forward(X_batch)
                    self.backward(P, Y_batch, opt)

                self.visualize(X, Y, X_test, Y_test)
                if self.keep_best_network(eS):
                    break
                bar()

        if self.best_network:
            self.layers = self.best_network

    def predict(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Predict the labels of the data.

        Args:
            X (np.ndarray): The Z data.
            Y (np.ndarray): The true labels.
        """

        Y = Y.astype(int)

        Y_one = one_hot(Y, 2)

        table = [
            ["Loss", self.loss(self.forward(X), Y_one)],
            ["Accuracy", accuracy_score(Y, np.argmax(self.forward(X), axis=1))],
            ["F1 score", f1_score(Y, np.argmax(self.forward(X), axis=1))],
            ["Recall", recall_score(Y, np.argmax(self.forward(X), axis=1))],
            ["Precision", precision_score(Y, np.argmax(self.forward(X), axis=1))],
        ]
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
