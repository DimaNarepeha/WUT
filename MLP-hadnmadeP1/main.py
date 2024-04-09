import numpy as np
import tensorboard as t
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
BATCH_SIZE = 32


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward_pass(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backpropagation(self, X, y):
        learning_rate = 0.01

        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update the weights and biases
        self.weights2 += self.hidden.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            # Mini-batch training
            for i in range(0, X.shape[0], BATCH_SIZE):
                X_batch = X[i:i + BATCH_SIZE]
                y_batch = y[i:i + BATCH_SIZE]

                self.forward_pass(X_batch)
                self.backpropagation(X_batch, y_batch)
                total_loss += mse_loss(y_batch, self.output)

            average_loss = total_loss / (X.shape[0] / BATCH_SIZE)
            if epoch % 100 == 0:
                writer.add_scalar('loss', average_loss, epoch)
                print(f"Epoch {epoch}, Loss: {average_loss}")

    def predict(self, X):
        output = self.forward_pass(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        actual_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == actual_classes)
        return accuracy


def load_iris_dataset(file_path):
    # Initialize lists to store features and labels
    features = []
    labels = []

    # Dictionary to encode species to integers
    species_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Open and read the dataset file
    with open(file_path, 'r') as file:
        for line in file:
            if line == '\n':
                break
            # Split each line into components: 4 features and 1 label
            parts = line.strip().split(',')
            # Convert features to float and add to the list
            features.append([float(part) for part in parts[:-1]])
            # Encode the label and add to the list
            labels.append(species_to_int[parts[-1]])

    # Convert lists to NumPy arrays
    features_np = np.array(features)
    labels_np = np.array(labels).reshape(-1, 1)  # Reshape for consistency

    return features_np, labels_np


def one_hot_encode(labels):
    n_classes = np.max(labels) + 1
    one_hot = np.zeros((labels.size, n_classes))
    one_hot[np.arange(labels.size), labels[:, 0]] = 1
    return one_hot


file_path = 'data/iris.data'  # Update with the actual path to your data file
features, labels = load_iris_dataset(file_path)

labels_onehot = one_hot_encode(labels)

# TRAINING
mlp = MLP(input_size=4, hidden_size=4, output_size=3)
mlp.train(features, labels_onehot, epochs=100000)

# TESTING
accuracy = mlp.evaluate(features, labels_onehot)  # Make sure you've defined X_test and y_test_onehot
print(f"Test Accuracy: {accuracy * 100:.2f}%")
