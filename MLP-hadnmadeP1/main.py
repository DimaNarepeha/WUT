import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))



def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class MLP:
    def __init__(self, layers, activation='sigmoid'):
        self.layers = layers
        self.activation = sigmoid if activation == 'sigmoid' else tanh
        self.activation_derivative = sigmoid_derivative if activation == 'sigmoid' else tanh_derivative
        self.weights = []
        self.biases = []
        self.velocity_w = []  # for momentum
        self.velocity_b = []  # for momentum

        for i in range(len(layers) - 1):
            stddev = np.sqrt(2 / (layers[i] + layers[i + 1])) if activation == 'tanh' else np.sqrt(1 / layers[i])
            self.weights.append(np.random.normal(0, stddev, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))
            self.velocity_w.append(np.zeros((layers[i], layers[i + 1])))
            self.velocity_b.append(np.zeros((1, layers[i + 1])))

    def backpropagation(self, X, y, learning_rate, momentum):
        output_error = y - self.activations[-1]
        deltas = [output_error * self.activation_derivative(self.activations[-1])]

        for i in reversed(range(len(self.activations) - 2)):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * self.activation_derivative(self.activations[i + 1])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            weight_gradient = self.activations[i].T.dot(deltas[i])
            bias_gradient = np.sum(deltas[i], axis=0, keepdims=True)

            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * weight_gradient
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * bias_gradient

            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def forward_pass(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.activation(net))
        return self.activations[-1]

    def train(self, X, y, epochs, batch_size, learning_rate, momentum):
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.forward_pass(X_batch)
                self.backpropagation(X_batch, y_batch, learning_rate, momentum)
                total_loss += mse_loss(y_batch, self.activations[-1])

            average_loss = total_loss / (X.shape[0] / batch_size)
            loss_history.append(average_loss)
            writer.add_scalar('loss', average_loss, epoch)
            print(f"Epoch {epoch}, Loss: {average_loss}")
        return loss_history

    def evaluate(self, X, y):
        output = self.forward_pass(X)
        predictions = np.argmax(output, axis=1)
        actual_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == actual_classes)
        return accuracy


def get_positive_integer(prompt):
    """ Utility function to get a positive integer from the user. """
    while True:
        try:
            value = int(input(prompt))
            if value < 1:
                print("Please enter a positive integer.")
            else:
                return value
        except ValueError:
            print("Invalid input; please enter an integer.")

# User inputs
hidden_layers = get_positive_integer("Enter number of hidden layers: ")
neurons_per_layer = []
# Ask the user for the number of neurons in each layer
for i in range(1, hidden_layers + 1):
    prompt = f"Enter the number of neurons for hidden layer {i}: "
    neurons = get_positive_integer(prompt)
    neurons_per_layer.append(neurons)

activation = input("Enter activation function (sigmoid/tanh): ")
epochs = int(input("Enter number of epochs: "))
batch_size = int(input("Enter batch size: "))
learning_rate = float(input("Enter learning rate: "))
momentum = float(input("Enter momentum: "))

#activation = 'sigmoid'
#epochs = 200
#batch_size = 32
#learning_rate = 0.005
#momentum = 0.2


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


def load_wine_dataset(file_path):
    features = []
    labels = []

    # Open and read the dataset file
    with open(file_path, 'r') as file:
        for line in file:
            if line == '\n':
                continue
            # Wine dataset usually starts with the label followed by features
            parts = line.strip().split(',')
            labels.append(int(parts[0]))  # Assuming the first column is the label
            features.append([float(part) for part in parts[1:]])

    features_np = np.array(features)
    labels_np = np.array(labels).reshape(-1, 1)

    # Calculate mean and std dev for each feature
    means = np.mean(features_np, axis=0)
    std_devs = np.std(features_np, axis=0)

    # Normalize features
    normalized_features = (features - means) / std_devs

    return normalized_features, labels_np


features, labels = None, None
file_path = 'data/iris.data'
# Ask the user which dataset to load
dataset_choice = input("Type 'iris' to load the Iris dataset or 'wine' to load the Wine dataset: ").lower().strip()

if dataset_choice == "iris":
    file_path = 'data/iris.data'  # Update this path to your Iris dataset location
    features, labels = load_iris_dataset(file_path)
    print("Loaded Iris dataset.")
elif dataset_choice == "wine":
    file_path = 'data/wine.csv'  # Update this path to your Wine dataset location
    features, labels = load_wine_dataset(file_path)
    print("Loaded Wine dataset.")
else:
    raise ValueError("Invalid dataset choice. Please type 'iris' or 'wine'.")

labels_onehot = one_hot_encode(labels)
input_size = 0
output_size = 0

if dataset_choice == "iris":
    input_size = 4
    output_size = 3
elif dataset_choice == "wine":
    input_size = 13
    output_size = 4 #even though we have 3 classes our one hot is output 4 shape
else:
    raise ValueError("Invalid dataset choice. Please type 'iris' or 'wine'.")

hidden_layers = 2
layers = [input_size]  # Input size for Iris/wine dataset
for n_n in neurons_per_layer:
    layers.append(n_n)
layers.append(output_size)  # Output size for Iris/wine dataset
#


#
# Initialize the MLP with user-defined settings
mlp = MLP(layers=layers, activation=activation)
# Training the model
print("Starting training...")
training_loss = mlp.train(features, labels_onehot, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                          momentum=momentum)

# Plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the trained model
accuracy = mlp.evaluate(features, labels_onehot)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Closing the SummaryWriter for TensorBoard
writer.close()
