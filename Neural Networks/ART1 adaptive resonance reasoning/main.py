import numpy as np
import logging
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ART1:
    def __init__(self, input_size, rho):
        self.input_size = input_size  # Number of input neurons
        self.rho = rho  # Vigilance parameter
        self.weights = []  # List to store weights for each cluster
        logging.info(f'ART1 initialized with input size {input_size} and vigilance parameter {rho}')

    def train(self, inputs):
        logging.info(f'Starting training with {len(inputs)} input vectors')
        cluster_assignments = []

        # Initialize weights
        n = self.input_size
        bottom_up_weights = np.ones((n, 0)) * (1 / (1 + n))  # Initially no clusters
        top_down_weights = np.ones((0, n))  # Initially no clusters

        for i, input_vector in enumerate(inputs):
            logging.debug(f'Training on input vector {i}')
            input_vector = self._binarize(input_vector)

            # Present input pattern and calculate activations
            activations = np.dot(bottom_up_weights.T, input_vector)

            # Active set A contains all nodes
            A = list(range(len(activations)))

            while A:
                # Select node with largest activation
                j = A[np.argmax(activations[A])]

                # Compute s*
                s_star = np.minimum(top_down_weights[j], input_vector)

                # Calculate similarity
                similarity = np.sum(s_star) / np.sum(input_vector)

                if similarity < self.rho:
                    # Remove j from A
                    A.remove(j)
                else:
                    # Update weights
                    bottom_up_weights[:, j] = (top_down_weights[j] * input_vector) / (
                            0.5 + np.sum(top_down_weights[j] * input_vector))
                    top_down_weights[j] = np.minimum(top_down_weights[j], input_vector)

                    # Assign pattern to cluster
                    cluster_assignments.append(j)
                    break

            if not A:
                # Create new node
                new_top_down_weight = input_vector
                new_bottom_up_weight = input_vector / (0.5 + np.sum(input_vector))

                top_down_weights = np.vstack([top_down_weights, new_top_down_weight])
                bottom_up_weights = np.column_stack([bottom_up_weights, new_bottom_up_weight])

                # Assign pattern to new cluster
                cluster_assignments.append(len(top_down_weights) - 1)

        logging.info('Training completed')
        self.bottom_up_weights = bottom_up_weights
        self.top_down_weights = top_down_weights
        return cluster_assignments

    def _binarize(self, vector):
        logging.debug('Binarizing input vector')
        return np.where(vector > 0.5, 1, 0)


def analyze_clusters(clusters, labels, rho):
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'Cluster': clusters, 'Label': labels})
    cluster_distribution = df.groupby(['Cluster', 'Label']).size().unstack(fill_value=0)

    # Print the number of clusters
    num_clusters = cluster_distribution.shape[0]
    print(f'Number of clusters for vigilance {rho}: {num_clusters}')

    # Plot the cluster distribution heatmap
    plot_cluster_heatmap(cluster_distribution, rho)

    return num_clusters  # Return the number of clusters


def plot_cluster_heatmap(cluster_distribution, rho):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cluster_distribution, annot=True, cmap='viridis', cbar=True)
    plt.title(f'Cluster Distribution Heatmap for vigilance {rho}', fontsize=20)
    plt.xlabel('Digit Class', fontsize=16)
    plt.ylabel('Cluster', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


# Function to train ART1 and analyze clusters for different vigilance values
def experiment_vigilance_values(vigilance_values, inputs, labels):
    results = []
    for rho in vigilance_values:
        logging.info(f'Experimenting with vigilance parameter: {rho}')
        art1 = ART1(input_size=784, rho=rho)
        cluster_assignments = art1.train(inputs)
        print(f'Results for vigilance parameter: {rho}')
        num_clusters = analyze_clusters(cluster_assignments, labels, rho)
        results.append((rho, num_clusters))
        print('-----------------------------------------')
    return results


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
images = mnist.data.values
labels = mnist.target.astype(int)

# Normalize images
images = images / 255.0

# Binarize images
threshold = 0.5
binarized_images = (images > threshold).astype(int)

# Train ART1 network on a subset of the dataset
train_subset = binarized_images[:1000]
true_labels_subset = labels[:1000]

# List of vigilance values to experiment with
vigilance_values = [0.02, 0.05, 0.1, 0.5, 0.8, 0.9]

# Train and analyze for different vigilance values
results = experiment_vigilance_values(vigilance_values, train_subset, true_labels_subset)

# Plot vigilance vs. number of clusters
vigilance, num_clusters = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(vigilance, num_clusters, marker='o')
plt.title('Vigilance vs. Number of Clusters', fontsize=20)
plt.xlabel('Vigilance', fontsize=16)
plt.ylabel('Number of Clusters', fontsize=16)
plt.grid(True)
plt.show()
