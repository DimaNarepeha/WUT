import os
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

def plot_cnn_weight_distribution(model, dir_name, model_name):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            plt.title(f"Weight distribution for {model_name} - layer: {layer.name}")
            print(f"Conv Layer: Name = {layer.name}, Filters = {layer.filters}, Kernel Size = {layer.kernel_size}")
            n_array = layer.weights[0].numpy()
            np.save(f'{dir_name}_{model_name}_{layer.name}.npy', n_array)
            plt.figure(figsize=(10, 4))
            plt.hist(n_array.flatten(), bins=30)  # Flatten weights to 1D for histogram
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")
            plt.savefig(dir_name + model_name+".jpg")
            plt.show()


def extract_cnn_weights(model):
    cnn_weights = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Conv Layer: Name = {layer.name}, Filters = {layer.filters}, Kernel Size = {layer.kernel_size}")
            n_array = layer.weights[0].numpy()
            cnn_weights.append(n_array)
    return cnn_weights


def extract_features(cnn_weights_list):
    # Initialize feature vector for a model
    features = []
    
    for weights in cnn_weights_list:
        # Flatten weights for analysis
        flat_weights = np.concatenate([w.flatten() for w in weights if w.size > 1])
        
        # Calculate skewness, kurtosis, and entropy
        weight_skew = skew(flat_weights)
        weight_kurtosis = kurtosis(flat_weights)
        hist, _ = np.histogram(flat_weights, bins=30, density=True)
        weight_entropy = entropy(hist)
        
        # Append calculated metrics to the feature list
        features.extend([weight_skew, weight_kurtosis, weight_entropy])
    return features

def feature_selection_conv(model):
    cnn_weights = extract_cnn_weights(model)
    features = extract_features(cnn_weights)
    return features


def process_models_in_directory(directory, is_good_model):
    X, y = [], []
    # List all files that start with "model_epoch" and have the .h5 or .keras extension
    for filename in os.listdir(directory):
        if filename.startswith("model_epoch") and (filename.endswith(".h5") or filename.endswith(".keras")):
            model_path = os.path.join(directory, filename)
            model = load_model(model_path)
            features = feature_selection_conv(model)
            # Add features and label to dataset
            X.append(features)
            y.append(is_good_model)  # Default label to 0 if filename not found in bad_good
            print(f"Processed model: {filename}, Label: {is_good_model}")
    return X, y


def load_model_predict(new_model):
    features = feature_selection_conv(new_model)
    with open('svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        # Apply the same scaling as during training
    features_scaled = scaler.transform(features)
    # Make a prediction
    prediction = svm_model.predict(x_scaled)
    print(f"Prediction for new feature x: {prediction[0]}")
    return prediction[0]


