from feature_extraction import process_models_in_directory
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
#
# Directories containing the models
GOOD_WEIGHT_DIRECTORY = "../Cats_dogs_good_weight_SGD"
BAD_WEIGHT_DIRECTORY = "../Cats_dogs_bad_weight_ADAM"
# Process models in "good" and "bad" directories
X_good, y_good = process_models_in_directory(GOOD_WEIGHT_DIRECTORY, True)
X_bad, y_bad = process_models_in_directory(BAD_WEIGHT_DIRECTORY, False)

# Combine "good" and "bad" model features and labels
X = X_good + X_bad
y = y_good + y_bad

# Convert lists to numpy arrays for processing
X = np.array(X)
y = np.array(y)

print(f"X: {X}")
print(f"y: {y}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the SVM model with a linear kernel (since the dataset is small)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_scaled, y)

# Apply Leave-One-Out Cross-Validation (LOOCV) for a small dataset
loo = LeaveOneOut()
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=loo)

# Output cross-validation results
print(f"LOOCV Accuracy: {np.mean(cv_scores):.2f}")
print(f"Individual CV scores: {cv_scores}")

with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
print("Model saved as 'svm_model.pkl'")
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler saved as 'scaler.pkl'")
