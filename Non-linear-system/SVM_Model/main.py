from feature_extraction import feature_selection_conv
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf

def load_model_predict(new_model):
    features = feature_selection_conv(new_model)
    with open('svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        # Apply the same scaling as during training
    print(features)
    features_scaled = scaler.transform([features])
    # Make a prediction
    prediction = svm_model.predict(features_scaled)
    print(f"Prediction for new feature x: {prediction[0]}")
    return prediction[0]

def load_tf_model():
    #model_path = input("Please enter the path to the saved TensorFlow model: ")
    model_path = "test_good_model.keras"
    try:
        # Load the model using the path
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'")
        return model
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return None

if __name__ == "__main__":
    model = load_tf_model()
    if model:
        load_model_predict(model)
    


