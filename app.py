from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Welcome to the Linear Regression API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get the data from frontend

        # Validate and process input features
        required_features = ["sizes", "log_sqft", "bath", "balcony", "area_type_Carpet", "area_type_Plot", "area_type_Super"]
        
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        input_values = np.array([[data["sizes"], data["log_sqft"], data["bath"], data["balcony"],
                                  data["area_type_Carpet"], data["area_type_Plot"], data["area_type_Super"]]])

        # Make prediction
        prediction = model.predict(input_values).tolist()
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
