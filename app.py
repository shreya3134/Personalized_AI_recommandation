from flask import Flask, request, jsonify
import pickle

# Load the model and columns
with open('model/disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/columns.pkl', 'rb') as file:
    columns = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Disease Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data sent from frontend
    symptoms = data['symptoms']  # List of selected symptoms
    
    # Make the prediction using the model
    prediction = model.predict([symptoms])
    
    # Return the predicted disease as JSON
    return jsonify({"disease": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
