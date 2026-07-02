import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# FindYourCrop page route
@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        output = prediction[0]

        return render_template('findyourcrop.html',
                               prediction_text='Best crop for given conditions is {}'.format(output))
    except Exception as e:
        return render_template('findyourcrop.html',
                               prediction_text='Error: {}'.format(str(e)))

# Main function to run the application
if __name__ == "__main__":
    app.run(debug=True)