from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('stock_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume = float(request.form['volume'])

    # Prepare the input for the model
    input_data = np.array([[open_price, high_price, low_price, volume]])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_price = prediction[0]

    return render_template('result.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
