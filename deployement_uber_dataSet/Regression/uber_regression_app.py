from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load your models
model = joblib.load('model_joblib.pkl')
model_classification = joblib.load('model_classification.pkl')

@app.route('/')
def home():
    return render_template('dashboard1.html')

@app.route('/calculate_fare', methods=['POST'])
def calculate_fare():
    try:
        # Log the incoming request data
        logging.debug("Request data: %s", request.form)

        # Get data from the form
        passenger_count = int(request.form['passenger_count'])
        distance_km = float(request.form['distance_km'])
        pickup_day = int(request.form['pickup_day'])
        pickup_hour = int(request.form['pickup_hour'])
        is_weekend = int(request.form['is_weekend'])
        season_winter = int(request.form['season_winter'])
        season_spring = int(request.form['season_spring'])
        season_summer = int(request.form['season_summer'])
        season_autumn = int(request.form['season_autumn'])

        # Extract and convert input data
        pickup_longitude = float(request.form['pickup_longitude'])
        pickup_latitude = float(request.form['pickup_latitude'])
        dropoff_longitude = float(request.form['dropoff_longitude'])
        dropoff_latitude = float(request.form['dropoff_latitude'])
        day_of_week = int(request.form['day_of_week'])
        month = int(request.form['month'])
        pickup_minute = int(request.form['pickup_minute'])
        # Prepare data for fare prediction model
        input_data_regression = np.array([[pickup_day, pickup_hour, passenger_count, is_weekend,
                                distance_km, season_winter, season_spring,
                                season_summer, season_autumn]])

        # Prepare input data for classification model
        input_data_classification = np.array([[distance_km,pickup_latitude,pickup_longitude, dropoff_latitude,  dropoff_longitude,
                                                pickup_hour,pickup_day,month,pickup_minute,  day_of_week,is_weekend,
                                                 season_winter, season_spring, season_summer, season_autumn,passenger_count ]])

        # Get the fare amount prediction
        fare_amount = model.predict(input_data_regression)[0]  # This could be a NumPy type
        # Get classification label
        classification_label = model_classification.predict(input_data_classification)[0]  # Assuming this is a string

        # Convert fare_amount to a standard Python type
        fare_amount = float(fare_amount)  # Ensure it's a float
        # Convert classification label to a standard Python type if necessary
        classification_label = str(classification_label)  # Ensure it's a string

        # Return result to the frontend as a single JSON object
        return jsonify({
            'fare_amount': fare_amount,
            'fare_amount_classification': classification_label  # No conversion needed
        })

    except Exception as e:
        logging.error("Error processing request: %s", str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(port=3000, debug=True)