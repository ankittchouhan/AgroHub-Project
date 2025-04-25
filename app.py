from flask import Flask, request, render_template, redirect, url_for
import pickle
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# ========================
# Load Models
# ========================

# Load Crop Recommendation Model
with open('models/crop_recommendation_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

# Load Crop Yield Prediction Model
yield_model = joblib.load('models/random_forest_model.pkl')
encoded_columns = yield_model.feature_names_in_

# Load Fertilizer Recommendation Model
with open('models/fertilizer_model.pkl', 'rb') as f:
    fertilizer_model = pickle.load(f)

# Load Label Encoders
with open('models/fertilizer_label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
soil_encoder = encoders['soil']
crop_encoder = encoders['crop']
fertilizer_encoder = encoders['fertilizer']


# ========================
# Routes
# ========================

# Home route - landing page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

@app.route('/choice')
def choice():
    return render_template('choice_page.html')


# --------------------------
# CROP RECOMMENDATION ROUTES
# --------------------------

@app.route('/crop_recommendation')
def crop_form():
    return render_template('crop_input.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Get form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = crop_model.predict(input_features)

        return render_template('crop_result.html', crop=prediction[0])

    except Exception as e:
        return f"Crop Recommendation Error: {str(e)}"


# ----------------------------
# CROP YIELD PREDICTION ROUTES
# ----------------------------

@app.route('/yield_prediction')
def yield_form():
    return render_template('yield_input.html')

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        # Get the form data
        crop = request.form['crop']
        season = request.form['season']
        state = request.form['state']
        area = float(request.form['area'])
        production = float(request.form['production'])
        annual_rainfall = float(request.form['annual_rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])

        # Preprocess input
        input_data = {
            'Crop': [crop],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        }

        input_df = pd.DataFrame(input_data)
        input_df_encoded = pd.get_dummies(input_df, columns=['Crop', 'Season', 'State'])

        for col in encoded_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0

        input_df_encoded = input_df_encoded[encoded_columns]

        # Predict yield
        predicted_yield = yield_model.predict(input_df_encoded)

        return render_template('yield_result.html', prediction=predicted_yield[0])

    except Exception as e:
        return f"Yield Prediction Error: {str(e)}"


# --------------------------------
# FERTILIZER RECOMMENDATION ROUTES
# --------------------------------

@app.route('/fertilizer_recommendation')
def fertilizer_form():
    return render_template('fertilizer_input.html')

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # Get form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])

        # Encode soil and crop type
        encoded_soil = soil_encoder.transform([soil_type])[0]
        encoded_crop = crop_encoder.transform([crop_type])[0]

        # Form input features
        input_features = np.array([[temperature, humidity, moisture, encoded_soil, encoded_crop, nitrogen, phosphorous, potassium]])

        prediction_encoded = fertilizer_model.predict(input_features)[0]
        prediction = fertilizer_encoder.inverse_transform([prediction_encoded])[0]

        return render_template('fertilizer_result.html', fertilizer=prediction)

    except Exception as e:
        return f"Fertilizer Recommendation Error: {str(e)}"


# ========================
# Run the app
# ========================
if __name__ == '__main__':
    app.run(debug=True)