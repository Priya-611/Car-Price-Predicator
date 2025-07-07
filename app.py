from flask import Flask, render_template, request, jsonify
import pandas as pd
import os     #for file checking
import pickle   #o load/save model & encoders
from task1 import load_data, prepare_features, create_encoders, encode_features, train_model, load_model_and_encoders, predict_price    #all ML-related functions (training, loading, prediction)
import json

app = Flask(__name__)    #Initializes the Flask app.


def load_or_train_model():  #Load the saved model & encoders or train a new one if not found.
    
    model, encoders = load_model_and_encoders()    #load from model.pkl and encoders.pkl
    
    if model is None or encoders is None:
        print("Training new model...")
        try:      #Console log for debugging.
            # Load and prepare data
            data = load_data()
            features, target = prepare_features(data)   #Splits into input (X) and output (y).
            
            #Creates label encoders for categorical features.
            encoders = create_encoders(features)
            print("Encoder keys:", list(encoders.keys()))
            
            # Encode features and train model
            features_encoded = encode_features(features, encoders)
            model = train_model(features_encoded, target)
            
            #Saves model and encoders for reuse 
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('encoders.pkl', 'wb') as f:
                pickle.dump(encoders, f)
                
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    
    return model, encoders

def load_suggestions():
    #Loads unique dropdown values for car name, fuel type, body.
    try:
        data = load_data()
        suggestions = {     #Creates dictionaries of dropdown options.
            'car_names': sorted(data['CarName'].unique().tolist()),
            'fuel_types': sorted(data['fueltype'].unique().tolist()),
            'car_bodies': sorted(data['carbody'].unique().tolist())
        }
        print(f"Dropdown options loaded: {len(suggestions['car_names'])} car names, {len(suggestions['fuel_types'])} fuel types, {len(suggestions['car_bodies'])} body types")
        return suggestions
    except Exception as e:
        print(f"Error loading suggestion data: {str(e)}")
        return {
            'car_names': [],
            'fuel_types': [],
            'car_bodies': []
        }

#initialize the model and dropdown options when the app starts.
try:
    model, encoders = load_or_train_model()
    suggestions = load_suggestions()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    model = None
    encoders = None
    suggestions = {'car_names': [], 'fuel_types': [], 'car_bodies': []}


#'/' — Render HTML page, Loads the homepage and passes suggestion data to index.html
@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        car_names=suggestions['car_names'],
        fuel_types=suggestions['fuel_types'],
        car_bodies=suggestions['car_bodies']
    )

#'/predict' — Handle Prediction, Accepts user input when the form is submitted.
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None:
        return jsonify({'error': 'Prediction model is not available. Please try again later.'}), 500
    
    try: #retrieves JSON input sent by JavaScript.
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        #Extracts and cleans form data.
        car_name = data.get('car_name', '').strip()
        fuel_type = data.get('fuel_type', '').strip()
        car_body = data.get('car_body', '').strip()
        engine_size = data.get('engine_size', '')
        stroke = data.get('stroke', '')
        horsepower = data.get('horsepower', '')
        
        # Validate inputs
        if not all([car_name, fuel_type, car_body, engine_size, stroke, horsepower]):
            return jsonify({'error': 'All fields are required'}), 400
        
        try:     #Checks that numeric fields are valid and positive.
            engine_size = float(engine_size)
            stroke = float(stroke)
            horsepower = float(horsepower)
            
            if engine_size <= 0 or stroke <= 0 or horsepower <= 0:
                return jsonify({'error': 'All numerical values must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Please enter valid numbers for engine size, stroke, and horsepower'}), 400
        
        #Ensures the user entered a value that exists in the training data.
        for col, value in [('CarName', car_name), ('fueltype', fuel_type), ('carbody', car_body)]:
            if value not in encoders[col].classes_:
                return jsonify({'error': f'Invalid {col}: {value}. Please enter a valid option.'}), 400
        

        #run prediction
        #Calls predict_price() from task1.py with cleaned data.
        predicted_price = predict_price(
            car_name=car_name,
            fuel_type=fuel_type,
            car_body=car_body,
            engine_size=engine_size,
            stroke=stroke,
            horsepower=horsepower,
            model=model,
            encoders=encoders
        )
        
        #returns predicted price as JSON to the front-end.
        return jsonify({
            'success': True,
            'prediction': f"${predicted_price:,.2f}",
            'price': predicted_price,
            'input_summary': {
                'car_name': car_name,
                'fuel_type': fuel_type,
                'car_body': car_body,
                'engine_size': engine_size,
                'stroke': stroke,
                'horsepower': horsepower
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

#starts Flask server on port 5000., debug=True helps during development by showing errors in browser.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
