# Car Price Prediction using Random Forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_data():
    """Load the car price dataset"""
    csv_path = r"C:\Users\HP\OneDrive\Documents\Project(Data Science)\OASIS\CarPrice_Assignment.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    return pd.read_csv(csv_path)

def prepare_features(data):
    """Prepare features and target for the model"""
    features = data[['CarName', 'fueltype', 'carbody', 'enginesize', 'stroke', 'horsepower']]
    target = data['price']
    return features, target

def create_encoders(features):
    """Create LabelEncoders for categorical features"""
    encoders = {
        'CarName': LabelEncoder().fit(features['CarName']),
        'fueltype': LabelEncoder().fit(features['fueltype']),
        'carbody': LabelEncoder().fit(features['carbody'])
    }
    return encoders

def encode_features(features, encoders):
    """Encode categorical features using LabelEncoders"""
    features_encoded = features.copy()
    for col, encoder in encoders.items():
        features_encoded[col] = encoder.transform(features[col])
    return features_encoded

def train_model(features_encoded, target):
    """Train the Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_encoded, target)
    return model

def evaluate_model(model, features_encoded, target):
    """Evaluate the model performance"""
    x_train, x_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=45)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r_sq = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R² Score: {r_sq:.4f}")
    
    return mse, r_sq

def save_model_and_encoders(model, encoders):
    """Save the trained model and encoders"""
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Model and encoders saved successfully!")

def load_model_and_encoders():
    """Load the trained model and encoders"""
    if os.path.exists('model.pkl') and os.path.exists('encoders.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    return None, None

def predict_price(car_name, fuel_type, car_body, engine_size, stroke, horsepower, model, encoders):
    """Predict car price using the trained model"""
    # Prepare input features
    input_features = pd.DataFrame([{
        'CarName': car_name,
        'fueltype': fuel_type,
        'carbody': car_body,
        'enginesize': engine_size,
        'stroke': stroke,
        'horsepower': horsepower
    }])
    
    # Encode categorical features
    for col in ['CarName', 'fueltype', 'carbody']:
        input_features[col] = encoders[col].transform([input_features[col].iloc[0]])[0]
    
    # Make prediction
    predicted_price = model.predict(input_features)[0]
    return predicted_price

def main():
    """Main function to train and test the model"""
    print("Loading car price dataset...")
    data = load_data()
    print(f"Dataset loaded: {data.shape[0]} cars, {data.shape[1]} features")
    
    print("\nPreparing features...")
    features, target = prepare_features(data)
    
    print("Creating encoders...")
    encoders = create_encoders(features)
    
    print("Encoding features...")
    features_encoded = encode_features(features, encoders)
    
    print("Training Random Forest model...")
    model = train_model(features_encoded, target)
    
    print("Evaluating model...")
    mse, r_sq = evaluate_model(model, features_encoded, target)
    
    print("Saving model and encoders...")
    save_model_and_encoders(model, encoders)
    
    # Test prediction
    print("\nTesting prediction with sample data...")
    test_price = predict_price(
        car_name='alfa-romero giulia',
        fuel_type='gas',
        car_body='sedan',
        engine_size=120,
        stroke=3.47,
        horsepower=95,
        model=model,
        encoders=encoders
    )
    print(f"Test prediction: ${test_price:,.2f}")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()



