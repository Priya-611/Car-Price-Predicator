# AutoValue - Car Price Predictor

An AI-powered web application that predicts the price of a car based on user input such as fuel type, car body, engine size, and more. It uses a **machine learning model (Random Forest Regressor)** trained on real-world car data, and a responsive **Flask + HTML interface** to deliver real-time price predictions.
The frontend (HTML) and backend (Flask API) were built using the help of an **AI tool to speed up development**.

---

## Dataset Source

This project uses the **Car Price Prediction** dataset from Kaggle:
- Dataset: [Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- Description: The dataset contains car specifications including car name, engine details, fuel type, body style, and actual price.

We used this dataset to **train a Random Forest Regressor model** for price prediction.

---

## Project Background

This project began with an initial script named `car_prediction.py`, where:

- The dataset from [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) was loaded and explored
- A **Random Forest Regressor** was trained using features like:
  - Car Name
  - Fuel Type
  - Car Body
  - Engine Size
  - Stroke
  - Horsepower
- **Label encoding** was used to convert categorical values to numerical format
- The model was evaluated using:
  - Mean Squared Error (MSE)
  - R² Score
- Predictions were made using **manual user input from the console**
- Basic EDA like heatmaps were created to check feature correlation

Later, the logic from `car_prediction.py` was modularized into `task1.py` to enable:

- Code reusability
- Integration with a Flask web app (`app.py`)
- Web-based input via `index.html` instead of terminal-based interaction

This evolution allowed the project to grow from a script to a fully functional **AI-powered car price prediction web app**.

---

## Project Workflow

1. Load and clean the Kaggle dataset
2. Encode categorical variables (`CarName`, `fueltype`, `carbody`) using `LabelEncoder`
3. Train a **Random Forest Regressor** model
4. Save the model and encoders as `.pkl` files using `pickle`
5. Build a **Flask backend** to:
   - Load the model
   - Accept input from users via a web form
   - Return predictions via JSON
6. Create a **responsive HTML UI** using Tailwind CSS and Bootstrap Icons

---

## Technologies Used

- **Python 3**
- **Flask** (web framework)
- **Scikit-learn** (ML)
- **Pandas & NumPy** (data manipulation)
- **Tailwind CSS** (styling)
- **Bootstrap Icons**
- **JavaScript** (form validation & fetch API)
- **HTML5**

---

## Project Structure

├── app.py `Flask backend`

├── car_prediction.py `initial script where model was trained and tested with console input`

├── task1.py `ML logic (train, encode, predict)`

├── model.pkl `Trained model`

├── encoders.pkl `LabelEncoders`

├── CarPrice_Assignment.csv `Source dataset from Kaggle`

├── requirements.txt `Python dependencies`

├── templates/

│ └── index.html `Frontend UI`

---

## How to Run the Project 

1. **Download the project files**
   - Make sure you have these files in one folder:
     - `app.py`
     - `car_prediction.py`
     - `task1.py`
     - `model.pkl`
     - `encoders.pkl`
     - `CarPrice_Assignment.csv`
     - `requirements.txt`
     - `templates/index.html` (inside a folder named `templates`)

2. **Install Python (if not already installed)**
   - Download [Python](https://www.python.org/downloads/)

3. **Open the folder in Command Prompt or Terminal**
   - Use `cd` to go to the folder location. Example:
     ```bash
     cd C:\Users\YourName\Desktop\car-price-predictor
     ```

4. **Install required Python libraries**
   - Run this command to install all required packages:
     ```bash
     pip install -r requirements.txt
     ```

5. **Run the Flask application**
   - Start the app by running:
     ```bash
     python app.py
     ```

6. **Open the app in your browser**
   - Go to this URL:
     ```
     http://localhost:5000
     ```
   - Fill in car details and click “Predict Price” to see the estimated price.


