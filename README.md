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



