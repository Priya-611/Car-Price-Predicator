# Random Forest can learn complex logic (It uses multiple trees to make the prediction more stable)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  #it works best with mixed data(both categorical and numerical features) [handles non-linear patterns as car price don't increase in straight line]
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder #LabelEncoder turns words into numbers so the model can understand and work with them.


#Loading dataset
d= pd.read_csv("C:\\Users\\HP\\Downloads\\CarPrice_Assignment.csv"  )

print(d.head())
print(d.info())
print(d.describe())
print(d.isnull().sum())

print(d.columns)
# ['car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration',
#        'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
#        'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
#        'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
#        'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg',
#        'price']


#heatmap for checking correlation amoung columns (to know which factor is contributing much in predicting car price)
sns.heatmap(d.corr(numeric_only=True),annot=True)
plt.show()


#Value of X[columns contributing in result] and Y[result]

x=d[['CarName', 'fueltype', 'carbody', 'enginesize','stroke','horsepower']].copy()

y=d['price']

#Machine learning models can’t understand text — they only work with numbers.
#LabelEncoder to convert these text values into numbers.
carname_encoder=LabelEncoder()
fueltype_encoder=LabelEncoder()
carbody_encoder=LabelEncoder()


x['CarName']=carname_encoder.fit_transform(x['CarName'])
x['fueltype']=fueltype_encoder.fit_transform(x['fueltype'])
x['carbody']=carbody_encoder.fit_transform(x['carbody'])

#Model training
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=45)
model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


#Evaluation
mse=mean_squared_error(y_test,y_pred)   #average difference between predicted and actual ratings (lower is better)
r_sq=r2_score(y_test,y_pred)   #Closer to 1 is better[perfect prediction]

print("Mean Square Error: ",mse )
print("r2_square", r2_score)    




#evaluating model on user input
print("Enter car details: ")

car_name=input("Enter car name: ").strip()  #strip() removes any leading and trailing whitespace from the input string.
fuel_ty=input("Enter fuel type: ").strip()
car_B=input("Enter car Body: ").strip()
engine_s=float(input("Enter engineSize: "))
stroke_input=float(input("Enter stroke value: "))
hp_input=float(input("Enter horsepower: "))


#again tranforming text data into numbers for model to predict
carname_encoded = carname_encoder.transform([car_name])[0]
fueltype_encoded = fueltype_encoder.transform([fuel_ty])[0]
carbody_encoded = carbody_encoder.transform([car_B])[0]


#creating a DataFrame (like a table) with one row that contains the user's input, ready to be used by the model.
#column names so that they match the column names used during model training, which is important for accurate prediction.
input_features = pd.DataFrame([[carname_encoded, fueltype_encoded, carbody_encoded, engine_s, stroke_input, hp_input]],columns=['CarName', 'fueltype', 'carbody', 'enginesize', 'stroke', 'horsepower'])


#Predict (y_pred=model.predict(x_test))
#model.predict() gives a list of predictions (even if there's only 1 row),, [0] grabs the first (and only) predicted valu
predicted_price = model.predict(input_features)[0]


#print predicted price till 2 decimal point
print("Predicted Car Price: ",round(predicted_price,2))

