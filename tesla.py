import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.pyplot as plt


#create the dataframe

data = pd.read_csv('TSLA.csv')  



#Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

#Select features and target
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
y = data['Close']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

linear_regressor = LinearRegression() 
linear_regressor.fit(X_train, y_train)  
Y_pred = linear_regressor.predict(X_train)  


#Make predictions for a specific date
specific_date = datetime.datetime(2011, 7, 29)  
specific_data = data[data['Date'] == specific_date]

specific_features = specific_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
predicted_price = linear_regressor.predict(specific_features)

# Calculate the prediction percentage score
mse = mean_squared_error(y_test, linear_regressor.predict(X_test))
prediction_percentage = 100 * (1 - mse)

print(f"Predicted price for {specific_date}: {predicted_price[0]:.2f}")
print(f"Prediction Percentage Score: {prediction_percentage:.2f}%")