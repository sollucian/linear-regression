# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace with your dataset)
data = {
    'Weather_Condition': ['Clear', 'Rain', 'Snow', 'Clear', 'Rain'],
    'Road_Type': ['Highway', 'Urban', 'Rural', 'Urban', 'Highway'],
    'Time_of_Day': ['Day', 'Night', 'Day', 'Night', 'Day'],
    'Vehicle_Type': ['Car', 'Truck', 'Motorcycle', 'Car', 'Truck'],
    'Driver_Age': [25, 30, 22, 35, 28],
    'Accident_Severity': [1, 2, 3, 2, 1]
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Weather_Condition', 'Road_Type', 'Time_of_Day', 'Vehicle_Type'], drop_first=True)

# Define dependent (target) and independent (features) variables
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model for future use
import joblib
joblib.dump(model, 'accident_severity_model.pkl')
