
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv('accident_data.csv')

# Define features and target
X = df[['Vehicle_Speed', 'Weather_Condition', 'Road_Surface', 'Lighting_Condition', 'Driver_Age', 'Time_of_Day']]
y = df['Accident_Severity']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')
print("Model saved as 'accident_severity_model.pkl'")

# Load model for prediction
loaded_model = joblib.load('accident_severity_model.pkl')

# Hypothetical new input
new_data = pd.DataFrame([{
    'Vehicle_Speed': 80,
    'Weather_Condition': 1,
    'Road_Surface': 2,
    'Lighting_Condition': 1,
    'Driver_Age': 32,
    'Time_of_Day': 20
}])

# Predict and show result
prediction = loaded_model.predict(new_data)
print(f"Predicted Accident Severity: {prediction[0]:.2f}")
