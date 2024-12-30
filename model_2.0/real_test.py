import joblib
import pandas as pd

# Load the trained model
model = joblib.load('sleep_time_predictor_model.pkl')

# Input data for a new user (example)
new_data = pd.DataFrame({
    'Age': [24],
    'Gender': ['Female'],
    'Occupation': ['White-Collar'],
    'WorkoutTime': [1],
    'ReadingTime': [0],
    'PhoneTime': [6],
    'WorkHours': [10],
    'CaffeineIntake': [0],
    'RelaxationTime': [3]
})

# Predict the sleep time
predicted_sleep_time = model.predict(new_data)

print(f"Predicted Sleep Time: {predicted_sleep_time[0]:.2f} hours/night")
