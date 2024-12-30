import pandas as pd
import numpy as np
import random

# Define the number of rows for synthetic data
num_rows = 1000  # Adjust the number of rows based on your needs

# Define the synthetic data ranges
ages = list(range(18, 80))  # Age between 18 and 80
genders = ['Male', 'Female', 'Non-Binary']  # Gender categories

# Define the occupational categories
occupational_categories = ['Blue-Collar', 'White-Collar', 'Pink-Collar', 'Green-Collar', 'Grey-Collar']

# Define additional lifestyle parameters (previously mentioned ones)
workout_time_range = (0, 3)  # hours per day
reading_time_range = (0, 2)  # hours per day
phone_time_range = (0, 12)  # hours per day
work_hours_range = (6, 12)  # hours per day
caffeine_intake_range = (0, 400)  # mg per day
relaxation_time_range = (0, 4)  # hours per day
sleep_time_range = (4, 10)  # hours per night (initial range)

# Generate synthetic data
data = {
    'Age': [random.choice(ages) for _ in range(num_rows)],
    'Gender': [random.choice(genders) for _ in range(num_rows)],
    'Occupation': [random.choice(occupational_categories) for _ in range(num_rows)],
    
    # Add the additional lifestyle data
    'WorkoutTime': [random.uniform(*workout_time_range) for _ in range(num_rows)],
    'ReadingTime': [random.uniform(*reading_time_range) for _ in range(num_rows)],
    'PhoneTime': [random.uniform(*phone_time_range) for _ in range(num_rows)],
    'WorkHours': [random.uniform(*work_hours_range) for _ in range(num_rows)],
    'CaffeineIntake': [random.uniform(*caffeine_intake_range) for _ in range(num_rows)],
    'RelaxationTime': [random.uniform(*relaxation_time_range) for _ in range(num_rows)]
}

# Create a DataFrame
df_synthetic = pd.DataFrame(data)

# Simulate the Sleep Time based on Age, Gender, and Occupation
def simulate_sleep_time(row):
    sleep_time = 8  # Start with an average of 8 hours for a healthy sleep

    # Age impact: Older people sleep less
    if row['Age'] > 60:
        sleep_time -= random.uniform(0.5, 1.5)  # Sleep time reduced by up to 1.5 hours for older people
    elif row['Age'] < 30:
        sleep_time += random.uniform(0.5, 1.0)  # Sleep time increased for younger people

    # Gender impact: Slight difference between male and female
    if row['Gender'] == 'Male':
        sleep_time -= random.uniform(0.1, 0.5)  # Slight reduction for males
    elif row['Gender'] == 'Female':
        sleep_time += random.uniform(0.1, 0.5)  # Slight increase for females

    # Occupation impact: Blue-collar workers sleep less due to manual labor
    if row['Occupation'] == 'Blue-Collar':
        sleep_time -= random.uniform(0.5, 1.5)  # Sleep time reduced for blue-collar workers
    elif row['Occupation'] == 'White-Collar':
        sleep_time += random.uniform(0.0, 0.5)  # White-collar workers may sleep more

    # Bound the sleep time to be between 4 and 10 hours for sanity
    sleep_time = max(4, min(sleep_time, 10))

    return sleep_time

# Apply the function to simulate SleepTime
df_synthetic['SleepTime'] = df_synthetic.apply(simulate_sleep_time, axis=1)

# Save to CSV (if you want to export)
df_synthetic.to_csv('synthetic_lifestyle_data_with_sleep_time.csv', index=False)

# Check the first few rows of the generated data
print(df_synthetic.head())
