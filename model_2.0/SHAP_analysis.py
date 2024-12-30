import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('/Users/navinishrao/shopee_data/sleep_time_predictor_model.pkl')

# Load the dataset
data = pd.read_csv('synthetic_lifestyle_data_with_sleep_time.csv')

# Separate features and target
# Ensure all required features are included
feature_columns = ['Age', 'Gender', 'Occupation', 'WorkHours', 'ReadingTime', 
                   'RelaxationTime', 'PhoneTime', 'CaffeineIntake', 'WorkoutTime']
X = data[feature_columns]
y = data['SleepTime']

# Extract the preprocessor and the model from the pipeline
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

# Preprocess the data
X_processed = preprocessor.transform(X)

# Convert the processed data back to a DataFrame
processed_feature_names = (
    preprocessor.named_transformers_['num'].feature_names_in_.tolist() + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out())
)
X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

# Explain the model predictions using SHAP
explainer = shap.Explainer(model, X_processed_df)
shap_values = explainer(X_processed_df)

# Summary plot
shap.summary_plot(shap_values, X_processed_df)

# Dependence plot for a specific feature (e.g., Age)
shap.dependence_plot("Age", shap_values, X_processed_df)

# Optional: Force plot for a specific prediction
shap.force_plot(explainer.expected_value, shap_values[0, :], X_processed_df.iloc[0, :])
