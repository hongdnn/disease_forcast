import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


keras = tf.keras

# Load the trained model
model = keras.models.load_model('disease_prediction_model.keras')

data = pd.read_csv('./dataset/Disease_Forcast_2024.csv')

print(data)

# Features and target
X_new = data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']]

# Scale features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Predict with the model
predictions = model.predict(X_new_scaled)

disease_prediction = (predictions >= 0.5).astype(int)  # 1 for disease, 0 for no disease

# Append predictions to new_data for viewing
data['is_disease'] = disease_prediction

# Print results. The model predicts there is a disease in a particular city if is_disease = 1
print(data[['city', 'illness', 'is_disease']])
