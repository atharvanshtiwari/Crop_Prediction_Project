import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

print("Starting the Machine Learning training process...")

# 1. Load the Data
# We look for the file inside the 'data' folder as per our project structure
file_path = 'data/Crop_recommendation.csv'

# A quick safety check just in case saved it in the main folder instead)
if not os.path.exists(file_path):
    file_path = 'Crop_recommendation.csv'

try:
    crop_data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}. Please check your folder structure.")
    exit()

# 2. Separate Features (X) and Labels (y)
# X = The questions (Nitrogen, Temp, Rainfall, etc.)
# y = The answers (The actual crop names)
X = crop_data.drop('label', axis=1)
y = crop_data['label']

# 3. Split the Data
# We hide 20% of the data (test_size=0.2) so we can test the model fairly later.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Model
# We are using a Random Forest with 100 "decision trees" voting together.
print("Training the Random Forest Classifier... please wait.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model's Performance
# We ask it to guess the crops for our hidden 20% test pile.
predictions = model.predict(X_test)

# We compare its guesses (predictions) to the real answers (y_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Training Complete! The model's accuracy score is: {accuracy * 100:.2f}%")

# 6. Save the Model (The most crucial step for web development)
model_filename = 'crop_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Success! The trained model's 'brain' has been saved as '{model_filename}'")
