
import joblib
import pandas as pd

# Load the saved model and unseen records
model = joblib.load("model.pkl")
unseen_data = pd.read_csv("inference_unseen.csv")

# Select one unseen passenger
passenger = unseen_data.iloc[[0]]

# Generate prediction
prediction = int(model.predict(passenger)[0])

print("ONLINE INFERENCE RESULT")
print("-----------------------")
print("Passenger details:")
print(passenger.to_string(index=False))
print("Predicted class:", prediction)

if prediction == 1:
    print("Prediction: Passenger Survived")
else:
    print("Prediction: Passenger Did Not Survive")
