
import joblib
import pandas as pd

model = joblib.load("model.pkl")
unseen_data = pd.read_csv("inference_unseen.csv")

# Select 20 unseen passengers
batch_data = unseen_data.iloc[:20].copy()

# Predict all records together
predictions = model.predict(batch_data)

batch_data["Predicted_Survived"] = predictions.astype(int)
batch_data["Prediction_Label"] = batch_data[
    "Predicted_Survived"
].map({
    1: "Survived",
    0: "Did not survive"
})

# Save results
batch_data.to_csv("batch_predictions.csv", index=False)

print("BATCH INFERENCE COMPLETED")
print("Total predictions:", len(batch_data))
print(batch_data.to_string(index=False))
print("Saved: batch_predictions.csv")
