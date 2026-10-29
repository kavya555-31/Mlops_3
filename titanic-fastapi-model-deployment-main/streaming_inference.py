
import joblib
import pandas as pd
import time

model = joblib.load("model.pkl")
unseen_data = pd.read_csv("inference_unseen.csv")

stream_data = unseen_data.iloc[:10]

print("STREAMING INFERENCE STARTED")

for i in range(len(stream_data)):
    passenger = stream_data.iloc[[i]]
    prediction = int(model.predict(passenger)[0])

    label = (
        "Survived" if prediction == 1
        else "Did not survive"
    )

    print(f"Record {i + 1}: {prediction} ({label})")

    if i < len(stream_data) - 1:
        time.sleep(1)

print("Streaming inference completed!")
print("Total records processed:", len(stream_data))
