
import requests

API_URL = "http://127.0.0.1:8000"

passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(
    f"{API_URL}/predict",
    json=passenger,
    timeout=10
)

print("HTTP Status:", response.status_code)
print("API Response:", response.json())

response.raise_for_status()
