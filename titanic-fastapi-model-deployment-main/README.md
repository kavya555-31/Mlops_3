# ML Model Deployment Strategies and Model Serving using FastAPI

## Project Overview
This project demonstrates machine learning model deployment using the Titanic dataset. A classification pipeline predicts whether a passenger survived or did not survive.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Requests

## Project Features
1. Train a machine learning classification pipeline.
2. Perform online inference for a single passenger.
3. Perform batch inference for multiple passengers.
4. Demonstrate streaming inference with a one-second delay.
5. Deploy the model using FastAPI.
6. Validate API inputs using Pydantic.
7. Test API endpoints using Swagger UI and a Python client.

## API Endpoints
- `GET /health` — Checks the API and model status.
- `POST /predict` — Predicts passenger survival.

## Project Files
- `model.pkl` — Trained model pipeline.
- `inference_unseen.csv` — Unseen passenger records.
- `batch_predictions.csv` — Batch prediction results.
- `app.py` — FastAPI application.
- `client.py` — API testing client.
- `online_inference.py` — Online inference program.
- `batch_inference.py` — Batch inference program.
- `streaming_inference.py` — Streaming inference program.
- `requirements.txt` — Python dependencies.

## How to Run
Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open the Swagger UI in your browser:

`http://127.0.0.1:8000/docs`

Run the API client in another terminal:

```bash
python client.py
```

## Conclusion
This project demonstrates online, batch, and streaming inference, along with REST API-based machine learning model serving and input validation.