# CardioRisk

CardioRisk is a full-stack heart disease risk prediction demo. The backend is a FastAPI service that loads a trained scikit-learn pipeline, predicts disease probability, and explains the strongest model factors with SHAP values. The frontend is a React/Vite clinical console for entering patient features and reviewing the prediction response.

This project is for learning and prototyping. It is not a diagnostic tool and should not be used for medical decisions.

## Features

- FastAPI `/predict` endpoint for heart disease risk prediction
- Trained scikit-learn `RandomForestClassifier` pipeline saved as `model_pipeline.pkl`
- SHAP-based top factor explanations aggregated back to readable clinical fields
- Optional Groq-powered plain-language summary generation
- React/Vite frontend with patient input form, risk meter, summary, warnings, and top factors
- Vite dev proxy from `/api` to the FastAPI backend

## Project Structure

```txt
.
|-- app.py                         # FastAPI app and prediction endpoint
|-- main.py                        # model training / exploration script
|-- model_pipeline.pkl             # saved trained model pipeline
|-- requirements.txt               # Python dependencies
|-- data/
|   `-- heart_disease_uci.csv      # training dataset
`-- frontend/
    |-- index.html
    |-- package.json
    |-- package-lock.json
    |-- vite.config.js
    `-- src/
        |-- App.jsx
        |-- main.jsx
        `-- styles.css
```

## Backend Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Optional: create a `.env` file if you want Groq summaries:

```txt
GROQ_API_KEY=your_key_here
```

If `GROQ_API_KEY` is not set or Groq is unavailable, the API still returns predictions and uses a local fallback summary.

Run the backend:

```bash
uvicorn app:app --reload
```

FastAPI docs will be available at:

```txt
http://127.0.0.1:8000/docs
```

## Frontend Setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --port 5174
```

Open:

```txt
http://127.0.0.1:5174
```

The frontend calls `/api/predict`, and `frontend/vite.config.js` proxies those requests to:

```txt
http://127.0.0.1:8000/predict
```

## API Example

Request:

```json
{
  "age": 29,
  "sex": "Male",
  "cp": "typical angina",
  "trestbps": 100,
  "chol": 190,
  "fbs": true,
  "restecg": "normal",
  "thalch": 100,
  "exang": false,
  "oldpeak": 0,
  "slope": "upsloping",
  "ca": 0,
  "thal": "normal",
  "has_missing": false
}
```

Response shape:

```json
{
  "probability": 0.39,
  "top_factors": [
    {
      "feature": "thalch",
      "value": 100,
      "shap_impact": 0.082,
      "display_feature": "Max heart rate achieved"
    }
  ],
  "summary": "Plain-language explanation of the prediction.",
  "warnings": []
}
```

## Model Notes

The training target is derived from the UCI heart disease `num` column:

```python
target = (num > 0).astype(int)
```

The saved API model uses balanced class weights. Earlier experimentation with heavier disease-class weighting improved disease recall but made raw probabilities too aggressive, so the final saved pipeline uses a more conservative weighting for the app.

SHAP values are computed on the transformed model input, then grouped back from one-hot encoded columns to the original clinical features before being returned to the frontend.

## Useful Commands

Run backend syntax check:

```bash
python -m py_compile app.py main.py
```

Build frontend:

```bash
cd frontend
npm run build
```

Run frontend dev server:

```bash
cd frontend
npm run dev -- --port 5174
```

## Git Notes

The repository ignores generated/local files such as:

- `.venv/`
- `.env`
- `__pycache__/`
- `frontend/node_modules/`
- `frontend/dist/`
- frontend log files

Commit source files, package manifests, the dataset, and the saved model artifact. Do not commit API keys or dependency folders.
