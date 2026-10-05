# Titanic Survival Predictor 🚢

An end-to-end machine learning project: exploratory analysis and feature engineering on the Kaggle Titanic dataset, model selection across 20+ classifiers, and a **FastAPI** web app that serves the final model.

**[Live demo](https://titanic-survival-predictor-jny8.onrender.com/)** · [EDA notebook](EDA.ipynb) · [Model selection notebook](Model_Selection.ipynb)

![App screenshot](static/Images/Demo.png)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/deployed%20on-Render-46E3B7?logo=render&logoColor=white)

## Highlights
- **Feature engineering:** passenger `Title` extracted from names (Mr, Mrs, Miss, Master, Rare) and `Family_size` bucketed into Alone / Small / Large.
- **Model selection:** 20+ scikit-learn classifiers compared with 5-fold cross-validation inside a single preprocessing `Pipeline` (MinMax scaling, one-hot and ordinal encoding, quantile binning).
- **Final model:** hard-voting ensemble of Ridge, LDA and Gradient Boosting — **≈82% accuracy** on the held-out test split.
- **Serving:** the whole pipeline is serialised with `joblib` and served by FastAPI with typed, validated inputs (Pydantic) and a responsive HTML front end.

## Project structure
```
├── app.py                 # FastAPI app: UI, /predict and /health endpoints
├── model.joblib           # Trained preprocessing + voting-classifier pipeline
├── templates/index.html   # Front end
├── static/Images/         # Background and screenshot
├── input/                 # Raw and cleaned Kaggle data
├── EDA.ipynb              # Exploratory analysis and feature engineering
├── Model_Selection.ipynb  # Cross-validated model comparison and final model
├── tests/test_app.py      # API tests
├── requirements.txt       # Runtime dependencies (app only)
├── requirements-dev.txt   # Notebook and test dependencies
└── render.yaml            # Render deployment config
```

## Run locally
```bash
git clone https://github.com/rjuzair/titanic-survival-prediction-api.git
cd titanic-survival-prediction-api
pip install -r requirements.txt
uvicorn app:app --reload        # or: python app.py
```
Open <http://localhost:8000>, or call the API directly:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Sex":"female","Age":29,"Pclass":1,"Fare":80,"Embarked":"C","Title":"Mrs","Family_size":"Small"}'
# {"prediction": 1}
```
Interactive API docs are available at `/docs`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Possible next steps
- Report precision/recall and a confusion matrix alongside accuracy, and tune the ensemble with a grid search.
- Return survival **probability** (soft voting) instead of a hard label.
- Add a GitHub Actions workflow to run the tests on every push.

## License
[Apache 2.0](LICENSE)
