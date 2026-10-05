from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PASSENGER = {
    "Sex": "female", "Age": 29, "Pclass": 1, "Fare": 80.0,
    "Embarked": "C", "Title": "Mrs", "Family_size": "Small",
}


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Titanic Survival Predictor" in response.text


def test_predict_returns_binary_label():
    response = client.post("/predict", json=PASSENGER)
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)


def test_first_class_woman_survives_third_class_man_does_not():
    survivor = client.post("/predict", json=PASSENGER).json()["prediction"]
    victim = client.post("/predict", json={
        **PASSENGER, "Sex": "male", "Title": "Mr", "Pclass": 3, "Fare": 7.25,
        "Embarked": "S", "Family_size": "Alone", "Age": 30,
    }).json()["prediction"]
    assert (survivor, victim) == (1, 0)


def test_invalid_category_is_rejected():
    response = client.post("/predict", json={**PASSENGER, "Embarked": "X"})
    assert response.status_code == 422
