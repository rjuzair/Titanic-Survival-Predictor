# Titanic Survivor Predictor 🚢

A simple machine learning inference web app built with [FastAPI](https://fastapi.tiangolo.com/) for predicting Titanic passenger survival.  
This project demonstrates how to serve ML models with FastAPI and a modern HTML frontend.

## 🚀 Features

- FastAPI backend for model inference
- Pre-trained Titanic survival model ([model.joblib](model.joblib))
- User-friendly web interface ([index.html](templates/index.html))
- Static assets served via FastAPI
- Example EDA and model selection notebooks

## 📸 Demo

![App Screenshot](static/Images/Demo.png)

## 🏗️ Project Structure

```
├── app.py                # FastAPI application
├── model.joblib          # Pre-trained ML model
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Main HTML UI
├── static/
│   └── Images/
│       └── image.jpg     # Background image
├── input/                # Data files
├── EDA.ipynb             # Exploratory Data Analysis
├── Model_Selection.ipynb # Model selection notebook
└── ...
```

## 🛠️ Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/yourusername/titanic-survivor-predictor.git
cd titanic-survivor-predictor
pip install -r requirements.txt
```

## ⚡ Usage

Start the FastAPI server:

```sh
python app.py
```

Visit [http://localhost:8000](http://localhost:8000) in your browser to use the app.

## 📝 Notebooks

- [EDA.ipynb](EDA.ipynb): Data exploration and visualization
- [Model_Selection.ipynb](Model_Selection.ipynb): Model training and evaluation

## 📂 Data

- Place your Titanic dataset CSVs in the `input/` directory.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

