# Crop Recommendation System

An end-to-end, AI-powered agricultural advisory application designed to provide data-driven crop recommendations based on soil health and environmental conditions.

## 🚀 Live Demo

URL: https://crop-prediction-crenv2qee8yoyejwzf5rjg.streamlit.app/

## 📝 Project Overview
This project leverages Machine Learning and Generative AI to bridge the gap between complex soil data and actionable agricultural advice. Instead of just predicting a crop, the system provides personalized, context-aware expert notes to help farmers maximize their yields.

## ✨ Key Features
* **Intelligent Prediction**: Uses a trained Random Forest Classifier to predict the optimal crop based on Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
* **Generative AI Integration**: Automatically generates expert-level agricultural notes using the Gemini API based on the model's output.
* **Interactive Visualization**: Features dynamic Plotly donut charts to visualize model confidence across various potential crop options.
* **Modern UI/UX**: Built with **Streamlit** and enhanced with animated **Lottie** visuals for a professional, user-friendly experience.
* **Secure Architecture**: Implements secure credential management using `.streamlit/secrets.toml` to protect API keys during deployment.

## 🛠 Tech Stack
* **Language**: Python
* **ML Framework**: scikit-learn (Random Forest Classifier)
* **Web Framework**: Streamlit
* **AI Integration**: Google GenAI SDK
* **Data Processing**: Pandas, NumPy
* **Visualization**: Plotly, Lottie

## 📂 Project Structure
```text
├── .streamlit/
│   └── secrets.toml        # Secure API key storage
├── data/
│   └── Crop_recommendation.csv # Training dataset
├── app.py                  # Main web application
├── model_training.py       # ML model training script
├── crop_model.pkl          # Exported trained ML model
├── requirements.txt        # Dependencies
└── .gitignore              # Security configurations
