# Instagram Engagement Prediction & Segmentation

## 🔴 Live Demo
After deploying this repository from your GitHub account on Streamlit Community Cloud, add your generated Streamlit app URL here.

**GitHub Repository:** https://github.com/youssefzizo757-yz/Instagram-Engagement-Prediction-and-Segmentation

## 📌 Overview
Machine Learning project for predicting Instagram post engagement and routing posts to specialized models using a K-Means + Mixture of Experts approach.

## ✨ Streamlit App
The app accepts follower count, hashtags, caption, image count, post date/time, post type, previous post count, and historical median engagement, then predicts engagement.

Pipeline:
1. Feature engineering
2. Scaling
3. K-Means cluster routing
4. Cluster-specific expert model prediction
5. Predicted engagement display

## 🧠 ML Techniques
- Data preprocessing
- Feature engineering
- EDA and visualization
- HistGradientBoostingRegressor
- RandomForestRegressor
- K-Fold Cross Validation
- K-Means clustering
- PCA
- Mixture of Experts
- Model evaluation with R², RMSE, MAE and MSE

## 📊 Saved Model Holdout Metrics
- R²: 0.865
- RMSE: 183,020
- MAE: 30,967

## 📁 Structure
```text
├── app.py
├── instagram_engagement_model.ipynb
├── requirements.txt
├── assets/
│   ├── icon.svg
│   └── style.css
├── ml/
│   ├── model_artifacts.joblib
│   └── train_model.py
└── data/
    └── README.md
```

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit deployment
Use this repository in Streamlit Community Cloud and set the main file to `app.py`.

## ⚠️ Data
Original CSV datasets are not included. Do not upload private or restricted data to a public repository.

## 👨‍💻 Author
**Youssef Mohamed Abd El Aziz**
