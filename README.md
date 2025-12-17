#🏠 House Price Predictor

This is a Flask-based machine learning web application that predicts house prices in Gurgaon using housing and location-related features. The project covers model training, data preprocessing, and a simple web interface for making predictions.

##🚀 Features
- House price prediction through a web interface
- Machine learning model built using Scikit-Learn
- Data preprocessing using pipelines
- Simple and clean UI built with HTML and CSS
- Well-structured and modular codebase

##🧠 Machine Learning Details
- Numerical data is handled using median imputation and feature scaling
- Categorical data (`ocean_proximity`) is encoded using one-hot encoding
- A Random Forest Regressor is used for prediction
- Stratified sampling is used while splitting the dataset

##📊 Dataset
The dataset contains housing-related information such as:
- Location details (longitude, latitude)
- Housing data (total rooms, bedrooms, households)
- Income-related features
- Ocean proximity information
