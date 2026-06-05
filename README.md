# Fertilizer-Recommendations-System
Fertilizer Recommendation System

Project Overview

The Fertilizer Recommendation System is a Machine Learning project that predicts the most suitable fertilizer based on environmental conditions, soil type, crop type, and nutrient values.

Problem Statement

Farmers often face difficulties in selecting the correct fertilizer for different crops and soil conditions. This project helps recommend the appropriate fertilizer using machine learning techniques.

Dataset Information

Dataset Name: Fertilizer Prediction Dataset

Features

* Temperature
* Humidity
* Moisture
* Soil Type
* Crop Type
* Nitrogen
* Potassium
* Phosphorous

Target Variable

* Fertilizer Name

Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset structure analysis
* Missing value detection
* Duplicate record checking
* Statistical summary generation
* Soil type distribution analysis
* Crop type distribution analysis
* Fertilizer distribution analysis
* Correlation analysis

Data Preprocessing

* Handled missing values
* Encoded categorical features using Label Encoding
* Prepared data for machine learning

Machine Learning Model

Algorithm Used:

* Random Forest Classifier

Model Evaluation

Evaluation Metrics:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Feature Importance Analysis

Visualizations

The project includes:

1. Fertilizer Distribution Chart
2. Soil Type Distribution Chart
3. Crop Type Distribution Chart
4. Nitrogen vs Fertilizer Box Plot
5. Correlation Heatmap

Files Included

* Fertilizer Prediction.csv
* train.py
* predict.py
* model.pkl
* graph1.png
* graph2.png
* graph3.png
* graph4.png
* graph5.png
* README.md

How to Run

Install Required Libraries

pip install pandas numpy matplotlib seaborn scikit-learn

Train the Model

python train.py

Predict Fertilizer

python predict.py

Sample Output

The model predicts the most suitable fertilizer based on the input values provided.

Future Enhancements

* Deploy using Streamlit
* Add real-time farmer recommendations
* Improve accuracy using hyperparameter tuning
* Integrate larger agricultural datasets

Conclusion

This project demonstrates the application of machine learning in agriculture by recommending suitable fertilizers based on crop, soil, and nutrient conditions. The system can help improve fertilizer selection and support data-driven farming decisions.
