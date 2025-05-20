# Urban Heat Island (UHI) Temperature Prediction

This repository contains the implementation of a machine learning model to predict temperature variations associated with the Urban Heat Island (UHI) phenomenon. The project involves multiple regression models and clustering techniques, aiming to uncover the relationship between environmental variables and temperature patterns in urban areas. 

## Project Overview

The primary goal of this project is to predict urban heat island (UHI) temperatures using various machine learning models. The data collected consists of various environmental features, including temperature, population density, urban greenness ratio, air quality, and humidity, from urban areas.

The following sections outline the workflow of the project and provide insight into the analysis and results.

## Table of Contents
- [Data Description and Preprocessing](#data-description-and-preprocessing)
- [Model Implementation and Evaluation](#model-implementation-and-evaluation)
- [Model Visualization](#model-visualization)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Directory Structure](#directory-structure)
- [Installation](#installation)

## Data Description and Preprocessing

The dataset used for this project is a collection of urban environmental metrics such as:
- **Temperature (°C)**
- **Urban Greenness Ratio (%)**
- **Population Density (people/km²)**
- **Humidity (%)**
- **Air Quality Index (AQI)**

### Data Cleaning and Feature Engineering
1. **Data Loading**: The data is loaded from CSV files containing time-series and multivariate features.
2. **Handling Missing Values**: The dataset contains no missing values. However, preprocessing steps include scaling and normalizing data to prepare it for model training.
3. **Feature Engineering**: The features were engineered by applying one-hot encoding for categorical variables and scaling continuous variables to standardize their range.

### Preprocessing Techniques
- **Standardization**: Numerical features are standardized using `StandardScaler` to ensure that all features contribute equally to the model training.
- **Train-Test Split**: Data is split into an 80-20% ratio for training and testing.

## Model Implementation and Evaluation

The following regression models were used to predict UHI temperatures:
1. **Linear Regression**
2. **Random Forest**
3. **XGBoost**
4. **MLP (Multi-layer Perceptron)**
5. **Stacking Ensemble**
6. **Voting Ensemble**
7. **SVR (Support Vector Regression)**
8. **LightGBM**
9. **Bayesian Ridge**

### Model Training Process
1. **Hyperparameter Tuning**: Key models such as Random Forest were tuned using `GridSearchCV` for optimal parameters.
2. **Performance Metrics**: Each model's performance was evaluated using several metrics, including **RMSE**, **R²**, **MAE**, and **CV RMSE**.
3. **Clustering**: K-Means and GMM (Gaussian Mixture Model) were used to cluster data points based on their features. The silhouette score was used to evaluate clustering quality.

### Model Evaluation
- **Cross-Validation**: 10-fold cross-validation was used to evaluate model stability.
- **Results**: The **Stacking Ensemble** model emerged as the top performer, followed by Random Forest and XGBoost. The **MLP** and **SVR** models showed higher RMSE, indicating less stability.

## Model Visualization

Several plots were generated to visualize the results and insights from the models:
1. **Predicted vs. Actual Temperature**: Scatter plots comparing predicted and actual temperatures for each model.
2. **Feature Importance**: Bar plots displaying the importance of different features in the model predictions.
3. **Correlation Heatmaps**: Visualizing the relationships among features.
4. **Learning Curves**: Showing training vs. test error over time.
5. **Residual Plots**: To analyze the prediction error distribution for each model.

## Results and Analysis

Key findings from the models:
- **Stacking Ensemble**: Achieved the best performance with RMSE = 0.07 and CV RMSE = 0.07, indicating excellent generalization across different datasets.
- **Random Forest**: Tuning improved the performance to an RMSE of 0.06, showing strong predictive power.
- **XGBoost**: Despite good performance, XGBoost showed slightly higher RMSE (0.18) than Random Forest.
- **K-Means vs. GMM**: Both clustering algorithms identified distinct temperature clusters, with GMM providing a more refined separation of urban areas.

## Conclusion

- **Best Model**: The **Stacking Ensemble** model was found to be the most reliable for UHI temperature prediction due to its robustness and strong performance in cross-validation.
- **Future Work**: Recommendations for future research include:
  - Expanding the dataset to include more cities and regions for better model generalization.
  - Enhancing the **MLP** model by adjusting its architecture and hyperparameters to stabilize performance.
  - Developing a real-time application for UHI prediction using the best-performing model.
