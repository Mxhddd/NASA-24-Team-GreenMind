# Mental Health Prediction Visualization

## Overview
The **Mental Health Prediction Visualization** application offers an interactive platform to analyze the impact of various environmental and social factors on mental health scores. It leverages machine learning, specifically the Random Forest Regressor, to predict mental health outcomes based on input data.

## Features
- **Data Upload**: Users can upload a CSV file containing relevant data.
- **Data Validation**: The application checks for necessary columns and informs users if any are missing.
- **Model Training**: The application trains a Random Forest model to predict mental health scores.
- **Visualizations**: 
  - Actual vs Predicted Mental Health Scores
  - Error Distribution
  - Feature Importance
  - Correlation Heatmap
  - Summary of Key Metrics
- **Qualitative Analysis**: Provides descriptive insights based on the uploaded data.
- **Algorithmic Transparency**: Explains the model's workings and ethical considerations.

## Requirements
The CSV file must contain the following columns:
- `air_quality_index`: Float
- `noise_level`: Float
- `green_space_area`: Float
- `land_surface_temp`: Float
- `temperature`: Float
- `humidity`: Float
- `precipitation`: Float
- `population_density`: Float
- `crime_rate`: Float
- `mental_health_score`: Integer (0-100)

