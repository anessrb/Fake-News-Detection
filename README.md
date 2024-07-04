# Fake News Detection Project

## Introduction
This project aims to classify news articles as fake or true using various machine learning techniques. It includes binary and multi-class classifications to improve the accuracy and reliability of fake news detection.

![Screenshot of Deployment](path_to_screenshot)

## Overview
1. **Data Preprocessing**: 
    - Data is loaded from provided CSV files.
    - Irrelevant columns are dropped.
    - Missing values are handled.
    - Text data is cleaned and normalized using a custom `MyCleanText` function.
  
2. **Feature Extraction**:
    - `TF-IDF Vectorization` is used to convert text data into numerical features for the models.

3. **Class Imbalance Handling**:
    - Downsampling is used to balance the classes for training.
    - SMOTE (Synthetic Minority Over-sampling Technique) is also used for some models.

4. **Classification Models**:
    - **Binary Classification**: 
        - Models used: SVC, Logistic Regression, KNN, RandomForest, XGBoost, AdaBoost, GradientBoosting.
    - **Multi-class Classification**:
        - Categories: True, False, Mixture, Other.
    - Models were trained and hyperparameters were tuned using GridSearchCV.

5. **Model Explanation**:
    - ELI5 is used to explain the model predictions and understand feature importances.

## Results and Discussion
- **Binary Classification**: Achieved satisfactory results with the combination of text and title using SVC.
- **Multi-class Classification**: Faced challenges due to the complexity and reduced dataset size from downsampling, resulting in lower accuracy.
- **Downsampling**: While necessary for class balance, it significantly reduced the dataset size, impacting model performance, especially for the multi-class classification.
- **Upsampling**: Tried but led to overfitting, indicating the need for a larger dataset.

## Potential Improvements
- **Increase Dataset Size**: Larger datasets can help improve model performance, especially for complex classifications.
- **Advanced Techniques**: Implementing advanced models like neural networks could enhance classification accuracy.
- **Hyperparameter Tuning**: Further refinement of hyperparameters and experimenting with different algorithms could yield better results.
- **Explainability Techniques**: Using ELI5, SHAP, and similar techniques to better understand and adjust models based on their decision-making process.

## Deployment
The model was deployed using Flask, allowing users to input a news article title and text to receive a prediction on its authenticity.

## Conclusion
This project highlights the importance of data preprocessing, class balancing, and model explainability in building reliable fake news detection systems. Despite challenges with dataset size and class imbalance, the project demonstrates promising approaches and provides a foundation for further improvements and exploration in fake news detection.

![Screenshot of Deployment](Deploiement.png)
