# Lung Cancer Prediction 

This project aims to predict lung cancer using several models, the  Bernoulli Naive Bayes model worked best. The dataset used for training and testing the model is the Lung Cancer dataset from the UCI Machine Learning Repository, which contains features such as age, smoking history, and air pollution exposure.

# Table of Contents
### Introduction
### Dataset
### Bernoulli Naive Bayes Model
### Training Process

Introduction
Lung cancer is a leading cause of cancer death worldwide. Early detection and diagnosis of lung cancer can significantly improve the chances of successful treatment and recovery. Machine learning algorithms can be used to analyze medical data and predict lung cancer.
This project explores the use of machine learning models for lung cancer prediction. The models were trained on a dataset of lung cancer patients and non-patients, and can predict the likelihood of a patient having lung cancer based on their demographic and medical information.

Dataset
The Lung Cancer dataset used in this project contains 309 records of patients and 15 attributes. The attributes include demographic information such as age, gender, and smoking history, as well as medical information allergy.
The dataset is available in the lung_cancer.csv file in the data directory. The data is preprocessed to remove any missing values and categorical variables are converted to numerical values.

Bernoulli Naive Bayes Model
The Bernoulli Naive Bayes model is a probabilistic model that is used for binary classification problems. It assumes that the features are binary and independent of each other, and uses Bayes' theorem to calculate the probability of a data point belonging to a particular class.
The Bernoulli Naive Bayes model is well-suited for datasets with small sample sizes, such as the Lung Cancer dataset. It also works well with sparse datasets, where the features are not present in every instance.

Training Process
The Bernoulli Naive Bayes model is trained on the Lung Cancer dataset using a 80/20 train-test split. The training data is used to fit the model parameters, while the test data is used to evaluate the model's performance.


The model was tested and performance metrics such as  accuracy, precision, recall, roc_auc_score and F1 score were calculated.

Conclusion
This project demonstrates the use of a Bernoulli Naive Bayes model for lung cancer prediction. The model is trained on a small dataset of lung cancer patients and non-patients, and can predict the likelihood of a patient having lung cancer based on their demographic and medical information. The project can be extended by using more advanced machine learning models or by incorporating additional features into the dataset.
