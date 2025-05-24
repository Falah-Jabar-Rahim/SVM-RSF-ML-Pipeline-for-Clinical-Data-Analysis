# SVM and RSF Pipeline for Clinical Data Analysis

![WSI-QA](SVM-RSF.png)

<p align="justify"> This pipeline provides a general framework for analyzing clinical data using both classification and survival analysis models. It begins with constructing a feature matrix, where each row corresponds to a patient and each column represents a clinical variable. Missing values are addressed through imputation, and features are normalized to ensure comparability. The processed data is then used to train two types of models: a classification model-Support Vector Machine, and a survival model-Random Survival Forest, to estimate individual survival probabilities over time. Model performance is evaluated using 5-fold cross-validation, ensuring reliability and generalizability across subsets of the data. This flexible pipeline supports a wide range of clinical prediction tasks, from risk group classification to time-to-event forecasting. </p>
