
###  Project Visualizations
####     1. Feature Correlation Matrix
<img width="843" height="490" alt="Screenshot 2025-10-26 055229" src="https://github.com/user-attachments/assets/fac82e54-ec0a-408b-84f3-74e6019114c5" />

Correlation matrix showing relationships between key diabetic features.


####     2. Classification Models

####      2i. Classification Models Performance Evaluation: 

A suite of machine learning classifiers including logistic regression (LR), stochastic gradient descent (SGD), decision tree (DT), k-nearest neighbors (KNN), random forest (RF), and extreme gradient boosting (XGBoost) were evaluated. A comparison of accuracy, precision, recall, F1 and receiver operating characteristic area under the curve (ROC-AUC) across the different models before and after model optimization via hyper parameter tuning, class weight and threshold optimization was performed.

<img width="983" height="327" alt="Screenshot 2026-01-14 040741" src="https://github.com/user-attachments/assets/43d72c53-f7f2-48e9-94bf-a5c7421c9917" />


####      2ii.  ROC Curves 
The plot presents the receiver operating characteristic (ROC) curves for the six optimized classification models evaluated on the test dataset

<img width="625" height="447" alt="Screenshot 2026-01-14 111302" src="https://github.com/user-attachments/assets/a5de1076-f0a7-4356-b9d4-46be0eca180f" />

####      2iii.  Feature Importance

Feature Importance bar plot showing the impact of features on XGBoost model's performance by ranking their influence on the model's output.

<img width="606" height="361" alt="Screenshot 2026-01-14 044710" src="https://github.com/user-attachments/assets/5aced5ba-3d78-408a-a06c-26368d04f7c8" />


####      2iv.  SHAP analysis
Mean absolute SHAP values for the top 5 features influencing XGBoost model's predictions of diabetes

<img width="695" height="387" alt="Screenshot 2026-01-14 045659" src="https://github.com/user-attachments/assets/b76dd287-f193-45b8-a812-f193b1e865f4" />


####      2v.  SHAP Summary Plot

SHAP summary (beeswarm) plot for top five features 

<img width="690" height="291" alt="Screenshot 2026-01-14 133216" src="https://github.com/user-attachments/assets/b1dd5490-3457-4d19-a68a-65908a8e3ffd" />


####  3.  Regression Models 

####      3i. Regression Models Performance Evaluation: 

Six regression models were evaluated using cross-validated root mean squared error (CV RMSE) and independent test-set metrics, including coefficient of determination (R²), absolute and squared error measures, and correlation-based statistics

<img width="860" height="250" alt="Screenshot 2026-01-14 060603" src="https://github.com/user-attachments/assets/b33be3c9-ce77-458b-908d-738658c04b61" />


####     3ii. Feature Importance
<img width="556" height="318" alt="Screenshot 2026-01-14 121041" src="https://github.com/user-attachments/assets/09e22b0c-85f5-48c6-8a10-4d894c99b7b1" />
   
####    3iii.   SHAP Summary Plot

SHAP summary plot showing both the magnitude and direction of each feature’s influence on the model’s predictions for diabetes

<img width="702" height="284" alt="Screenshot 2026-01-14 125432" src="https://github.com/user-attachments/assets/54955dbb-bb5d-4e3c-87b6-9a44a0c4c404" />









