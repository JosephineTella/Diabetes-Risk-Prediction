
###  Project Visualizations
####     1. Feature Correlation Matrix
<img width="843" height="490" alt="Screenshot 2025-10-26 055229" src="https://github.com/user-attachments/assets/fac82e54-ec0a-408b-84f3-74e6019114c5" />

Correlation matrix showing relationships between key diabetic features.


####     2. Classification Models

####      2i. Classification Models Performance Evaluation: 

A suite of machine learning classifiers including logistic regression (LR), stochastic gradient descent (SGD), decision tree (DT), k-nearest neighbors (KNN), random forest (RF), and extreme gradient boosting (XGBoost) were evaluated. A comparison of accuracy, precision, recall, F1 and receiver operating characteristic area under the curve (ROC-AUC) across the different models before and after model optimization via hyper parameter tuning, class weight and threshold optimization was performed.

<img width="983" height="327" alt="Screenshot 2026-01-14 040741" src="https://github.com/user-attachments/assets/43d72c53-f7f2-48e9-94bf-a5c7421c9917" />

####        Insights

•	Among all evaluated models, XGBoost demonstrated the most favorable balance between precision, recall, and overall discrimination. It achieved the highest ROC-AUC and F1-score under both default and optimized thresholds, while maintaining exceptionally high precision even after threshold adjustment. This stability suggests well-calibrated probability estimates and robust modeling of non-linear feature interactions.

•	Random forest also exhibited strong discriminative performance but showed greater sensitivity to threshold changes, transitioning from a conservative classifier under the default threshold to a highly recall-oriented model after optimization. Linear models (LR and SGD) displayed comparatively stable behavior across thresholds, with consistently high recall but limited precision, reflecting the constraints of linear decision boundaries.

•	KNN and decision tree models demonstrated weaker robustness, with threshold optimization offering limited benefit for KNN and inducing instability in decision trees due to coarse probability estimates.


####      2ii.  ROC Curves 

<img width="625" height="447" alt="Screenshot 2026-01-14 111302" src="https://github.com/user-attachments/assets/a5de1076-f0a7-4356-b9d4-46be0eca180f" />

The plot presents the receiver operating characteristic (ROC) curves for six optimized classification models evaluated on the test dataset. The ROC analysis demonstrates that all models achieve substantial discriminative ability, with areas under the curve (AUCs) well above the no-discrimination reference line (AUC = 0.5), confirming robust separation between positive and negative classes following optimization.

####  Insight

•	Among the evaluated models, the Random Forest classifier achieves the highest discriminative performance (AUC = 0.978), exhibiting a consistently dominant ROC curve across the full range of false positive rates. This indicates superior sensitivity–specificity trade-offs and highlights the benefit of ensemble learning in capturing nonlinear relationships within the data. Closely following is the XGBoost model (AUC = 0.976), whose curve nearly overlaps that of the Random Forest, suggesting comparable and near-optimal classification performance.

•	The Decision Tree model (AUC = 0.973) also demonstrates strong discrimination, particularly at lower false positive rates, but its performance is marginally inferior to the ensemble methods, reflecting reduced stability and generalization capacity. In contrast, the Logistic Regression and SGD classifiers (both AUC = 0.961) show nearly identical ROC profiles, consistent with their shared linear modeling assumptions. While these models achieve high overall discrimination, they require comparatively higher false positive rates to attain very high true positive rates, indicating more conservative class separation.

•	The k-Nearest Neighbors (k-NN) model exhibits the lowest performance (AUC = 0.922), with a flatter ROC curve, particularly in the low false positive region. This suggests reduced ranking ability and less reliable probability estimates relative to the other approaches.

•	Overall, the ROC analysis indicates that ensemble-based methods (Random Forest and XGBoost) provide the most robust and clinically relevant discrimination, making them the most suitable candidates for high-stakes prediction tasks and downstream interpretability analyses such as SHAP. Linear models remain competitive as transparent baselines, whereas k-NN appears less optimal for this application


####      2iii.  Feature Importance

<img width="606" height="361" alt="Screenshot 2026-01-14 044710" src="https://github.com/user-attachments/assets/5aced5ba-3d78-408a-a06c-26368d04f7c8" />

From the plot, XGBoost model relied primarily on blood sugar-related measures (HbA1c and blood glucose) which suggests that for this model and dataset, biochemical indicators are far more predictive than factors like gender or smoking history.


####      2iv.  SHAP analysis
Mean absolute SHAP values for the top 5 features influencing XGBoost model's predictions of diabetes

<img width="695" height="387" alt="Screenshot 2026-01-14 045659" src="https://github.com/user-attachments/assets/b76dd287-f193-45b8-a812-f193b1e865f4" />


####      2v.  Beeswarm Summary Plot

<img width="695" height="387" alt="Screenshot 2026-01-14 045659" src="https://github.com/user-attachments/assets/938ee209-e453-4e06-b9c5-46ffdfd61446" />

The SHAP summary (beeswarm) plot for the top five features shows both the importance and the direction of influence of each feature on the model's predictions. It shows that elevated HbA1c and blood glucose levels are the strongest drivers of increased model predictions, while age has a moderate effect and BMI and hypertension contribute minimally


####  3.  Regression Models Performance Evaluation

####      3i. Regression Models Performance Evaluation: 

Six regression models were evaluated using cross-validated root mean squared error (CV RMSE) and independent test-set metrics, including coefficient of determination (R²), absolute and squared error measures, and correlation-based statistics

<img width="860" height="250" alt="Screenshot 2026-01-14 060603" src="https://github.com/user-attachments/assets/b33be3c9-ce77-458b-908d-738658c04b61" />

####    Insights

•	Among the evaluated models, the Random Forest classifier achieves the highest discriminative performance (AUC = 0.978), exhibiting a consistently dominant ROC curve across the full range of false positive rates. This indicates superior sensitivity–specificity trade-offs and highlights the benefit of ensemble learning in capturing nonlinear relationships within the data. Closely following is the XGBoost model (AUC = 0.976), whose curve nearly overlaps that of the Random Forest, suggesting comparable and near-optimal classification performance.


•	The Decision Tree model (AUC = 0.973) also demonstrates strong discrimination, particularly at lower false positive rates, but its performance is marginally inferior to the ensemble methods, reflecting reduced stability and generalization capacity. In contrast, the Logistic Regression and SGD classifiers (both AUC = 0.961) show nearly identical ROC profiles, consistent with their shared linear modeling assumptions. While these models achieve high overall discrimination, they require comparatively higher false positive rates to attain very high true positive rates, indicating more conservative class separation.


•	The k-Nearest Neighbors (k-NN) model exhibits the lowest performance (AUC = 0.922), with a flatter ROC curve, particularly in the low false positive region. This suggests reduced ranking ability and less reliable probability estimates relative to the other approaches.


•	Overall, the ROC analysis indicates that ensemble-based methods (Random Forest and XGBoost) provide the most robust and clinically relevant discrimination, making them the most suitable candidates for high-stakes prediction tasks and downstream interpretability analyses such as SHAP. Linear models remain competitive as transparent baselines, whereas k-NN appears less optimal for this application.


####     3ii. Feature Importance
<img width="556" height="318" alt="Screenshot 2026-01-14 121041" src="https://github.com/user-attachments/assets/09e22b0c-85f5-48c6-8a10-4d894c99b7b1" />

 The model relied mainly on Blood sugar-related measures (HbA1c and blood glucose) for its predicitions less on BMI and age with no reliance on gender    

####    3iii.   SHAP Summary Plot

SHAP summary plot showing both the magnitude and direction of each feature’s influence on the model’s predictions for diabetes

<img width="702" height="284" alt="Screenshot 2026-01-14 125432" src="https://github.com/user-attachments/assets/54955dbb-bb5d-4e3c-87b6-9a44a0c4c404" />

The SHAP analysis reveals that the Random forest regression model is also primarily driven by metabolic indicators, particularly HbA1c and blood glucose, with demographic and clinical factors such as age, BMI, and hypertension playing progressively smaller roles. This hierarchy aligns well with established clinical understanding and supports the model’s interpretability and face validity for health-related risk prediction







