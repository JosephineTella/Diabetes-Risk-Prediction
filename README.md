
###  Project Visualizations
####     1. Feature Correlation Matrix

Correlation matrix showing relationships between key diabetic features.
<img width="843" height="490" alt="Screenshot 2025-10-26 055229" src="https://github.com/user-attachments/assets/fac82e54-ec0a-408b-84f3-74e6019114c5" />

The correlation analysis indicated that advancing age was moderately associated with higher BMI (r = 0.34) and showed positive relationships with hypertension (r = 0.25), heart disease (r = 0.23), and diabetes (r = 0.26), suggesting increased cardiometabolic risk with aging. Hypertension and heart disease were positively but weakly correlated (r = 0.12), reflecting a modest yet clinically relevant cardiovascular link. Diabetes showed its strongest associations with blood glucose level (r = 0.42) and HbA1c level (r = 0.40), underscoring their central role in diabetes diagnosis and risk assessment. BMI was also moderately correlated with diabetes (r = 0.21), reinforcing excess body weight as an important risk factor. Overall, diabetes was most strongly associated with elevated blood glucose and HbA1c levels, followed by older age, higher BMI, and hypertension.

####     2. Classification Models

####      2i. Classification Models Performance Evaluation: 

A suite of machine learning classifiers (logistic regression (LR), stochastic gradient descent (SGD), decision tree (DT), k-nearest neighbors (KNN), random forest (RF), and extreme gradient boosting (XGBoost)) were evaluated. A comparison of accuracy, precision, recall, F1 and receiver operating characteristic area under the curve (ROC-AUC) across the different models before and after model optimization via hyper parameter tuning, class weight and threshold optimization was performed.


<img width="983" height="327" alt="Screenshot 2026-01-14 040741" src="https://github.com/user-attachments/assets/43d72c53-f7f2-48e9-94bf-a5c7421c9917" />


Across all evaluated models, XGBoost achieved the best overall performance, demonstrating the strongest balance between precision, recall, and discrimination, with the highest ROC-AUC and F1-score under both default and optimized thresholds and consistently high precision, indicating stable calibration and effective modeling of non-linear feature interactions. Random forest also showed strong discriminative ability but was more sensitive to threshold adjustments, shifting from a conservative to a recall-oriented classifier after optimization. In contrast, linear models (logistic regression and SGD) exhibited stable performance across thresholds with high recall but lower precision, reflecting the limitations of linear decision boundaries, while KNN and decision tree models showed weaker robustness, with minimal gains from threshold optimization for KNN and increased instability for decision trees due to coarse probability estimates.


####      2ii.  ROC Curves 
The plot presents the receiver operating characteristic (ROC) curves for the six optimized classification models evaluated on the test dataset

<img width="625" height="447" alt="Screenshot 2026-01-14 111302" src="https://github.com/user-attachments/assets/a5de1076-f0a7-4356-b9d4-46be0eca180f" />

All models achieved high AUC scores above 0.92, indicating strong discriminative ability.

####      2iii.  Feature Importance

Feature Importance bar plot showing the impact of features on XGBoost model's performance by ranking their influence on the model's output.

<img width="606" height="361" alt="Screenshot 2026-01-14 044710" src="https://github.com/user-attachments/assets/5aced5ba-3d78-408a-a06c-26368d04f7c8" />


Feature importance showed that the XGBoost model relied heavily on blood sugar measurements (HbA1c and glucose levels), which together account for a high percentage of the total importance. This suggests that the model was primarily using clinical indicators of glucose metabolism to make its predictions, while other factors played supporting roles.

####      2iv.  SHAP analysis
Mean absolute SHAP values for the top 5 features influencing XGBoost model's predictions of diabetes

<img width="695" height="387" alt="Screenshot 2026-01-14 045659" src="https://github.com/user-attachments/assets/b76dd287-f193-45b8-a812-f193b1e865f4" />


The SHAP (SHapley Additive exPlanations) feature importance chart for the XGBoost model, showed the top 5 most important features based on their average impact on model predictions. The ranking was consistent between both the feature importance and SHAP charts, but SHAP values reveal that age had a more significant role than simple feature importance suggested. The dominance of HbA1c was more pronounced in this SHAP analysis, emphasizing that blood sugar control was the primary driver of the model's predictions.


####      2v.  SHAP Summary Plot

SHAP summary (beeswarm) plot for top five features 

<img width="690" height="291" alt="Screenshot 2026-01-14 133216" src="https://github.com/user-attachments/assets/b1dd5490-3457-4d19-a68a-65908a8e3ffd" />

The beeswarm plot revealed that high HbA1c and blood glucose levels were strong, consistent drivers of positive predictions. Age and BMI showed expected directional effects (higher = higher risk), but with more variability. The tight clustering of glucose-related features versus the spread in age suggested that the model relied most heavily on direct metabolic markers while using demographic factors as modifying influences.

####  3.  Regression Models 

####      3i. Regression Models Performance Evaluation: 

Six regression models were evaluated using cross-validated root mean squared error (CV RMSE) and independent test-set metrics, including coefficient of determination (R²), absolute and squared error measures, and correlation-based statistics

<img width="860" height="250" alt="Screenshot 2026-01-14 060603" src="https://github.com/user-attachments/assets/b33be3c9-ce77-458b-908d-738658c04b61" />

The table compared six regression models using cross-validated RMSE and independent test-set metrics, including error measures and correlation statistics, and showed a clear advantage for tree-based approaches over linear and regularized regressions. The Random Forest regressor delivered the strongest performance, explaining 71.0% of the outcome variance (R² = 0.710), achieving the lowest errors (MAE = 0.045; RMSE = 0.150), and exhibiting a strong linear agreement between predicted and observed values (Pearson r = 0.843, p < 0.001). The Decision Tree regressor performed comparably (R² = 0.702; RMSE = 0.152), underscoring the importance of nonlinear decision rules, though its slightly weaker results reflected the expected variance reduction benefit of ensemble averaging in Random Forests. In contrast, linear and regularized models (Linear, Ridge, Lasso, Elastic Net) performed substantially worse, explaining only about 32% of the variance (R² ≈ 0.319) with higher prediction errors (RMSE ≈ 0.231), indicating that regularization did not overcome model misspecification and that the predictor–outcome relationship is largely nonlinear. Correlation analyses reinforce these conclusions, with tree-based models showing markedly stronger Pearson correlations, while Spearman correlations were moderate across all models, reflecting only partial monotonic agreement.

####     3ii. Feature Importance
<img width="556" height="318" alt="Screenshot 2026-01-14 121041" src="https://github.com/user-attachments/assets/09e22b0c-85f5-48c6-8a10-4d894c99b7b1" />

The feature importance of the Random Forest Regressor model showed that it also relied heavily on direct metabolic markers (blood sugar measurements), treating demographic and comorbidity factors as nearly irrelevant.
   
####    3iii.   SHAP Summary Plot

SHAP summary plot showing both the magnitude and direction of each feature’s influence on the model’s predictions for diabetes

<img width="702" height="284" alt="Screenshot 2026-01-14 125432" src="https://github.com/user-attachments/assets/54955dbb-bb5d-4e3c-87b6-9a44a0c4c404" />


The SHAP summary plot for the Random Forest regression model showed that predictions were primarily driven by metabolic indicators, with HbA1c and blood glucose emerging as the most influential features, where higher values consistently increased predicted risk and lower values reduce it. Features are ranked by mean absolute SHAP value, highlighting a clear hierarchy in which glycemic markers dominated the model's decisions, followed by moderate contributions from age and smaller, incremental effects from BMI. Hypertension has the least influence, with SHAP values clustered near zero, indicating minimal marginal impact once other factors were considered. 







