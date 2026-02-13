### 1. Introduction

<img width="612" height="408" alt="image" src="https://github.com/user-attachments/assets/115351a8-1a7f-4490-a144-f20e97e7c10f" />


Diabetes, which is one of the dominant and chronic diseases across the globe, comes with life-threatening challenges if it is not detected early enough. Timely diagnosis with risk prediction is an effective way of preventing and managing the complications associated with the disease. In this study, machine learning techniques were utilized in developing predictive models that were capable of detecting individuals who are at the risk of having diabetes based on key factors such as health and lifestyle. 

Robust machine learning models were optimized for predicting diabetes in individuals from the onset based on data collated on their lifestyle and health. The methodology employed incorporated comprehensive data preprocessing, exploratory data analysis, multiple supervised learning algorithms such as Logistic Regression, Random Forest, Gradient Boosting etc were trained and hyperparameter tuning and k-fold cross-validation were employed to enhance their performance. Each model were evaluated for their performance before and after hyper parameter tuning and k-fold cross-validation using metrics such as accuracy, precision, recall and F1-score.

Feature importance analysis and SHAP (SHapley Additive exPlanations) values were also utilized to enhance model interpretability and provide transparency in model decision-making. These techniques via their ability to identify the most influential diabetes predictors, assist in developing reliable and interpretable predictive models which can aid early risk detection, preventive interventions, allocation of healthcare resources and minimising the global burden of diabetes


### 2. Problem Statement

Globally, diabetes mellitus is one of the fastest-growing chronic diseases with serious health complications such as cardiovascular disease, nerve damage and kidney failure. A large number of people remain undiagnosed with it, despite the insignificant progress that has been made in healthcare. This is due to the heavy dependence on traditional diagnostic methods, which often rely on laboratory tests and reduces the chances of detecting and preventing the disease early

Utilizing machine learning techniques has paved the way for addressing this gap by its ability to identify hidden patterns in datasets useful for the early prediction of diabetes. However, the challenge lies in the selection of the most efficient model that can achieve a high accuracy while reducing false predictions to the barest minimum. This project seeks to evaluate various machine learning algorithms to determine the most reliable model suitable for the prediction of diabetes based on key health indicators


### 3. Significance of the Study

  -  The early detection of diabetes for timely medical intervention and lifestyle changes.
    
  -  Data-driven decisions via predictive insights derived from health datasets.
    
  -  Identification of the most accurate and reliable model for improved diagnostic precision and decision-making.
    
  -  Reduced time and cost related to traditional screening and diagnostic methods.

This research aims to highlight the potential of using artificial intelligence and data science to transform healthcare via predictive analytics and data-driven solutions

### 4. Methodology Overview

A systematic, step-by-step approach was utilized in this project for the prediction of diabetes using machine learning techniques. The processes involved from data acquisition to model interpretation and explainability are outlined below:

- **Data Acquisition:** The dataset which was gotten from kaggle had relevant patient information such as age, gender, BMI, blood glucose level, HbA1c level, smoking history, and diabetes status. 

- **Data Cleaning and Preprocessing:** Data cleaning process involved handling missing values, removing duplicates and correcting data types. Encoding of categorical variables and normalization of numerical features were done were appropriate. 

-	**Exploratory Data Analysis (EDA):** This was performed to gain insights into how the data was distributed and the relationships among variables. Data was visualized using visualization tools such as histograms, bar plots, and scatter plots which were used to examine the trends and understand how features relate to diabetes occurrence.

-	**Feature Engineering:**  involved improving the model’s predictive capability by transforming existing features and creating new ones 

- **Model Development:** Several machine learning algorithms were trained and tested on the dataset, including:

  - Logistic Regression (LR)
    
  - Support Vector Classifier (SVC)
    
  - Decision Tree (DT)
    
  - Random Forest (RF)
    
  - Stochastic Gradient Descent (SGD)
  
  - Extreme Gradient Boosting (XGBoost)

-	**Hyperparameter Tuning:** Each model was fine-tuned to enhance its performance and predictive strength. GridSearchCV was used to optimize the model’s performance metric

- **Model Evaluation:** Model performance was evaluated utilizing evaluation metrics such as accuracy, precision, recall, and f1-Score. Confusion matrices were also produced, which help to visualize the true positives, false positives, true negatives, and false negatives. Cross-validation was applied to ensure the models’ stability and consistency

- **Model Interpretation and Explainability:** Feature importance and SHAP values were employed to evaluate the extent to which each feature affects the model’s prediction.

This project employed a systematic, step-by-step approach to predict the likelihood of diabetes using machine learning techniques. The methodology outlines all processes from data acquisition to model interpretation and explainability to ensure accuracy, reliability, and reproducibility.

### 5. Libraries and tools	
- **Data Manipulation and Analysis:**
  - Pandas: For cleaning, manipulation, and structuring data.
  - NumPy: For efficient numerical operations and array computations.
- **Data Visualization:**
  - Matplotlib: For generating static, interactive, and animated plots.
  - Seaborn: For creating refined and informative statistical graphics.
- **Machine Learning:**
  - Scikit-Learn: Core library for preprocessing, model training, evaluation, and hyperparameter tuning.

###  6. Results
###  Project Visualizations
####     6a. Feature Correlation Matrix

Correlation matrix showing relationships between key diabetic features.
<img width="843" height="490" alt="Screenshot 2025-10-26 055229" src="https://github.com/user-attachments/assets/fac82e54-ec0a-408b-84f3-74e6019114c5" />

The correlation analysis indicated that advancing age was moderately associated with higher BMI (r = 0.34) and showed positive relationships with hypertension (r = 0.25), heart disease (r = 0.23), and diabetes (r = 0.26), suggesting increased cardiometabolic risk with aging. Hypertension and heart disease were positively but weakly correlated (r = 0.12), reflecting a modest yet clinically relevant cardiovascular link. Diabetes showed its strongest associations with blood glucose level (r = 0.42) and HbA1c level (r = 0.40), underscoring their central role in diabetes diagnosis and risk assessment. BMI was also moderately correlated with diabetes (r = 0.21), reinforcing excess body weight as an important risk factor. Overall, diabetes was most strongly associated with elevated blood glucose and HbA1c levels, followed by older age, higher BMI, and hypertension.

####    6b. Classification Models

####      6bi. Classification Models Performance Evaluation: 

A suite of machine learning classifiers (logistic regression (LR), stochastic gradient descent (SGD), decision tree (DT), k-nearest neighbors (KNN), random forest (RF), and extreme gradient boosting (XGBoost)) were evaluated. A comparison of accuracy, precision, recall, F1 and receiver operating characteristic area under the curve (ROC-AUC) across the different models before and after model optimization via hyper parameter tuning, class weight and threshold optimization was performed.


<img width="983" height="327" alt="Screenshot 2026-01-14 040741" src="https://github.com/user-attachments/assets/43d72c53-f7f2-48e9-94bf-a5c7421c9917" />


Across all evaluated models, XGBoost achieved the best overall performance, demonstrating the strongest balance between precision, recall, and discrimination, with the highest ROC-AUC and F1-score under both default and optimized thresholds and consistently high precision, indicating stable calibration and effective modeling of non-linear feature interactions. Random forest also showed strong discriminative ability but was more sensitive to threshold adjustments, shifting from a conservative to a recall-oriented classifier after optimization. In contrast, linear models (logistic regression and SGD) exhibited stable performance across thresholds with high recall but lower precision, reflecting the limitations of linear decision boundaries, while KNN and decision tree models showed weaker robustness, with minimal gains from threshold optimization for KNN and increased instability for decision trees due to coarse probability estimates.


####      6bii.  ROC Curves 
The plot presents the receiver operating characteristic (ROC) curves for the six optimized classification models evaluated on the test dataset

<img width="625" height="447" alt="Screenshot 2026-01-14 111302" src="https://github.com/user-attachments/assets/a5de1076-f0a7-4356-b9d4-46be0eca180f" />

All models achieved high AUC scores above 0.92, indicating strong discriminative ability.

####      6biii.  Feature Importance

Feature Importance bar plot showing the impact of features on XGBoost model's performance by ranking their influence on the model's output.

<img width="606" height="361" alt="Screenshot 2026-01-14 044710" src="https://github.com/user-attachments/assets/5aced5ba-3d78-408a-a06c-26368d04f7c8" />


Feature importance showed that the XGBoost model relied heavily on blood sugar measurements (HbA1c and glucose levels), which together account for a high percentage of the total importance. This suggests that the model was primarily using clinical indicators of glucose metabolism to make its predictions, while other factors played supporting roles.

####      6biv.  SHAP analysis
Mean absolute SHAP values for the top 5 features influencing XGBoost model's predictions of diabetes

<img width="695" height="387" alt="Screenshot 2026-01-14 045659" src="https://github.com/user-attachments/assets/b76dd287-f193-45b8-a812-f193b1e865f4" />


The SHAP (SHapley Additive exPlanations) feature importance chart for the XGBoost model, showed the top 5 most important features based on their average impact on model predictions. The ranking was consistent between both the feature importance and SHAP charts, but SHAP values reveal that age had a more significant role than simple feature importance suggested. The dominance of HbA1c was more pronounced in this SHAP analysis, emphasizing that blood sugar control was the primary driver of the model's predictions.


####      6bv.  SHAP Summary Plot

SHAP summary (beeswarm) plot for top five features 

<img width="690" height="291" alt="Screenshot 2026-01-14 133216" src="https://github.com/user-attachments/assets/b1dd5490-3457-4d19-a68a-65908a8e3ffd" />

The beeswarm plot revealed that high HbA1c and blood glucose levels were strong, consistent drivers of positive predictions. Age and BMI showed expected directional effects (higher = higher risk), but with more variability. The tight clustering of glucose-related features versus the spread in age suggested that the model relied most heavily on direct metabolic markers while using demographic factors as modifying influences.

####  6c.  Regression Models 

####      6ci. Regression Models Performance Evaluation: 

Six regression models were evaluated using cross-validated root mean squared error (CV RMSE) and independent test-set metrics, including coefficient of determination (R²), absolute and squared error measures, and correlation-based statistics

<img width="860" height="250" alt="Screenshot 2026-01-14 060603" src="https://github.com/user-attachments/assets/b33be3c9-ce77-458b-908d-738658c04b61" />

The table compared six regression models using cross-validated RMSE and independent test-set metrics, including error measures and correlation statistics, and showed a clear advantage for tree-based approaches over linear and regularized regressions. The Random Forest regressor delivered the strongest performance, explaining 71.0% of the outcome variance (R² = 0.710), achieving the lowest errors (MAE = 0.045; RMSE = 0.150), and exhibiting a strong linear agreement between predicted and observed values (Pearson r = 0.843, p < 0.001). The Decision Tree regressor performed comparably (R² = 0.702; RMSE = 0.152), underscoring the importance of nonlinear decision rules, though its slightly weaker results reflected the expected variance reduction benefit of ensemble averaging in Random Forests. In contrast, linear and regularized models (Linear, Ridge, Lasso, Elastic Net) performed substantially worse, explaining only about 32% of the variance (R² ≈ 0.319) with higher prediction errors (RMSE ≈ 0.231), indicating that regularization did not overcome model misspecification and that the predictor–outcome relationship is largely nonlinear. Correlation analyses reinforce these conclusions, with tree-based models showing markedly stronger Pearson correlations, while Spearman correlations were moderate across all models, reflecting only partial monotonic agreement.

####     6cii. Feature Importance
<img width="556" height="318" alt="Screenshot 2026-01-14 121041" src="https://github.com/user-attachments/assets/09e22b0c-85f5-48c6-8a10-4d894c99b7b1" />

The feature importance of the Random Forest Regressor model showed that it also relied heavily on direct metabolic markers (blood sugar measurements), treating demographic and comorbidity factors as nearly irrelevant.
   
####    6ciii.   SHAP Summary Plot

SHAP summary plot showing both the magnitude and direction of each feature’s influence on the model’s predictions for diabetes

<img width="702" height="284" alt="Screenshot 2026-01-14 125432" src="https://github.com/user-attachments/assets/54955dbb-bb5d-4e3c-87b6-9a44a0c4c404" />


The SHAP summary plot for the Random Forest regression model showed that predictions were primarily driven by metabolic indicators, with HbA1c and blood glucose emerging as the most influential features, where higher values consistently increased predicted risk and lower values reduce it. Features are ranked by mean absolute SHAP value, highlighting a clear hierarchy in which glycemic markers dominated the model's decisions, followed by moderate contributions from age and smaller, incremental effects from BMI. Hypertension has the least influence, with SHAP values clustered near zero, indicating minimal marginal impact once other factors were considered. 


### Conclusion
Across correlation, classification, and regression analyses, the findings consistently demonstrate that diabetes prediction is driven primarily by direct metabolic indicators (HbA1c and blood glucose). Age and BMI contribute meaningfully but secondarily, while hypertension and heart disease show modest additional influence. From a modeling perspective, tree-based ensemble methods (XGBoost and Random Forest) substantially outperform linear approaches, indicating that nonlinear interactions among predictors are central to accurately modeling diabetes risk. The convergence of statistical correlation, machine learning performance, and SHAP-based explainability strengthens the robustness and clinical interpretability of these findings.




