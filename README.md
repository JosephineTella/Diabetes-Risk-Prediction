###  Introduction

Diabetes, which is one of the dominant and chronic diseases across the globe, comes with life-threatening challenges if it is not detected early enough. Timely diagnosis with risk prediction is an effective way of preventing and managing the complications associated with the disease. In this study, machine learning techniques were utilized in developing predictive models that were capable of detecting individuals who are at the risk of having diabetes based on key factors such as health and lifestyle. 

Robust machine learning models were optimized for predicting diabetes in individuals from the onset based on data collated on their lifestyle and health. The methodology employed incorporated comprehensive data preprocessing, exploratory data analysis, multiple supervised learning algorithms such as Logistic Regression, Random Forest, Gradient Boosting etc were trained and hyperparameter tuning and k-fold cross-validation were employed to enhance their performance. Each model were evaluated for their performance before and after hyper parameter tuning and k-fold cross-validation using metrics such as accuracy, precision, recall and F1-score.

Feature importance analysis and SHAP (SHapley Additive exPlanations) values were also utilized to enhance model interpretability and provide transparency in model decision-making. These techniques via their ability to identify the most influential diabetes predictors, assist in developing reliable and interpretable predictive models which can aid early risk detection, preventive interventions, allocation of healthcare resources and minimising the global burden of diabetes


###  Project Visualizations
####     i. Feature Correlation Matrix
<img width="843" height="490" alt="Screenshot 2025-10-26 055229" src="https://github.com/user-attachments/assets/fac82e54-ec0a-408b-84f3-74e6019114c5" />

Correlation matrix showing relationships between key diabetic features.

####     ii. Model Performance Metrics
<img width="864" height="409" alt="Screenshot 2025-10-26 055410" src="https://github.com/user-attachments/assets/42d12858-897d-422c-97f7-8ae85b77007d" />

Comparison of accuracy, precision, recall and F1 scores across different models before and after model optimization via hyper parameter tuning and class weight

####   iii. Feature Importances
<img width="704" height="304" alt="Screenshot 2025-10-26 204530" src="https://github.com/user-attachments/assets/b5ea395e-84bf-4f8d-a29d-2923b70577f8" />

Bar plot providing insights into how features contribute to model predicition

<img width="841" height="368" alt="Screenshot 2025-10-26 204653" src="https://github.com/user-attachments/assets/addbaa71-dd7d-4c68-880a-de6d6f8226d8" />

Mean absolute SHAP values for the top 5 features influencing the model’s predictions of diabetes

<img width="745" height="261" alt="Screenshot 2025-10-26 204720" src="https://github.com/user-attachments/assets/bee83d91-ccfc-4852-8fad-9cfb3a1caba5" />

SHAP summary plot showing both the magnitude and direction of each feature’s influence on the model’s predictions for diabetes




