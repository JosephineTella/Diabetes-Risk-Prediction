# Diabetes-Risk-Prediction

<img width="285" height="190" alt="image" src="https://github.com/user-attachments/assets/bad9a0fd-267b-4c7a-87f8-dcd39e6cb335" />

### 1. Introduction

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
- **Model Optimization:**
  - XGBoost: High-performance Gradient Boosting library with scalability and accuracy.
  - Logistic Regression, Decision Tree, Random Forest, SGDClassifier: Core algorithms for baseline and optimized predictive modeling.
- **Model Explainability:**
  - Feature Importance: Used to identify the relative contribution of each predictor variable to the model’s performance.
  - SHAP (SHapley Additive exPlanations): Provides model interpretability by explaining individual predictions and highlighting the impact of features on risk estimation.
- **Development Environment and Workflow:**
  - Jupyter Notebook: Interactive platform for code development, analysis, and documentation.
  - Anaconda: Comprehensive distribution for efficient package management and deployment in data science projects.

### 6. Project Visualizations
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

### 7. Results

Key findings include:

  - **Correlation matrix:** the correlation analysis highlights that diabetes is most strongly associated with elevated HbA1c and blood glucose levels, followed by higher BMI, older age and hypertension.

  - **Model Evaluation:** model performance revealed random forest as the best performing model 
  
  - **Feature Importance:** Key indicators of diabetes prediction included HbA1c level, blood glucose level, age, bmi and hypertension 

### 8. Future Work

  -  Incorporating deep learning architectures such as attention-based neural networks) with interpretable frameworks such as SHAP or LIME might provide transparency and predictive power

### 9. Conclusion

This study developed and evaluated various machine learning models to predict diabetes cases with the aid of structured clinical data and specific focus on the interpretability of the best-performing model. Random Forest Classifier had the best predictive performance across the various metrics tested. 

Application of SHAP (SHapley Additive exPlanations) made the model’s decision-making process transparent and interpretable. It identified the most influential features contributing to diabetes cases as HbA1c level, blood glucose level, BMI, age and hypertension. The feature importance ranking aligned well with established clinical evidence, thereby strengthening the biological and diagnostic validity of the model’s predictions.


### References

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608.
Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. Nature Machine Intelligence, 1(9), 389–399. https://doi.org/10.1038/s42256-019-0088-2

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6), 1–35. https://doi.org/10.1145/3457607

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. Artificial Intelligence, 267, 1–38. https://doi.org/10.1016/j.artint.2018.07.007

Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. New England Journal of Medicine, 380(14), 1347–1358. https://doi.org/10.1056/NEJMra1814259

Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence. Nature Medicine, 25(1), 44–56. https://doi.org/10.1038/s41591-018-0300-7

Zou, J., Schiebinger, L., & Obermeyer, Z. (2023). Ensuring fairness in machine learning for health care. Nature Reviews Genetics, 24(3), 173–188. https://doi.org/10.1038/s41576-022-00560-1
