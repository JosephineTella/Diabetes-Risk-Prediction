# Diabetes-Risk-Prediction

<img width="285" height="190" alt="image" src="https://github.com/user-attachments/assets/bad9a0fd-267b-4c7a-87f8-dcd39e6cb335" />

### 1. Introduction
Diabetes is one of the most prevalent chronic diseases worldwide, posing serious health challenges if not detected early. Timely diagnosis and risk prediction are essential for effective management and prevention of complications. In this project, machine learning techniques are applied to develop predictive models capable of identifying individuals at risk of diabetes based on key health and lifestyle factors such as age, BMI, blood glucose levels, and smoking history.

This study presents the development of a robust machine learning framework for predicting the onset of diabetes in individuals based on health and lifestyle data. The methodology incorporates comprehensive data preprocessing, including missing value imputation, normalization of numerical features, and encoding of categorical variables. Exploratory data analysis is conducted to identify trends, correlations, and anomalies that may inform predictive modeling. Multiple supervised learning algorithms—including Logistic Regression, Random Forest, and Gradient Boosting—are trained, with hyperparameter tuning and k-fold cross-validation employed to optimize performance. Model evaluation is conducted using established metrics such as accuracy, precision, recall and F1-score

As a supportive component of the methodology, feature importance analysis and SHAP (SHapley Additive exPlanations) values are applied to enhance interpretability and provide transparency in model decision-making. These techniques assist in identifying the most influential predictors of diabetes onset, offering supplementary insights for clinical interpretation. The anticipated outcome is the development of a reliable and interpretable predictive model that facilitates early risk identification, supports preventive interventions, improves healthcare resource allocation, and contributes to reducing the global burden of diabetes.

### 2. Problem Statement
Diabetes mellitus is one of the fastest-growing chronic diseases worldwide, leading to severe health complications such as cardiovascular disease, kidney failure, and nerve damage. Despite significant advances in healthcare, many individuals remain undiagnosed until symptoms become severe. Traditional diagnostic methods often rely on laboratory testing after clinical signs have appeared, which limits early detection and prevention.

Machine learning provides an opportunity to address this gap by identifying hidden patterns in health data that can predict diabetes risk at an early stage. However, the challenge lies in selecting the most effective model that achieves high accuracy and generalization while minimizing false predictions. This project, therefore, seeks to evaluate multiple machine learning algorithms to determine the most reliable model for diabetes prediction based on key health indicators.

### 3. Significance of the Study

•	    Early Detection: Enabling timely medical intervention and lifestyle changes before the onset of severe symptoms.

•	    Data-Driven Decision-Making: Supporting clinicians with predictive insights derived from historical health data.

•	    Model Optimization: Identifying the most accurate and interpretable model to improve diagnostic precision.

•	    Resource Efficiency: Reducing the cost and time associated with traditional screening methods.

•	    Public Health Impact: Contributing to preventive medicine efforts by helping identify high-risk individuals in the population.

Ultimately, this research emphasizes the potential of artificial intelligence and data science to transform healthcare through predictive analytics and evidence-based solutions

### 4. Methodology Overview

This project employed a systematic, step-by-step approach to predict the likelihood of diabetes using machine learning techniques. The methodology outlines all processes from data acquisition to model interpretation and explainability to ensure accuracy, reliability, and reproducibility.

- **Data Acquisition:**
The dataset used for this project was obtained from kaggle and it contains relevant patient information such as age, gender, BMI, blood glucose level, HbA1c level, smoking history, and diabetes status. The dataset forms the foundation for model training and evaluation.

- **Data Cleaning and Preprocessing:**
Data cleaning involved handling missing values, removing duplicates, correcting data types, and addressing outliers. Categorical variables such as gender and smoking history were encoded, and numerical features were normalized where appropriate. These steps ensured that the dataset was consistent, complete, and suitable for model training.

- **Exploratory Data Analysis (EDA):**
EDA was performed to gain insights into the data distribution and relationships among variables. Visualization tools such as histograms, bar plots, and scatter plots were used to examine trends (e.g., the relationship between age, BMI, and blood glucose levels) and to understand how features relate to diabetes occurrence. 

- **Feature Engineering:**
Feature engineering involved transforming existing features and creating new ones to enhance the model’s predictive capability.

- **Model Development:**
Several machine learning algorithms were trained and tested on the dataset, including:
  -  Logistic Regression (LR)
  -  Support Vector Classifier (SVC)
  -  Decision Tree (DT)
  -  Random Forest (RF)
  -  Stochastic Gradient Descent (SGD)
  -  Extreme Gradient Boosting (XGBoost)

- **Hyperparameter Tuning:**
Each model was fine-tuned using GridSearchCV to optimize performance metrics, including accuracy, precision, recall, and F1-score. The tuning process aims to enhance model performance and predictive strength.

- **Model Evaluation:**
Model performance was assessed using multiple evaluation metrics, including Accuracy, Precision, Recall (Sensitivity), and F1-Score. Confusion matrices were generated to visualize true positive, false positive, true negative, and false negative rates. Cross-validation was also applied to ensure the models’ stability and consistency. The best-performing model is selected based on balanced performance across all metrics, with special attention to recall (to detect diabetic cases accurately).

- **Model Interpretation and Explainability:**
Feature importance and SHAP values were used to understand how each feature affects predictions. This step ensures transparency, builds trust, and provides insights into which factors most influence diabetes risk.

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

Potential areas of future development include:

-  integrating multi-modal data sources, combining clinical, biochemical, and genomic data to develop more robust and personalized predictive models for diabetes. The application of deep learning architectures (e.g., attention-based neural networks) with interpretable frameworks like SHAP or LIME could provide both predictive power and transparency.
  
-  comparative analyses between SHAP and other explainable AI (XAI) methods (such as Integrated Gradients or Counterfactual Explanations) could offer deeper insights into model behavior and the consistency of feature attributions across frameworks

### 9. Conclusion

This study developed and evaluated multiple machine learning models to predict diabetes occurrence using structured clinical data, with a particular focus on the interpretability of the best-performing model. Among the tested algorithms, the Random Forest Classifier demonstrated superior predictive performance across multiple metrics, including accuracy, precision, recall and F1-score, confirming its robustness and adaptability for binary classification tasks in healthcare domains.

Through the application of SHAP (SHapley Additive exPlanations), the model’s decision-making process was rendered transparent and interpretable. SHAP analysis identified the most influential features contributing to diabetes risk, with HbA1c level, blood glucose level, BMI, age and hypertension, emerging as the dominant predictors. The feature importance ranking aligned well with established clinical evidence, thereby strengthening the biological and diagnostic validity of the model’s predictions.

### References

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608.
Jobin, A., Ienca, M., & Vayena, E. (2019). The global landscape of AI ethics guidelines. Nature Machine Intelligence, 1(9), 389–399. https://doi.org/10.1038/s42256-019-0088-2

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6), 1–35. https://doi.org/10.1145/3457607

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. Artificial Intelligence, 267, 1–38. https://doi.org/10.1016/j.artint.2018.07.007

Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. New England Journal of Medicine, 380(14), 1347–1358. https://doi.org/10.1056/NEJMra1814259

Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence. Nature Medicine, 25(1), 44–56. https://doi.org/10.1038/s41591-018-0300-7

Zou, J., Schiebinger, L., & Obermeyer, Z. (2023). Ensuring fairness in machine learning for health care. Nature Reviews Genetics, 24(3), 173–188. https://doi.org/10.1038/s41576-022-00560-1
