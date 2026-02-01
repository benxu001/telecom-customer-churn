# Telecom-Customer-Churn-Machine-Learning

# Project Summary

Customer churn is a major driver of revenue loss in the telecom industry. In this project, I analyze a telecom customer dataset to identify **key factors associated with churn**, quantify their impact, and build predictive models to support **data-driven retention strategies**.

While machine learning models are used, the primary focus of this project is **data analysis, exploratory data analysis (EDA), feature understanding, and translating insights into business recommendations.**


## Business Problem

Telecom companies face high customer acquisition costs, making customer retention significantly more cost-effective than acquisition.

### Objective:
* Understand _why_ customers churn
* Identify high-risk customer segments
* Provide actionable insights that could inform retention strategies

## Dataset Overview

**Source**: [Telco Customer Churn from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
* Observations: ~7,000 customers
* Features: Customer demographics, account information, service usage, and billing data
* Target Variable: Churn (Yes / No)

**Key feature categories:**
* Demographics (e.g., gender, senior citizen status)
* Account information (tenure, contract type, payment method)
* Service usage (internet type, add-on services)
* Billing data (monthly and total charges)



## Exploratory Data Analysis (EDA)

Key findings from EDA include:
* **Short-tenure customers** churn at significantly higher rates than long-tenure customers
* **Month-to-month contracts** have much higher churn compared to one-year or two-year contracts
* Customers with **higher monthly charges** are more likely to churn
* **Electronic check** payment method is associated with elevated churn
* Customers without additional services (e.g., tech support) churn more frequently

These insights highlight both **pricing sensitivity** and **contract structure** as major churn drivers.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/322aaf2b-7acc-43e6-8e80-ea39ac7dbcf4" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/2d83e1ca-538d-4f0e-84ce-9005bdcc38a4" />
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/1234d043-bbdc-4ae5-8262-68f7f3e82d1a" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/57a01dfd-a230-43b4-97a6-abf170f7b31b" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/33d4d729-6075-4ed6-94f7-a5d81b4bb4f6" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/acb8d7e5-2336-44a5-ba1a-6888538e453c" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/19701082-3932-4ec1-8593-903fc4fed3fc" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/3fa180b4-2b0c-4c24-a750-d9fd15844df5" />
<img width="420" height="250" alt="image" src="https://github.com/user-attachments/assets/141cc9d1-862e-4145-924c-acaf597265e1" />

## Data Preparation & Feature Engineering

Steps included:
* Handling missing and improperly formatted values
* Encoding categorical variables using one-hot encoding
* Scaling numerical features where appropriate
* Ensuring the dataset was suitable for both analysis and modeling

While feature engineering was kept intentionally simple, the emphasis was on **interpretability** and **analytical clarity**.

## Model Training and Evaluation
Three different classification models were trained and evaluated. The goal was not to optimize a production-grade model, but to validate analytical insights, identify which customer attributes consistently influence churn, and compare interpretability vs. predictive power

### Random Forest Classifier
*   **Accuracy:** 0.78
*   **ROC AUC Score:** 0.8219
*   **Recall (Churn Class):** 0.47
    *   Showed good overall performance but struggled with recalling actual churners (false negatives).

### Linear Regression (adapted for classification)
*   **Accuracy:** 0.80
*   **ROC AUC Score:** 0.8297
*   **Recall (Churn Class):** 0.52
    *   Achieved competitive results, slightly outperforming Random Forest in terms of ROC AUC and churn recall, despite being a linear model.

### Logistic Regression (Default 0.5 threshold)
*   **Accuracy:** 0.80
*   **ROC AUC Score:** 0.8415
*   **Recall (Churn Class):** 0.56
    *   This model demonstrated the best overall performance with the highest ROC AUC score and the best recall for the churn class among the three models using the default threshold.

### Logistic Regression (0.4 threshold)
*   **Accuracy:** 0.78
*   **ROC AUC Score:** 0.8415
*   **Recall (Churn Class):** 0.67
  *   Adjusting the classification threshold to 0.4 significantly improved the recall for the churn class (meaning more actual churners were identified), though this came with a slight decrease in precision (more false positives).

<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/58245ace-e286-42c2-83d9-5040f9d4f5ee" />
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/0ef138f9-5931-40fb-94f5-f71532d1e863" />

## Predictive Insights
*   **Feature Importance (Logistic Regression):**
    *   Features like **Contract_Two year**, **OnlineSecurity_Yes**, and **TechSupport_Yes** had negative coefficients, indicating they **decrease** the likelihood of churn.
    *   Features like **InternetService_Fiber optic**, **PaperlessBilling_Yes**, and **PaymentMethod_Electronic check** had positive coefficients, indicating they **increase** the likelihood of churn.
    *   Model results aligned closely with trends observed during EDA, reinforcing confidence in the findings
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/6ee08c6b-73be-4628-bdfe-130ee810f044" />

## Business Insights & Recommendations
*   **Contract Length & Services:** Customers on longer contracts and those utilizing additional services like online security and technical support are less likely to churn. **Recommendation:** Offer incentives for longer-term contracts and promote value-added services.
*   **Payment & Billing:** Customers using electronic checks and paperless billing showed a higher propensity to churn. **Recommendation:** Investigate the customer experience associated with these methods and potentially offer alternative payment incentives or support.
*   **Early Customer Engagement:** Newer customers (lower tenure) are a high-risk group. **Recommendation:** Target this group with onboarding incentives, implement early engagement programs, and monitor satisfaction closely during the initial months.
*   **High Monthly Charges:** Customers with higher monthly charges are more prone to churn. **Recommendation:** Review pricing strategies, offer personalized plans, or provide additional benefits to high-paying customers to justify costs and increase perceived value.
*   **Model Choice:** Logistic Regression with a fine-tuned threshold (e.g., 0.4) emerged as the most effective model for identifying churners, balancing the need for high recall to enable proactive retention efforts.

## Tools & Technologies
* Python
* pandas, NumPy
* matplotlib, seaborn
* scikit-learn
* Jupyter Notebook
