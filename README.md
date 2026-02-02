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

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/e1a89df3-c9a0-4a33-90fd-404f86a03ddc" />
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/10bf19a1-11d6-44bf-b941-9f0eaf910229" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/38e45678-71aa-41a7-ade6-bfdcca848040" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/86fd3e7d-1ef7-4177-803e-4d42def80495" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/7c899adf-7ee6-4235-b52b-73466eeae8d8" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/7abf19cf-27a4-40ee-9b65-455bdec985d2" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/e1967b8a-e27c-45c9-9af7-b601b1c20ba5" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/cf1956cf-b62c-4412-ae19-736e0502031b" />
<img width="333" height="200" alt="image" src="https://github.com/user-attachments/assets/c14c5fe9-090e-4be0-88d4-ee98c8a120fc" />


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


<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/6762f4af-232a-4172-8ce2-97378e86b325" />
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/2a2bc9f1-4c65-4e40-b465-ea2bca7bee5b" />


## Predictive Insights
*   **Feature Importance (Logistic Regression):**
    *   Features like **Contract_Two year**, **OnlineSecurity_Yes**, and **TechSupport_Yes** had negative coefficients, indicating they **decrease** the likelihood of churn.
    *   Features like **InternetService_Fiber optic**, **PaperlessBilling_Yes**, and **PaymentMethod_Electronic check** had positive coefficients, indicating they **increase** the likelihood of churn.
    *   Model results aligned closely with trends observed during EDA, reinforcing confidence in the findings

<img width="1118" height="611" alt="image" src="https://github.com/user-attachments/assets/e7e9d769-511b-49f4-b1ad-82c30c756505" />


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
