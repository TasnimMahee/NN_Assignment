# Bank Term Deposit Prediction (Logistic Regression)

## 📌 Overview
This project aims to predict whether a bank customer will subscribe to a term deposit based on their demographic and banking behavior data.

## 📊 Dataset
The dataset used is the Bank Marketing Dataset, which contains 17 attributes including:
- Age
- Job
- Marital status
- Education
- Balance
- Housing loan
- Loan
- Contact type
- Campaign details
- Previous outcome
- Target variable (y): yes/no

## ⚙️ Methodology

1. Data Preprocessing:
   - Converted target variable (yes/no → 1/0).
   - Encoded categorical variables using Label Encoding.
   - Handled dataset without missing values.

2. Feature Engineering:
   - Selected all relevant features except target.

3. Model Used:
   - Logistic Regression.

4. Training:
   - Dataset split into 80% training and 20% testing.
   - StandardScaler used for normalization.

## 🧰 Tools & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn

## 📈 Results
- The model predicts whether a customer will subscribe to a term deposit.
- Accuracy achieved: 88.8%.


This is likely due to class imbalance in the dataset, where most customers did not subscribe to the term deposit.

## 🔍 Findings
- Customer demographics and previous campaign outcomes influence subscription.
- Logistic Regression performs well for binary classification tasks.

   The model achieved an accuracy of approximately 88.8%. However, the recall for the positive class (customers who subscribed) is low (22%), indicating that the model struggles to correctly identify customers who will subscribe.