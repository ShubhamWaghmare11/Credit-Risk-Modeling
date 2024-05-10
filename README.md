# Credit Risk Modeling

This project focuses on developing a credit risk modeling solution to assist banks in evaluating loan applications. 
By leveraging machine learning techniques, we aim to predict the likelihood of loan approval for each applicant, 
thereby aiding financial institutions in making informed lending decisions.


## Features
1. Feature Engineering: Comprehensive analysis and engineering of relevant features to capture the key factors influencing credit risk.
2. Model Training: Utilization of machine learning algorithms to train a predictive model on historical loan data.
3. Risk Classification: Classification of loan applicants into distinct risk categories (P1, P2, P3, P4) based on their likelihood of loan approval.
4. Model Interpretability: Interpretation of model predictions to provide insights into the factors contributing to credit risk.


## Deployment

The credit risk model is deployed using Streamlit, a Python library for creating web applications with minimal effort. 
The deployment process involves setting up a Streamlit server to host the application. 
To deploy the model, follow these steps:

1. Install Streamlit: pip install streamlit
2. Clone the repository: git clone [repository-url]
3. Navigate to the project directory: cd credit-risk-model
4. Run the Streamlit app: streamlit run app.py
5. Access the application in your browser at the provided URL (typically http://localhost:8501)


## Usage
To use the credit risk model via the Streamlit web application, follow these steps:

1. Open the deployed Streamlit application in your web browser.
2. Input the relevant applicant information directly into the provided form fields, or upload an Excel/CSV file containing the required columns.
3. Click the "Predict" button to obtain the model's prediction on the likelihood of loan approval.
4. Review the predicted risk category (P1, P2, P3, or P4) and associated confidence score.
5. Optionally, download the predicted labels in an Excel file for further analysis or record-keeping.
