Smart Delivery Intelligence Using Machine Learning and Deep Learning
1. Project Overview

Smart Delivery Intelligence is an end-to-end Machine Learning and Deep Learning based platform developed to analyze and predict different aspects of the delivery process.

The project focuses on three major delivery prediction problems:

Delivery Charge Prediction
Delivery Time Prediction
Rider Acceptance Prediction

In addition to delivery prediction, the project includes a Natural Language Processing based Customer Review Analysis module.

The review analysis system uses DistilBERT Deep Learning models to perform:

Sentiment Classification
Customer Review Topic Classification

The complete solution is integrated into an interactive Streamlit dashboard containing four modules.

2. Problem Statement

Delivery platforms generate large amounts of operational and customer-related data.

Important business questions include:

What delivery charge should be expected for an order?
How long will the delivery take?
Will a rider accept the delivery request?
What are customers saying about their delivery experience?
What are the major issues mentioned in customer reviews?

Analyzing these questions manually can be difficult when the volume of data is large.

The objective of this project is to develop an AI-based platform that uses historical delivery data and customer reviews to generate predictions and insights.

3. Objectives

The main objectives of this project are:

To perform data cleaning and preprocessing on delivery data.
To perform exploratory data analysis.
To identify important relationships between delivery features.
To build a regression model for delivery charge prediction.
To build a regression model for delivery time prediction.
To build a classification model for rider acceptance prediction.
To evaluate the Machine Learning models using appropriate metrics.
To preprocess and analyze customer reviews.
To develop a DistilBERT based sentiment classification model.
To develop a DistilBERT based topic classification model.
To classify customer reviews into different sentiment categories.
To identify the major topic discussed in each review.
To integrate all trained models into a Streamlit application.
To provide an interactive business prediction platform.
4. Dataset

The main delivery dataset contains approximately 81,280 records and 39 columns.

The dataset contains information related to orders, delivery conditions, riders, locations, weather, traffic, and historical delivery information.

The major categories of features include:

Order Information
Order amount
Order date and time
Membership information
Promotional information
Order-related attributes
Delivery Information
Delivery distance
Delivery zone
City
Road condition
Traffic index
Rainfall
Weather condition
Peak hour
Weekend
Festival day
Rider Information
Rider experience
Rider rating
Vehicle type
Rider acceptance
Incentive
Historical Information
Historical delivery cost
Historical delivery time
Other historical delivery-related features
5. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics and relationships within the dataset.

The following analysis was performed:

Dataset shape analysis
Data type analysis
Missing value analysis
Duplicate analysis
Numerical feature distributions
Categorical feature distributions
Outlier analysis
Correlation analysis
Target variable distribution
Relationship between distance and delivery charge
Relationship between incentive and delivery charge
Relationship between delivery time and delivery charge

The analysis helped identify important variables for model development and highlighted data quality issues that needed to be addressed during preprocessing.

6. Data Cleaning and Preprocessing

Several data quality issues were identified in the dataset.

The preprocessing process included handling missing values, invalid values, incorrect data types, and outliers.

Examples of identified issues included:

Missing order_placed_at
Missing traffic_index
Missing rainfall_mm
Missing rider ratings
Invalid rider experience values
Invalid rider rating values
Invalid weight values
Invalid monetary values
Incorrect categorical data types
Traffic values stored in percentage/string formats

The data was cleaned and converted into appropriate formats before model training.

7. Handling Missing Values

Missing values were identified in several columns.

For example:

order_placed_at
traffic_index
rainfall_mm
rider_rating

Appropriate preprocessing and imputation techniques were applied depending on the feature type.

Numerical features were processed using numerical imputation methods, while categorical features were handled using suitable categorical preprocessing.

8. Outlier Detection and Treatment

Outlier analysis was performed on important numerical variables, particularly delivery charge.

The Interquartile Range method was used to identify extreme values.

For delivery charge, the original distribution contained very high values.

Before outlier treatment:

Mean    ≈ 55.41
Median  ≈ 52.37
Minimum = 10
Maximum ≈ 284.37

IQR-based capping was applied to reduce the effect of extreme values.

After capping:

Mean ≈ 54.50
Maximum ≈ 95.04

This preprocessing helped create a more stable dataset for analysis and modeling.

9. Feature Engineering

Feature engineering was performed to create useful variables from the available delivery information.

The project uses features related to:

Order timing
Distance
Traffic
Weather
Delivery zone
City
Vehicle type
Membership
Rider characteristics
Peak hour
Weekend
Festival day
Historical delivery information

Time-related information was also used to derive features such as:

Order month
Peak hour
Weekend indicator
Festival day indicator

These features were used by the prediction models where applicable.

10. Delivery Charge Prediction
Problem Type

Regression

Target Variable
delivery_charge

The objective is to predict the expected delivery charge based on order, delivery, rider, location, and environmental characteristics.

11. Delivery Charge Model Development

Different regression approaches were evaluated.

The project evaluated relationships between delivery charge and important variables such as:

Delivery distance
Incentive
Delivery time
Order characteristics
Rider characteristics
Traffic
Location-related variables

Polynomial Regression was used to model non-linear relationships between the input variables and delivery charge.

A Degree-2 Polynomial Regression model was selected for the final implementation.

12. Delivery Charge Model Evaluation

The Degree-2 Polynomial Regression model achieved approximately:

Test MAE : 4.87
Test R²  : 0.893

Mean Absolute Error

MAE represents the average absolute difference between the actual delivery charge and the predicted delivery charge.

A lower MAE indicates smaller prediction errors.

R² Score

R² represents the proportion of variation in the target variable explained by the model.

The model achieved an R² of approximately 0.893 on the test dataset.

13. Delivery Charge Model Deployment

The trained model was saved as:

Polynomial_Regression_Degree2.pkl

The saved model is loaded into the Streamlit application and used to generate delivery charge predictions based on user input.

14. Delivery Time Prediction
Problem Type

Regression

Target Variable
delivery_time_minutes

The objective is to estimate the expected delivery time in minutes based on delivery and order characteristics.

Important factors include:

Delivery distance
Traffic
Weather
Road condition
Rider characteristics
Vehicle type
Location
Peak hour
Weekend
Festival day
Order characteristics
15. Delivery Time Model

A regression pipeline was developed for delivery time prediction.

The pipeline handles the required preprocessing and model prediction steps.

The final model was saved as:

Delivery_TimeMin_pipeline_PR_2.pkl

The pipeline is integrated into the Streamlit dashboard.

16. Rider Acceptance Prediction
Problem Type

Binary Classification

Target Variable
rider_accepted

The target contains two classes:

1 = Accepted
0 = Not Accepted

The objective is to predict whether a rider will accept a delivery request.

17. Rider Acceptance Data Distribution

The target distribution was approximately:

Accepted     : 48,599
Not Accepted : 31,401

The corresponding percentages were approximately:

Accepted     : 60.75%
Not Accepted : 39.25%

This distribution was considered during classification model development and evaluation.

18. Rider Acceptance Model Development

Classification algorithms were evaluated for rider acceptance prediction.

The models included approaches such as:

Logistic Regression
Random Forest
Gradient Boosting

The classification models achieved approximately 87%–89% accuracy during evaluation.

The final trained model was saved as:

rider_acceptance_model.pkl
19. Rider Acceptance Prediction

The model predicts whether the rider is expected to accept the delivery.

The Streamlit application displays the prediction and acceptance probability.

The prediction is presented as:

Accepted

or

Not Accepted

along with the corresponding model probability.

20. Customer Review Analysis

The project includes a separate Natural Language Processing module for analyzing customer reviews.

The purpose of this module is to automatically understand:

Customer sentiment
Main topic of the review

Instead of manually reading every customer review, the Deep Learning models classify each review automatically.

21. Review Dataset

The review dataset was cleaned and duplicate review texts were removed.

After text deduplication, the review dataset contained approximately:

6,472 reviews

The dataset was divided into:

Training : 5,177
Testing  : 1,295

The sentiment labels were:

Negative = 0
Neutral  = 1
Positive = 2
22. Review Data Preprocessing

The following preprocessing steps were performed:

Duplicate review text removal
Invalid rating handling
Review text preparation
Sentiment label preparation
Topic label preparation
Train-test splitting
Tokenization using the DistilBERT tokenizer

The review dataset also contained English and Tanglish style customer feedback.

23. Sentiment Classification

The sentiment model classifies customer reviews into three categories:

Negative
Neutral
Positive

The objective is to understand the overall emotional orientation of a customer review.

For example:

"The delivery was very fast and the rider was polite."

can be classified as:

Positive
24. DistilBERT Sentiment Model

DistilBERT was selected for sentiment classification.

DistilBERT is a smaller and computationally efficient version of BERT that can be fine-tuned for text classification tasks.

The model was fine-tuned using the customer review dataset.

Training configuration included:

Model       : DistilBERT
Task        : Sentiment Classification
Classes     : 3
Epochs      : 3
25. Sentiment Model Evaluation

The trained DistilBERT sentiment model achieved approximately:

Validation Accuracy : 99.9%
Validation F1 Score : 99.9%

The model was evaluated using:

Accuracy
Precision
Recall
F1 Score

The trained model was saved locally as:

Final_Sentiment_DistilBERT
26. Customer Review Topic Classification

In addition to sentiment classification, a separate Deep Learning model was developed to identify the main topic of a review.

The topic model contains 14 categories.

The categories are:

Product Quality
Delivery Experience
Customer Support
Order Issues
Payment Issues
App Experience
Website Experience
Refund Issues
Membership
Coupons & Offers
Product Availability
Packaging
Pricing & Fees
Overall Experience
27. DistilBERT Topic Model

A separate DistilBERT model was trained for topic classification.

The topic classification model receives the customer review as input and predicts the most relevant topic.

The trained model was saved as:

Final_Topic_DistilBERT

This model is loaded independently from the sentiment model in the Streamlit application.

28. Review Prediction Workflow

The customer review analysis follows the below workflow:

Customer Review
       ↓
DistilBERT Tokenizer
       ↓
Sentiment Model
       ↓
Sentiment Prediction
       ↓
Topic Model
       ↓
Topic Prediction
       ↓
Confidence Calculation
       ↓
Streamlit Dashboard

The application displays both predictions along with their confidence values.

29. Streamlit Dashboard

The complete project was integrated into a Streamlit dashboard.

The main application file is:

Dashboard.py

The dashboard contains four tabs.

30. Tab 1 — Delivery Charge Prediction

The first tab allows the user to enter the required delivery and order information.

The application loads:

Polynomial_Regression_Degree2.pkl

and generates the predicted delivery charge.

The prediction is displayed interactively in the dashboard.

31. Tab 2 — Delivery Time Prediction

The second tab predicts the expected delivery time.

The application uses:

Delivery_TimeMin_pipeline_PR_2.pkl

The predicted result is displayed in minutes.

32. Tab 3 — Rider Acceptance Prediction

The third tab predicts whether the rider will accept the delivery.

The application loads:

rider_acceptance_model.pkl

The dashboard displays:

Acceptance prediction
Prediction probability
33. Tab 4 — Customer Review Analysis

The fourth tab allows the user to enter a customer review.

For example:

The delivery was very fast and the rider was polite.

The system predicts:

Sentiment: Positive
Topic: Delivery Experience

The dashboard also displays:

Sentiment confidence
Topic confidence
Review summary
Business-oriented interpretation
34. Model Integration

The Machine Learning models were saved using Joblib.

The Streamlit application loads the models using:

joblib.load()

The NLP models are loaded using Hugging Face Transformers:

AutoTokenizer.from_pretrained()
AutoModelForSequenceClassification.from_pretrained()

The NLP models are stored locally and loaded using local model files.

This allows the Streamlit application to perform predictions using the trained models without retraining them.

35. Technologies Used

The major technologies used in this project are:

Programming

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

Polynomial Regression

Logistic Regression

Random Forest

Gradient Boosting

Deep Learning

PyTorch

Hugging Face Transformers

DistilBERT

Natural Language Processing

Text Classification

Sentiment Analysis

Topic Classification

Tokenization

Deployment

Streamlit

Joblib

Development Environment

Google Colab

Visual Studio Code

Windows

36. Model Files

The final project contains the following trained models:

Delivery Charge
Polynomial_Regression_Degree2.pkl
Delivery Time
Delivery_TimeMin_pipeline_PR_2.pkl
Rider Acceptance
rider_acceptance_model.pkl
Sentiment Analysis
Final_Sentiment_DistilBERT/
Topic Classification
Final_Topic_DistilBERT/
37. Project Architecture

The overall project architecture can be represented as:

                 Smart Delivery Intelligence
                           |
                           ↓
                  Delivery Dataset
                           |
                           ↓
                 Data Preprocessing
                           |
                           ↓
                    Exploratory Analysis
                           |
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        Delivery       Delivery       Rider
         Charge          Time        Acceptance
             ↓             ↓             ↓
             └─────────────┼─────────────┘
                           |
                           ↓
                    ML Models
                           |
                           |
                  Customer Reviews
                           |
                           ↓
                    NLP Processing
                           |
                           ↓
                     DistilBERT
                     /         \
                    ↓           ↓
               Sentiment       Topic
                    |           |
                    └─────┬─────┘
                          ↓
                 Streamlit Dashboard
                          |
                          ↓
                 Business Predictions
38. Business Applications

The developed platform can potentially support several delivery-business use cases.

Delivery Charge Prediction

Can provide an estimated delivery charge based on order and delivery characteristics.

Delivery Time Prediction

Can provide an estimated delivery time to support delivery planning.

Rider Acceptance

Can help analyze rider acceptance behavior for delivery requests.

Customer Review Analysis

Can automatically identify customer sentiment and major review topics.

This can help businesses organize large volumes of customer feedback for further analysis.

39. Example Customer Review Analysis
Example 1

Input:

The delivery was very fast and the rider was polite.

Possible model output:

Sentiment:
Positive

Topic:
Delivery Experience
Example 2

Input:

The food arrived late and the package was damaged.

Possible output:

Sentiment:
Negative

Topic:
Delivery Experience
Example 3

Input:

The application is easy to use and ordering is simple.

Possible output:

Sentiment:
Positive

Topic:
App Experience
40. Model Evaluation

The major model evaluation results obtained during the project include:

Delivery Charge
Model:
Polynomial Regression Degree 2

Test MAE:
≈ 4.87

Test R²:
≈ 0.893
Rider Acceptance
Classification Accuracy:
Approximately 87%–89%
Sentiment Classification
Validation Accuracy:
≈ 99.9%

Validation F1 Score:
≈ 99.9%

The delivery time model was also evaluated during model development and integrated into the final Streamlit application.

41. Limitations

The project has some limitations.

The prediction performance depends on the quality and distribution of the historical dataset.

Changes in delivery behavior, traffic conditions, weather patterns, rider behavior, or business rules may affect future predictions.

The customer review models may also encounter difficulty with:

Very short reviews
Unclear language
Sarcasm
Mixed sentiment
Text that differs significantly from the training data
New topics not represented in the training dataset

The current application is an interactive local Streamlit application and is not a full-scale production deployment.

42. Future Improvements

The project can be further enhanced by:

Integrating real-time traffic data.
Integrating real-time weather information.
Adding GPS-based delivery tracking.
Connecting the application to a production database.
Adding real-time rider availability information.
Adding automated model retraining.
Implementing model monitoring.
Adding SHAP-based model explainability.
Adding customer review trend analysis.
Adding sentiment distribution dashboards.
Adding topic frequency visualizations.
Supporting additional languages.
Deploying the application on a cloud platform.
Developing a mobile application.
Adding real-time alerts for unusual delivery patterns.
43. Project Workflow

The complete project workflow was:

1. Understand the business problem
              ↓
2. Load the delivery dataset
              ↓
3. Perform data quality analysis
              ↓
4. Clean missing and invalid values
              ↓
5. Perform exploratory data analysis
              ↓
6. Perform feature engineering
              ↓
7. Develop delivery charge model
              ↓
8. Develop delivery time model
              ↓
9. Develop rider acceptance model
              ↓
10. Evaluate Machine Learning models
              ↓
11. Save trained ML models
              ↓
12. Prepare customer review dataset
              ↓
13. Remove duplicate reviews
              ↓
14. Prepare sentiment and topic labels
              ↓
15. Fine-tune DistilBERT sentiment model
              ↓
16. Train DistilBERT topic model
              ↓
17. Evaluate NLP models
              ↓
18. Save NLP models
              ↓
19. Develop Streamlit dashboard
              ↓
20. Integrate all five trained models
              ↓
21. Test all four dashboard modules
              ↓
22. Generate final predictions
44. Project Outcome

The final project successfully integrates Machine Learning and Deep Learning models into a single interactive Streamlit platform.

The system provides four major capabilities:

1. Delivery Charge Prediction
2. Delivery Time Prediction
3. Rider Acceptance Prediction
4. Customer Review Analysis

The Customer Review Analysis module further performs:

Sentiment Classification
        +
Topic Classification

The final dashboard allows users to enter delivery information or customer reviews and receive model-based predictions interactively.

45. Conclusion

Smart Delivery Intelligence demonstrates how Machine Learning and Deep Learning can be combined to solve multiple problems in the delivery domain.

The project includes regression models for delivery charge and delivery time, a classification model for rider acceptance, and Transformer-based Deep Learning models for customer review analysis.

The use of DistilBERT allows customer reviews to be analyzed for both sentiment and topic, while the Streamlit dashboard provides an interactive interface for accessing all predictions.

Overall, the project demonstrates an end-to-end workflow covering:

Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Deep Learning / NLP
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Streamlit Integration
      ↓
Interactive Prediction

The project provides a practical demonstration of applying Machine Learning, Deep Learning, NLP, and deployment techniques to a real-world delivery analytics problem.
