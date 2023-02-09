# Credit_Risk_Analysis
# Background
Jill commends me for all my hard work. Piece by piece, I’ve been building up my skills in data preparation, statistical reasoning, and machine learning. I am now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks me to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the `SMOTEENN` algorithm. Next, I’ll compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once I am done, I’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

# Deliverable 1: Use Resampling Models to Predict Credit Risk

Using my knowledge of the imbalanced-learn and scikit-learn libraries, I’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, I’ll use the oversampling RandomOverSampler and SMOTE algorithms, and then I’ll use the undersampling ClusterCentroids algorithm. Using these algorithms, I’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Follow the instructions below and use the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 1.

Open the `credit_risk_resampling_starter_code.ipynb` file, rename it `credit_risk_resampling.ipynb`, and save it to your Credit_Risk_Analysis folder.

Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:

  - Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
  - Create the target variables.
  - Check the balance of the target variables.
Next, begin resampling the training data. First, use the oversampling `RandomOverSampler` and `SMOTE` algorithms to resample the data, then use the undersampling `ClusterCentroids` algorithm to resample the data. For each resampling algorithm, do the following:

  - Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
  - Calculate the accuracy score of the model.
  - Generate a confusion matrix.
  - Print out the imbalanced classification report.
Save my `credit_risk_resampling.ipynb` file to my Credit_Risk_Analysis folder.

# Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

Using my knowledge of the `imbalanced-learn` and `scikit-learn` libraries, I’ll use a combinatorial approach of over- and undersampling with the `SMOTEENN` algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the `SMOTEENN` algorithm, I’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 2.

  1. Continue using my `credit_risk_resampling.ipynb` file where I have already created my training and target variables.
  2. Using the information we have provided in the starter code, resample the training data using the `SMOTEENN` algorithm.
  3. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
  4. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save my `credit_risk_resampling.ipynb`file to my Credit_Risk_Analysis folder.

# Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Using my knowledge of the imblearn.ensemble library, I’ll train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, I’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 3.

  1. Open the credit_risk_ensemble_starter_code.ipynb file, rename it credit_risk_ensemble.ipynb, and save it to your Credit_Risk_Analysis folder.
  2. Using the information we have provided in the starter code, create my training and target variables by completing the following:
 
    - Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
    - Create the target variables.
    - Check the balance of the target variables.
   
  3. Resample the training data using the `BalancedRandomForestClassifier` algorithm with 100 estimators.
    - Consult the following Random Forest documentation Links to an external site.for an example.
 
  4. After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
  5. Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
  6. Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
    - Consult the following Easy Ensemble documentation Links to an external site.for an example.
  7. After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save my `credit_risk_ensemble.ipynb` file to your Credit_Risk_Analysis folder.

# Deliverable 4: Written Report on the Credit Risk Analysis

For this deliverable, I’ll write a brief summary and analysis of the performance of all the machine learning models used in this Challenge.

The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
