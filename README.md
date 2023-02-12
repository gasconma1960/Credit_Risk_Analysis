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
 
![image](https://user-images.githubusercontent.com/112348240/218282232-4b0dd7ac-68f6-4f97-a9c4-ad5e31b672b9.png)
  
  - Check the balance of the target variables.
![image](https://user-images.githubusercontent.com/112348240/218282316-55d8ad74-2000-4f8b-a437-78b2816c6355.png)

Next, begin resampling the training data. First, use the oversampling `RandomOverSampler` and `SMOTE` algorithms to resample the data, then use the undersampling `ClusterCentroids` algorithm to resample the data. For each resampling algorithm, do the following:
## **Oversampling RandomOverSampler**
  - Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
  
![image](https://user-images.githubusercontent.com/112348240/218282382-a0a14dcf-b248-4506-a5ef-b86d12c1f6b5.png)

![image](https://user-images.githubusercontent.com/112348240/218282419-e63e0d65-a5e1-4fe7-88f0-eaeba811e89d.png)

  - Calculate the accuracy score of the model.

![image](https://user-images.githubusercontent.com/112348240/218282516-dbb5ddf8-f752-4147-a741-edaa196a5487.png)

  - Generate a confusion matrix.
 
![image](https://user-images.githubusercontent.com/112348240/218282619-5615a735-7554-45f6-8a0e-408c519e2740.png)

  - Print out the imbalanced classification report.

![image](https://user-images.githubusercontent.com/112348240/218282665-1ea50338-be54-4f21-bc01-532bcb7d17cf.png)

## **SMOTE Oversampling**

 - Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
 
![image](https://user-images.githubusercontent.com/112348240/218282906-2842202a-6ec9-4234-ae11-9574ad88a311.png)
![image](https://user-images.githubusercontent.com/112348240/218282958-193b78c5-755c-4479-b43e-f136209bf620.png)

 - Calculate the accuracy score of the model.

![image](https://user-images.githubusercontent.com/112348240/218283022-d36ee453-a8b7-4f4e-8dc4-43f3c639e560.png)

 - Generate a confusion matrix.

![image](https://user-images.githubusercontent.com/112348240/218283185-6d06bf33-735a-454f-ba3d-e1865454f368.png)

 - Print out the imbalanced classification report.
 
 ![image](https://user-images.githubusercontent.com/112348240/218286595-c3aa770e-a2bf-439e-a389-be0c942844d1.png)

## **Undersampling ClusterCentroids**

1. View the count of the target classes using Counter from the collections library.

![image](https://user-images.githubusercontent.com/112348240/218286775-2bb8edbc-c05f-42a7-90dd-920e71357d62.png)

2. Use the resampled data to train a logistic regression model.

![image](https://user-images.githubusercontent.com/112348240/218286814-abac83b6-dcf2-4ed7-aaad-51b9eb816ee1.png)

3. Calculate the balanced accuracy score from sklearn.metrics.

![image](https://user-images.githubusercontent.com/112348240/218286855-bf8d01b1-736b-4042-8d93-8cd3b15ff4ec.png)

4. Print the confusion matrix from sklearn.metrics.

![image](https://user-images.githubusercontent.com/112348240/218286889-f9da2183-2a4e-4ce6-8321-6c2b86921b3b.png)

5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn

![image](https://user-images.githubusercontent.com/112348240/218286906-04643592-0448-425e-b8e6-9a3e7b36458a.png)

Save my `credit_risk_resampling.ipynb` file to my Credit_Risk_Analysis folder.

# Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

Using my knowledge of the `imbalanced-learn` and `scikit-learn` libraries, I’ll use a combinatorial approach of over- and undersampling with the `SMOTEENN` algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the `SMOTEENN` algorithm, I’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 2.

## **Combination (Over and Under) Sampling**
  1. View the count of the target classes using Counter from the collections library.

![image](https://user-images.githubusercontent.com/112348240/218286999-b8449224-a068-4e81-bc51-3331d3a50cb7.png)

  2. Use the resampled data to train a logistic regression model.

![image](https://user-images.githubusercontent.com/112348240/218287071-387e463b-5721-4387-a4c0-9e4b82c7e8b1.png)

  3. Calculate the balanced accuracy score from sklearn.metrics.

![image](https://user-images.githubusercontent.com/112348240/218287132-3480420e-a337-4608-a78f-6e917e064812.png)

  4. Print the confusion matrix from sklearn.metrics.

![image](https://user-images.githubusercontent.com/112348240/218287173-6eeb6020-014d-404a-a11b-a33279260033.png)

  5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.

![image](https://user-images.githubusercontent.com/112348240/218287205-6ae07539-52a0-4308-beb3-33046cffa9c0.png)


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

## **Overview of the analysis**: Explain the purpose of this analysis.

## **Results**

## **Summary** : Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

# **SOURCES**:


# **MODULE 18 Challenge**
My link github page:

by **Marisol Gascon Linarez**

**UCF Bootcamp Data Visualization and Analytics**
