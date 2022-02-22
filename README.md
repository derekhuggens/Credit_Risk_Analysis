# Credit_Risk_Analysis
Class Imbalanced Modeling: Resampling, Combinatorial, and Ensemble Methods to Predict Credit Risk

## Overview of the Analysis:
The purpose of this analysis was to apply varying oversampling, undersampling, combinatorial sampling, and ensemble learning algorithms to predict loan risk.

* Oversampling methods: Naive Random Oversampling and SMOTE (Synthetic Minority Oversampling Technique)
* Undersampling methods: Cluster Centroids
* Combinatorial method: SMOTEENN (SMOTE with Edited Nearest Neighbors)
* Ensemble method: AdaBoost

The above methods were used on loan source data from Prospectus. <br>
Loan source data: Notes offered by Prospectus (https://www.lendingclub.com/info/prospectus.action) <br><br>
The many available columns within the dataset were preprocessed accordingly, i.e. changing strings to numeric form using `pd.get_dummies()`, so that our feature and target variables, X and y, were selected for future modeling. The dataset was then split using `sklearn` `.model_selection` `import train_test_split` into X_train, X_test, y_train, and y_test variables for further resampling using the different learning algorithms chosen. For each sampling/learning algorithm, a `sklearn.metrics` `balanced_accuracy_score`, `confusion_matrix`, and `imblearn.metrics` `classification_report_imbalanced` was analyzed for performance metrics.

## Results:
From https://www.scikit-yb.org/en/latest/index.html and https://dev.to/amananandrai/performance-measures-for-imbalanced-classes-2ojj
`Precision = TP/(TP+FP)`
`Recall = TP/(TP+FN)`
`Specificity = TN/(TN+FP)`
`F1-score = 2*Precision*Recall/(Precision+Recall)`
`Geometric Mean = √(Recall*Specificity) or Geometric Mean = √(True Positive Rate * True Negative Rate)`
`IBA or Index Balanced Accuracy = (1 + α*(Recall-Specificity))*(Recall*Specificity)`
`sup or Support = the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t change between models but instead diagnoses the evaluation process.`

`balanced_accuracy_score` is used for this analysis due to the fact that this is a very imbalanced classification dataset that needed resampling. Such is the difficulty in predicting credit risk as there is a skew towards `low_risk` classified target values compared to `high_risk` classified target values. You want to give weight to both classes.

Balanced Accuracy = (Sensitivity + Specificity) / 2

RandomOverSampler (Naive Random Oversampling)
  * balanced_accuracy_score: 0.585, the average of recall obtained on each class is 58.5%
  * precision "pre": high_risk = 0.01 , low_risk = 1.00
  * recall "rec": high_risk = 0.50, low_risk = 0.67

<br>

![ros](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/random_oversampler.png)<br><br>

SMOTE (Synthetic Minority Oversampling Technique)
  * balanced_accuracy_score: 0.617, the average of recall obtained on each class is 61.7%
  * precision "pre": high_risk = 0.01, low_risk = 1.00
  * recall "rec": high_risk = 0.51, low_risk = 0.72
<br>

![smote](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/smote.png)<br><br>

Cluster Centroids
  * balanced_accuracy_score: 0.542, the average of recall obtained on each class is 54.2%
  * precision "pre": high_risk = 0.01, low_risk = 0.99
  * recall "rec": high_risk = 0.55, low_risk = 0.54
<br>

![cc](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/cc.png)<br><br>

  
SMOTEENN (SMOTE with Edited Nearest Neighbors)
  * balanced_accuracy_score: 0.655, the average of recall obtained on each class is 65.5%
  * precision "pre": high_risk = 0.01, low_risk = 1.00
  * recall "rec": high_risk = 0.68, low_risk = 0.63
<br>

![smoteenn](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/smoteenn.png)<br><br>

  
BalancedRandomForestClassifier (Random Forest)
  * balanced_accuracy_score: 0.759, the average of recall obtained on each class is 75.9%
  * precision "pre": high_risk = 0.04, low_risk = 1.00
  * recall "rec": high_risk = 0.62, low_risk = 0.89
<br>

![rf](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/random_forest.png)<br><br>

  
EasyEnsembleClassifier (AdaBoost)
  * balanced_accuracy_score: 0.917, the average of recall obtained on each class is 97.1%
  * precision "pre": high_risk = 0.10, low_risk = 1.00
  * recall "rec": high_risk = 0.89, low_risk = 0.95
<br>

![adaboost](https://github.com/derekhuggens/Credit_Risk_Analysis/blob/18a4d0663852d46064c72dd22cb8f02a47d567f1/README_IMAGES/ada_boost.png)<br><br>

<br>

## Summary:

For every algorithm, the precision results for both high and low risk show that the models proportion of positive identifications was near 0 for high risk and near 100% for low risk. When looking at the metric of recall, we can see that RandomOversampler, SMOTE, and Cluster Centroids could not predict actual positives for high_risk at greater than 60%, whereas SMOTEENN, Balanced Random Forest, and EasyEnsemble could predict actual positives for high risk at greater than 60%. To predict actual positives of low risk correctly, the ascending order of performance was: Cluster Centroids, SMOTEENN, RandomSampler, SMOTE, Random Forest, then EasyEnsemble. Precision and recall are at odds with each other. The best model in this analysis is one with the highest precision and recall that befits the practical or meaningful use, for this analysis that would be the EasyEnsembleClassifier (AdaBoost), whos creators Yoav Freund and Robert Schapire won a prize for its creation. I would not use the models that could not acheieve a high risk precision greater than 1%, which was every model other than Random Forest and EasyEnsemble. For recall, I would only use models that could provide a proportion of actual positives identified correctly above 70%, which would again be Random Forest and EasyEnsemble. Between Random Forest and EasyEnsemble I chose either and would fine-tune them until the most practical model was found to be appropriate for the business.
