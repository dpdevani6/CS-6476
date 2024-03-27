---
layout: default
title: Midterm Report
nav_order: 2
has_children: false
permalink: /midterm-report
---

# Midterm Report

## Introduction/Background:
Phishing attacks are a social engineering tactic used to trick users into giving away personal data. In 2022, the Internet Crime Complaint Center received more than 300,000 complaints of phishing attacks, the most of any type of internet crime [1]. After obtaining sensitive information, attackers can steal the user's identity, access their bank accounts, and install malware.

Traditional approaches like two-factor authentication, personnel training and blacklists are either expensive, insufficient to handle deceptive phishing attacks, or unable to deal with short-lived webpages in time [2].


## Problem Definition:
The “Phishing Dataset for Machine Learning” [3] collects 48 features from 5000 phishing and 5000 legitimate webpages. This dataset is for anti-phishing researchers performing feature analysis, conducting proof-of-concept experiments and benchmarking their models.
Our goal is to develop an accurate and efficient classifier that can screen a web page and alert the user if they are about to access a phishing webpage. This will protect the user’s personal information and enhance their online security.


## Data Collection:
The dataset contains a comprehensive collection from both phishing and legitimate webpages. There are about 10,000 webpages, and it is evenly split between 5,000 phishing and 5,000 legitimate sites. The webpages were harvested from January to May 2015, and then again from May to June 2017. With the usage of Selenium, we collected data such that each webpage in the dataset is roughly described by 48 unique features, which should be sufficient when it comes to analyzing phishing features and also working on our models.

## Methods

### Unsupervised:

One of the algorithms we will use to identify phishing attacks is a Gaussian Mixture Model (GMM). We will be attempting to group the data into two clusters, phishing websites and legitimate websites. By using a GMM, it will allow us to capture complex clustering shapes. We will be using the sklearn.mixture package which contains a GaussianMixture object that can learn a Gaussian Mixture Model from the training dataset and give predictions. 

### Supervised:

We will also be using Support Vector Machines (SVM), supervised learning algorithms that are well suited for binary classification. Our goal will be to create a hyperplane that maximizes the margin between the points that correspond to phishing webpages and the points that do not. We’ll use the sklearn.svm package for training, testing , and evaluation, and specifying kernel and hyperparameters. SVMs work effectively in higher dimensional spaces. 
Finally, we’ll use a Random Forest that will aggregate the results of randomized decision trees. Randomization makes the model less prone to overfitting. We can use the sklearn.ensemble library, create a Classifier object and modify hyperparameters like the number of trees. 

## Result and Discussion:

We cleaned up the dataset by fixing the cases where there were missing values as well as detecting outliers. Roughly over 20% of the dataset contained outliers, and so after cleaning and dropping all the rows that have an outlier feature, we ended up having a total of 1103 entries. Once we cleaned up the outliers, we created the visualization for seeing the Feature Boxplots, as shown below.

<img src="docs/project_proposal/images/boxPlots.png" alt="Box Plots"/>


For pre-processing the data, we employed PCA and Lasso on our dataset. 
For PCA, we applied the StandardScalar to normalize the features and to help ensure that the PCA results aren't skewed by a few large features. We constructed a PCA dataframe that included each datapoint expressed in terms of its PCA values and then generated plots/visualizations for how the PCA performed on the phishing dataset. We applied 2 principal components to create a visualization and then we noticed that the first 2 principal components were in the same region and that the overall variance as explained by the 7 principal components is under 50%. Our results showed that each feature contributes fairly evenly to the overall variance, since we found that it takes about 40 principal components to reach the 100% variance explained, which is almost the total number of features in the whole dataset. The following visualizations were generated based on our analysis with PCA on the dataset.


<img src="docs/project_proposal/images/pca.png" alt="PCA image" />

<img src="docs/project_proposal/images/expVarGraph.png" alt="n_components vs Explained Variance Ratio" />


For Lasso, we utilized this technique to help refine the dataset as it would help in identifying the most significant features that would contribute to classifying a webpage as phishing or legitimate. By performing Lasso with an alpha of 0.001, we strived to help minimize overfitting and to help enhance the model's ability to generalize. The performance of the model gave us a mean squared error of 0.0354 and the coefficient analysis shows that some features had a negligible role in the phishing detection (as the coefficient was reduced to zero) while the non-zero coefficients represented important features. The bar plot depicts each feature's coefficient magnitude, helping to show the features Lasso determined to be most and least relevant for the phishing classification.

Here are the following features that ended up having the coefficient reduced to zero.

UrlLength, AtSymbol, TildeSymbol, NumHash, NoHttps, IpAddress, DomainInSubdomains, HttpsInHostname, DoubleSlashInPath, EmbeddedBrandName, ExtFormAction, AbnormalFormAction, PctNullSelfRedirectHyperlinks, FakeLinkInStatusBar, PopUpWindow, ImagesOnlyInForm

<img src="docs/project_proposal/images/Coefficient.png" alt="coefficient image" />

*Supervised Learning*

For a Random Forest classifier on a phishing dataset, we wanted to optimize the model's performance by varying the max_depth and estimators parameters. The Random Forest algorithm is particularly effective for such datasets due to its ability to handle large data sets with higher dimensionality. The goal was to maximize the combined score of accuracy, F1 score, and precision.
We used different combinations of best_depth and best_estimators (ranging from 1 to 11). The results showed a trend where both the accuracy and other evaluation metrics like F1 score and precision increased with higher values of these parameters, up to a certain point. There was a plateau observed after both of these values reached between 7-9, which shows that increasing these values more does not benefit the model's ability to classify phishing attempts much better. This plateau was likely due to the model reaching its capacity to learn from the data. We can see the accuracy, F1 score, balanced accuracy, MCC, precision, recall, and false positive rate as below:


### Metrics

- **Accuracy**: 0.9868
- **F1 score**: 0.9868473495416502
- **Balanced Accuracy**: 0.9868015978882556
- **MCC (Matthews Correlation Coefficient)**: 0.9736312367644127
- **Precision**: 0.9829297340214371
- **Recall**: 0.990796318527411
- **False Positive Rate (FPR)**: 0.01719312275089964

Classification Report:

<img src="docs/project_proposal/images/classReport.png" alt="image of the report" />


Confusion Matrix:


<img src="docs/project_proposal/images/ConfusionMatrix.png" alt="Confusion image" />

## Citations
[1] “Internet crime complaint center releases 2022 statistics,” FBI, https://www.fbi.gov/contact-us/field-offices/springfield/news/internet-crime-complaint-center-releases-2022-statistics (accessed Oct. 4, 2023).

[2] S. Hawa Apandi, J. Sallim, and R. Mohd Sidek, “Types of anti-phishing solutions for phishing attack,” IOP Conference Series: Materials Science and Engineering, vol. 769, no. 1, p. 012072, 2020. doi:10.1088/1757-899x/769/1/012072

[3] K. L. Chiew, C. L. Tan, K. Wong, K. S. C. Yong, and W. K. Tiong, “A New Hybrid Ensemble Feature Selection Framework for Machine Learning-based phishing detection system,” Information Sciences, vol. 484, pp. 153–166, 2019. doi:10.1016/j.ins.2019.01.064 

[4]  S. Alnemari and M. Alshammari, “Detecting phishing domains using machine learning,” Applied Sciences, vol. 13, no. 8, p. 4649, 2023. doi:10.3390/app13084649


## Contribution:

<img src="docs/project_proposal/images/newContrTable.png" alt="Contribution Table" width="1006" height="540"/>
