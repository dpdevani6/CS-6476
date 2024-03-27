---
layout: default
title: Final Report
nav_order: 1
has_children: true
permalink: /final-report
---

# Final Report

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

Additionally, we will also be using a Multi-layer Perceptron (MLP). MLP is good for binary classification because it has the ability to learn complex nonlinear relationships. This is important because the decision boundary between data points that are phishing and are not phishing is not necessarily a linear function. Our output layer will give us a probability with which a point belongs to a class which will round to get a predicted label. 

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

# Supervised Learning

## Random Forest

For a Random Forest classifier on a phishing dataset, we wanted to optimize the model's performance by varying the max_depth and estimators parameters. The Random Forest algorithm is particularly effective for such datasets due to its ability to handle large data sets with higher dimensionality. The goal was to maximize the combined score of accuracy, F1 score, and precision.
We used different combinations of best_depth and best_estimators (ranging from 1 to 11). The results showed a trend where both the accuracy and other evaluation metrics like F1 score and precision increased with higher values of these parameters, up to a certain point. There was a plateau observed after both of these values reached between 7-9, which shows that increasing these values more does not benefit the model's ability to classify phishing attempts much better. This plateau was likely due to the model reaching its capacity to learn from the data. We can see the accuracy, F1 score, balanced accuracy, MCC, precision, recall, and false positive rate as below:


Metrics:

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

## Support Vector Machine
Another supervised learning model that we used was SVM, which we retrieved from sklearn SVC. We can see from the results below that SVM was extremely accurate with a high accuracy, high f1 score, high balanced accuracy, and high MCC even compared to Random Forest. The precision and a recall show that the model is highly effective in both identifying positive cases and minimizing false positives. The uniformity in precision, recall, and F1 score across both classes (0 and 1) in the classification report implies that the model can identify both classes without bias.

Metrics:

- **Accuracy**: 0.999
- **F1 score**: 0.9990005996402158
- **Balanced Accuracy**: 0.9990003998400641
- **MCC (Matthews Correlation Coefficient)**: 0.9980019972834349
- **Precision**: 0.9980031948881789
- **Recall**: 1.0
- **False Positive Rate (FPR)**: 0.0019992

Classification Report:

<img src="docs/project_proposal/images/svmClassReport.png" alt="SVM Classification Report"/>



Confusion Matrix:


<img src="docs/project_proposal/images/svmConfusionMatrix.png" alt="SVM Confusion Matrix"/>

## Multi-layer Perceptron
Multi-layer perceptron (MLP) is a type of neural network designed for supervised learning. We defined a MLP with two hidden layers, the first layer with 5 neurons and the second with 2 neurons. From the results, we can see that this model is very accurate with 94% accuracy. The consistent and high precision, recall, and f1-score indicates that the model is not biased towards a class and minimizes the amount of negative predictions. This can be visually confirmed by the confusion matrix which shows an overwhelming amount of positive cases.

Metrics:

- **Accuracy**: 0.94
- **F1 score**: 0.0.94
- **Balanced Accuracy**: 0.9372009499521521
- **MCC (Matthews Correlation Coefficient)**: 0.8744102942847655
- **Precision**: 0.94
- **Recall**: 0.935
- **False Positive Rate (FPR)**: 0.03361008537761366

Classification Report:

<img src="docs/project_proposal/images/mlpClassReport.png" alt="MLP Classification Report"/>


Confusion Matrix:


<img src="docs/project_proposal/images/mlpConfusionMatrix.png" alt="MLP Confusion Matrix"/>

Receiver Operating Characteristic (ROC) Curve:

<img src="docs/project_proposal/images/mlpROCCurve.png" alt="MLP ROC Curve"/>

Receiver Operator Characteristic Curve represents the performance of binary classification on different thresholds. The tradeoff is between true positive rate (sensitivity) and false positive rate (specificity). Area under the curve quantifies overall performance at all thresholds. In our case, the area under the curve (AUC) is .98, which is close to 1. This indicates near perfect classification of test labels. 

# Unsupervised Learning

## Gaussian Mixture Model
An unsupervised learning algorithm that we used is called Gaussian Mixture Model (GMM). A notable aspect of using GMM is its sensitivity to outliers, with the removal of outliers resulting in increased accuracy. Despite this, the GMM has shown some unpredictability in its performance, resulting in us modifying its parameters and introducing a random state to stabilize its behavior.
The iterative process of tuning the GMM yielded varying results. The first iteration achieved an accuracy of 0.67, indicating moderate effectiveness in classifying phishing attempts. The second iteration showed a significant drop in performance with an accuracy of 0.317, emphasizing the algorithm's sensitivity to parameter changes and the stochastic nature of its learning process.

First Iteration Accuracy: 0.67

<img src="docs/project_proposal/images/gmmFirstIteration.png" alt="GMM First Iteration"/>

Second Iteration Accuracy: 0.317

<img src="docs/project_proposal/images/gmmSecondIteration.png" alt="GMM Second Iteration"/>

Metrics:

The metrics used for the supervised learning section (accuracy, f1 score, balanced accuracy, MCC, precision, recall, and false positive rate) are not applicable to unsupervised learning. Therfore, for GMM we will only be using Silhoutte score as the metric.
- **Silhouette Score (without outliers)**: 0.561
- **Silhouette Score (with outliers)**: 0.134

GMM is probabilistic in how it assigns data points to clusters. It is also an unsupervised technique without any awareness of the true labels, so metrics like accuracy, precision and F1 are not applicable. This is evidenced by the inconsistency of accuracy results that we see above. We instead used the silhouette coefficient, which compares each combination of the predicted and ground truth labels and counts the number of true positive and true negative (0s with 0s and 1s with 1s). This number is divided by the total number of combinations and we are left with the silhouette coefficient. We found that removing outliers improved the performance of our GMM clustering because there wasn’t a need to assign outlier points to clusters. We received a silhouette score of .561, which demonstrates that the clusters are not well separated. This means that the points in one cluster are similar to points in the other cluster. The GMM’s lack of separation means that the 2 clusters can’t be mapped to a class label. This means GMM is not as good a predictor for phishing detection as our models are. However, this is an improvement compared to the silhouette score of .134 we received when running the model with outliers.

<img src="docs/project_proposal/images/gmmClusters.png" alt="GMM Clusters Image"/>

Created Clusters - Silhouette score of .561 which indicates that the clusters are not very well separated and likely do not have the same predictive ability as the other models. 

<img src="docs/project_proposal/images/gmmBarChart.png" alt="GMM Bar Chart"/>


# Conclusions

## Comparison Between Models

Each model has their own unique benefits. SVM has a very high accuracy and recall, making it appear to be highly reliable for critical phishing detection tasks. Another factor we could consider is MCC, in which SVM scores highest, which suggests that it has a good quality of binary classifications. SVM also shows strong recall and precision which means it can identify positive cases highly and has a low rate of false positives. Random Forest is good for accuracy and interpretability, which is ideal for scenarios where understanding the model's decision process is important. GMM, while less accurate than the other models, provides insights into the underlying structure of the data itself and this is good for exploratory data analysis and finding unique patterns that exist in the phishing dataset itself. MLP uses a neural network architecture with a high accuracy and good balance between the precision and the recall, but this does come with more complexity and then a slight reduction in interpretability as opposed to models like Random Forest and SVM.

All of the results from above are compiled into the following table:

<img src="docs/project_proposal/images/metricsTable.png" alt="Metrics Table"/>

## Next Steps

Our next steps would be to find another dataset that we could merge with the current Kaggle dataset in order to increase the number of samples in our dataset. This would allow us to better generalize our models to real world scenarios. Another way we could train our model on more samples is by using a better unsupervised approach. This would entail either enhancing our current GMM model or exploring other approaches such as hierarchical clustering and DBSCAN.
Beyond improving the performance of our model, it would be interesting to create a browser extension that could classify a URL as a phishing website or not, proactively alerting the user.

## Citations
[1] “Internet crime complaint center releases 2022 statistics,” FBI, https://www.fbi.gov/contact-us/field-offices/springfield/news/internet-crime-complaint-center-releases-2022-statistics (accessed Oct. 4, 2023).

[2] S. Hawa Apandi, J. Sallim, and R. Mohd Sidek, “Types of anti-phishing solutions for phishing attack,” IOP Conference Series: Materials Science and Engineering, vol. 769, no. 1, p. 012072, 2020. doi:10.1088/1757-899x/769/1/012072

[3] K. L. Chiew, C. L. Tan, K. Wong, K. S. C. Yong, and W. K. Tiong, “A New Hybrid Ensemble Feature Selection Framework for Machine Learning-based phishing detection system,” Information Sciences, vol. 484, pp. 153–166, 2019. doi:10.1016/j.ins.2019.01.064 

[4]  S. Alnemari and M. Alshammari, “Detecting phishing domains using machine learning,” Applied Sciences, vol. 13, no. 8, p. 4649, 2023. doi:10.3390/app13084649


## Contribution:

<img src="docs/project_proposal/images/finalContributionTable.png" alt="Contribution Table" width="1006" height="540"/>
