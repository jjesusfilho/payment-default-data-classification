# Classification on Payment Default Data

[Data source: Universit of California, Irvine, Machine Learning respository](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Performance Summary:

| Model                                    | Bootstrapped Accuracy | Bootstrapped TPR | Boostrapped FPR | Area Under ROC Curve |
| ---------------------------------------- |:---------------------:|:----------------:|:---------------:|:--------------------:|
| Logistic Regression                      | 81.86%                | 35.84%           | 5.08%           | 0.7237               |
| Linear Discriminant Analysis             | 66.67%                | 16.85%           | 19.23%          | 0.6438               |
| Linear Discriminant Analysis (Collinear) | 28.30%                | 87.08%           | 88.41%          | 0.6868               |
| K Nearest Neighbours                     | 82.40%                | 38.26%           | 5.05%           |                      |
| Quadratic Discriminant Analysis          | 78.65%                | 32.26%           | 8.24%           | 0.6084               |

## Interpretation:

Logistic regression (LASSO) only eliminated 2 features, but clearly placed more importance on predictors such as
`SEX`, `EDUCATION`, `MARRIAGE` and `PAY_X`, which makes intuitive sense about a debtor's ability to make payments. This 
also suggests that the probability of default is likely not very closely related to the actual amount of debt and past
payments. Information on whether past defaults occured or not is strong enough to come up with predictions of future
defaults. 

<br/>

KNN was performed with the top 10 features from LASSO. Unfortunately, `rknn::rknn` did not support randomly breaking 
ties and therefore was not possible to perform subset selection. In order to avoid the problem of data points not having
"near neighbours" in high dimensions, I opted for using the same subset of features from LASSO. Cross validation
indicated that 23 neighbours yielded the best accuracy, but there was no clear improvement from 18 neighbours to 23
neighbours. To avoid high variance, I chose 18 instead. KNN and logistic regression performed very similarly, with good
boostrapped accuracy and false-positive rate, but poor sensitivity. Depending on the objective of the prediction model,
we can adjust the decision boundary to trade accuracy for TPR. 

<br/>

`sda::sda.ranking` came up with a very strange ranking of features, which seemed to completely contradict LASSO.
Comparing the model's performance to our previous models, it seemed to have traded accuracy for no increase in
sensitivity. Note that the percentage of non-defaults in our data is 78%. Therefore simply saying that no one will
default is a more accurate prediction than the LDA model. 

Since `sda::sda.ranking` ranks based on each feature's individual contribution, I thought perhaps the discrepancy
between the selected features could be attributed to some collinear interaction between the features. The results are
even worse. The model believes that nearly everyone will default, and therefore has a very high TPR and FPR. The high
FPR renders this model practically useless. In short, I don't understand why `sda::sda.ranking` resulted in poor feature
selection.

<br/>

QDA offered slightly better results than the prior mean of non-defaults. Its TPR and FPR are similar to those of
logistic regression, but no better, which doesn't suggest that the true relationship is quadratic
