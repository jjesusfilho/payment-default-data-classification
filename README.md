# Classification on Payment Default Data

[Data source: Universit of California, Irvine, Machine Learning respository](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Performance Summary:

| Model                                    | Bootstrapped Accuracy | Bootstrapped TPR | Boostrapped FPR | Area Under ROC Curve |
| ---------------------------------------- |:---------------------:|:----------------:|:---------------:|:---------------------|
| Logistic Regression                      | 81.86%                | 35.84%           | 5.075%          | 0.7237               |
| Linear Discriminant Analysis             | 66.67%                | 16.85%           | 19.23%          | 0.6438               |
| Linear Discriminant Analysis (Collinear) |                       |                  |                 | 0.6868               |
| K Nearest Neighbours                     |                       |                  |                 |                      |
| Quadratic Discriminant Analysis          |                       |                  |                 | 0.6084               |
