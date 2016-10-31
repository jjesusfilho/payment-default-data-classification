library(ISLR)
library(glmnet)
library(class)
library(sda)
library(MASS)

set.seed(1)

data = read.csv('default.csv', header = TRUE)
data$ID = NULL
data$PAY_1 = data$PAY_0 # this is a correction in the column names
data$PAY_0 = NULL

##
 # This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable.
 # This study reviewed the literature and used the following 23 variables as explanatory variables:
 # X1: Amount of the given credit (NT dollar):
 # 	it includes both the individual consumer credit and his/her family (supplementary) credit.
 # X2: Gender (1 = male; 2 = female).
 # X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
 # X4: Marital status (1 = married; 2 = single; 3 = others).
 # X5: Age (year).
 # X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows:
 # 	X6 = the repayment status in September, 2005;
 # 	X7 = the repayment status in August, 2005; . . .;
 # 	X11 = the repayment status in April, 2005.
 # 	The measurement scale for the repayment status is: -1 = pay duly;
 # 		1 = payment delay for one month; 2 = payment delay for two months; . . .;
 # 		8 = payment delay for eight months;
 # 		9 = payment delay for nine months and above.
 # X12-X17: Amount of bill statement (NT dollar).
 # 	X12 = amount of bill statement in September, 2005;
 # 	X13 = amount of bill statement in August, 2005; . . .;
 # 	X17 = amount of bill statement in April, 2005.
 # X18-X23: Amount of previous payment (NT dollar).
 # 	X18 = amount paid in September, 2005;
 # 	X19 = amount paid in August, 2005; . . .;
 # 	X23 = amount paid in April, 2005.

#############################################
#                                           #
# 		Logistic Regression (LASSO)         #
#                                           #
#############################################
x = model.matrix(default.payment.next.month ~ ., data)[, -1]
y = data$default.payment.next.month
train = sample(1:nrow(x), nrow(x) / 2)
test = (-train)
y.test = y[test]

glmmod = glmnet(x, y, alpha = 1, family = 'binomial')
plot(glmmod, xvar = 'lambda')

cv = NULL
cv.glmmod = cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.glmmod)
bestlam <- cv.glmmod$lambda.min # 0.001298234

# build model using full data
glmmod.best = glmnet(x, y, alpha = 1, lambda = bestlam, family = 'binomial')
coef(glmmod.best)
##
 # 24 x 1 sparse Matrix of class "dgCMatrix"
 #                        s0
 # (Intercept) -7.418014e-01
 # LIMIT_BAL   -7.529287e-07
 # SEX         -9.709918e-02
 # EDUCATION   -9.008426e-02
 # MARRIAGE    -1.439956e-01
 # AGE          6.923942e-03
 # PAY_1        5.786396e-01
 # PAY_2        7.926692e-02
 # PAY_3        7.659470e-02
 # PAY_4        2.442465e-02
 # PAY_5        3.412764e-02
 # PAY_6        8.669790e-03
 # BILL_AMT1   -2.287605e-06
 # BILL_AMT2    .           
 # BILL_AMT3    1.221684e-07
 # BILL_AMT4    .           
 # BILL_AMT5    5.306804e-07
 # BILL_AMT6    3.195263e-07
 # PAY_AMT1    -1.014663e-05
 # PAY_AMT2    -7.572916e-06
 # PAY_AMT3    -2.693596e-06
 # PAY_AMT4    -3.681782e-06
 # PAY_AMT5    -2.994034e-06
 # PAY_AMT6    -1.875046e-06
 ##

glm.probs = predict(glmmod.best, newx = x[test, ], type = 'response')
glm.pred = rep(0, nrow(x[test, ]))

# use k-fold validation on decision boundary
kfolds <- 10
folds = sample(1:kfolds, nrow(x), replace = TRUE)
cv.accuracy = matrix(NA, kfolds, 100)
cv.fp = matrix(NA, kfolds, 100)
cv.tp = matrix(NA, kfolds, 100)


##
 # cv.accuracy[j, i] is the accuracy percentage of the j-th fold with (i /100) as
 # the decision boundary
 ##
getMetrics = function(pred, l, total) {
    ret = NULL
    TN = 0
    FN = 0
    TP = 0
    FP = 0

    t = table(pred, l)

    if (length(t[, 1]) == 2) {
        TN = t[1, 1]
        FN = t[1, 2]
        FP = t[2, 1]
        TP = t[2, 2]
    } else if ('1' %in% rownames(t)) {
        # no 0's were predicted
        FP = t[1, 1]
        TP = t[1, 2]
    } else {
        # no 1's were predicted
        TN = t[1, 1]
        FN = t[1, 2]
    }

    ret$accuracy = (TP + TN) / total
    ret$fp = FP / (FP + TN)
    ret$tp = TP / (TP + FN)
    ret
}

plotROC = function(fp, tp) {
    fpr = colMeans(fp)
    tpr = colMeans(tp)
    plot(fpr, tpr, type = 'l', xlab = 'False Positive Rate', ylab = 'True Positive Rate', main = 'ROC Curve')
    lines(x = seq(0, 1), y = seq(0, 1), col = 'red')
}

for (j in 1:kfolds) {
	foldx = x[folds == j, ]
	foldy = y[folds == j]
	glm.pred = rep(0, nrow(foldx))
	glm.probs = predict(glmmod.best, newx = foldx, type = 'response')

	for (i in 100:0) {
        glm.pred[glm.probs > (i / 100)] = 1
        metrics = getMetrics(glm.pred, foldy, nrow(foldx))
        cv.accuracy[j, i] = metrics$accuracy
        cv.fp[j, i] = metrics$fp
        cv.tp[j, i] = metrics$tp
	}
}

plotROC(cv.fp, cv.tp)
decBounds = colMeans(cv.accuracy)
# performance trend of decision boundary
plot(decBounds, xlab = 'Decision Boundary * 100', ylab = 'Accuracy')
# decision boundary is the max average
glm.decBound <- which.max(decBounds) / 100 # 0.42


######################################
#                                    #
# 		K Nearest Neighbours         #
#                                    #
######################################
##
 # Using the rknn package to perform feature selection seems to frequently run
 # into too many ties, even after adding noise. Therefore, we are going to use 
 # LASSO to perform feature selection, be selecting features whose coefficients
 # is greater than 0.001 in the LASSO model.
 #
 # Without subset selection, KNN can suffer from the problem of data points not
 # being "close" to other points simply because too many dimensions are used in
 # calculating Euclidean distance.
 #
 # TODO: Revisit feature selection for KNN after having learned forests.
 ##

subsetCols = c('SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3',
    'PAY_4', 'PAY_5', 'PAY_6', 'default.payment.next.month')
data.subset = data[, subsetCols]
data.subset.x = model.matrix(default.payment.next.month ~ ., data.subset)[, -1]
data.subset.y = data.subset$default.payment.next.month

##
 # cross-validation to find the best value for k
 # cv.accuracy[j, i] is the accuracy percentage of the j-th fold with i as the
 # number of neighbours in the model
 ##
cv.accuracy = matrix(NA, kfolds, ncol(data.subset.x))
for (j in 1:kfolds) {
    newx = data.subset.x[folds != j, ]
    newy = data.subset.y[folds != j]
    foldx = data.subset.x[folds == j, ]
    foldy = data.subset.y[folds == j]

    for (i in 1:ncol(data.subset.x)) {
        knn.pred = knn(newx, foldx, newy, k = i)
        cv.accuracy[j, i] = getMetrics(knn.pred, foldy, nrow(foldx))$accuracy
    }
    
}

neighbours = colMeans(cv.accuracy)
plot(neighbours, xlab = 'Number of Neighbours', ylab = 'Accuracy')
bestk <- which.max(neighbours) # 10



######################################################
#                                                    #
#      Shrinkage Linear Discriminant Analysis        #
#                                                    #
######################################################
# Rank features according to correlation-adjusted t-scores
ranking.LDA = sda.ranking(x[train, ], y[train], diagonal = FALSE)
plot(ranking.LDA)
# take the top 10 features
idx = ranking.LDA[1:10, 'idx']
xLDA = x[, idx]

# build model using full dataset
sda.fit = sda(xLDA, y, diagonal = FALSE) # we will not explore DDA regression for now
sda.fit$beta
##
 #      LIMIT_BAL      PAY_AMT1      PAY_AMT4      PAY_AMT3      PAY_AMT2 
 # 0  6.427124e-07  1.286178e-06  8.950597e-07  5.431451e-07  4.750909e-07
 # 1 -2.262589e-06 -4.527830e-06 -3.150945e-06 -1.912074e-06 -1.672498e-06
 #
 #      PAY_AMT5     BILL_AMT6     BILL_AMT1      PAY_AMT6     EDUCATION
 #   1.083355e-06 -8.137378e-07  1.925636e-07  1.891543e-07  7.645156e-05
 #  -3.813816e-06  2.864662e-06 -6.778961e-07 -6.658942e-07 -2.691382e-04
 ##

# use cross validation to find best decision boundary
cv.accuracy = matrix(NA, kfolds, 100)
cv.tp = matrix(NA, kfolds, 100)
cv.fp = matrix(NA, kfolds, 100)

for (j in 1:kfolds) {
    foldx = xLDA[folds == j, ]
    foldy = y[folds == j]

    sda.pred = rep(0, nrow(foldx))
    sda.probs = predict.sda(sda.fit, foldx)
    for (i in 100:0) {
        # sda.probs$posterior[, 2] is the probabilities for default
        sda.pred[sda.probs$posterior[, 2] > (i / 100)] = 1

        metrics = getMetrics(sda.pred, foldy, nrow(foldx))
        cv.accuracy[j, i] = metrics$accuracy
        cv.fp[j, i] = metrics$fp
        cv.tp[j, i] = metrics$tp
    }
}

plotROC(cv.fp, cv.tp)
decBounds = colMeans(cv.accuracy)
plot(decBounds, xlab = 'Decision Boundary * 100', ylab = 'Accuracy')
decBoundLDA = which.max(decBounds) # 0.34


################################################
#                                              #
#       Quadratic Discriminant Analysis        #
#                                              #
################################################
qda.fit = qda(default.payment.next.month ~ ., data = data)
##
 # Call:
 # qda(default.payment.next.month ~ ., data = data)
 #
 # Prior probabilities of groups:
 #      0      1 
 # 0.7788 0.2212 
 #
 # Group means:
 #   LIMIT_BAL      SEX EDUCATION MARRIAGE      AGE      PAY_2      PAY_3      PAY_4
 # 0  178099.7 1.614150  1.841337 1.558637 35.41727 -0.3019175 -0.3162558 -0.3556326
 # 1  130109.7 1.567058  1.894665 1.528029 35.72574  0.4582580  0.3621157  0.2545208
 #
 #       PAY_5      PAY_6 BILL_AMT1 BILL_AMT2 BILL_AMT3 BILL_AMT4 BILL_AMT5
 #  -0.3894881 -0.4056240  51994.23  49717.44  47533.37  43611.17  40530.45
 #   0.1678722  0.1121157  48509.16  47283.62  45181.60  42036.95  39540.19
 #
 #   BILL_AMT6 PAY_AMT1 PAY_AMT2 PAY_AMT3 PAY_AMT4 PAY_AMT5 PAY_AMT6      PAY_1
 # 0  39042.27 6307.337 6640.465 5753.497 5300.529  5248.22 5719.372 -0.2112224
 # 1  38271.44 3397.044 3388.650 3367.352 3155.627  3219.14 3441.482  0.6681736
 ##

cv.accuracy = matrix(NA, kfolds, 100)
cv.fp = matrix(NA, kfolds, 100)
cv.tp = matrix(NA, kfolds, 100)

for (j in 1:kfolds) {
    foldx = data[folds == j, ]
    foldy = y[folds == j]

    qda.pred = rep(0, nrow(foldx))
    qda.probs = predict(qda.fit, foldx)
    for (i in 100:0) {
        qda.pred[qda.probs$posterior[, 2] > (i / 100)] = 1

        metrics = getMetrics(qda.pred, foldy, nrow(foldx))
        cv.accuracy[j, i] = metrics$accuracy
        cv.fp[j, i] = metrics$fp
        cv.tp[j, i] = metrics$tp
    }
}

plotROC(cv.fp, cv.tp)
decBounds = colMeans(cv.accuracy)
plot(decBounds, xlab = 'Decision Boundary * 100', ylab = 'Accuracy')
decBoundQDA = which.max(decBounds) # 0.99


#############################
#                           #
#       Boostrapping        #
#                           #
#############################
getResampleMSE = function(method, modelObject, threshold, sampleSize, numSamples) {
    BSErrors <- vector(, numSamples)

    for (i in 1:numSamples) {
        resample = data[sample(1:rows, sampleSize, replace = TRUE), ]
        resample.y = resample$default.payment.next.month
        resample.x = model.matrix(default.payment.next.month ~ ., resample)[, -1]

        pred = rep(0, sampleSize)
        if (method == 'logistic') {
            probs = predict(glmmod.best, newx = resample.x, type = 'response')
        } else if (method == 'lda') {
            probs = predict.sda(sda.fit, resample.x)
        } else if (method == 'knn') {
            probs = knn.pred(data.subset.x, resample.x, data.subset.y, k = threshold)
        } else if (method == 'qda') {
            probs = predict(qda.fit, resample.x)
        } else {
            break;
        }

        if (method != 'knn') {
            pred[probs > threshold] = 1
        }

        BSErrors[i] = getMetrics(pred, resample.y, sampleSize)
    }

    hist(BSErrors, main = paste(method, 'bootstrapping', sep = ' '))
    mean(BSErrors)
}

sampleSize <- 1000
numSamples <- 1000

