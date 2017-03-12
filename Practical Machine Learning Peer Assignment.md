---
title: "Practical Machine Learning Peer Assignment"
author: "Tan Tong Sheng"
date: "March 10, 2017"
output: html_document
---
*Title : Practical Machine Learning Peer Assignment*

*1) Executive Summary:*

*This report uses machine learning algorithms to predict the manners of Human Activity Recognition. Six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). This report will describe how the data captured are used to identify the parameters involved in predicting the movement involved based on the above classification, and then to predict the movement for 20 test cases. Three machine learning algorithm are used in prediction such as Random Forest, Decision Tree, Generalized Boosted Model. Random forest is found to be the most accuracy and used to answer 20 quiz.*

*2) Data Processing*

*(a) Loading packages and Reading data* 

```r
#Setting environment
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
set.seed(12345)

#Data loading for train and test data
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Download the dataset
training <- read.csv(url(UrlTrain))
testing <- read.csv(url(UrlTest))

#Create a partition with the training dataset 
inTrain <- createDataPartition(training$classe, p=0.7, list = FALSE)
TrainSet <- training[inTrain, ]
TestSet <- training[-inTrain, ]
dim(TrainSet)
```

```
## [1] 13737   160
```

```r
dim(TestSet)
```

```
## [1] 5885  160
```

*Both dataset shows there are 160 variables. They have plenty of NA that needed to remove from the dataset.*
*(b) Cleaning dataset*

```r
#Remove variables with Nearly Zero Variables
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet <- TestSet[, -NZV]
dim(TrainSet)
```

```
## [1] 13737   106
```

```r
dim(TestSet)
```

```
## [1] 5885  106
```

```r
#Remove variables that are mostly NA
AllNA <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```

```
## [1] 13737    59
```

```r
dim(TestSet)
```

```
## [1] 5885   59
```

```r
# remove identification only variables (columns 1 to 5)
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```

```
## [1] 13737    54
```

```r
dim(TestSet)
```

```
## [1] 5885   54
```
*After cleaning the dataset, the number of variables for the analysis has been reduced to 54.*

*(c) Correlation Analysis*
*A correlation among variables is analysed before proceed to further analysis.*

```r
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = 'FPC', method = 'color',type = 'lower',t1.cex = 0.8, t1.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png)

*The highly correlated variables are shown in dark colours. However, as the correlations are quite few, Principal Component Analysis is not needed for the pre-processing.*

*3) Prediction Model Building*

*Random forest, decision trees and generalized boosted model are used to model the regression (in the Train dataset) and model with higher accuracy will be applied for the quiz questions. A confusion matrix is plotted at the end of each analysis to visualize the accuracy of the models.*

*(a) Random Forest*

```r
#model fit
set.seed(12345)
controlRF <- trainControl(method='cv',number=3,verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method='rf',trControl=controlRF)
modFitRandForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    5 2652    1    0    0 0.0022573363
## C    0    5 2390    1    0 0.0025041736
## D    0    0    7 2245    0 0.0031083481
## E    0    1    0    5 2519 0.0023762376
```

```r
#prediction on Test dataset
predictRandForest <- predict(modFitRandForest,newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1133    4    0    0
##          C    0    1 1022    8    0
##          D    0    0    0  956    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9961   0.9917   0.9972
## Specificity            0.9988   0.9992   0.9981   0.9994   1.0000
## Pos Pred Value         0.9970   0.9965   0.9913   0.9969   1.0000
## Neg Pred Value         1.0000   0.9987   0.9992   0.9984   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1925   0.1737   0.1624   0.1833
## Detection Prevalence   0.2853   0.1932   0.1752   0.1630   0.1833
## Balanced Accuracy      0.9994   0.9969   0.9971   0.9955   0.9986
```

```r
#plot matrix results
plot(confMatRandForest$table, col=confMatRandForest$byClass, main=paste("Random Forest - Accuracy =",round(confMatRandForest$overall['Accuracy'],4)))
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png)

*(b) Decision Trees*

```r
#model fit
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)

```r
#prediction on Test dataset
predictDecTree <- predict(modFitDecTree,newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree,TestSet$classe)
confMatDecTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530  269   51   79   16
##          B   35  575   31   25   68
##          C   17   73  743   68   84
##          D   39  146  130  702  128
##          E   53   76   71   90  786
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7368         
##                  95% CI : (0.7253, 0.748)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6656         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9140  0.50483   0.7242   0.7282   0.7264
## Specificity            0.9014  0.96650   0.9502   0.9100   0.9396
## Pos Pred Value         0.7866  0.78338   0.7543   0.6131   0.7305
## Neg Pred Value         0.9635  0.89051   0.9422   0.9447   0.9384
## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
## Detection Rate         0.2600  0.09771   0.1263   0.1193   0.1336
## Detection Prevalence   0.3305  0.12472   0.1674   0.1946   0.1828
## Balanced Accuracy      0.9077  0.73566   0.8372   0.8191   0.8330
```

```r
#plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, main = paste("Decision Tree - Accuracy =",
round(confMatDecTree$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-2.png)

*(c) Generalized Boosted Model*

```r
#model fit
set.seed(12345)
controlGBM <- trainControl(method='repeatedcv',number=5,repeats=1)
modFitGBM <- train(classe ~ ., data=TrainSet, method="gbm",trControl=controlGBM,verbose=FALSE)
modFitGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 43 had non-zero influence.
```

```r
#predict on Test dataset
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    9    0    3    0
##          B    2 1117   20    2    1
##          C    0   11 1004   14    3
##          D    1    2    2  944   11
##          E    1    0    0    1 1067
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9859          
##                  95% CI : (0.9825, 0.9888)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9822          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9807   0.9786   0.9793   0.9861
## Specificity            0.9972   0.9947   0.9942   0.9967   0.9996
## Pos Pred Value         0.9929   0.9781   0.9729   0.9833   0.9981
## Neg Pred Value         0.9990   0.9954   0.9955   0.9959   0.9969
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1898   0.1706   0.1604   0.1813
## Detection Prevalence   0.2858   0.1941   0.1754   0.1631   0.1816
## Balanced Accuracy      0.9974   0.9877   0.9864   0.9880   0.9929
```

```r
#plot matrix results
plot(confMatGBM$table,col=confMatGBM$byClass,main=paste("GBM - Accuracy=",round(confMatGBM$overall['Accuracy'],4)))
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png)

*4)Applying the Selected Model to the Test Data*

*The accuracy of the 3 regression modeling methods above are:*

*(a) Random Forest: 0.9964*

*(b) Decision Tree: 0.7368*

*(c) GBM: 0.9859*

*Therefore, the random forest model will be applied to predict the 20 quiz results as shown below*

```r
predictTEST <- predict(modFitRandForest,newdata=testing)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

