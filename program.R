# load libraries
library(caret)
library(klaR)
library(mlbench)
library(caretEnsemble)
library(dplyr)
library(ROCR)
library(ggplot2)
library(MLeval)
library(pROC)
library(rpart)
library(plotROC)


###-----------------------------------------

# load the dataset 
dataset <- read.csv("C:/Users/jayasaigoutheman/Desktop/Datamining/Dataset/seeds.csv")

dataset = dataset[dataset$Wheat!="Canadian", ]
dataset$Wheat = factor(dataset$Wheat)

#randomise order
dataset <- dataset[sample(1:nrow(dataset)), ]

#understanding data
summary(dataset)
head(dataset)
str(dataset)
glimpse(dataset)
boxplot(dataset)

#check if dataset contains any missing value
sum(is.na(dataset))


# calculate correlation matrix
correlationMatrix <- cor(dataset[,1:7])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# define an 80%/20% train/test split of the dataset
split=0.80
trainIndex <- createDataPartition(dataset$Wheat, p=split, list=FALSE)
data_train <- dataset[ trainIndex,]
data_test <- dataset[-trainIndex,]

###Bagging
###-----------------------------------------
set.seed(12345)
metric <- "Accuracy"

# define training control
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
model <- train(Wheat~., data=data_train, trControl=control, method="nb")
print(model)

# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


model.treebag <- train(Wheat~., data=data_train, method="treebag",  metric=metric, trControl=control)
model.rf <- train(Wheat ~., data = data_train, method="rf", metric=metric, trControl=control)

# Summarize the results
print(model.treebag)
print(model.rf)


bagging_results <- resamples(list(treebag=model.treebag, rf=model.rf))

summary(bagging_results)  
dotplot(bagging_results)


#prediction
pred.treebag <- predict(model.treebag, data_test , type='raw')
pred.rf <- predict(model.rf, data_test)

pred.treebag
pred.rf

#confusion matrix
result.treebag <- confusionMatrix(data_test$Wheat,pred.treebag)
result.rf <- confusionMatrix(data_test$Wheat,pred.rf)

result.treebag
result.rf

#Precision & Recall estimation
result.treebag$byClass
result.rf$byClass

#Precision Recall, ROC & RUAC : treebag
pred.treebag <- prediction(as.numeric(pred.treebag), as.numeric(data_test$Wheat))
RP.treebag  <- performance(pred.treebag, "prec", "rec")
pred.treebag
plot (RP.treebag)
abline(a = 0, b = 1)

ROC.treebag <- performance(pred.treebag, "tpr", "fpr");
ROC.treebag
plot (ROC.treebag)
abline(a = 0, b = 1)

auc.treebag <- performance(pred.treebag, measure = "auc")
auc.treebag <- auc.treebag@y.values[[1]]
auc.treebag
----------
pred.rf <- prediction(as.numeric(pred.rf), as.numeric(data_test$Wheat))
RP.rf <- performance(pred.rf, "prec", "rec")
plot (RP.rf)
abline(a = 0, b = 1)

ROC.rf <- performance(pred.rf, "tpr", "fpr");
plot (ROC.rf)
abline(a = 0, b = 1)

auc.rf <- performance(pred.rf, measure = "auc")
auc.rf<- auc.rf@y.values[[1]]
auc.rf



###-----------------------------------------

start.time.train <- Sys.time()
model.treebag <- train(Wheat~., data=data_train, method="treebag",  metric=metric, trControl=control)
end.time.train <- Sys.time()
time.taken.train <- end.time.train - start.time.train

# Time taken to train Treebag
time.taken.train

###-----------------------------------------

start.time.train <- Sys.time()
model.rf <- train(Wheat ~., data = data_train, method="rf", metric=metric, trControl=control)
end.time.train <- Sys.time()
time.taken.train <- end.time.train - start.time.train

# Time taken to train Random Forest
time.taken.train

###-----------------------------------------

start.time.test <- Sys.time()
pred.treebag = predict(model.treebag, data_test)
end.time.test  <- Sys.time()
time.taken.test  <- end.time.test  - start.time.test 

# Time taken to test treebag
time.taken.test

###-----------------------------------------

start.time.test <- Sys.time()
pred.rf = predict(model.rf, data_test)
end.time.test  <- Sys.time()
time.taken.test  <- end.time.test  - start.time.test 

# Time taken to test Random Forest
time.taken.test


###Stacking 
###-----------------------------------------

algorithmList <- c('rpart', 'knn', 'nb')
set.seed(12345)
models <- caretList(Wheat~., data=data_train, trControl=control, methodList=algorithmList)
results <- resamples(models)

summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)

# stack using knn
stack <- caretStack(models, method="knn", metric="Accuracy", trControl=control)
stack



#prediction: rpart
stackpredicted.rpart <- predict(models$rpart, data_test)
result.rpart <- confusionMatrix(stackpredicted.rpart, data_test$Wheat)

#prediction: knn
stackpredicted.knn <- predict(models$knn, data_test)
result.knn <- confusionMatrix(stackpredicted.knn, data_test$Wheat)

#prediction: nb
stackpredicted.nb <- predict(models$nb, data_test)
result.nb <- confusionMatrix(stackpredicted.nb, data_test$Wheat)

#confusion table
result.rpart
result.knn
result.nb

#Precision & Recall estimation
result.rpart$byClass
result.knn$byClass
result.nb$byClass



#ROC & RUAC : RPART
predicted.rpart <- prediction(as.numeric(stackpredicted.rpart), as.numeric(data_test$Wheat))
perf.treebag <- performance(predicted.rpart, measure = "tpr", x.measure = "fpr")
plot(perf.treebag, main = "ROC curve",colorize = T)
abline(a = 0, b = 1)

auc.treebag <- performance(predicted.rpart, measure = "auc")
pred.treebag <- auc.treebag@y.values[[1]]
pred.treebag


#ROC & RUAC : KNN
predicted.knn <- prediction(as.numeric(stackpredicted.knn), as.numeric(data_test$Wheat))
perf.knn <- performance(predicted.knn, measure = "tpr", x.measure = "fpr")
plot(perf.knn, main = "ROC curve",colorize = T)
abline(a = 0, b = 1)

auc.knn <- performance(predicted.knn, measure = "auc")
stackpredicted.knn <- auc.knn@y.values[[1]]
stackpredicted.knn


#ROC & RUAC : NB
predicted.nb <- prediction(as.numeric(stackpredicted.nb), as.numeric(data_test$Wheat))
perf.nb <- performance(predicted.nb, measure = "tpr", x.measure = "fpr")
plot(perf.nb, main = "ROC curve",colorize = T)
abline(a = 0, b = 1)

auc.nb <- performance(predicted.nb, measure = "auc")
stackpredicted.nb <- auc.knn@y.values[[1]]
stackpredicted.nb


###-----------------------------------------


start.time.train <- Sys.time()
models <- caretList(Wheat~., data=data_train, trControl=control, methodList=algorithmList)
end.time.train <- Sys.time()
time.taken.train <- end.time.train - start.time.train

# Time taken to train stacking
time.taken.train

start.time.test <- Sys.time()
stackpredicted.rpart <- predict(models$rpart, data_test)
end.time.test  <- Sys.time()
time.taken.test  <- end.time.test  - start.time.test

# Time taken to test treebag
time.taken.test

start.time.test <- Sys.time()
stackpredicted.knn <- predict(models$knn, data_test)
end.time.test  <- Sys.time()
time.taken.test  <- end.time.test  - start.time.test

# Time taken to test treebag
time.taken.test

start.time.test <- Sys.time()
stackpredicted.nb <- predict(models$nb, data_test)
end.time.test  <- Sys.time()
time.taken.test  <- end.time.test  - start.time.test 

# Time taken to test treebag
time.taken.test



