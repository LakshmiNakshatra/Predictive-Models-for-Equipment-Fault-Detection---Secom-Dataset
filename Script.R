## Loading the datasets
secom.data=read.table("secom.data")
secom.labels=read.table("secom_labels.data")

## Analyzing the datasets
dim(secom.data)
head(secom.data)
str(secom.data)
sum(is.na(secom.data))
dim(secom.labels)
head(secom.labels)
str(secom.labels)
sum(is.na(secom.labels))

## Cleaning and Imputing data
# Removing columns where missing values are greater than or equal to 50%
secom.cleandata <- secom.data[, -which(colMeans(is.na(secom.data)) >= 0.5)]
dim(secom.cleandata)

# Imputing columns with less than 50% missing vales
install.packages("zoo")
library(zoo)
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = T)+rnorm(sum(is.na(x)))*sd(x, na.rm = T))
secom.procdata= replace(secom.cleandata, TRUE, lapply(secom.cleandata, NA2mean))
sum(is.na(secom.procdata))
dim(secom.procdata)

#Removing columns with zero variance/constant values
secom.procdata=secom.procdata[,apply(secom.procdata, 2, var, na.rm=TRUE) != 0]
dim(secom.procdata)

## PCA and Dimensionality Reduction
# Fitting Principal Components
pr.out=prcomp(secom.procdata,scale=TRUE)
names(pr.out)
summary(pr.out)
dim(pr.out$x)
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)
plot(cumsum(pve), col="red", xlab="Principal Component",ylab="Cumulative Proportion of Variance Explained",ylim=c(0,1),type="b")

# First 100 Principal Components which explain about 80% of the variation are selected for further analysis
loadings <- as.data.frame(pr.out$rotation[,1:100])
dim(loadings)

## Adding response variable to PCs
data.final=as.data.frame(pr.out$x[,1:100])
data.final=cbind(data.final,secom.labels$V1)
colnames(data.final)[101]="Response"
yield=ifelse(data.final$Response==-1,"Pass","Fail")
data.final=data.frame(data.final,yield)
dim(data.final)
head(data.final)

## Train-Test split
install.packages("ISLR")
library(ISLR)
attach(data.final)

# The entire dataset is split into train set containing 80% of data and test set containing 20% of data 
set.seed(65)
train=sample(1:nrow(data.final),nrow(data.final)*0.80)
train.data=data.final[train,]
test.data=data.final[-train,]
dim(train.data)
dim(test.data)

# Response variable of test set is assigned to a new variable 
Yield.test=test.data$yield

## Applying SMOTE function to Oversample Minority class(Fail) and Undersample majority class(Pass) 
install.packages("DMwR")
library(DMwR)
set.seed(60)
data.smote <- SMOTE(yield ~ .-Response, data=train.data, perc.over = 100, perc.under=400)
table(data.smote$yield)

## Fitting Decision Tree
install.packages("tree")
library(tree)
set.seed(1)
tree.finaldata=tree(yield~.-Response,data=data.smote)
summary(tree.finaldata)

# Plotting the tree
plot(tree.finaldata)
text(tree.finaldata,pretty=0)

# Predicting test set results
tree.pred=predict(tree.finaldata,test.data,type="class")
table(tree.pred,Yield.test)
mean(tree.pred==Yield.test)
mean(tree.pred!=Yield.test)

## Tree Pruning
# Applying Cross Validation to find the optimal tree size
set.seed(1)
cv.finaldata=cv.tree(tree.finaldata,FUN=prune.misclass)
plot(cv.finaldata$size,cv.finaldata$dev,type="b")

# Fitting and plotting pruned tree
prune.finaldata=prune.misclass(tree.finaldata,best=16)
summary(prune.finaldata)
plot(prune.finaldata)
text(prune.finaldata,pretty=0)

# Predicting test set results using pruned tree
prunetree.pred=predict(prune.finaldata,test.data,type="class")
table(prunetree.pred,Yield.test)
mean(prunetree.pred==Yield.test)
mean(prunetree.pred!=Yield.test)

## Random Forest
install.packages("randomForest")
library(randomForest)
set.seed(133)
rf.finaldata=randomForest(yield~.-Response, data=train.data,mtry=10,ntree=5,importance=TRUE)
varImpPlot(rf.finaldata)

# Predicting test set results 
rf.pred=predict(rf.finaldata,test.data,type="class")
table(rf.pred,Yield.test)
mean(rf.pred==Yield.test)
mean(rf.pred!=Yield.test)

## Bagging
#install.packages("randomForest")
#library(randomForest)
set.seed(168)
bag.finaldata=randomForest(yield~.-Response, data=train.data,mtry=100,ntree=5,importance=TRUE)
varImpPlot(bag.finaldata)

# Predicting test set results
bag.pred=predict(bag.finaldata,test.data,type="class")
table(bag.pred,Yield.test)
mean(bag.pred==Yield.test)
mean(bag.pred!=Yield.test)

## Boosting
# Data pre-processing
Yield.boosttrain=ifelse(data.smote$yield=="Pass",1,0)
Yield.boosttest=ifelse(test.data$yield=="Pass",1,0)
data.boosttrain=data.frame(data.smote,Yield.boosttrain)
data.boosttest=data.frame(test.data [,1:100],Yield.boosttest)

# Fitting boosting algorithm
install.packages("gbm")
library(gbm)
set.seed(100)
boost.finaldata=gbm(Yield.boosttrain~.-yield-Response,data=data.boosttrain,distribution="bernoulli",
                    n.trees=100,interaction.depth=4)
summary(boost.finaldata)

# Predicting test set results
boost.prob=predict(boost.finaldata,newdata=data.boosttest,type="response",n.trees=100)
boost.pred=rep("0",nrow(data.boosttest))
boost.pred[boost.prob>=0.665]="1"
table(boost.pred,Yield.boosttest)
mean(boost.pred==Yield.boosttest)
mean(boost.pred!=Yield.boosttest)

## kNN
# Data pre-processing
names(train.x)
names(data.smote)
train.x= data.smote[,1:100]
yield.train.knn=data.smote$yield
test.x=test.data[,1:100]

# Fitting kNN algorithm
install.packages("class")
library(class)
set.seed(10)
knn.pred=knn(train.x,test.x,yield.train.knn,k=129)
table(knn.pred,Yield.test)
print(mean(knn.pred==Yield.test))

## Naive Bayes
install.packages("e1071")
library(e1071)
set.seed(1)
naive.fit=naiveBayes(yield~.-Response,data=train.data)
summary(naive.fit)

# Predicting test set results
naive.pred=predict(naive.fit,test.data)
table(naive.pred,Yield.test)
mean(naive.pred==Yield.test)
mean(naive.pred!=Yield.test)

## SVM-Radial
#library(e1071)
# Tuning to obtain the best parameter estimates
tune.out=tune(svm,yield~.-Response,data=data.smote,kernel="radial",ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),gamma=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out)

# Fitting SVM-Radial
radial.svmfit=svm(yield~.-Response,data=data.smote,kernel="radial",cost=5,gamma=.01)

# Predicting test set results
radial.pred=predict(radial.svmfit,test.data)
table(radial.pred, Yield.test)
mean(radial.pred==Yield.test)
mean(radial.pred!=Yield.test)

## SVM-Polynomial
#library(e1071)
# Tuning to obtain the best parameter estimates
tune.out=tune(svm,yield~.-Response,data=data.smote,kernel="polynomial",ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),degree=c(1,2,3,4,5)))
summary(tune.out)

# Fitting SVM-Polynomial
poly.svmfit=svm(yield~.-Response,data=data.smote,kernel="polynomial",cost=10,degree=3)

# Predicting test set results
poly.pred=predict(poly.svmfit,test.data)
table(poly.pred,Yield.test)
mean(poly.pred==Yield.test)
mean(poly.pred!=Yield.test)

## SVM-Linear
#library(e1071)
# Tuning to obtain the best parameter estimates
tune.out=tune(svm,yield~.-Response,data=data.smote,kernel="linear",ranges=list(cost=c(0.1,1,5,10,100)))
summary(tune.out)

# Fitting SVM-Linear
linear.svmfit=svm(yield~.-Response,data=data.smote,kernel="linear",cost=1)
summary(linear.svmfit)

# Predicting test set results
linear.pred=predict(linear.svmfit, test.data)
table(linear.pred, Yield.test)
mean(linear.pred==Yield.test)
mean(linear.pred!=Yield.test)

## LDA
install.packages("MASS")
library(MASS)
set.seed(1)
lda.fit=lda(yield~.-Response,data=data.smote)
summary(lda.fit)

# Predicting test set results
lda.pred=predict(lda.fit,test.data)
lda.class=lda.pred$class
table(lda.class,Yield.test)
mean(lda.class==Yield.test)
mean(lda.class!=Yield.test)

## Logistic Regression
#library(MASS)
set.seed(100)
logistic.fit=glm(yield~.-Response,data=data.smote,family = binomial)
summary(logistic.fit)

# Analyzing VIF
install.packages("car")
library(car)
vif(logistic.fit)

# Fitting Logistic Regression model after eliminating PCs with VIF values greater than 5 
set.seed(100)
logistic.fit=glm(yield~.-Response-PC63-PC64-PC71-PC55-PC56-PC38-PC40-PC41-PC43-PC47-PC34-PC35-PC36-PC31
                 -PC29-PC27-PC25-PC13-PC15-PC16-PC17-PC18-PC19-PC20-PC21-PC22-PC23-PC24-PC1-PC3-PC4-PC5-PC6-PC7-PC8-PC9, 
                 data=data.smote,family = binomial)
summary(logistic.fit)

# Predicting test set results
logistic.probs=predict(logistic.fit,test.data, type="response")
logistic.pred=rep("Fail",314)
logistic.pred[logistic.probs>0.63]="Pass"
table(logistic.pred, Yield.test)
mean(logistic.pred==Yield.test)
mean(logistic.pred!=Yield.test)
