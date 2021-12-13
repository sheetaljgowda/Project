##########################################################################
## Name:Sheetal Jayaram Gowda
## Created:12/01/2021
## Edited:12/10/2021
##########################################################################


########################################################################################################
###Question 1
########################################################################################################
##A pen-based handwritten digit recognition (pendigits) was obtained from 44
##writers, each of whom handwrote 250 examples of the digits 0,10,2,..,9 in a
##random order. The raw data consists of handwritten digits extracted from tablet
##coordinates of the pen at fixed time intervals. The last column in the dataset are
##the class labels (digits). The data can be found here:

##(a)Compute the variance of each of the variables and show that they are very similar.
rm(list=ls())
getwd()
load('C:/Users/Admin/Documents/UB/statistical_data_mining/HW/FINAL HW/pendigits.Rdata')
head(pendigits)
dim(pendigits)
##str(pendigits)

##Storing it another variable as Digits_data
Digits_data<-pendigits

##Splitting the data into inputs and outputs where output will have the class column which has the digits 1 to 9
inputs<-Digits_data[,1:16]
head(inputs)
output<-as.factor(Digits_data[,17])
head(output)



##scale the data
dig_inputs<-scale(inputs)
head(dig_inputs)

?prcomp
pc_dig<-prcomp(dig_inputs,center=FALSE,scale=FALSE)
summary(pc_dig)
plot(pc_dig)
pc_dig
names(pc_dig)

##percent of variation explained
pc_var<-(pc_dig$sdev)^2
per_var_exp<-(pc_var/(sum(pc_var)))*100
barplot(per_var_exp,main="PC variance explained",ylab="% variation explained",xlab="PCs")

percent_variations<-pc_var/(sum(pc_var))*100
##PC1 to PC5 contribute to 80 percent of the total variations
sum(per_var_exp[1:5])>80

##PC1 to PC8 contribute to 90 percent of the total variations
sum(per_var_exp[1:8])>90

head(pc_dig$x)        
##plot(pc_dig$x[,1],pc_dig$x[,2],xlab="PC1scores",ylab="PC2scores")
plot(pc_dig$x[,1:5],xlab="PC1scores",ylab="PC2scores")
unique(output)

my_col<-rep("black",length(dig_inputs[,1]))
c0<-which(output==0)
my_col[c0]<-"dark green"
c1<-which(output==1)
my_col[c1]<-"red"
c2<-which(output==2)
my_col[c2]<-"blue"
c3<-which(output==3)
my_col[c3]<-"orange"
c4<-which(output==4)
my_col[c4]<-"violet"
c5<-which(output==5)
my_col[c5]<-"yellow"
c6<-which(output==6)
my_col[c6]<-"green"
c7<-which(output==7)
my_col[c7]<-"pink"
c8<-which(output==8)
my_col[c8]<-"brown"
c9<-which(output==9)
my_col[c9]<-"Light blue"


plot(pc_dig$x[,1:5],xlab="PC1scores",ylab="PC2scores",col=my_col,main="coloured score plot")

?biplot
biplot(pc_dig)

install.packages("plotly")
library(plotly)
library(ggplot2)

plot_ly(x=pc_dig$x[,1],y=pc_dig$x[,2],z=pc_dig$x[,3],type="scatter3d",mode="markers",color=output)


dim(Digits_data)
names(Digits_data)

###Splitting data into test and the training data
###(b) Divide the data into test and training. Fit a kNN model over a range of "k" to the (a) raw data, and 
##(b) PCs from part (A) that capture at least 80% of the variation.Comment on your results.
set.seed(123)
indexes=sample(1:nrow(Digits_data),size=0.8*nrow(Digits_data),replace = FALSE)
Digits_data_train<-Digits_data[indexes,]
Digits_data_test<-Digits_data[-indexes,]


Knn_train_data <-Digits_data_train[,-c(17)]
Knn_test_data <-Digits_data_test[,-c(17)]


res_train_data <-Digits_data_train[,c(17)]
res_test_data <-Digits_data_test[,c(17)]
??knn_pred
require(class)
k_mod <- c(1,3,5,7,9)
k_mod.error <- rep(NA, length(k_mod))
for (i in 1:length(k_mod)) {
  knn_pred <- knn(Knn_train_data, Knn_test_data, res_train_data, k_mod[i])
  k_mod.error[i] <- mean(knn_pred != res_test_data)
}
compare<-cbind(knn_pred,res_test_data)
?knn
as.matrix(k_mod.error)
total_error <- matrix(c(k_mod.error), ncol = 1)
x11()
plot(c(1, 10), c(0, 1.1 * max(total_error)), type = "n", main = "Error Comparing", 
     ylab = "Error", xlab = "k", las=2)
points(k_mod, k_mod.error, col = 1)
lines(k_mod, k_mod.error, col = 2, lty = 3)

new_X<-pc_dig$x[,1:5]

set.seed(123)
indexes=sample(1:nrow(new_X),size=0.8*nrow(new_X),replace = FALSE)
data_train<-new_X[indexes,]
data_test<-new_X[-indexes,]
dim(data_train)
k_mod <- c(1,3,5,7,9)
k_mod.error <- rep(NA, length(k_mod))
for (i in 1:length(k_mod)) {
  knn_pred <- knn(data_train, data_test, res_train_data, k_mod[i])
  k_mod.error[i] <- mean(knn_pred != res_test_data)
}
compare<-cbind(knn_pred,res_test_data)
?knn
as.matrix(k_mod.error)
total_error_p <- matrix(c(k_mod.error), ncol = 1)
x11()
plot(c(1, 10), c(0, 1.1 * max(total_error_p)), type = "n", main = "Error Comparing for PC1 to PC5", 
     ylab = "Error", xlab = "k", las=2)
points(k_mod, k_mod.error, col = 1)
lines(k_mod, k_mod.error, col = 2, lty = 3)

#######################################################################################
#LDA
##################################################################################

library(MMST)
library(lattice)
library(caret)

set.seed(123)
indexes=sample(1:nrow(Digits_data),size=0.8*nrow(Digits_data),replace = FALSE)
Digits_data_train<-Digits_data[indexes,]
Digits_data_test<-Digits_data[-indexes,]

ldafit<-lda(class~.,data = Digits_data, family= binomial)

lda_predtest<-predict(ldafit, Digits_data_test)
lda_predtrain<-predict(ldafit, Digits_data_train)

table(lda_predtest$class, Digits_data_test$class)

True_test<-(Digits_data_test$class)
True_train<-(Digits_data_train$class)

lda_pred_test<-(lda_predtest$class)
lda_pred_train<-(lda_predtrain$class)

conf_lda_test <- confusionMatrix(as.factor(lda_pred_test), as.factor(True_test))
conf_lda_test

conf_lda_train <- confusionMatrix(as.factor(lda_pred_train), as.factor(True_train))
conf_lda_train

## Model Accuracy
mean(lda_pred_test==True_test)
## 0.8781264
mean(lda_pred_train==True_train)
## 0.8768338

## LDA Error
mean(lda_pred_test!=True_test)
## 0.1218736
mean(lda_pred_train!=True_train)
## 0.1231662


#####################################################################################################################
###         Question 3
#######################################################################################################################

###3Q: This problem involves the OJ data set which is part of the ISLR2 package.
##(a)Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
rm(list = ls())
##install.packages("ROCR")
library(ISLR2)
library(e1071)
library(MASS)
library(ROCR)

data(OJ)
dim(OJ)
?OJ
my_data<-OJ
set.seed(123)
split <- sample(1:nrow(my_data), 800, replace = FALSE)
training <- my_data[split, ]
dim(training)
test <- my_data[-split, ]
dim(test)


head(my_data)
#######################################
## SVM with a linear kernel
#######################################
class(my_data$Purchase) ##response is as factor
fit <- svm(Purchase~., data = training, kernel = "linear", cost = .01, scale = FALSE)
summary(fit)
names(fit)
fit$index

##plot(fit, data = training, formula = STORE ~ Purchase, fill = FALSE)


##(c)What are the training and test error rates?

# predict the test data 
y_hat <- predict(fit, newdata = test)
y_true <- test$Purchase
acc <- length(which(y_hat == y_true))/length(y_true)
acc ## 0.762963
test_error <- length(which(y_hat != y_true))/length(y_true)
test_error ## 0.237037
test_error_1=0.237037
table(y_hat, y_true)

# predict the train data 
y_hat <- predict(fit, newdata = training)
y_true <- training$Purchase
acc <- length(which(y_hat == y_true))/length(y_true)
acc ## 0.775
training_error <- length(which(y_hat != y_true))/length(y_true)
training_error ## 0.225
table(y_hat, y_true)

##(d) Use the tune() function to select an optimal cost. Consider values in the range 0.01 to 10.
tune.model <- tune(svm, Purchase~., data = training, kernel = "linear",ranges = list(cost=c(.01, .1, 1, 5, 10)))
summary(tune.model)
##Error estimation of 'svm' using 10-fold cross validation: 0.16875
bestmod <- tune.model$best.model


##(e)Compute the training and test error rates using this new value for cost
# predict the test data 
y_hat <- predict(bestmod, newdata = test)
y_true <- test$Purchase
acc <- length(which(y_hat == y_true))/length(y_true)
acc ## 0.8407407
test_error <- length(which(y_hat != y_true))/length(y_true)
test_error ##0.1592593
test_error_2=0.1592593
table(y_hat, y_true)

# predict the train data 
y_hat <- predict(bestmod, newdata = training)
y_true <- training$Purchase
acc <- length(which(y_hat == y_true))/length(y_true)
acc ## 0.8375
training_error <- length(which(y_hat != y_true))/length(y_true)
training_error ##0.1625
table(y_hat, y_true)

#(f) Repeat parts (b) through (e) using a support vector machine with a radial
#kernel. Use the default value for gamma.


#################################
## SVM with a radial kernel
#################################
##tune.model.rad <- tune(svm, Purchase ~., data = training, kernel = "radial",ranges = list(cost = c(.01, .1, 1, 5, 10), gamma = c(.1, .5, 1, 2, 3, 4)))
?tune
tune.model.rad <- tune(svm, Purchase ~., data = training, kernel = "radial",ranges = list(cost = c(.01, .1, 1, 5, 10)))
summary(tune.model.rad)
names(tune.model.rad)
bestmodrad <- tune.model.rad$best.model
bestmodrad

# predict the test data 
y_hat <- predict(bestmodrad, newdata = test)
y_true <- test$Purchase
acc.rad <- length(which(y_hat == y_true))/length(y_true)
acc.rad ## 0.8111111
test_error <- length(which(y_hat != y_true))/length(y_true)
test_error ##0.1888889
test_error_3=0.1888889

# predict the train data 
y_hat <- predict(bestmodrad, newdata = training)
y_true <- training$Purchase
acc.rad <- length(which(y_hat == y_true))/length(y_true)
acc.rad ## 0.86125
training_error <- length(which(y_hat != y_true))/length(y_true)
training_error ##0.86125

library(caret)
##confusionMatrix(y_hat, y_true)

?svm
#(g) Repeat parts (b) through (e) using a support vector machine with a
#polynomial kernel. Set degree = 2.

################################
## SVM with a polynomial kernel
#################################
tune.model.poly <- tune(svm, Purchase ~., data = training, kernel = "polynomial",
                        ranges = list(cost = c(.01, .1, 1, 5, 10)),degree=2)
?tune
summary(tune.model.poly)
names(tune.model.rad)
bestmodpoly <- tune.model.poly$best.model
bestmodpoly


# predict the test data 
y_hat <- predict(bestmodpoly, newdata = test)
y_true <- test$Purchase
acc.rad <- length(which(y_hat == y_true))/length(y_true)
acc.rad ##  0.7962963
test_error <- length(which(y_hat != y_true))/length(y_true)
test_error ##0.2037037
test_error_4=0.2037037

# predict the train data 
y_hat <- predict(bestmodpoly, newdata = training)
y_true <- training$Purchase
acc.rad <- length(which(y_hat == y_true))/length(y_true)
acc.rad ## 0.85625
training_error <- length(which(y_hat != y_true))/length(y_true)
training_error ## 0.14375

#(h) Overall, which approach seems to give the best results on this data?
overall_error<-cbind(test_error_1,test_error_2,test_error_3,test_error_4)

#####  test_error_1 test_error_2 test_error_3 test_error_4
#[1,]     0.237037    0.1592593    0.1888889    0.2037037
##We can see test_error_2 is the least obtained from the 2nd method 


#####################################################################################################################
###         Question 5
#######################################################################################################################
###   5) Fit a series of random-forest classifiers to the SPAM data, to explore the
###   sensitivity to m (the number of randomly selected inputs for each tree). Plot both the
###   OOB error as well as the test error against a suitably chosen range of values for m.
rm(list=ls())
##install.packages("ElemStatLearn")
library(ElemStatLearn)
library(randomForest)
dim(spam)
## 4601   58
head(spam)
str(spam)
spam_data<-spam
set.seed(333)
split<- sample(1:nrow(spam_data),round(0.80*nrow(spam_data)),replace = FALSE)
spam_data_train<-spam_data[split,]
dim(spam_data_train)
##3681   58
spam_data_test<-spam_data[-split,]
dim(spam_data_test)
##920  58
?randomForest
rm_model=randomForest(spam~., data=spam_data_train,mtry =5,ntree=1000)
mean(rm_model$err.rate)
ooj<-rm_model$err.rate
varImpPlot(rm_model)
summary(rm_model)

range <- c(3,5,7,8,9,13,23,33,43,53)
num<-1000
rm_test_error <- rep(NA, length(range))
Error_OOB<- rep(NA, length(range))
for (i in 1:length(range)) {
  rm_model = randomForest(spam~., data=spam_data_train,mtry=range[i],ntree=1000)
  rmpred <- predict(rm_model, spam_data_test, type = "class")
  rm_test_error[i] <- mean(rmpred != spam_data_test$spam)
  Error_OOB[i]<-as.numeric(rm_model$err.rate[num,1])
}


as.matrix(rm_test_error)
as.matrix(Error_OOB)
cbind(rm_test_error,Error_OOB)
x11()
plot(rm_test_error~range, col = "Dark blue", type = "b", xlab = "Values of m", ylab = "Rate")
lines(Error_OOB~range, col = "dark green", type = "b")
legend(40, 0.055, legend=c("rm_test_error", "Error_OOB"),col=c("Dark blue", "dark green"), lty=1:2, cex=0.8)








