#Setting directory
setwd("D:/Learning/Kaggle/grupo bimbo")




#Libraries used
library(randomForest)
library(data.table)

library(ggplot2)
library(plyr)
library(dplyr)
library(xgboost)


#Setting the seed in case for random sampling later
set.seed(100)


#freading the data into data table
train <- fread("train.csv")
test <- fread("test.csv")
product <- fread("producto_tabla.csv")
client <- fread("cliente_tabla.csv")
town <- fread("town_state.csv")


train<-merge(train,product,by="Producto_ID")
train<-merge(train,client,by="Cliente_ID")
train<-merge(train,town,by="Agencia_ID")


test<-merge(test,product,by="Producto_ID")
test<-merge(test,client,by="Cliente_ID")
test<-merge(test,town,by="Agencia_ID")