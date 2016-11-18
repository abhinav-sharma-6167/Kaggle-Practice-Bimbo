
library(data.table)
library(ggplot2)
library(plotly)
library(animation)
library(corrplot)

train <- fread('train.csv')
test <- fread('test.csv')
product <- fread("producto_tabla.csv")
client <- fread("cliente_tabla.csv")
town <- fread("town_state.csv")


train<-merge(train,product,by="Producto_ID")
train<-merge(train,client,by="Cliente_ID")
train<-merge(train,town,by="Agencia_ID")


test<-merge(test,product,by="Producto_ID")
test<-merge(test,client,by="Cliente_ID")
test<-merge(test,town,by="Agencia_ID")

preprocessed<-fread('preprocessed_products.csv')
setnames(preprocessed,"product_name","NombreProducto")
preprocessed<-as.data.table(preprocessed)
preprocessed[,ID:=NULL]

sol <-fread('submit1.csv')


#Sampling things
id <- 1:nrow(train)
set.seed(500)
train_sample <-sample(id,0.4*nrow(train))
train<-train[train_sample]

train<-merge(train,preprocessed,by="NombreProducto")
test<-merge(test,preprocessed,by="NombreProducto")



