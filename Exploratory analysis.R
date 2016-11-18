
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
train_user_id<-as.factor(unique(train_data$user_id))
train_user_id_sample <- (unclass(train_user_id[sample(1:length(train_user_id),100000,replace = F)]))


new_train <- subset(train_data, subset = user_id %in% train_user_id_sample)
new_train<-new_train[order(user_id)]


train<-merge(train,preprocessed,by="NombreProducto")
test<-merge(test,preprocessed,by="NombreProducto")



