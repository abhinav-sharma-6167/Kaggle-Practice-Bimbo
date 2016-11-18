#very bad results..1.12 on lb


library(data.table)
train <- fread('train.csv', select = c('Cliente_ID','Producto_ID','Demanda_uni_equil','Agencia_ID','Canal_ID','Ruta_SAK'),
             colClasses=c(Cliente_ID="numeric",Producto_ID="numeric",Demanda_uni_equil="numeric",Agencia_ID = "numeric",Canal_ID="numeric",Ruta_SAK="numeric"))
median <- train[, median(Demanda_uni_equil)]
setkey(train, Producto_ID, Cliente_ID)
median_Prod <- train[, median(Demanda_uni_equil), by = Producto_ID]
setnames(median_Prod,"V1","M2")

median_Client_Prod <- train[, median(Demanda_uni_equil),by = .(Producto_ID,Cliente_ID)]
setnames(median_Client_Prod,"V1","M3")



median_Agency <- train[, median(Demanda_uni_equil),by = .(Agencia_ID,Canal_ID,Ruta_SAK,Producto_ID)]
median_Agency_two <- train[, median(Demanda_uni_equil),by = .(Agencia_ID,Cliente_ID)]

# median_Client_Prod<-median_Client_Prod[median_Client_Prod$M3>=2]
# median_Prod<-median_Prod[median_Prod$M2>=2]
# median_Agency<-median_Agency[median_Agency$V1>=3]

test <- fread('test.csv', 
              select = c('id','Cliente_ID','Producto_ID','Agencia_ID','Canal_ID','Ruta_SAK'),
              colClasses=c(id="numeric",Cliente_ID="numeric",Producto_ID="numeric",Agencia_ID = "numeric",Canal_ID="numeric",Ruta_SAK="numeric"))


setkey(test, Producto_ID, Cliente_ID)

# Create table called submit that joins medians (in field M3) by Product and Client to test data set
submit <- merge(test, median_Client_Prod, all.x = TRUE)

# add column M2 that contains median by Product.....look out for the M2 column at the end of the code
submit$M2 <- merge(test, median_Prod, by = "Producto_ID", all.x = TRUE)$M2




setkey(train, Agencia_ID , Canal_ID,Ruta_SAK,Producto_ID )
setkey(test,Agencia_ID,Canal_ID,Ruta_SAK,Producto_ID)

submit$M1 <- merge(test, median_Agency, all.x = TRUE)$V1



setkey(train, Agencia_ID ,Cliente_ID )
setkey(test,Agencia_ID ,Cliente_ID)

submit$M0 <- merge(test, median_Agency_two, all.x = TRUE)$V1




# Now create Predictions column; intially set to be M3 which contains median by product and client
submit$Pred <- submit$M3

# where median by product and client is null use median by product (M2)
submit[is.na(M3)]$Pred <- submit[is.na(M3)]$M2
submit[is.na(M3)]$Pred <- submit[is.na(M3)]$M1
submit[is.na(M3)]$Pred <- submit[is.na(M3)]$M0
# where median by product is null use overall median
submit[is.na(Pred)]$Pred <- median

# now relabel columns ready for creating submission
setnames(submit,"Pred","Demanda_uni_equil")

# check all looks OK
head(submit)


options(scipen = 3)
# Write out submission file.
# Any results you write to the current directory are saved as output.
write.csv(submit[,.(id,Demanda_uni_equil)],"submit_the_benchmark.csv", row.names = FALSE)

submission <-fread("submit_the_benchmark.csv")
