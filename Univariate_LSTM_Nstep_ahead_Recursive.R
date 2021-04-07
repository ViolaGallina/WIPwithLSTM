library(fst)
library(data.table)
library(lubridate)
library(tidyr)
library(ggplot2)
library(keras)
library(tensorflow)
library(caret)


rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memory and report the memory usage.
setwd("C:/Users/X")
WIP = read.fst('WIP.fst', as.data.table = T)
DT <- WIP[,c("timeScale", "WIP")]
##### data preparation #####

ggplot(DT, aes(x=timeScale, y=WIP/1000))+
  geom_line(colour="#179c7d", size=1)+   # Fh green
  labs(x="Time", y="WIP[t] ", 
       title ="Work in progress for 4201")+
  theme(plot.title = element_text(hjust=0.5)) # plot in to the middle

##### split: train + test  & normalization #####
# the first 70 is the training set, the last 30 is the test set
k <- round(nrow(DT) * 0.7, digits = 0)
train <- DT[1:k, ]
test <- DT[(k+1):nrow((DT)),]
#normalization: data should be between 0-1 or -1-1
scale_data <- function(train, test, feature_range = c(0,1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = (x - min(x)) / (max(x) - min(x))
  std_test = (test - min(x)) / (max(x) - min(x))
  
  scaled_train = std_train * (fr_max - fr_min) + fr_min
  scaled_test = std_test * (fr_max - fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
}
#Function to reverse scale data for prediction
reverse_scaling <- function(scaled, scaler, feature_range = c(0,1)) {
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for(i in 1:t) {
    X = (scaled[i] - mins) / (maxs - mins)
    rawValues = X * (max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}
#scaling
Scaled <- scale_data(train$WIP, test$WIP, c(0,1))

# determining the time window
window = 15   # the number of historical data used for prediciton is defined
# To cover all elements in a vector, we'll add a 'window' into the last part of  'a'
# vector by replicating the last element.
# Creating x - input, and y - output data: training and test sets
#training
x_train = NULL
y_train = NULL
for(i in 1:nrow(train)){
  s = i-1+window
  x_train = rbind(x_train,Scaled$scaled_train[i:s])
  y_train = rbind(y_train,Scaled$scaled_train[s+1])
}
#check for NAs
which(is.na(x_train))
x_train <- x_train[1:(nrow(train)-window),]
which(is.na(y_train))
y_train <- y_train[1:(nrow(train)-window)]
#  Input data should be an array type, so we'll reshape it.
X_train = array(x_train, dim=c(nrow(x_train), window,1))
Y_train = array(y_train, dim= c(length(y_train), 1, 1))
# test
x_test = NULL
y_test = NULL
for(i in 1:nrow(test)){
  s = i-1+window
  x_test = rbind(x_test,Scaled$scaled_test[i:s])
  y_test = rbind(y_test,Scaled$scaled_test[s+1])
}
#check for NAs
which(is.na(x_test))
x_test <- x_test[1:(nrow(test)-window),]
which(is.na(y_test))
y_test <- y_test[1:(nrow(test)-window)]
#  Input data should be an array type, so we'll reshape it.
X_test = array(x_test, dim=c(nrow(x_test), window,1))
Y_test = array(y_test, dim= c(length(y_test), 1,1))


##### Building Keras LSTM model #####
# Next, we'll create Keras sequential model, add an LSTM layer,
# and compile it with defined metrics.
# defining the model
rm(model)
model = keras_model_sequential() %>% 
  layer_lstm(units=64, input_shape=c(window,1), activation="relu") %>%  
  layer_dense(units=256, activation = "relu") %>% 
  layer_dropout(rate = 0.25) %>%
  layer_dense(units=dim(Y_train)[2], activation = "linear")


# compile the model
model %>% compile(loss = 'mse',
                  optimizer = 'adam', # alternatives: sgd, nadam, adam, rmsprop 
                  metrics = list("mean_absolute_error")
)
model %>% summary()
##### Predicting and plotting the result #####
# Next, we'll train the model with X and y input data, predict X data, and check the errors.
model %>% fit(X_train,Y_train, epochs=100, batch_size=32, 
              shuffle = FALSE, validation_split=0.1,
              callbacks = list(
                callback_early_stopping(patience = 5),
                callback_reduce_lr_on_plateau(factor = 0.05))
)


#model run environment can be saved at this point

##### recursive prediction #####
steps=10
y_all_pred = NULL
y_all_pred_resc = NULL
all_scores = NULL

#for loop for recursive prediction for all steps
for(i in 1:steps){
  y_all_pred <- cbind(y_all_pred, model%>% predict(X_test))
  all_scores <- rbind(all_scores, model %>% evaluate(X_test, y_test, verbose = 0))
  # let`s drop the oldest value and update with our prediction
  x_test <- x_test[,-1] # delete first column containing the oldest data
  x_test <- cbind(x_test, y_all_pred[,i]) # add the predictions in the last column
  X_test = array(x_test, dim=c(nrow(x_test), window,1))
}

# rescaling of the prediction results
y_all_pred_resc <- NULL
for (j in 1:ncol(y_all_pred)){
  y_all_pred_resc <- cbind(y_all_pred_resc,reverse_scaling(y_all_pred[,j], Scaled$scaler, c(0,1)))
}

#evaluation
# creating a table with test data and prediction data
results <- as.data.frame(test$timeScale)
names(results)[ncol(results)] <- "timeScale"
results <- cbind(results, test$WIP)
names(results)[ncol(results)] <- "WIP"
for (i in 1:steps) {
  results[, ncol(results)+1] <- NA
  results[(window+i):(window+dim(y_all_pred_resc)[1]+i-1), ncol(results)]<- y_all_pred_resc[,i]
  names(results)[ncol(results)] <- paste0("lstm-step-", i)
}
results <- results[(window+1+steps):nrow(test),]

# calculating different error measures
library(caret)
library(hydroGOF)
library(ie2misc)

MAPEs_lstm <- NULL
R2s_lstm <- NULL
NRMSEs_lstm <- NULL
for(j in (3:ncol(results))){
  MAPEs_lstm[j-2] <- mape(results$WIP, results[,j])
  R2s_lstm[j-2] <- R2(results$WIP, results[,j], form="traditional")
  NRMSEs_lstm[j-2] <- nrmse(results$WIP, as.numeric(results[,j]))
}

error_summary <- data.frame(MAPEs_lstm, R2s_lstm, NRMSEs_lstm)
setattr(error_summary, "row.names", c(paste0("step-",1:steps)))


library(ggplot2)
colors <- c("LSTM" = "#179c7d", "orig" = "black")
ggplot(data=results, aes(x=timeScale))+
  geom_line(aes(y=WIP, colour="orig"), size=1)+   
  labs(x="Time", y="WIP[kg]", 
       title ="5 step-ahead prediction with recursive LSTM",
       subtitle = "real and predicted values",
       color="Legend") +
  geom_line(aes(y=`lstm-step-5`, color="LSTM"), size=1)+
  theme(plot.title = element_text(hjust=0.5))+ # plot in to the middle
  theme(plot.subtitle = element_text(hjust=0.5))+ # plot in to the middle
  scale_color_manual(values = colors)

