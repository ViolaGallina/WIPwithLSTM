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
# the first 70 is the training set, the last is the test set
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
window = 2   # the number of historical data used for prediction is defined
steps = 10# definition of time steps (shifts) for which the prediction will be done
# To cover all elements in a vector, we'll add a 'window' into the last part of  'a'
# vector by replicating the last element.
# Creating x - input, and y - output data: training and test sets
#training
x_train = NULL
y_train = NULL
for(i in 1:nrow(train)){
  s = i-1+window
  x_train = rbind(x_train,Scaled$scaled_train[i:s])
  y_train = rbind(y_train,Scaled$scaled_train[(s+1):(s+steps)])
}
#check for NAs
which(is.na(x_train))
x_train <- x_train[1:(nrow(train)-(window+steps+1)),]
which(is.na(y_train))
y_train <- y_train[1:(nrow(train)-(window+steps+1)),] 
#  Input data should be an array, so we'll reshape it.
X_train <- array(x_train, dim=c(nrow(x_train), window,1))
#X_train <- as.matrix(x_train)
Y_train <- array(y_train, dim= c(nrow(y_train), steps, 1))
#Y_train <- as.matrix(y_train)
# test
x_test = NULL
y_test = NULL
for(i in 1:nrow(test)){
  s = i-1+window
  x_test = rbind(x_test,Scaled$scaled_test[i:s])
  y_test = rbind(y_test,Scaled$scaled_test[(s+1):(s+steps)])
}
#check for NAs
which(is.na(x_test))
x_test <- x_test[1:(nrow(test)-(window+steps+1)),]
which(is.na(y_test))
y_test <- y_test[1:(nrow(test)-(window+steps+1)),]
#  reshape
X_test = array(x_test, dim = c(nrow(x_test), window,1))
#X_test = as.matrix(x_test)
#Y_test = matrix(y_test, nrow = nrow(y_test))
#Y_test = as.matrix(y_test)
Y_test <- array(y_test, dim=c(nrow(y_test), steps,1))
##### Building Keras LSTM model #####
# Next, we'll create Keras sequential model, add an LSTM layer,
# and compile it with defined metrics.
# defining the model dimensions
# defining the model
##### fitting the model and predicting #####
# Next, we'll train the model with X and y input data, predict X data, and check the errors.
score_summary <- NULL
y_pred_results <- NULL
for(i in 1:steps){
  model = keras_model_sequential() %>% 
    layer_lstm(units=64, input_shape=c(window,1), activation="relu") %>%  
    layer_dense(units=256, activation = "relu") %>% 
    layer_dropout(rate = 0.25) %>%
    layer_dense(units=1, activation = "linear")  
  # compile the model
  model %>% compile(loss = 'mse',
                    optimizer = 'adam', # alternatives: sgd, nadam, adam, rmsprop 
                    metrics = list("mean_absolute_error")
  )
  #model %>% summary()
  model %>% fit(X_train,as.data.frame(Y_train)[,i], epochs=100, batch_size=32, 
                shuffle = FALSE, validation_split=0.1,
                callbacks = list(
                  callback_early_stopping(patience = 5),
                  callback_reduce_lr_on_plateau(factor = 0.05))
  )
  
  y_pred_results <- cbind(y_pred_results,model %>% predict(X_test))
  score_summary <- rbind(score_summary, model %>% evaluate(X_test, as.data.frame(Y_test)[,i], verbose=0))
    print(i)
  rm(model)
  
}

##### rescaling and evaluation #####
#rescaling
y_pred_rescaled <- reverse_scaling(y_pred_results, Scaled$scaler, c(0,1))
dim(y_pred_rescaled) <- c(dim(y_test)[1], dim(y_test)[2])

#evalutation
results <- as.data.frame(test$timeScale)
names(results)[ncol(results)] <- "timeScale"
results <- cbind(results, test$WIP)
names(results)[ncol(results)] <- "WIP"
for (i in 1:steps) {
  results[, ncol(results)+1] <- NA
  results[(window+i):(window+dim(y_pred_rescaled)[1]+i-1), ncol(results)]<- y_pred_rescaled[,i]
  names(results)[ncol(results)] <- paste0("lstm-step-", i)
}
results <- results[(window+steps):(nrow(results)),] # delete NAs at the beginning
results <- results[1:(min(which(is.na(results$`lstm-step-1`)))-1),] # delete NAs at the end

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
       title ="10 step-ahead prediction with direct multistep LSTM",
       subtitle = "real and predicted values",
       color="Legend") +
  geom_line(aes(y=`lstm-step-10`, color="LSTM"), size=1)+
  theme(plot.title = element_text(hjust=0.5))+ # plot in to the middle
  theme(plot.subtitle = element_text(hjust=0.5))+ # plot in to the middle
  scale_color_manual(values = colors)

