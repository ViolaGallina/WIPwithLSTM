library(fst)
library(data.table)
library(lubridate)
library(tidyr)
library(ggplot2)
library(keras)
library(tensorflow)


rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memory and report the memory usage.
setwd("C:/Users/X")
WIP = read.fst('WIP.fst', as.data.table = T)
DT <- WIP[,c("timeScale", "WIP")]
##### data preparation #####

ggplot(DT, aes(x=timeScale, y=WIP/1000))+
  geom_line(colour="#179c7d", size=1)+   # Fh green
  labs(x="Time", y="WIP[t] ", 
       title ="Work in progress for x")+
  theme(plot.title = element_text(hjust=0.5)) # plot in to the middle

##### split: train + test  & normalization #####
# the first 70 is the training set, the last is the test set
k <- round(nrow(DT) * 0.7, digits = 0)
train <- DT[1:k, ]
test <- DT[(k+1):nrow((DT)),]
# create data set for accuracy check after training
res <- data.frame("time"= DT$timeScale[(k+1):(nrow(DT))],
                  "original" = DT$WIP[(k+1):(nrow(DT))])
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

# grid search for time window
#grid_search_tw <- c(2,3,4,5,6,7,8,9,10,12,15,20)
#for (step in grid_search_tw) {
step=2
  x_train = NULL
  y_train = NULL
  for(i in 1:nrow(train)){
    s = i-1+step
    x_train = rbind(x_train,Scaled$scaled_train[i:s])
    y_train = rbind(y_train,Scaled$scaled_train[s+1])
  }
  #delete NAs
  x_train <- x_train[1:(nrow(train)-step),]
  y_train <- y_train[1:(nrow(train)-step)]
  #  Input data should be an array type, so we'll reshape it.
  X_train = array(x_train, dim=c(nrow(x_train), step,1))
  # test
  x_test = NULL
  y_test = NULL
  for(i in 1:nrow(test)){
    s = i-1+step
    x_test = rbind(x_test,Scaled$scaled_test[i:s])
    y_test = rbind(y_test,Scaled$scaled_test[s+1])
  }
  #delete NAs
  x_test <- x_test[1:(nrow(test)-step),]
  y_test <- y_test[1:(nrow(test)-step)]
  #  Input data should be an array type, so we'll reshape it.
  X_test = array(x_test, dim=c(nrow(x_test), step,1))

  ##### Building Keras LSTM model #####
  # Next, we'll create Keras sequential model, add an LSTM layer,
  # and compile it with defined metrics.
  # defining the model
  model = keras_model_sequential() %>% 
    layer_lstm(units=128, input_shape=c(step,1), activation="relu") %>%  
    layer_dense(units=64, activation = "relu") %>% 
    layer_dense(units=32) %>% 
    layer_dense(units=1, activation = "linear")
  
  # compile the model
  model %>% compile(loss = 'mse',
                    optimizer = 'adam', # alternatives: sgd, nadam, adam, rmsprop 
                    metrics = list("mean_absolute_error")
  )
  ##### Predicting and plotting the result #####
  # Next, we'll train the model with X and y input data, predict X data, and check the errors.
  model %>% fit(X_train,y_train, epochs=25, batch_size=32, shuffle = FALSE, validation_split=0.1)

  ##### prediction and rescaling#####
  y_pred = model %>% predict(X_test)
  y_pred_rescaled <- reverse_scaling(y_pred, Scaled$scaler, c(0,1))
  # save prediction data into result dataset
  res[(step+1):(nrow(res)),ncol(res)+1] <- y_pred_rescaled
  names(res)[ncol(res)] <- paste("lstm_windowsize-", step, sep="")
  rm(model)
#}

# evaluation of the time windows
res <- res[(step+1):641,] #delete first elements (number of steps-because they are NAs), last element (because they are zero)
library(caret)
library(hydroGOF)
library(ie2misc)
  
MAPEs <- NULL
R2s <- NULL
RMSEs <- NULL
NRMSEs <- NULL
for(j in (1:1)){
  MAPEs[j] <- mape(res$original, res[,2+j])
  R2s[j] <- R2(res$original, res[,2+j], form="traditional")
  RMSEs[j] <- rmse(res$original, res[,2+j])
  NRMSEs[j] <- nrmse(res$original, as.numeric(res[,2+j]))
}  
  
error_summary <- data.frame(MAPEs, R2s, RMSEs, NRMSEs)
setattr(error_summary, "row.names", c(paste0("time window-",step)))
error_summary[,(ncol(error_summary)+1)] <- step
names(error_summary)[ncol(error_summary)] <- "tw"


# plot the result with the best model
colors <- c("LSTM" = "#179c7d", "orig" = "black")
options(scipen=700000) #This works by setting a higher penalty for deciding to use scientific notation
ggplot(data=res, aes(x=time))+
  geom_line(aes(y=original, colour="orig"), size=1)+   
  labs(x="Time", y="WIP[kg]", 
       title ="Work in progress - real and predicted values",
       color="Legend") +
  geom_line(aes(y=`lstm_windowsize-2`, color="LSTM"), size=1, alpha= 0.8)+
  theme(plot.title = element_text(hjust=0.5))+ # plot in to the middle
  scale_color_manual(values = colors)
