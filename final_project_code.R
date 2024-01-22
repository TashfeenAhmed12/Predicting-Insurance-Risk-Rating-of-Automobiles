###### The code is divided into two parts - Classification models and Clustering ######

## Note after running classification model remove all variables and then run clustering 
#as both have same variable names and might cause wrong variables to be used

# Import Dataset
auto_data = read.csv("D:\\Google Drive\\McGill\\Fall Semester\\MGSC 661\\Final Project\\Dataset 5 — Automobile data.csv")
attach(auto_data)


#Inspect Data
#install.packages('tidyverse')
library(tidyverse)
glimpse(auto_data)
summary(auto_data)

# Check for missing values in each column
missing_values <- sapply(auto_data, function(x) sum(is.na(x) | x == "?"))
print(missing_values)

#normalized losses has 41 missing values out of 205 rows so remove it
#remove ? rows from num of doors columns

# Function to clean non-numeric values
clean_data <- function(column) {
  as.numeric(gsub("\\?", NA, column))
}

selected_features <- auto_data %>%
  mutate(across(c(bore, stroke, horsepower, peak.rpm, price), clean_data)) %>%
  filter(!if_any(c(bore, stroke, horsepower, peak.rpm, price), is.na)) %>%
  na.omit() %>%
  select(symboling, fuel.type, aspiration, num.of.doors,
         body.style, drive.wheels, engine.location, wheel.base,
         length, width, height, curb.weight, engine.type,
         num.of.cylinders, engine.size, fuel.system, bore, stroke,
         compression.ratio, horsepower, peak.rpm, city.mpg,
         highway.mpg, price) %>%
  mutate(across(c(fuel.type, aspiration, num.of.doors, body.style,
                  drive.wheels, engine.location, engine.type,
                  num.of.cylinders, fuel.system), as.factor))
selected_features <- selected_features %>%
  mutate(num.of.doors = as.character(num.of.doors)) %>%
  filter(num.of.doors != "?")
selected_features <- selected_features %>%
  mutate(num.of.doors = factor(num.of.doors))

# Convert symboling to a binary factor for classification
selected_features$symboling <- factor(ifelse(selected_features$symboling > 0, 1, 0), levels = c(0, 1))
table(selected_features$symboling)

# Check the processed data
glimpse(selected_features)
summary(selected_features)
#install.packages('corrplot')
library(corrplot)

# PLot

library(GGally)

# Selected features for the pair plot
selected_cols <- c("length", "width", "height", "engine.size", "horsepower", "city.mpg","price")

# Generate the pair plot
ggpairs(selected_features[selected_cols])

# Calculate the correlation matrix for numeric variables
correlation_matrix <- cor(selected_features %>% select(where(is.numeric)))
print(correlation_matrix)

library(corrplot)

# Plot correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust",
         addCoef.col = "black", 
         tl.col = "black",      
         tl.srt = 45,           
         tl.cex = 0.6,          
         diag = FALSE)          


#curb.weight and engine.size (correlation: 0.8502362)
#city.mpg and highway.mpg (correlation: 0.97096425)
#Length and Curb Weight: Correlation of 0.8827 
#Width and Curb Weight: Correlation of 0.8676
#Engine Size and Price: Correlation of 0.8888
# Removing curb.weight and highway.mpg, engine size due to high correlation with engine.size and city.mpg respectively
# Remove highly correlated variables
selected_features <- selected_features %>% 
  select(-curb.weight, -highway.mpg,-engine.size) 

# Check class balance in 'symboling'
table(selected_features$symboling)

#Degree of Imbalance: The ratio of the two classes is fairly close (approximately 1:1.19), which is not a severe imbalance. 
#Often, severe imbalances are more like 1:10 or more extreme

#PCA

library(tidyverse)
library(ggplot2)
library(ggfortify)

# Standardize the numeric features
selected_features_standardized <- selected_features %>%
  mutate(across(where(is.numeric), scale))

# Convert factor variables to numeric for PCA
selected_features_numeric <- selected_features_standardized %>%
  mutate(across(where(is.factor), as.numeric))

# Perform PCA to find highly correlated variables 
pca_result <- prcomp(selected_features_numeric, center = TRUE, scale. = TRUE)
pca_result
# Scree plot to show variance explained by each principal component
autoplot(pca_result, data = selected_features_numeric, loadings = FALSE)

# Biplot to show principal components and loadings
library(ggfortify)
#install.packages('plotly')
library(plotly)

p <- autoplot(pca_result, data = selected_features_numeric, loadings = TRUE, loadings.label = TRUE)

# Convert the ggplot object to a plotly object for interactive visualization
ggplotly(p)
#no highly correlated variables

### Modelling

# Build a random forest model
#install.packages("caret")
library(caret)

library(randomForest)
myforest <- randomForest(symboling ~ engine.location+fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = selected_features, ntree=500, importance=TRUE, na.action=na.omit, do.trace=50)
myforest
#OOB estimate of  error rate: 9.09%

# Plot variable importance
varImpPlot(myforest)
importance(myforest)

# remove variables less important  engine.location
# Remove less important variables 
selected_features_reduced <- selected_features %>% 
  select( -engine.location)

# Rebuild the Random Forest model with the revised set of features
set.seed(123) 
folds <- createFolds(selected_features_reduced$symboling, k = 5)
metrics_rf <- data.frame(Accuracy = double(5), Error = double(5))

for(i in 1:5) {
  train_set <- selected_features_reduced[-folds[[i]], ]
  test_set <- selected_features_reduced[folds[[i]], ]
  
  myforest <- randomForest(symboling ~ fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = train_set, ntree = 500, importance = TRUE, na.action = na.omit)
  predictions <- predict(myforest, newdata = test_set)
  conf_mat <- confusionMatrix(predictions, test_set$symboling)
  
  metrics_rf[i, "Accuracy"] <- conf_mat$overall["Accuracy"]
  metrics_rf[i, "Error"] <- 1 - conf_mat$overall["Accuracy"]
}

mean_metrics_rf <- colMeans(metrics_rf)
print(mean_metrics_rf)

revised_forest <- randomForest(symboling ~ fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = selected_features_reduced, 
                               ntree=500, importance=TRUE, na.action = na.omit, do.trace=50)
revised_forest
#OOB estimate of  error rate: 6.74%
# Optimal number of tress: 200
#Accuracy      Error 
#0.94304993 0.05695007 


## Decision tree 

library(rpart)
library(rpart.plot)

# Build an initial decision tree model
# cp=0.01 sets a complexity parameter to avoid overfitting.
mytree <- rpart(symboling ~ fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = selected_features_reduced, method = "class", control = rpart.control(cp=0.01))
rpart.plot(mytree) # Visualize the tree
summary(mytree) # Get a detailed summary of the tree

# Finding the optimal cp (complexity parameter)
# Step 1: Build an overfitted tree
# A lower cp (0.001) is likely to produce a more complex tree, which might overfit the data.
overfitted_tree <- rpart(symboling ~ fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = selected_features_reduced, method = "class", control = rpart.control(cp=0.001))
rpart.plot(overfitted_tree) # Visualize the overfitted tree

# Step 2: Display complexity parameter values
# This helps in understanding how the tree complexity affects model performance.
printcp(overfitted_tree) # Print the CP table
plotcp(overfitted_tree) # Plot the CP table

# Step 3: Find optimal cp
# Identify the cp value that minimizes the cross-validated error.
opt_cp <- overfitted_tree$cptable[which.min(overfitted_tree$cptable[,"xerror"]), "CP"]



library(caret)
set.seed(123) # for reproducibility

folds <- createFolds(selected_features_reduced$symboling, k = 5)
metrics_dt <- data.frame(Accuracy = double(5), Error = double(5))

for(i in 1:5) {
  train_set <- selected_features_reduced[-folds[[i]], ]
  test_set <- selected_features_reduced[folds[[i]], ]
  
  mytree <- rpart(symboling ~ fuel.type+aspiration+num.of.doors+body.style+drive.wheels+wheel.base+length+width+height+engine.type+num.of.cylinders+fuel.system+bore+stroke+compression.ratio+horsepower+peak.rpm+city.mpg+price, data = train_set, method = "class", control = rpart.control(cp = opt_cp))
  predictions <- predict(mytree, newdata = test_set, type = "class")
  conf_mat <- confusionMatrix(predictions, test_set$symboling)
  
  metrics_dt[i, "Accuracy"] <- conf_mat$overall["Accuracy"]
  metrics_dt[i, "Error"] <- 1 - conf_mat$overall["Accuracy"]
}

mean_metrics_dt <- colMeans(metrics_dt)
print(mean_metrics_dt)

#Accuracy     Error 
#0.8446694 0.1553306

best_tree <- rpart(symboling ~ ., data = selected_features, method = "class", control = rpart.control(cp=opt_cp))
rpart.plot(best_tree) # Visualize the best tree
best_tree


###Clustering###
#install.packages('tidyverse')
#install.packages('factoextra')
#install.packages('NbClust')
#install.packages('reshape2')
library(tidyverse)
library(factoextra)
library(NbClust)
library(reshape2)

# Read and preprocess the dataset
auto_data <- read.csv("D:\\Google Drive\\McGill\\Fall Semester\\MGSC 661\\Final Project\\Dataset 5 — Automobile data.csv")

# Function to clean non-numeric values and convert to numeric
clean_data <- function(column) {
  column <- as.character(column) 
  column <- gsub("\\?", NA, column) # Replace "?" with NA
  as.numeric(column) # Convert to numeric
}

selected_features <- auto_data %>%
  mutate(across(c(bore, stroke, horsepower, peak.rpm, price), clean_data)) %>%
  drop_na() %>%
  select(-normalized.losses, -num.of.doors) %>%
  mutate(across(where(is.numeric), ~ (.-min(.))/(max(.) - min(.)))) # Normalization

# Check if there are any NAs left in the data
sum(is.na(selected_features))

selected_features <- selected_features %>%
  select_if(~is.numeric(.))


glimpse(selected_features)

# Determine the optimal number of clusters using Elbow and Silhouette methods
set.seed(123) # for reproducibility

# Elbow method
elbow_method <- fviz_nbclust(selected_features, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")
print(elbow_method)

# Silhouette method
silhouette_method <- fviz_nbclust(selected_features, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")
print(silhouette_method)

# Optimal k=3
k <- 3 
km_result <- kmeans(selected_features, centers = k, nstart = 25)

selected_features$cluster <- as.factor(km_result$cluster)

# Plotting Cluster Centroids
cluster_centers <- as.data.frame(km_result$centers)
cluster_centers$cluster <- rownames(cluster_centers)
long_centers <- melt(cluster_centers, id.vars = 'cluster')

scatter_plot <- ggplot(long_centers, aes(x = variable, y = value, colour = cluster)) +
  geom_jitter(width = 0.2, height = 0, size = 3, alpha = 0.8) +
  labs(title = 'Cluster-wise Centroids for each Variable',
       colour = 'Cluster') +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(scatter_plot)




