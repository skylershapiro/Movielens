############################################################################
############################################################################
###                                                                      ###
###                              SECTION 1:                              ###
###       CREATE EDX SET, VALIDATION SET (FINAL HOLD-OUT TEST SET)       ###
###                                                                      ###
############################################################################
############################################################################


# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(bannerCommenter)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

mlens <- edx
val <- validation


###########################################################################
###########################################################################
###                                                                     ###
###                              SECTION 2:                             ###
###                         PROCESSING THE DATA                         ###
###                                                                     ###
###########################################################################
###########################################################################

# Separate year from title
mlens <- mlens %>% mutate(movie_year = as.numeric(str_sub(title,-5,-2)))
val <- val %>% mutate(movie_year = as.numeric(str_sub(title,-5,-2)))

# Convert timestamp column to date format
mlens <- mutate(mlens, date = as_datetime(timestamp), rating_year = year(date))
val <- mutate(val, date = as_datetime(timestamp), rating_year = year(date))

# Create Age by subtracting movie_year from rating_year
mlens <- mutate(mlens, age = rating_year - movie_year)
val <- mutate(val, age = rating_year - movie_year)

# Convert to factor
#mlens$userId <- as.factor(mlens$userId)
#mlens$movieId <- as.factor(mlens$movieId)


#val$userId <- as.factor(val$userId)
#val$movieId <- as.factor(val$movieId)
#val$genres <- as.factor(val$genres)

# Partition edx dataset
test_index <- createDataPartition(y = mlens$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- mlens[-test_index,]
test_set <- mlens[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

############################################################################
############################################################################
###                                                                      ###
###                              SECTION 3:                              ###
###                           DATA EXPLORATION                           ###
###                                                                      ###
############################################################################
############################################################################

# Number of observations and variables
dim(mlens)

# Variables 
colnames(mlens)

# Number of unique users and movies
mlens %>% summarize (n_users = n_distinct(mlens$userId), 
                     n_movies = n_distinct(mlens$movieId))

#Check for ratings out of range (should be between 0.5 and 5.0)
summary(mlens$rating)

#Ratings per user
mlens %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black") +
  scale_x_log10() +
  xlab("Ratings Given") + 
  ylab("Users") +
  ggtitle("Number of Ratings Given by Users")

# Average rating per week plotted against date
mlens %>% mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Date") + 
  ylab("Rating") +
  ggtitle("Average Rating per Week over Time ")

# Number of ratings per movie
n_movie_ratings <- mlens %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
n_movie_ratings %>% ggplot(aes(movieId)) +
  geom_histogram()

# Ratings per genre
genre_counts <- mlens %>% separate_rows(genres, sep = "\\|") %>%
       group_by(genres) %>%
       summarize(count = n()) %>%
       arrange(desc(count)) 
genre_counts %>% as.data.frame()

# Most rated movies
head(n_movie_ratings)

# Least rated Movies
tail(n_movie_ratings)

############################################################################
############################################################################
###                                                                      ###
###                              SECTION 4:                              ###
###                          MODEL CONSTRUCTION                          ###
###                                                                      ###
############################################################################
############################################################################

##----------------------------------------------------------------
##                    Defining RMSE Function                    --
##----------------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#################################################################
##                         Naive Model                         ##
#################################################################

mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

predictions <- rep(mu_hat, nrow(test_set))
RMSE(test_set$rating, predictions)

rmse_results <- data_frame(Method = "1: Naive Model", RMSE = naive_rmse)
rmse_results %>% knitr::kable

##################################################################
##                    Modeling Movie Effects                    ##
##################################################################
# Define Mu
mu <- mean(train_set$rating) 
# Build Model
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
# Predict Ratings
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="2: Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

##---------------------------------------------------------------
##                 Exploring and Visualizing Movie Effects                  --
##---------------------------------------------------------------

# Plot of Movie Effects
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# 10 largest mistakes from Movie Effects model
test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()

movie_titles <- mlens %>% 
  select(movieId, title, ) %>%
  distinct() 

# Best 10 movies according to Movie Effects model
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()
# Worst 10 movies from Movie Effects model 
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#################################################################
##               Modeling Movie and User Effects               ##
#################################################################

# Calculate mu
mu <- mean(train_set$rating) 

# Build model
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="3: Movie + User Effect Model",
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

# Visualizing User Effects
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#################################################################
##            Modeling Movie, User, and Age Effects            ##
#################################################################

# Calculate mu
mu <- mean(train_set$rating) 

# Build model
age_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(age) %>%
  summarize(b_a = mean(rating - mu - b_i - b_u))

#Predict Ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(age_avgs, by='age') %>%
  mutate(pred = mu + b_i + b_u + b_a) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="4: Movie + User + Age Effect Model",
                                 RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

##################################################################
##              Movie, User, Age, and Genre Effect              ##
##################################################################

# Calculate mu
mu <- mean(train_set$rating) 

# Build model
genre_avgs <- train_set %>% 
  left_join(movie_avgs,by="movieId") %>% 
  left_join(user_avgs,by="userId") %>%
  left_join(age_avgs,by='age') %>%
  group_by(genres) %>%	  
  summarize(b_g = mean(rating - mu - b_i - b_u - b_a))

# Predict ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(age_avgs, by='age') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_a + b_g) %>%
  .$pred

model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="5: Movie + User + Age + Genre Effect Model",
                                 RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()

############################################################################
############################################################################
###                                                                      ###
###                              SECTION 5:                              ###
###                            REGULARIZATION                            ###
###                                                                      ###
############################################################################
############################################################################

# Choose range of tuning parameter lambda to optimize model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  # Calculate mu
  mu <- mean(train_set$rating)
  
  # Regularize movie effect
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Regularize user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Regularize age effect
  b_a <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by='userId') %>%
    group_by(age) %>%
    summarize(b_a = sum(rating - b_u - b_i - mu)/(n()+l))
  
  # Regularize genre effect
    b_g <- train_set %>% 
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by='userId') %>%
      left_join(b_a, by='age') %>%
    group_by(genres) %>%
      summarize(b_g = sum(rating - b_a - b_u - b_i - mu)/(n()+l))
  
    # Predict ratings
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by='age') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_a + b_g) %>%
    pull(pred)
  
  # Calculate RMSE
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plot Lambdas versus RMSE values
qplot(lambdas, rmses)  

# Calculate lambda which minimizes RMSE
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="6: Regularized Movie + User + Age + Genre Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#################################################################
##             Testing Algorithm on Validation Set             ##
#################################################################

# Calculate RMSE on model with optimized lambda
final_rmse <- sapply(lambda, function(l){
 
   mu <- mean(mlens$rating)
  
   b_i <- mlens %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- mlens %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_a <- mlens %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by='userId') %>%
    group_by(age) %>%
    summarize(b_a = sum(rating - b_u - b_i - mu)/(n()+l))
  b_g <- mlens %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by='userId') %>%
    left_join(b_a, by='age') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_a - b_u - b_i - mu)/(n()+l))

  predicted_ratings <- val %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by='age') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_a + b_g) %>%
    pull(pred)
  
  return(RMSE(val$rating, predicted_ratings))
  
})

#model_6_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="VALIDATION SET: Regularized Movie + User + Age + Genre Effect Model",
                                 RMSE = final_rmse ))
rmse_results %>% knitr::kable()

