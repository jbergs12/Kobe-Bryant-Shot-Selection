library(vroom)
library(tidymodels)
library(tidyverse)

dat <- vroom("data.csv")
View(dat)

#### FOR KAGGLE NOTEBOOK
# dat <- vroom("/kaggle/input/dat-bryant-shot-selection/data.csv.zip")

# Create distance column
dat$shot_distance <- sqrt((dat$loc_x/10)^2 + (dat$loc_y/10)^2)

# Create angle column 
loc_x_zero <- dat$loc_x == 0
dat['angle'] <- rep(0,nrow(dat))
dat$angle[!loc_x_zero] <- atan(dat$loc_y[!loc_x_zero] / dat$loc_x[!loc_x_zero])
dat$angle[loc_x_zero] <- pi / 2

# Create one time variable 
dat$time_remaining <- (dat$minutes_remaining*60)+dat$seconds_remaining

# Home and Away
dat$matchup = ifelse(str_detect(dat$matchup, 'vs.'), 'Home', 'Away')

# Season
dat['season'] <- substr(str_split_fixed(dat$season, '-',2)[,2],2,2)

# Game number
dat$game_num <- as.numeric(dat$game_date)

# Delete columns
dat <- dat %>%
  select(-c('team_id', 'team_name', 'shot_zone_range', 'lon', 'lat',
            'seconds_remaining', 'minutes_remaining', 'game_event_id',
            'game_id', 'game_date','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))
# Train
dat_train <- dat %>%
  filter(!is.na(shot_made_flag))
# Test 
dat_test <- dat %>% 
  filter(is.na(shot_made_flag)) |> 
  select(-shot_made_flag)

# Make the response variable into a factor 
train$shot_made_flag <- as.factor(train$shot_made_flag)
recipe <- recipe(shot_made_flag ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())


# dat_test <- dat[is.na(dat$shot_made_flag), ] |>
#   select(-shot_made_flag)
# dat_train <- dat[!is.na(dat$shot_made_flag), ]
# head(dat_test)
# 
# dat_test$shot_made_flag
# dat_train[!is.na(dat_train$shot_made_flag), ]

vroom_write(dat_test, "./test.csv")
vroom_write(dat_train, "./train.csv")
