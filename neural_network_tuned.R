library(tidymodels)
library(vroom)
library(embed)

source("kobe_recipe.R")

kobe_train <- vroom("train.csv")
kobe_train$shot_made_flag <- as.factor(kobe_train$shot_made_flag)
kobe_test <- vroom("test.csv") |> 
  select(-shot_made_flag)

kobe_rec <- recipe(shot_made_flag~., data = kobe_train) |>
  step_rm(team_name, team_id, matchup) |>
  step_date(game_date, features = c("month", "dow", "year"), keep_original_cols = F) |>
  step_mutate(game_date_month=as.factor(game_date_month),
              game_date_dow=as.factor(game_date_dow),
              game_date_year=as.factor(game_date_year)) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag)) |>
  step_range(all_numeric_predictors(), min = 0, max = 1)

kobe_nn <- mlp(hidden_units = 15,
               epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

final_wf <- workflow() |>
  add_recipe(kobe_rec) |>
  add_model(kobe_nn)

final_fit <- final_wf %>%
  fit(data = kobe_train)

predictions <- predict(final_fit, new_data = kobe_test)

kaggle_submission <- predictions %>%
  bind_cols(kobe_test) %>%
  select(shot_id, .pred_class) %>%
  rename(shot_made_flag = .pred_class)

vroom_write(x = kaggle_submission, file = "./results/neuralnet.csv", delim = ",")
