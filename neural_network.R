library(tidymodels)
library(vroom)
library(embed)

source("kobe_recipe.R")

kobe_train <- vroom("train.csv")
kobe_train$shot_made_flag <- as.factor(kobe_train$shot_made_flag)
kobe_test <- vroom("test.csv")

kobe_rec <- recipe(shot_made_flag~., data = kobe_train) |> 
  step_rm(team_name, team_id) |> 
  step_date(game_date, features = c("month", "dow", "year"), keep_original_cols = F) |>
  step_mutate(game_date_month=as.factor(game_date_month),
              game_date_dow=as.factor(game_date_dow),
              game_date_year=as.factor(game_date_year)) |> 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag)) |> 
  step_range(all_numeric_predictors(), min = 0, max = 1)

kobe_nn <- mlp(hidden_units = tune(),
              epochs = 100) |> 
  set_engine("keras") |> 
  set_mode("classification")

nn_wf <- workflow() |> 
  add_recipe(kobe_rec) |>
  add_model(kobe_nn)

nn_grid <- grid_regular(hidden_units(range=c(1,15)),
                        levels = 5)

folds <- vfold_cv(kobe_train, v = 5, repeats = 1)

CV_results <- run_cv(nn_wf, folds, nn_grid, metric = metric_set(roc_auc, accuracy, mn_log_loss),
                     parallel = F)

# CV_results |> 
#   collect_metrics() |> 
#   filter(.metric=='mn_log_loss') |> 
#   ggplot(aes(x=hidden_units, y=mean)) +
#   geom_line()

bestTune <- CV_results |> 
  select_best(metric = "mn_log_loss")

bestTune$hidden_units # 15
# bestTune$epochs

final_wf <- nn_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=kobe_train)

nn_preds <- final_wf |> 
  predict(new_data = kobe_test,
          type = "class")

kaggle_submission <- nn_preds |>
  bind_cols(kobe_test) |>
  select(shot_id, .pred_class) |>
  rename(shot_made_flag = .pred_class)

vroom_write(x=kaggle_submission, file="./results/neuralnet_CV.csv", delim = ",")
