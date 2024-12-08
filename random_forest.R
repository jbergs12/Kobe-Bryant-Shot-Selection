library(tidymodels)
library(vroom)
library(embed)
library(ranger)

source("kobe_recipe.R")

kobe_train <- vroom("train.csv")
kobe_train$shot_made_flag <- as.factor(kobe_train$shot_made_flag)
kobe_test <- vroom("test.csv")

kobe_rec <- kobe_recipe(kobe_train)


### Random Forest

kobe_forest <- rand_forest(mtry = tune(),
                           min_n = tune(),
                           trees = 800) |> 
  set_engine("ranger") |> 
  set_mode("classification")

forest_wf <- workflow() |> 
  add_recipe(kobe_rec) |> 
  add_model(kobe_forest)

forest_grid <- grid_regular(
  mtry(range = c(1, ncol(juice(prep(kobe_rec))))),
  min_n(),
  levels = 5)

folds <- vfold_cv(kobe_train, v=5)

CV_results <- run_cv(forest_wf,
                     folds,
                     forest_grid,
                     metric = metric_set(roc_auc, mn_log_loss),
                     cores = 6)

# Log Loss

bestTune <- CV_results |> 
  select_best(metric="mn_log_loss")

bestTune$mtry # 30
bestTune$min_n # 40

final_wf <- forest_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=kobe_train)

forest_preds <- final_wf |>
  predict(new_data = kobe_test,
          type = "prob")

kaggle_submission <- forest_preds |> 
  bind_cols(kobe_test) |> 
  select(shot_id, .pred_1) |> 
  rename(shot_made_flag = .pred_1)

vroom_write(x=kaggle_submission, file="./results/random_forest_loss2.csv", delim = ",")

# # RMSE
# 
# bestTune <- CV_results |> 
#   select_best(metric="rmse")
# 
# bestTune$mtry
# bestTune$min_n
# 
# final_wf <- forest_wf |> 
#   finalize_workflow(bestTune) |> 
#   fit(data=kobe_train)
# 
# forest_preds <- final_wf |>
#   predict(new_data = kobe_test,
#           type = "prob")
# 
# kaggle_submission <- forest_preds |> 
#   bind_cols(kobe_test) |> 
#   select(shot_id, .pred_1) |> 
#   rename(shot_made_flag = .pred_1)
# 
# vroom_write(x=kaggle_submission, file="./results/random_forest_rmse.csv", delim = ",")

# roc_auc

bestTune <- CV_results |> 
  select_best(metric="roc_auc")

bestTune$mtry # 30
bestTune$min_n # 40

final_wf <- forest_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=kobe_train)

forest_preds <- final_wf |>
  predict(new_data = kobe_test,
          type = "prob")

kaggle_submission <- forest_preds |> 
  bind_cols(kobe_test) |> 
  select(shot_id, .pred_1) |> 
  rename(shot_made_flag = .pred_1)

vroom_write(x=kaggle_submission, file="./results/random_forest_roc2.csv", delim = ",")

