library(tidymodels)
library(vroom)
library(embed)
library(bonsai)

source("kobe_recipe.R")

kobe_train <- vroom("train.csv")
kobe_train$shot_made_flag <- as.factor(kobe_train$shot_made_flag)
kobe_test <- vroom("test.csv")

kobe_rec <- kobe_recipe(kobe_train)

### BART

kobe_bart <- parsnip::bart(trees = tune()) |>
  set_engine("dbarts") |>
  set_mode("classification")

bart_wf <- workflow() |>
  add_recipe(kobe_rec) |>
  add_model(kobe_bart)

bart_grid <- grid_regular(trees(),
                          levels = 5)

folds <- vfold_cv(kobe_train, v = 5, repeats = 1)

bart_CV <- run_cv(bart_wf, folds, bart_grid, metric = metric_set(mn_log_loss, roc_auc),
                  cores = 7)

# mn_log_loss

bestTune_bart <- bart_CV |> 
  select_best(metric = "mn_log_loss")

bestTune_bart$trees # 1

final_wf_bart <- bart_wf |> 
  finalize_workflow(bestTune_bart) |> 
  fit(data=kobe_train)

bart_preds <- final_wf_bart |> 
  predict(new_data = kobe_test,
          type = "prob")

kaggle_submission_bart <- bart_preds |> 
  bind_cols(kobe_test) |> 
  select(shot_id, .pred_1) |> 
  rename(shot_made_flag = .pred_1)

vroom_write(x=kaggle_submission_bart, file="./results/bart_loss.csv", delim = ",")


# roc_auc

bestTune_bart <- bart_CV |> 
  select_best(metric = "roc_auc")

bestTune_bart$trees # 1

final_wf_bart <- bart_wf |> 
  finalize_workflow(bestTune_bart) |> 
  fit(data=kobe_train)

bart_preds <- final_wf_bart |> 
  predict(new_data = kobe_test,
          type = "prob")

kaggle_submission_bart <- bart_preds |> 
  bind_cols(kobe_test) |> 
  select(shot_id, .pred_1) |> 
  rename(shot_made_flag = .pred_1)

vroom_write(x=kaggle_submission_bart, file="./results/bart_roc.csv", delim = ",")
