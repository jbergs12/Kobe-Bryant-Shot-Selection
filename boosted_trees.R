library(tidymodels)
library(vroom)
library(embed)
library(bonsai)

source("kobe_recipe.R")

kobe_train <- vroom("train.csv")
kobe_train$shot_made_flag <- as.factor(kobe_train$shot_made_flag)
kobe_test <- vroom("test.csv")

kobe_rec <- kobe_recipe(kobe_train)


### Boosted Trees

kobe_boost <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

boost_wf <- workflow() |> 
  add_recipe(kobe_rec) |>
  add_model(kobe_boost)

boost_grid <- grid_regular(tree_depth(),
                           trees(),
                           learn_rate(),
                           levels = 5)

folds <- vfold_cv(kobe_train, v = 5, repeats = 1)

boost_CV <- run_cv(boost_wf, folds, boost_grid, metric = metric_set(roc_auc, accuracy, mn_log_loss, rmse),
                   parallel = F)

bestTune <- boost_CV |> 
  select_best(metric = "accuracy")

bestTune$tree_depth # 1
bestTune$trees # 1000 # 500
bestTune$learn_rate # 0.1 

final_wf <- boost_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=kobe_train)

boost_preds <- final_wf |> 
  predict(new_data = kobe_test,
          type = "prob")

kaggle_submission <- boost_preds |> 
  bind_cols(kobe_test) |> 
  select(shot_id, .pred_1) |> 
  rename(shot_made_flag = .pred_1)

vroom_write(x=kaggle_submission, file="./results/boosted_trees.csv", delim = ",")
