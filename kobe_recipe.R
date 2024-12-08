kobe_recipe <- function(traindata){
  recipe(shot_made_flag~., data = traindata) |>
    update_role(shot_id, new_role = "ID") |>
    step_date(game_date, features = c("month", "dow", "year"), keep_original_cols = F) |>
    step_mutate(game_date_month=as.factor(game_date_month),
                game_date_dow=as.factor(game_date_dow),
                game_date_year=as.factor(game_date_year)) |>
    step_mutate(period = as.factor(period),
                season = as.factor(season),
                playoffs = as.factor(playoffs),
                #game_event_id = as.factor(game_event_id)
                ) |>
    step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag))
}

# kobe_recipe <- function(traindata){
#   recipe(shot_made_flag~., data = traindata) |>
#     update_role(shot_id, new_role = "id variable") |>
#     step_mutate(period = as.factor(period)) |>
#     step_novel(all_nominal_predictors()) |>
#     step_unknown(all_nominal_predictors()) |>
#     step_dummy(all_nominal_predictors())
# }



run_cv <- function(wf, folds, grid, metric=metric_set(rmse), cores=7, parallel = TRUE){
  if(parallel == TRUE){
    library(doParallel)
    
    cl <- makePSOCKcluster(cores)
    registerDoParallel(cl)
  }
  
  results <- wf |>
    tune_grid(resamples=folds,
              grid=grid,
              metrics=metric)
  if(parallel == TRUE){
    stopCluster(cl)
  }
  return(results)
}