kobe_recipe <- function(traindata){
  recipe(shot_made_flag~., data = traindata) |> 
    # step_rm(team_name, team_id) |> 
    step_date(game_date, features = c("month", "dow", "year"), keep_original_cols = F) |>
    step_mutate(game_date_month=as.factor(game_date_month),
                game_date_dow=as.factor(game_date_dow),
                game_date_year=as.factor(game_date_year),
                home_away = as.factor(ifelse(str_detect(matchup, 'vs.'), 'Home', 'Away'))) |> 
    # step_rm(matchup) |> 
    step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag)) |> 
    step_range(all_numeric_predictors(), min = 0, max = 1)
}



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