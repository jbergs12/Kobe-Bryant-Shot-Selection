library(vroom)

dat <- vroom("data.csv")
View(dat)

dat_test <- dat[is.na(dat$shot_made_flag), ]
dat_train <- dat[!is.na(dat$shot_made_flag), ]
head(dat_test)

dat_test$shot_made_flag
dat_train[!is.na(dat_train$shot_made_flag), ]

vroom_write(dat_test, "./test.csv")
vroom_write(dat_train, "./train.csv")
