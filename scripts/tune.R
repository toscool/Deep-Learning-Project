library(keras)
library(here)
library(cloudml)

setwd("scripts")


#-------------------------------------------------------------------------------

gs_copy(
  source = here::here("data"),
  destination = "gs://face_age",
  recursive = TRUE)

#-------------------------------------------------------------------------------
#tuning # c
cloudml_train("train-best-model.R", master_type = "standard_p100") 

#best model
cloudml_train("train-best-model.R", master_type = "standard_p100") 

#-------------------------------------------------------------------------------


#insert run with tuning
job_collect("cloudml_2020_05_27_175741554")
view_run("runs/cloudml_2020_05_27_175741554")

#train-best-model
job_collect("cloudml_2020_05_27_175741554")
view_run("runs/cloudml_2020_05_27_175741554")