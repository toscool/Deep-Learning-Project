library(keras)
library(here)
library(cloudml)
library(tidyverse)
library(kableExtra)


model=load_model_hdf5(here::here("results/model.hdf5"))

class_names=c("Adult", "Minor")

thomas=image_load(here::here("selfie-image-pred/thomas.jpg"))
thomas=image_array_resize(thomas, width = 200, height = 200)
thomas=thomas/255
thomas <- array_reshape(thomas, dim = c(1, 200, 200, 3))
proba_thomas=model %>%
  predict_proba(thomas)
class_thomas=model%>%
  predict_classes(thomas)
class_thomas=class_names[class_thomas+1]
pred_thomas=as.data.frame(proba_thomas)%>%
  mutate(Prob_Class=class_thomas)

leo=image_load(here::here("selfie-image-pred/leo.jpg"))
leo=image_array_resize(leo, width = 200, height = 200)
leo=leo/255
leo <- array_reshape(leo, dim = c(1, 200, 200, 3))
proba_leo=model %>% 
  predict_proba(leo)
class_leo=model%>%
  predict_classes(leo)
class_leo=class_names[class_leo+1]
pred_leo=as.data.frame(proba_leo)%>%
  mutate(Prob_Class=class_leo)

Iegor=image_load(here::here("selfie-image-pred/Iegor.jpg"))
Iegor=image_array_resize(Iegor, width = 200, height = 200)
Iegor=Iegor/255
Iegor <- array_reshape(Iegor, dim = c(1, 200, 200, 3))
pred_Iegor=model %>%
  predict_proba(Iegor)
class_Iegor=model%>%
  predict_classes(Iegor)
class_Iegor=class_names[class_Iegor+1]
pred_Iegor=as.data.frame(pred_Iegor)%>%
  mutate(Prob_Class=class_Iegor)

child=image_load(here::here("selfie-image-pred/18-.jpg"))
child=image_array_resize(child, width = 200, height = 200)
child=child/255
child <- array_reshape(child, dim = c(1, 200, 200, 3))
proba_child=model %>% 
  predict_proba(child)
class_child=model%>%
  predict_classes(child)
class_child=class_names[class_child+1]
pred_child=as.data.frame(proba_child)%>%
  mutate(Prob_Class=class_child)

table_pred= rbind(pred_thomas,
                  pred_leo,
                  pred_Iegor,
                  pred_child) 

table_pred= table_pred %>%
              rename(Prob_Adult= V1, Prob_Minor= V2 )

table_pred=table_pred%>% 
  mutate(Picture_of=c("Thomas", "Léo", "Iegor", "Child")) %>% 
  select(Picture_of, Prob_Adult,Prob_Minor,Prob_Class )


table_pred %>% write.csv(file = here::here("table_pred.csv"))

