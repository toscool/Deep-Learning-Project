library(keras)
library(here)
library(cloudml)

# 0. Define hyperparameter flags
#-------------------------------------------------------------------------------

FLAGS <- flags(
  flag_numeric("n_nodes", 300),
  flag_numeric("rate", 0.5),
  flag_numeric("lr", 0.000004)
)
#-------------------------------------------------------------------------------
generator_augmented <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 30,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)


train <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://face_age/data/train"),
  generator = generator_augmented,
  target_size = c(200, 200),
  batch_size = 8,
  shuffle = TRUE, 

)

valid <- flow_images_from_directory(
  directory = gs_data_dir_local("gs://face_age/data/validation"),
  generator = image_data_generator(rescale = 1 / 255),
  target_size = c(200, 200),
  batch_size = 8,
  shuffle = TRUE,
 
)

test= flow_images_from_directory(
  directory = gs_data_dir_local("gs://face_age/data/test"),
  generator = image_data_generator(rescale = 1 / 255),
  target_size = c(200, 200),
  batch_size = 8,
)

#-------------------------------------------------------------------------------

model_base <- application_vgg16(
  include_top = FALSE,
  weights = "imagenet",
  input_shape = c(200, 200, 3)
)

freeze_weights(model_base)

#-------------------------------------------------------------------------------

model <- keras_model_sequential() %>%
  model_base %>%
  layer_flatten() %>%
  layer_dense(units = FLAGS$n_nodes, 
              activation = "relu",
              ) %>%
  layer_dropout(rate = FLAGS$rate) %>%
  
  
  layer_dense(units = FLAGS$n_nodes, 
              activation = "relu",
              ) %>%
  layer_dropout(rate = FLAGS$rate) %>%
  
  
  layer_dense(units = 2, activation = "softmax")

#-------------------------------------------------------------------------------

model %>% compile(
  optimizer = optimizer_rmsprop(lr = FLAGS$lr),
  loss = 'categorical_crossentropy',
  metric = "accuracy"
)

model %>% fit_generator(
  generator = train,
  steps_per_epoch = train$n / train$batch_size,
  epochs = 100,
  callbacks = callback_early_stopping(patience = 7,
                                      restore_best_weights = TRUE),
  validation_data = valid,
  validation_steps = valid$n / valid$batch_size
)

unfreeze_weights(object = model_base, from = "block5_conv1")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = FLAGS$lr),
  loss = 'categorical_crossentropy',
  metric = "accuracy"
)

model %>% fit_generator(
  generator = train,
  steps_per_epoch = train$n / train$batch_size,
  epochs = 100,
  callbacks = callback_early_stopping(patience = 5,
                                      restore_best_weights = TRUE),
  validation_data = valid,
  validation_steps = valid$n / valid$batch_size
)



score= evaluate_generator(model, generator = test, steps = test$n / test$batch_size )

model%>% save_model_hdf5(here::here("results/model.hdf5"))

