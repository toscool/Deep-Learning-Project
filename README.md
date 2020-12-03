# Deep-Learning-Project
Deep learning project during my master degree. The aim of the project was to recognize if a person is over 18 years old or not.

•	I achieved 91% accuracy while training CNNs on 10000 facial images to predict whether a person is older than 18 years old or not. 

•	The model used transfer learning with VGG16 pretrained model in a fine-tuning approach. In order to find the best accuracy, I augmented the training set and launched a grid search of 125 combinations including different nodes, learning rates, dropout rates (activation functions ReLu, Softmax).




The data file is too large for github but can be found here: https://www.kaggle.com/frabbisw/facial-age (just have to split the data into +18 and -18 and split
the dataset into 3 folders, test, train, validation with respectively 2 files each time, +18 and -18)
The model in /results, and the training runs are too large for GitHub. Please contact me for further explanation: thomas@baeumlin.fr
