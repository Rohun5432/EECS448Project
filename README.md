# EECS448Project

Note:
This was the following structure of the Project directory, but GitHub won't allow for all of it to be uploaded, so the important files are uploaded to Github where Data is our annotated Data and Prediction Data is the 200k Kaggle Prediction Humor Dataset

Data Folder
    The Raw Data folder contains all of the folders for all collected data for standup humor. Each folder contains the original youtube audio wav file, the csv file, and an wav directory folder named from 0 to n, n being proportional to the number of entries for the particular data. The Rjokes data folder contains the train, val, and test for the RJokes written humor dataset, split into four classes that represent the humor of the joke. This was used when trying to use written classification as a task for transfer learning. Warning: This dataset contains jokes that are offensive, so read with caution. The Prediction Data folder contains the train, val, and test for the Kaggle 200k humor prediction dataset. Used for the task of transfer learning using prediction humor. FinalData folder contains a Combined.csv that combines all the csv files from RawData, a Changed.csv that was used in the task to see how the Prediction transfer learning model could be used to predict humor in our annotated data: 0 for not funny and 1 for funny, a FinalFeatures.csv, the features for our dataset that were directly fed into the final model, a MFCC.csv that holds all the mfcc matrices for all data entries, and a TrainFinal, ValFinal, and TestFinal for our datasetthat were used to train and test a final model.
    
Results Folder 
    holds all the results, graphs and performance metrics, of the final model, the final model with weights,the prediction model, and the failed written humor transfer learning attempt.


Names.txt 
    the text file with all the folder names in RawData it was used in DataFeaturesConvert.py to automate the process of converting features into data that could be fed through the model through our experiments.
    
SavedModels
    The Final folder holds all the saved models at each epoch for Final, Final with Weight, and the Prediciton machine learning models 
    
UnecessaryPreviousWorkFolder
    Contains all saved previous work that was deemed unecessary in the final model
    
DataFeaturesConvert.py
    Python File that was used in the initial processing of data, and contains all code that was used in testing how to convert the data into features to be used in a machine learning model.

MachineLearningModelTraining.py
    Python File that was used to train, validate, and test any of the machine learning models that were used in the course of the project.

