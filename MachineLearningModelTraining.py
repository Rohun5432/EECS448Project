import numpy as np
import pandas as pd
import sklearn
import json
import spacy
import nltk
import time
import os
import shutil
import math
import sns

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import seaborn as sns
from datetime import datetime
from scipy.io.wavfile import read
from python_speech_features import mfcc
from sklearn import preprocessing
from scipy.io import wavfile
from collections import Counter
import torch
import torchtext
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from torchmetrics.classification import BinaryAccuracy,MulticlassF1Score,MulticlassAccuracy, BinaryF1Score
from sklearn.metrics import confusion_matrix

from transformers import logging


logging.set_verbosity_warning()

# Loads in Stopwords
StopWords = set(stopwords.words('english'))


# loads in spacy Glove 
nlp = spacy.load("en_core_web_lg")

#  Loads in Pre-Trained BERT model
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)




# Previous work (not usefull)
"""
def CombineRawText():
    lis= []
    with open('Names.txt') as f:
          for line in f:
              key = "MFCC/MFCCFeatures"+line.strip() + "Features.csv"
              df = pd.read_csv(key)
              lis.append(df)
          df = pd.concat(lis, ignore_index=True)
          df.to_csv("MFCC.csv")




def preprocessing(fileText, fileAudio):
    df = pd.read_csv(fileText)
    Text = df['Transcript at Timestamp']
    text_emb = []

    for text in Text.values:
            Upd = text.lower()
            Upd = word_tokenize(Upd)
            temp = torch.zeros((len(Upd),300))
            for i in range(len(Upd)):
                temp[i] = torch.from_numpy(nlp(Upd[i]).vector)
            text_emb.append(temp)

    text_emb = torch.nn.utils.rnn.pad_sequence(text_emb, batch_first=True, padding_value=0.0)

    return text_emb
"""


# Converts the sentance to replace any newlines, lowers each sentance, takes out all punctuation, essentially preprocessinf
def ConvertSentance(sentance):
    sentance = sentance.replace('\n', ' ')
    sentance = sentance.lower()
    punc = '''!()-[]{};:'"\/,.?@#$%^&*_~'''
    for element in sentance:
        if element in punc:
            sentance = sentance.replace(element,"")
    Lis = word_tokenize(sentance)
    Lis = [w.lower() for w in Lis if w.lower() not in StopWords and len(w) != 0]
    return Lis



# Data wrapper for Prediction 
class DatasetMaper(Dataset):

   def __init__(self, data):
      self.df = data
      
   def __len__(self):
      return len(self.df)
      
   def __getitem__(self, idx):

        df = self.df.iloc[idx]
        text = df['X']
        text = ConvertSentance(text)

        text_emb = torch.zeros((len(text),300))
        for i in range(len(text)):
             text_emb[i] = torch.from_numpy(nlp(text[i]).vector)

        labels = torch.tensor(df['Y'])

        return text_emb,labels



# TextCNN model for Prediction
class TextCNN(nn.Module):
    def __init__(self,kernel_size, num_channel):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(300, num_channel, k,
            padding=k // 2) for k in kernel_size])

        cnn_out_dim = (len(kernel_size) * num_channel)

        self.Linear = nn.Linear(cnn_out_dim,2)


        # adding another hidden layer and a dropfitting layer, for more depth and regularization


    def forward(self, x):
        output = None
        
        output = torch.transpose(x, 1, 2)
        output = [F.relu(conv(output)) for conv in self.convs]
        output = [i.max(dim=2)[0] for i in output]
        output = torch.cat(output, 1)

        output = self.Linear(output)

        
        return output






# Work with written classification, this was use of a BERT model that would use the output from a BERT last hidden state and try to feed into a TEXTCNN model
class BertTextCNN(nn.Module):
    def __init__(self,kernel_size, num_channel):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(768, num_channel, k,
            padding=k // 2) for k in kernel_size])

        cnn_out_dim = (len(kernel_size) * num_channel)

        self.Linear1 = nn.Linear(cnn_out_dim,4)


        # adding another hidden layer and a dropfitting layer, for more depth and regularization


    def forward(self, x):
        output = None
        
        output = torch.transpose(x, 1, 2)
        output = [F.relu(conv(output)) for conv in self.convs]
        output = [i.max(dim=2)[0] for i in output]
        output = torch.cat(output, 1)

        output = self.Linear1(output)

        
        return output


# Padded sequences used for when calling a function for dataloader
def Padd(batch):
    text_emb = []
    labels = []
    for i in batch:
        text_emb.append(i[0]) # dialogue
        labels.append(i[1])
        
    text_emb = torch.nn.utils.rnn.pad_sequence(text_emb, batch_first=True, padding_value=0.0)
    return text_emb,labels


# Written Humor classification Neural Net using the output of BERT model

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()


        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        # 2/3

        self.drop = nn.Dropout(p=0.15)
        #self.input = nn.Linear(768,512)
        self.out = nn.Linear(512,4)
        
        #self.text = BertTextCNN([3,4,5],600)


        # adding another hidden layer and a dropfitting layer, for more depth and regularization


    def forward(self, input_ids, attention_mask):
        output = None
        
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]

        
        output = self.drop(pooled_output)

        output = self.out(output)

        #output = F.relu(self.input(output))
        #output = self.out(output)

        
        return output


# Useless work, did not amount to anything
"""
def Process():
    file = "dataset.csv"
    df = pd.read_csv(file)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in range(len(df['humor'])):
        if df.at[i,'humor'] == False:
            df.at[i,'humor'] = 0
        else:
            df.at[i,'humor'] = 1
        
        df.at[i,'text'] = df.at[i,'text'].lower()
        
 
        for ele in df.at[i,'text']:
            if ele in punc:
                    df.at[i,'text'] =  df.at[i,'text'].replace(ele, "")

    df.to_csv("Pro.csv")
"""


# created the different train, val, and test sets for our final data used
def CreateDistinctionsOnlyForFinalFeatures():
    # Process data.csv
    file = "Data/FinalData/FinalFeatures.csv"
    df = pd.read_csv(file)




    train,test = sklearn.model_selection.train_test_split(df, test_size=0.20, random_state=42)

    train, val = sklearn.model_selection.train_test_split(train, test_size=0.20, random_state=42)

  

    train.to_csv("TrainFinal.csv")

    val.to_csv("ValFinal.csv")

    test.to_csv("TestFinal.csv")



# dataclass wrapper for BERT, used for written classification transfer learning tests, would, padded and truncated sequences to different lengths
class BERTClass(Dataset):
  def __init__(self, data):
       self.df = data

  def __len__(self):
       return len(self.df)

  def __getitem__(self, idx):

       df = self.df.iloc[idx]
       text = df['X']

       labels = torch.tensor(df['Y'])

       encoding = tokenizer.encode_plus(text, max_length=25, truncation=True, padding='max_length', add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt',) # from 250 to 25 to 45 bact to 25


       return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten(), labels




# Data wrapper for final features dataset 
class FinalClass(Dataset):
  def __init__(self, data):
       self.df = data

  def __len__(self):
       return len(self.df)

  def __getitem__(self, idx):

      df = self.df.iloc[idx]

      features = torch.zeros(20)

      for i in range(20):
          features[i] = df[str(i)]

      labels = torch.tensor(df['Score'])


      return features,labels



# Final Neural Network model
class FinalNet(nn.Module):
    def __init__(self):
        super().__init__()


       
        self.one = nn.Linear(20,200) # 100 and 150, found best at 200
        self.two = nn.Linear(200,5)
      

    def forward(self,features):
        output = None
        
      

        output = F.relu(self.one(features))
        output = self.two(output)

        
        return output



# This is the Main function where we trained classes and saw performance in validation data, if we wanted to train any model we would make some modifications before trainign the model, right now this is set to train the FinalModel 

def main():

    #4 min and 1- min for power settings

    TrainingLoss, ValidationLoss, TrainingAccuracy, ValidationAccuracy, TrainingF1, ValidationF1,best_train_loss,best_val_loss = [],[],[],[],[],[],[],[]

    Trainfile = "Data/FinalData/TrainFinal.csv"
    ValidationFile = "Data/FinalData/ValFinal.csv"
    TestingFile = "Data/FInalData/TestFinal.csv"


    epochs = 150

    TrainData = pd.read_csv(Trainfile)
    ValidationData = pd.read_csv(ValidationFile)
    TestingData = pd.read_csv(TestingFile)

   
    train = FinalClass(TrainData)
    val = FinalClass(ValidationData)
    test = FinalClass(TestingData)

    #train =  BERTClass(TrainData)
    #val = BERTClass(ValidationData)
    #test = BERTClass(TestingData)

    print("Finishes Wrapper")

    loader_train = DataLoader(train, batch_size=32, shuffle=True)
    loader_val = DataLoader(val, batch_size=32, shuffle=True)
    loader_test = DataLoader(test, batch_size=32, shuffle=True)

    #loader_train = DataLoader(train, batch_size=32, shuffle=True)
    #loader_val = DataLoader(val, batch_size=32, shuffle=True)
    #loader_test = DataLoader(test, batch_size=32, shuffle=True)

    print("Finished DataLoader")


    #model= TextCNN([3,4,5],250)
    model= FinalNet()
    #model = NeuralNet()

    print("Model Initialization finished")


    class_weights = torch.tensor([(474/5 * 1216),  (139/5 * 1216), (200/5 * 1216),(240/5 * 1216), (163/5 * 1216)])

    criterion = nn.CrossEntropyLoss(ignore_index=-1,weight = class_weights) # larger learning rate
    accuracy = MulticlassAccuracy(num_classes=5, ignore_index=-1)
    f1 = MulticlassF1Score(num_classes = 5, ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr= 2e-4)
    

    print("____________________________ Training Begins ___________________________________")
    for e in range(epochs):
        t_start = time.time()
        print("Starting epoch : " + str(e))

        loss = 0
        y_true = []
        y_pred = []

        num_iter = 0
        # iteration 2600

        # loader_train has 4000

        global_min_loss = 100
        counter = 0
        Best_iter = 0
        
        Averageloss = []


        for text, labels in loader_train:


            optimizer.zero_grad()
            output = model(text)
            #labels = torch.stack(labels)

            loss = criterion(output,labels.long())
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)

            y_true.append(torch.flatten(labels))
            y_pred.append(torch.flatten(predictions))
            
            Averageloss.append(loss.item())

            if loss.item() < global_min_loss:
                global_min_loss = loss.item()


            if num_iter % 100 == 0:
                print("_____________________________")
                print("iteration "+ str(num_iter))
                print("Global min loss = " + str(global_min_loss))
                print("Training Loss " + str(loss.item()))
                print("_____________________________")

            num_iter += 1

          


        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        TrainingLoss.append(np.mean(np.array(Averageloss)))
        TrainingAccuracy.append(accuracy(y_true, y_pred).item())
        TrainingF1.append(f1(y_true, y_pred).item())
        best_train_loss.append(global_min_loss)

        print("Training Accuracy " + str(accuracy(y_true, y_pred).item()))
        print("Training F1 " + str(f1(y_true, y_pred).item()))
      

        y_true = []
        y_pred = []
        num_iter = 0
        Averageloss = []

        print("Training loop finished")


        global_val_loss = 100
        counter = 0

        for text, labels in loader_val:
            output = model(text)
            #labels = torch.stack(labels)

            loss = criterion(output, labels.long())
            predictions = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)
            y_true.append(torch.flatten(labels))
            y_pred.append(torch.flatten(predictions))

            Averageloss.append(loss.item())


            if loss.item() < global_val_loss:
                global_val_loss = loss.item()
       
             


            if num_iter % 100 == 0:
                print("_____________________________")
                print("iteration "+ str(num_iter))
                print("Global min loss = " + str(global_val_loss))
                print("Validation Loss " + str(loss.item()))
                print("_____________________________")

            num_iter += 1

            
           
            
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        ValidationLoss.append(np.mean(np.array(Averageloss)))
        ValidationAccuracy.append(accuracy(y_true, y_pred).item())
        ValidationF1.append(f1(y_true, y_pred).item())
        best_val_loss.append(global_val_loss)

        print("Validation Accuracy " + str(accuracy(y_true, y_pred).item()))
        print("Validation F1 " + str(f1(y_true, y_pred).item()))
        

        print("Validation looop finished")




        # Include Testing Part and at the very end remember to save the model TODO



        print("-----------------------Final Results----------------------")
        print("------------------Stats----------------")
        print("Training Loss " + str(TrainingLoss[e]))
        print("Validation Loss " + str(ValidationLoss[e]))
        print("Training Accuracy " + str(TrainingAccuracy[e]))
        print("Validation Accuracy " + str(ValidationAccuracy[e]))
        print("Training F1 " + str(TrainingF1[e]))
        print("Validation F1 " + str(ValidationF1[e]))
        print("------------------Stats----------------")
    

        PATH = "SavedModels/FinalWeight/modelFinal" + str(e) + ".pth"

        torch.save(model.state_dict(), PATH)



        t_end = time.time()

        print('Epoch lasted {0:.2f} minutes'.format((t_end - t_start)/60))

    

    X = range(epochs)


    print("Training Loss")
    plt.plot(X,TrainingLoss)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.savefig('TrainingLoss.png')
    plt.clf()

    print("Validation Loss")
    plt.plot(X,ValidationLoss)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.savefig('ValidationLoss.png')
    plt.clf()

    print("Training Accuracy")
    plt.plot(X,TrainingAccuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.savefig('TrainingAccuracy.png')
    plt.clf()

    print("Validation Accuracy")
    plt.plot(X,ValidationAccuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.savefig('ValidationAccuracy.png')
    plt.clf()

    print("Training F1")
    plt.plot(X,TrainingF1)
    plt.xlabel("Epochs")
    plt.ylabel("Training F1")
    plt.savefig('TrainingF1.png')
    plt.clf()

    print("Validation F1")
    plt.plot(X,ValidationF1)
    plt.xlabel("Epochs")
    plt.ylabel("Validation F1")
    plt.savefig('ValidationF1.png')
    plt.clf()

    print("Best Train loss")
    plt.plot(X,best_train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Best train loss")
    plt.savefig('TrainingLossBest.png')
    plt.clf()

    print("Best validation loss")
    plt.plot(X,best_val_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Best validation loss")
    plt.savefig('ValidationLossBest.png')
    plt.clf()

    #best_val_loss 

# run functions




# Datawrapper used when working with the written humor dataset without Transfer learning of preidiction or any features, just the text
class ComedianData(Dataset):

   def __init__(self, data):
      self.df = data
      
   def __len__(self):
      return len(self.df)
      
   def __getitem__(self, idx):

        df = self.df.iloc[idx]
        text = df['Transcript at Timestamp']
        text = word_tokenize(text)
        text_emb = torch.zeros((len(text),300))
        for i in range(len(text)):
             text_emb[i] = torch.from_numpy(nlp(text[i]).vector)

        labels = torch.tensor(df['Score'])

        return text_emb,labels


# this was used when I was trying to see how well the Prediction model could predict humor
def Customdf():
    file = "Data/FInalData/Combined.csv"
    df = pd.read_csv(file)

    for i in range(len(df['Score'].values)):
        if df.at[i,'Score'] > 0:
            df.at[i,'Score'] = 1

    df.to_csv("Changed.csv")






# this is how the data is tested
def testingOnData(PATH):

    AverageLoss = []
    y_true = []
    y_pred = []

    file = "Data/FinalData/TestFinal.csv"
    testdf = pd.read_csv(file)

    #model = TextCNN([3,4,5],250)
    model = FinalNet()
    model.load_state_dict(torch.load(PATH))
    model.eval()


    accuracy = MulticlassAccuracy(num_classes = 5, ignore_index=-1)
    f1 = MulticlassF1Score(num_classes = 5, ignore_index=-1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) 

    test =  FinalClass(testdf)
    loader_test = DataLoader(test, batch_size=32, shuffle=True)
    
    print("Finished loading model")
    iter = 0
    print(len(loader_test))
    for text,labels in loader_test:
        if iter % 10 == 0:
            print("Iteration " + str(iter))

        output = model(text)
        #labels = torch.stack(labels)

        loss = criterion(output, labels.long())

        AverageLoss.append(loss.item())

        predictions = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)
        y_true.append(torch.flatten(labels))
        y_pred.append(torch.flatten(predictions))
        iter += 1


    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    matrix = confusion_matrix(y_true.numpy(), y_pred.numpy())

    df_cm = pd.DataFrame(matrix / np.sum(matrix, axis=1)[:, None], index = [i for i in range(5)],
                     columns = [i for i in range(5)])

    ax = plt.subplot()
    sns.heatmap(df_cm, annot=True)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels');
    plt.savefig('output.png')

    Accuracy = accuracy(y_true, y_pred).item()
    F1 = f1(y_true, y_pred).item()

    print("Average Accuracy " + str(Accuracy))
    print("Average F1 Score " + str(F1))
    print("Average Loss " + str(np.mean(np.array(AverageLoss))))




# Cleaning the written jokes dataset
def CleaningData():
    df = pd.read_csv('jokes.csv')

    df = df.drop(df[ (df['selftext'] == '[removed]') |  (df['selftext'] == '[deleted]')].index, inplace=False)

    print("Dropping finished")

    Features = {}
    Features['text'] = df['title'] + " " + df['selftext']
    Features['score'] = df['score']

    print("Feature extractiion finished")

    New = pd.DataFrame(Features)

    New.to_csv("CleanJokes.csv")

    print("Finished!")



# distributing data into classes so that it can better fit our goal of classification through transfer learning
def ChangingData():
    df = pd.read_csv('Unecessary/CleanJokes.csv')

    scores = np.array(df['score'])

    Data =  df.loc[ (df['score'] < 1) & (df['text'].str.len() <= 250)]
    Zero = Data.sample(n=11000, random_state=448)
    Zero['score'] = 0

    Data =  df.loc[ (df['score'] < 5) & (df['score'] >= 1) & (df['text'].str.len() <=250)]
    One = Data.sample(n=11000, random_state=448)
    One['score'] = 1

    Data =  df.loc[ (df['score'] < 16) & (df['score'] >= 5) & (df['text'].str.len() <= 250)]
    Two = Data.sample(n=11000, random_state=448)
    Two['score'] = 2

    Data =  df.loc[ (df['score'] <= 142733) & (df['score'] >= 16) & (df['text'].str.len() <= 250)]
    Three = Data.sample(n=11000, random_state=448)
    Three['score'] = 3

    Finaldf = pd.concat([Zero, One,Two, Three], ignore_index=True)

    print(len(Zero))
    print(len(One))
    print(len(Two))
    print(len(Three))

    Finaldf = Finaldf. dropna()

    X = Finaldf['text']
    y = Finaldf['score']

    print("sampling + combining finished")

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=42)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    train = {}
    val = {}
    test = {}

    train['X'] = X_train
    train['Y'] = y_train

    val['X'] = X_val
    val['Y'] = y_val

    test['X'] = X_test
    test['Y'] = y_test

    temp = pd.DataFrame(train)
    temp.to_csv("Train.csv")

    temp = pd.DataFrame(val)
    temp.to_csv("Val.csv")

    temp = pd.DataFrame(test)
    temp.to_csv("Test.csv")
    
    print("Train, val, and test created")

    # min value of 0.0
    # 25 percentile of 1.0
    # 50 percentile of 5.0
    # 75% of 16.0
    # 100% of 142733.0


# this was just to test how to split the data using percentiles
"""
    print(np.percentile(scores,0))
    print(np.percentile(scores,25))
    print(np.percentile(scores,50))
    print(np.percentile(scores,75))
    print(np.percentile(scores,100))


    print(len(df['score'] < 1)) # 578637
    print(len( (df['score'] < 5) & (df['score'] >= 1) ))
    print(len( (df['score'] < 16) & (df['score'] >= 5) ))
    print(len( (df['score'] <= 142733) & (df['score'] >= 16) ))
"""






# This was a test to see how the Bert encoder would work
def BertEncodingCheck():
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

    encoding = tokenizer.encode_plus(
        sample_txt,
        max_length=32,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
)

    # encoding keys
    # input_ids, attention_mask

    print(len(encoding['input_ids'][0]))
    print(encoding['input_ids'][0])

    # this is how the tokenization process will happen
    # will tokenize the text
    # will convert tokens into unique token ids
    # tokens = tokenizer.tokenize(sample_txt)
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(tokens)
    # print(token_ids)




# Converts data into CSv file for FinalFeatures
def ConvertCSV():



    enc = OneHotEncoder(handle_unknown='ignore',categories=[['normal','laughing','annoyed','serious','sad','angry']])



    df = pd.read_csv('Data/FinalData/Combined.csv')



    data = {}
    
    data['0'] = []
    data['1'] = []
    data['Score'] = []
    data['BP'] = []

    data['walking'] = list(np.zeros(len(df)))
    data['standing'] = list(np.zeros(len(df)))
    data['speedwalking'] = list(np.zeros(len(df)))
    data['crouching'] = list(np.zeros(len(df)))
    data['turning'] = list(np.zeros(len(df)))
    data['dancing'] = list(np.zeros(len(df)))
    data['miming'] = list(np.zeros(len(df)))
    data['singing'] = list(np.zeros(len(df)))
    data['laying down'] = list(np.zeros(len(df)))


    data['normal'] = list(np.zeros(len(df)))
    data['laughing'] = list(np.zeros(len(df)))
    data['annoyed'] = list(np.zeros(len(df)))
    data['serious'] = list(np.zeros(len(df)))
    data['sad'] = list(np.zeros(len(df)))
    data['angry'] = list(np.zeros(len(df)))
    data['amused'] = list(np.zeros(len(df)))
    data['smiling'] = list(np.zeros(len(df)))


    model = TextCNN([3,4,5],250)
    PATH = "SavedModels/Prediction/Prediction1.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    for j in range(len(df)):


        if j % 10 == 0:
            print("Iteration "+ str(j))

        Newdf = df.iloc[j]

        text = Newdf['Transcript at Timestamp']
        text = word_tokenize(text)
        text_emb = torch.zeros((1,len(text),300))
        for i in range(len(text)):
                text_emb[0][i] = torch.from_numpy(nlp(text[i]).vector)

       

        output = model(text_emb)

        data['0'].append(output[0][0].item())
        data['1'].append(output[0][1].item())
        data['Score'].append(Newdf['Score'])

        if Newdf['buildup vs punchline'].strip() == "buildup":
            data['BP'].append(0)
        else:
            data['BP'].append(1)


        data[Newdf['action of comedian at timestamp'].strip()][j] = 1

        data[Newdf['emotion of comedian at timestamp'].strip()][j] = 1

    final = pd.DataFrame(data)
    final.to_csv('Data/FinalData/FinalFeatures.csv')






# Created length of text distribution plot
def DistributionPlot():
    df = pd.read_csv('Data/PredictionData/Train.csv')
    df1 = pd.read_csv('Data/PredictionData//Val.csv')
    df2 = pd.read_csv('Data/PredictionData/Test.csv')

    df = pd.concat([df, df1, df2])

    token_lens = []

    Max = 0



    for text in df['X'].values:
        tokens = word_tokenize(text)
        token_lens.append(len(tokens))
        Max = max(Max, len(tokens))


    sns.distplot(token_lens)
    plt.xlim([0, Max]);
    plt.xlabel('Token count');

    plt.show()


