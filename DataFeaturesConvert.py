import numpy as np
import pandas as pd
import sklearn
import json
import spacy
import nltk
import time
import os
import shutil
from nltk.corpus import stopwords
from sklearn.mixture import GaussianMixture
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
from collections import defaultdict

# loading in stopwords and glove dimensional through spacy
StopWords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")


# Gets the dimensional embedding for a sentance
def WordVector(sentance):
    sentance = ' '.join(sentance)
    vector = nlp(sentance)
    # takes the average of all the word vectors, sets dimensions to 300
    return vector.vector
    
# Preprocessing the sentance, same as TextAudioCNN.py
def ConvertSentance(sentance):
    sentance = sentance.replace('\n', ' ')
    sentance = sentance.lower()
    punc = '''!()-[]{};:'"\/,.?@#$%^&*_~'''
    for element in sentance:
        if element in punc:
            sentance = sentance.replace(element,"")
    Lis = list(sentance.split(" "))
    index = 0
    Lis = [w.lower() for w in Lis if w.lower() not in StopWords and len(w) != 0]
    return np.array(Lis)




# Converts text to a 300 dimensioal embedding
def ConvertText(df):
    corpus = []
    df1 = df['Transcript at Timestamp']
    for i in range(len(df1)):
        sentance = ConvertSentance(df1[i])
        corpus.append(WordVector(sentance))
    corpus = np.array(corpus)
    #Changed = TSNE(random_state=448).fit_transform(corpus)
    
    #X = Changed[:,0]
    #Y = Changed[:,1]
    #sns.scatterplot(x=X,y=Y,hue=df['laughter scale (subjective scale)'])
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    #plt.show()
    return corpus
    


# One hot encoding for category
def OneHotEncodingCategory(df):

    df1 = df['category 1 of text']
    df2 = df['category 2 of text (subcategory, if applicable)']
    enc = OneHotEncoder(handle_unknown='ignore', categories=[['audience', 'imitation', 'one man conversation', 'personal', 'unexpected','stereotype','pop reference','sarcasm']])
    df1 = np.array(df1)
    df1 = np.reshape(df1,(len(df1),1))

    df2 = np.array(df2)
    df2 = np.reshape(df2,(len(df2),1))

    fitted1 = enc.fit_transform(df1).toarray()
    fitted2 = enc.transform(df2).toarray()

    data = {}

    categories = np.array(enc.categories_)[0]
   
    for i in range(len(categories)):
        data[categories[i]] = np.zeros((len(df1)))

    for i in range(len(df1)):
        for j in range(6):
            if fitted1[i][j] == 1 or fitted2[i][j] == 1:
                data[categories[j]][i] = 1
            else:
                data[categories[j]][i] = 0

    

    return data


# Grammer processing for text
def GrammerProcess(df):
    df = df.fillna("")
    for i in range(len(df['category 1 of text'])):
        df.at[i,'category 1 of text'] = df.at[i,'category 1 of text'].strip()
        df.at[i,'category 2 of text (subcategory, if applicable)'] = df.at[i,'category 2 of text (subcategory, if applicable)'].strip()
        df.at[i,'emotion of comedian at timestamp'] = df.at[i,'emotion of comedian at timestamp'].strip()
        df.at[i,'action of comedian at timestamp'] = df.at[i,'action of comedian at timestamp'].strip()
        df.at[i,'gesticulating'] = df.at[i,'gesticulating'].strip()
        df.at[i,'buildup vs punchline'] = df.at[i,'buildup vs punchline'].strip()
    return df


# Binary ENcoding for Categories
def BinaryEncoding(df,name):
    df1 = df[name]
    Key = ""
    if name == "gesticulating":
        Key = "yes"
    else:
        Key = "buildup"

    Binary = []

    for i in range(len(df1)):
        if df1[i] == Key:
            Binary.append(0)
        else:
            Binary.append(1)
    return Binary


# One hot encoding for Action and Emotion
def OneHotEncodingAE(df,name):
    df1 = df[name]
    enc = 0
    if name == 'emotion of comedian at timestamp':
        enc = OneHotEncoder(handle_unknown='ignore',categories=[['normal','laughing','annoyed','serious','sad','angry']])
    else:
        enc = OneHotEncoder(handle_unknown='ignore',categories=[['walking','standing','speedwalking','crouching','turning','dancing','miming']])
    df1 = np.array(df1)
    df1 = np.reshape(df1,(len(df1),1))
    fitted1 = enc.fit_transform(df1).toarray()
    data = {}

    categories = np.array(enc.categories_)[0]
    #print(categories)
    
    for i in range(len(categories)):
        data[categories[i]] = np.zeros((len(df1)))

    for i in range(len(df1)):
        for j in range(len(categories)):
            if fitted1[i][j] == 1:
                data[categories[j]][i] = 1
            else:
                data[categories[j]][i] = 0
    return data


# Converting Audio to seperate wav files 
def AudioConversion(df, URL,WavFolder):
    c = 3
    Time = np.array(df['Timestamp'])

    Seconds = []
    index = 0
    for i in range(len(Time)):
        pt = datetime.strptime(Time[i],'%M:%S')
        total_seconds = pt.second + pt.minute*60
        Seconds.append(total_seconds)
    Seconds.append(Seconds[len(Seconds) - 1] + c)
    Converted = []

    for i in range( (len(Seconds) - 1)):
        Converted.append((Seconds[i], Seconds[i+1]))
    
    rate, data = wavfile.read(URL) 
    for pair in Converted:

        start = pair[0]
        end = pair[1]
        start = rate * start
        end = rate * end

        # split
        test = data[start:end]

        # save the result
        wavfile.write(str(WavFolder) + "/"+ str(index) + '.wav', rate, test)
        index += 1
    return index


# COnverting to MFCC
def ConverttoMFCC(fileName,length):
    MFCCFeatures = []
    for i in range(length):
        test= str(fileName) + str(i) + ".wav"
        samplerate, signal = read(test)
        mfcc_features = mfcc(signal,
                             samplerate,
                             winlen = 0.025,
                             winstep = 0.01,
                             numcep = 13,
                             nfft = 2400)

        #temp= np.mean(mfcc_features,axis=0)
        #temp = (temp - np.mean(temp))/np.std(temp)
        

        MFCCFeatures.append(temp)
    return np.array(MFCCFeatures)



  # COnverts all the features into a csv file
def ConvertFeatures(filename,WavFolder,URL,CSV):
    if os.path.exists(WavFolder):
        shutil.rmtree(WavFolder)
    os.makedirs(WavFolder)
    #print("Directory '% s' created" % WavFolder)
    # 35 features

    # Reading filename
    df = pd.read_csv(filename)
    
    #print("Converted " + str(filename) + " into pandas dataframe.")

    length = AudioConversion(df,URL,WavFolder)

    
    #print("Audio broken into segments")
    

    fileName = WavFolder + "/"
    #length = 51
    MF = ConverttoMFCC(fileName,length)
    
    MFCC = defaultdict(list)

    for i in range(len(MF)):
        for j in range(len(MF[i])):
            MFCC["M" + str(j)].append(MF[i][j])
            


    #print("Created MFCC Features")

    

    df = GrammerProcess(df)

    #print("Finished Grammer Process of dataframe")

    Changed= ConvertText(df)
    TextData = {}
    for i in range(300):
        TextData[str(i)] = Changed[:,i]

    #print("Finished Converting text")

    CategoryData = OneHotEncodingCategory(df)
    name = 'emotion of comedian at timestamp'
    EmotionData = OneHotEncodingAE(df,name)
    name = 'action of comedian at timestamp'
    ActionData = OneHotEncodingAE(df,name)

    #print("Finished Converting Emotion and ction")

    BinaryData = {}

    name = 'gesticulating'
    BinaryData['Hands'] = BinaryEncoding(df,name)
    name = "buildup vs punchline"
    BinaryData['Type'] = BinaryEncoding(df,name)

    #print("Finished Converting gesticulating and buildup vs punchline")
    
    
    FinalFeatures = {}
    FinalFeatures.update(MFCC)
    #FinalFeatures.update(TextData)
    #FinalFeatures.update(CategoryData)
    #FinalFeatures.update(EmotionData)
    #FinalFeatures.update(ActionData)
    #FinalFeatures.update(BinaryData)
    #FinalFeatures['Score'] = np.array(df['laughter scale (subjective scale)'])
    
    lis= [(k,len(v)) for k,v in FinalFeatures.items()]
    Createddf = pd.DataFrame(FinalFeatures)
    Createddf.to_csv(str("MFCCFeatures"+ CSV))
    print("Completed Feature Extraction")
    

   
# Combines all csv files into Data.csv
def Combined(filename):
    lis = []
    for i in range(25):
        URL = filename + str(i) + ".csv"
        df = pd.read_csv(URL)
        lis.append(df)
    df = pd.concat(lis, ignore_index=True)
    df.to_csv("Data.csv")


# Combins all data into features
def main():

    with open('Names.txt') as f:
        for line in f:
            key = line.strip()
            print("Working on " + str(key))

            name = "Data/RawData/" + key + "/"
            CSV = key + ".csv"
            FeatureName = key + "Features.csv"
            WavFile = key + "Wav"
            waveURL = key + ".wav"

            filename = name + CSV
            WavFolder = name + WavFile
            URL = name + waveURL
            CSvName = FeatureName
            ConvertFeatures(filename,WavFolder,URL,CSvName)

            print("Finished" + str(key))
            print("-"*20)
        f.close()
        print("Finished all Processing")

  
# Creates MFCC.csv
def Concatenaet():
     Df = pd.DataFrame()

     with open('Names.txt') as f:
        for line in f:
            key = line.strip()

            d = pd.read_csv('Data/RawData/' + key + "/" + key + ".csv")

            fileName = "Data/RawData/" + key + "/" + key + "Wav"

            length = len(os.listdir(fileName))



            for i in range(length):
                test = "Data/RawData/" + key + "/" + key + "Wav/" +str(i) + ".wav"

                print(test)

                samplerate, signal = read(test)
                mfcc_features = mfcc(signal,
                                     samplerate,
                                     winlen = 0.025,
                                     winstep = 0.01,
                                     numcep = 13,
                                     nfft = 2400)


                df = pd.DataFrame(mfcc_features, columns = ['M0','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12'])

                Df = pd.concat([Df, df])
        

            print("Finished converting " + str(key))

        Df.to_csv('MFCC.csv')
            
        print("CSV done!")
            
