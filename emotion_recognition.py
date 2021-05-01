# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:10:14 2021

@author: vyass
"""

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
            
        return result
    
#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob(r"speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        # print(file)
        file_name=os.path.basename(file)
        # print(file_name)
        emotion=emotions[file_name.split("-")[2]]
        # print(emotion)
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)        
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def load_file(path):
    file_name=os.path.basename(path)
    emotion=emotions[file_name.split("-")[2]]
    if emotion not in observed_emotions:
        print("Emotion not trained")
        return None,None
    feature=extract_feature(path, mfcc=True, chroma=True, mel=True)
    return feature,emotion

# print(emotion)

#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=300, epsilon=1e-08, hidden_layer_sizes=(350,), learning_rate='adaptive', max_iter=500)

#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Model is trained.")
print("Accuracy: {:.2f}%".format(accuracy*100))
print(observed_emotions)



#Predict Emotion
#2,3,6,7 trained for these files
path = input("Enter path to the audio file :-")
feature,emotion = load_file(path)

predict = model.predict(feature.reshape(1,-1))
print(f"Predicted emotion is {predict} and actual emotion is {emotion}")
