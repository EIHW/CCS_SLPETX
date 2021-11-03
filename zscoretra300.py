#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:12:45 2021

@author: tianhao
"""



import wave
import numpy as np
import python_speech_features as ps
import os
# import glob
import pickle
import librosa

train_path = "/home/tianhao/yth/dist_conf/data/train/"
# test_path = "/home/tianhao/yth/dist_conf/data/test/"
# dev_path = "/home/tianhao/yth/dist_conf/data/dev/"
# root_path = "/home/tianhao/yth/dist_conf/data/"

# negnum = 215
# posnum = 71
train_emt = {'neg':0,'pos':0}
# train_label = np.empty((train_num,1), dtype = np.int8)

def generate_label(emotion,classnum):
    label = -1
    if (emotion == 'neg'):
        label = 0
    elif (emotion == 'pos'):
        label = 1

    return label

# def read_file(filename):
#     file = wave.open(filename,'r')    
#     params = file.getparams()
#     nchannels, sampwidth, framerate, wav_length = params[:4]
#     str_data = file.readframes(wav_length)
#     wavedata = np.fromstring(str_data, dtype = np.short)
#     #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
#     time = np.arange(0,wav_length) * (1.0/framerate)
#     file.close()
#     return wavedata, time, framerate

def abstract_line(line):
#    print(line)
    words = line.split('.')[0].split('_')[1]
#    print(len(words))
#    print(words)
    # path = words[1]
    label = words

    return label

def read_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate
 
# def read_ComparE():
    # trnum = 286
    # tenum = 208
    # denum = 231
    # train_num = None
    # test_num = None
    # dev_num = None
    
    # train_num = 403
filter_num = 40

traindata1 = []
traindata2 = []
traindata3 = []
train_num = 0

train_datalist = []
for speaker in os.listdir(train_path):
    path = train_path + speaker
    # print(speaker)
    label = abstract_line(speaker)
    if (label == 'neg') or (label == 'pos'):
        train_datalist.append([path,label])
# print (train_datalist)
for i, datalist in enumerate (train_datalist):
    path = datalist[0]
    label = datalist[1]
    data, time, rate = read_file(path)
    mel_spec = ps.logfbank(data,rate,nfilt = filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    # y2, sr2 = librosa.load(path, sr = 16000)
    # y3 = librosa.feature.melspectrogram(y=y2, sr= 16000, hop_length= 256, ,nfft = n_mels= 40)
    # y3 = librosa.power_to_db(y3)
    # y4 = librosa.feature.delta(y3)
    # y5 = librosa.feature.delta(y4)
    # mel_spec = np.transpose(mel_spec)
    # delta1 = np.transpose(delta1)
    # delta2 = np.transpose(delta2)
            
    
    traindata1.append(mel_spec)
    traindata2.append(delta1) 
    traindata3.append(delta2)
    
    train_num = train_num + 1
    

 
print (train_num)
total1 = np.concatenate(traindata1,axis=0)
total2 = np.concatenate(traindata2,axis=0)
total3 = np.concatenate(traindata3,axis=0)
mean1 = np.mean(total1,axis=0)#axis=0纵轴方向求均值
std1 = np.std(total1,axis=0)
mean2 = np.mean(total2,axis=0)#axis=0纵轴方向求均值
std2 = np.std(total2,axis=0)
mean3 = np.mean(total3,axis=0)#axis=0纵轴方向求均值
std3 = np.std(total3,axis=0)
    
    
output = './zscoretra300'+str(filter_num)+'.pkl'
#output = './IEMOCAP'+str(m)+'_'+str(filter_num)+'.pkl'
f=open(output,'wb') 
pickle.dump((mean1,std1,mean2,std2,mean3,std3),f)
f.close() 
    
#     return
        
    
# if __name__=='__main__':
#     read_ComparE()    
    
    
    
    
    
    
    
    
    
    

    